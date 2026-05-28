"""In-context performative loop on macro ABM: no fine-tuning, history-in-prompt only.

Two modes via STATIC env var:
- STATIC=1: model gets only current-state prompt each round (no memory).
- STATIC=0 (default): model gets a rolling history of prior (recommendation, outcome) pairs in its prompt.

The performative effect (if any) emerges from in-context adaptation alone.
"""

from __future__ import annotations

import json
import os
import random as _random
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _wandb = None
    _HAS_WANDB = False

from perfsim.models.hf_causal_lm import HFCausalLMModel
from perfsim.scenarios.at_macro import make_macro_env


def _env_or(name, default=None):
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var {name!r} not set")
    return val


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def main() -> int:
    run_tag = _env_or("RUN_TAG")
    base_model = _env_or("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    n_rounds = _env_int("N_ROUNDS", 20)
    k_steps = _env_int("K_STEPS", 3)
    seed = _env_int("SEED", 0)
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.environ.get("OUT_DIR", f"runs/at_macro_in_context/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT")
    gen_batch_size = _env_int("GEN_BATCH_SIZE", 32)
    max_new_tokens = _env_int("MAX_NEW_TOKENS", 16)
    yaml_name = os.environ.get("MACRO_YAML", "config_100_agents.yaml")
    n_agents_cfg = _env_int("N_AGENTS", 100)
    group_prompting = os.environ.get("GROUP_PROMPTING", "1").lower() in ("1", "true", "yes")
    consumption_noise = _env_float("CONSUMPTION_NOISE", 0.0)
    static_mode = os.environ.get("STATIC", "0").lower() in ("1", "true", "yes")
    history_window = _env_int("HISTORY_WINDOW", 10)
    calibrated_uac_path = os.environ.get("CALIBRATED_UAC", "")

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag,
        "base_model": base_model,
        "n_rounds": n_rounds,
        "k_steps": k_steps,
        "seed": seed,
        "yaml_name": yaml_name,
        "n_agents": n_agents_cfg,
        "group_prompting": group_prompting,
        "consumption_noise": consumption_noise,
        "static_mode": static_mode,
        "history_window": history_window,
        "host": os.uname().nodename,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[run] {json.dumps(config)}", flush=True)

    wandb = None
    if wandb_project and _HAS_WANDB:
        wandb = _wandb
        wandb_suffix = os.environ.get("WANDB_RUN_SUFFIX", "")
        wandb_name = f"{run_tag}{wandb_suffix}" if wandb_suffix else run_tag
        wandb.init(project=wandb_project, name=wandb_name, config=config)

    def logit_signal_writer(runner, preds):
        if preds.ndim == 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        if consumption_noise > 0:
            preds = preds + torch.randn_like(preds) * consumption_noise
        p = preds.clamp(min=0.01, max=0.99)
        logit_p = torch.log(p / (1.0 - p))
        runner.state["agents"]["consumers"]["platform_signal"] = logit_p.detach().clone()

    torch.manual_seed(seed)
    print("[run] building macro env", flush=True)
    t0 = time.time()
    env = make_macro_env(
        init_seed=seed,
        yaml_name=yaml_name,
        n_agents=n_agents_cfg,
        signal_writer=logit_signal_writer,
        keep_trajectory=True,
    )
    n_agents = env.runner.state["agents"]["consumers"]["age"].shape[0]
    print(f"[run] env ready: {n_agents} agents in {time.time() - t0:.1f}s", flush=True)

    if calibrated_uac_path:
        uac_data = torch.load(calibrated_uac_path, weights_only=False)
        for tf_key in env.runner.initializer.transition_function:
            tf = env.runner.initializer.transition_function[tf_key]
            for _, module in tf.named_modules():
                if hasattr(module, "external_UAC"):
                    n_steps_local = module.external_UAC.shape[0]
                    with torch.no_grad():
                        module.external_UAC.copy_(uac_data[:n_steps_local])
                    print(
                        f"[run] loaded calibrated UAC from {calibrated_uac_path}",
                        flush=True,
                    )
                    break

    AGE_LABELS = ["20-29", "30-39", "40-49", "50-64", "65+", "Under 19"]
    GENDER_LABELS = ["female", "male"]
    ETHNICITY_LABELS = ["white", "other", "hispanic", "black", "asian", "native"]

    ages = env.runner.state["agents"]["consumers"]["age"].squeeze().long().tolist()
    genders = env.runner.state["agents"]["consumers"]["gender"].squeeze().long().tolist()
    ethnicities = env.runner.state["agents"]["consumers"]["ethnicity"].squeeze().long().tolist()

    profiles = pd.DataFrame({
        "age_bucket": ages,
        "age_label": [AGE_LABELS[min(int(a), len(AGE_LABELS) - 1)] for a in ages],
        "gender": [GENDER_LABELS[min(int(g), 1)] for g in genders],
        "ethnicity": [ETHNICITY_LABELS[min(int(e), len(ETHNICITY_LABELS) - 1)] for e in ethnicities],
        "agent_id": list(range(n_agents)),
    })

    SYSTEM_MSG = (
        "You are a financial advisor providing personalized consumption "
        "recommendations during ongoing economic uncertainty. Your goal "
        "is to help each household balance current spending with future "
        "financial security."
    )

    history_log: list[dict] = []

    def _get_econ_state():
        env_state = env.runner.state["environment"]
        inflation = 0.0
        pi = env_state.get("P_i")
        if pi is not None:
            try:
                val = float(pi[-1][-1].item())
                if abs(val) < 1.0:
                    inflation = val
            except Exception:
                pass
        unemployment = 0.0
        u = env_state.get("U")
        if u is not None:
            try:
                row = u[-1]
                nz = row.nonzero(as_tuple=True)[0]
                if len(nz):
                    unemployment = float(row[nz[-1]].item())
            except Exception:
                pass
        price = 0.0
        p = env_state.get("P")
        if p is not None:
            try:
                price = float(p[-1][-1].item())
            except Exception:
                pass
        return inflation, unemployment, price

    def _format_history_block() -> str:
        if not history_log or static_mode:
            return ""
        recent = history_log[-history_window:]
        lines = ["Previous rounds in this simulation:"]
        for h in recent:
            lines.append(
                f"  Round {h['round']}: recommended consumption ~{h['mean_rec']:.2f}, "
                f"resulting inflation {h['inflation'] * 100:.1f}%, "
                f"unemployment {h['unemployment']:.1f}%, "
                f"price index {h['price']:.1f}, "
                f"mean assets {h['mean_assets']:.0f}."
            )
        lines.append("")
        return "\n".join(lines)

    def prompt_builder(profile_row, tokenizer):
        inflation, unemployment, price = _get_econ_state()
        history_block = _format_history_block()
        user_msg_parts = [
            f"Household profile: age group {profile_row['age_label']}, "
            f"{profile_row['gender']}, {profile_row['ethnicity']}.",
            "",
            f"Current economic conditions: "
            f"inflation {inflation * 100:.1f}%, "
            f"unemployment {unemployment:.1f}%, "
            f"price index {price:.1f}.",
        ]
        if history_block:
            user_msg_parts.extend(["", history_block])
        user_msg_parts.append(
            "Respond with ONLY a single number between 0.00 and 1.00. "
            "0 means save everything, 1 means spend everything. "
            "No explanation."
        )
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": "\n".join(user_msg_parts)},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    print(f"[run] loading LM: {base_model} on {device}", flush=True)
    t0 = time.time()
    model = HFCausalLMModel(
        base_model_name=base_model,
        profiles=profiles,
        prompt_builder=prompt_builder,
        use_lora=False,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        max_new_tokens=max_new_tokens,
        gen_batch_size=gen_batch_size,
        group_prompting=group_prompting,
        load_now=True,
    )
    print(f"[run] LM loaded in {time.time() - t0:.1f}s", flush=True)

    _rng = _random.Random(seed)
    _sample_idx = _rng.sample(range(n_agents), min(10, n_agents))
    sample_prompt = model.build_prompt(model.profile_at(_sample_idx[0]))
    print(f"[diag] sample prompt at round 0 (agent {_sample_idx[0]}):", flush=True)
    print(sample_prompt, flush=True)

    trajectory = []

    for t in range(n_rounds):
        t_round = time.time()
        with torch.no_grad():
            x = env.runner.state["agents"]["consumers"]["age"].float()
            preds = model(x).squeeze().detach().cpu()

        env.run(model, n_steps=k_steps)

        inflation, unemployment, price = _get_econ_state()
        cur_assets = env.runner.state["agents"]["consumers"]["assets"].float().detach().mean(dim=1)
        mean_assets = float(cur_assets.mean().item())
        cons_used = env.runner.state["agents"]["consumers"]["consumption_propensity"].float().detach().squeeze()
        mean_cons = float(cons_used.mean().item())

        row = {
            "round": t,
            "mean_rec": float(preds.mean().item()),
            "pred_std": float(preds.std().item()),
            "pred_min": float(preds.min().item()),
            "pred_max": float(preds.max().item()),
            "mean_consumption_prop": mean_cons,
            "inflation": inflation,
            "unemployment": unemployment,
            "price": price,
            "mean_assets": mean_assets,
            "round_seconds": time.time() - t_round,
        }
        history_log.append(row)
        trajectory.append(row)
        if wandb is not None:
            wandb.log(row)
        print(
            f"[round {t}] rec={row['mean_rec']:.3f}+-{row['pred_std']:.3f} "
            f"cons={mean_cons:.3f} infl={inflation*100:.1f}% unemp={unemployment:.1f}% "
            f"price={price:.1f} assets={mean_assets:.0f} ({row['round_seconds']:.1f}s)",
            flush=True,
        )

    (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    print(f"[run] outputs in {out_dir}", flush=True)
    if wandb is not None:
        wandb.finish()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
