"""Single-config macro_economics + LM + KL-SFT run for condor sweep."""

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

from perfsim.core.learner import Learner
from perfsim.core.types import SUPERVISED_SCHEMA
from perfsim.learners.lm.kl_sft import KLSFTLearner
from perfsim.learners.lm.sft import SFTLearner
from perfsim.losses import MSELoss
from perfsim.models.hf_causal_lm import HFCausalLMModel
from perfsim.scenarios.at_macro import make_macro_env
from perfsim.simulator import Simulator


def _env_or(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var {name!r} not set")
    return val


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def main() -> int:
    run_tag = _env_or("RUN_TAG")
    kl_beta = _env_float("KL_BETA", 0.0)
    training_style = _env_or("TRAINING_STYLE", "sft_kl")
    base_model = _env_or("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    n_rounds = _env_int("N_ROUNDS", 5)
    k_steps = _env_int("K_STEPS", 3)
    seed = _env_int("SEED", 0)
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.environ.get("OUT_DIR", f"runs/at_macro_lm/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT")
    max_steps = _env_int("SFT_MAX_STEPS", 20)
    gen_batch_size = _env_int("GEN_BATCH_SIZE", 32)
    max_new_tokens = _env_int("MAX_NEW_TOKENS", 8)
    sft_batch_size = _env_int("SFT_BATCH_SIZE", 16)
    sft_full_epoch = os.environ.get("SFT_FULL_EPOCH", "0").lower() in ("1", "true", "yes")
    yaml_name = os.environ.get("MACRO_YAML", "config_100_agents.yaml")
    lora_r = _env_int("LORA_R", 32)
    use_lora = _env_int("USE_LORA", 1) == 1
    sft_lr = _env_float("SFT_LR", 1e-5)

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag,
        "kl_beta": kl_beta,
        "training_style": training_style,
        "base_model": base_model,
        "n_rounds": n_rounds,
        "k_steps": k_steps,
        "seed": seed,
        "device": device,
        "max_steps": max_steps,
        "sft_batch_size": sft_batch_size,
        "sft_full_epoch": sft_full_epoch,
        "gen_batch_size": gen_batch_size,
        "max_new_tokens": max_new_tokens,
        "yaml_name": yaml_name,
        "lora_r": lora_r,
        "use_lora": use_lora,
        "sft_lr": sft_lr,
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

    # Logit signal_writer mirrors the covid setup: PerfsimEarningDecision
    # applies sigmoid() to platform_signal. To make consumption_propensity
    # equal the LM's emitted value `p`, we write logit(p) so
    # sigmoid(logit(p)) = p. Recovers full [0, 1] expressivity instead of
    # being squashed to [sigmoid(0), sigmoid(1)] = [0.5, 0.73].
    def logit_signal_writer(runner, preds):
        if preds.ndim == 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        p = preds.clamp(min=0.01, max=0.99)
        logit_p = torch.log(p / (1.0 - p))
        runner.state["agents"]["consumers"]["platform_signal"] = logit_p.detach().clone()

    torch.manual_seed(seed)
    print("[run] building macro env", flush=True)
    t0 = time.time()
    env = make_macro_env(
        init_seed=seed,
        yaml_name=yaml_name,
        signal_writer=logit_signal_writer,
        keep_trajectory=True,
    )
    n_agents = env.runner.state["agents"]["consumers"]["age"].shape[0]
    print(f"[run] env ready: {n_agents} agents in {time.time() - t0:.1f}s", flush=True)

    # Per-agent profile DataFrame for prompt construction. Pulls features
    # from the AT runner state; the categorical labels come from
    # populations/NYC/mapping.json (loaded via the YAML's mapping_path).
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
    print(
        f"[run] profile features: age_buckets={sorted(set(ages))} "
        f"genders={sorted(set(genders))} ethnicities={sorted(set(ethnicities))}",
        flush=True,
    )

    # Chat-formatted prompt (mirroring covid's structure, swapped for the
    # macro-recommender framing). Single source of truth for both
    # generation and SFT, so train/gen format match (avoiding the
    # mismatch trap we hit on covid).
    SYSTEM_MSG = (
        "You are a financial advisor providing personalized consumption "
        "recommendations during ongoing economic uncertainty. Your goal "
        "is to help each household balance current spending with future "
        "financial security."
    )

    def prompt_builder(profile_row, tokenizer):
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": (
                    f"Household profile: age group {profile_row['age_label']}, "
                    f"{profile_row['gender']}, {profile_row['ethnicity']}.\n\n"
                    f"Output a single number between 0 and 1 (e.g. 0.50) "
                    f"where 0 means save all income and 1 means spend all "
                    f"available assets this month."
                ),
            },
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
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=2 * lora_r,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        max_new_tokens=max_new_tokens,
        gen_batch_size=gen_batch_size,
        load_now=True,
    )
    print(
        f"[run] LM training mode: {'LoRA r=' + str(lora_r) if use_lora else 'FULL fine-tuning'}",
        flush=True,
    )
    print(f"[run] SFT learning_rate={sft_lr} max_steps={max_steps}", flush=True)
    print(f"[run] LM loaded in {time.time() - t0:.1f}s", flush=True)

    # Pre-SFT diagnostic.
    _rng = _random.Random(seed)
    _sample_idx = _rng.sample(range(n_agents), min(20, n_agents))
    _sample_prompts = [model.build_prompt(model.profile_at(i)) for i in _sample_idx]
    print("[diag] sample LM outputs (pre-SFT):", flush=True)
    _sample_texts = model._generate(_sample_prompts)
    _sample_log = []
    for _idx, _prompt, _txt in zip(_sample_idx, _sample_prompts, _sample_texts):
        _parsed = model._parse(_txt)
        _sample_log.append({"agent_idx": int(_idx), "raw_text": _txt, "parsed": float(_parsed)})
        print(f"  agent {_idx}: text={_txt!r}  parsed={_parsed:.3f}", flush=True)
    (out_dir / "diagnostic_pre_sft.json").write_text(json.dumps(_sample_log, indent=2))

    if sft_full_epoch:
        effective_max_steps = -(-n_agents // sft_batch_size)
        print(
            f"[run] SFT_FULL_EPOCH=1: max_steps={effective_max_steps} "
            f"(= ceil({n_agents}/{sft_batch_size}))",
            flush=True,
        )
    else:
        effective_max_steps = max_steps

    # TODO: refine target extractor to be genuinely performative.
    _cit_init = env.runner.state["agents"]["consumers"]
    _age_init = _cit_init["age"].squeeze().long()
    _gender_init = _cit_init["gender"].squeeze().long()
    _ethnicity_init = _cit_init["ethnicity"].squeeze().long()
    _keys = (_age_init * 100 + _gender_init * 10 + _ethnicity_init).long()
    _unique_keys, _inverse = torch.unique(_keys, return_inverse=True)
    _n_buckets = int(_unique_keys.shape[0])

    # Base recommended consumption by age:
    # younger / mid-career → recommend higher consumption (build economy)
    # retirees / high-age → recommend lower (preserve savings)
    _base = torch.where(
        _age_init >= 4, torch.tensor(0.3),   # 65+ → conservative
        torch.where(
            _age_init >= 2, torch.tensor(0.5),  # 30-64 → balanced
            torch.tensor(0.7),                   # under 30 → growth-oriented
        ),
    ).float()

    def custom_state_extractor(runner):
        # Per-round target depends on agent's static profile (base) plus a
        # small adjustment from current inflation. Higher inflation =
        # recommend less consumption (preserve purchasing power).
        # Performative loop closes because inflation depends on past
        # consumption decisions.
        consumers = runner.state["agents"]["consumers"]
        env_state = runner.state["environment"]
        # Inflation tensor shape is (1, T+1) historically; take last value.
        inflation = env_state.get("P_i")
        if inflation is not None:
            try:
                inf_scalar = float(inflation[-1][-1].item())
            except Exception:
                inf_scalar = 0.0
        else:
            inf_scalar = 0.0
        target = (_base - 0.5 * inf_scalar).clamp(0.05, 0.95)
        return {
            "x": consumers["age"].float().detach(),
            "y": target.detach().reshape(-1, 1),
            "agent_idx": torch.arange(target.shape[0]),
        }
    env._state_extractor = custom_state_extractor  # noqa: SLF001
    _y0 = custom_state_extractor(env.runner)["y"].squeeze()
    print(
        f"[run] target = consumption_aware: n_buckets={_n_buckets} "
        f"init y min={float(_y0.min()):.3f} max={float(_y0.max()):.3f} "
        f"mean={float(_y0.mean()):.3f} std={float(_y0.std()):.3f}",
        flush=True,
    )

    loss = MSELoss()
    learner_kwargs = dict(
        model=model,
        loss=loss,
        max_steps=effective_max_steps,
        per_device_batch_size=sft_batch_size,
        output_dir=str(out_dir / "trl"),
        response_template="<|im_start|>assistant\n",
        learning_rate=sft_lr,
    )
    if training_style == "sft":
        learner = SFTLearner(**learner_kwargs)
    elif training_style == "sft_kl":
        learner = KLSFTLearner(
            **learner_kwargs,
            ref_model_name=base_model,
            kl_beta=kl_beta,
        )
    elif training_style == "frozen":
        class _FrozenLearner(Learner):
            accepted_schemas = (SUPERVISED_SCHEMA,)

            def __init__(self, model, loss):
                super().__init__(model, loss)

            def train(self, data):
                print("[FrozenLearner] skipping SFT (frozen baseline)", flush=True)

            def reset(self):
                pass

        learner = _FrozenLearner(model=model, loss=loss)
    else:
        raise ValueError(f"unknown TRAINING_STYLE: {training_style!r}")

    # Per-round metrics: capture macro state and mean consumption signal.
    # TODO: add inflation, unemployment, price_of_goods once you confirm
    # the AT macro field names match what we read here.
    def _mean_assets(sim_obj) -> float:
        a = sim_obj.env.runner.state["agents"]["consumers"]["assets"]
        return float(a.float().mean().item())

    def _mean_consumption_prop(sim_obj) -> float:
        c = sim_obj.env.runner.state["agents"]["consumers"].get("consumption_propensity")
        if c is None:
            return 0.0
        return float(c.float().mean().item())

    def _inflation(sim_obj) -> float:
        pi = sim_obj.env.runner.state["environment"].get("P_i")
        if pi is None:
            return 0.0
        try:
            return float(pi[-1][-1].item())
        except Exception:
            return 0.0

    def _unemployment(sim_obj) -> float:
        u = sim_obj.env.runner.state["environment"].get("U")
        if u is None:
            return 0.0
        try:
            return float(u[-1][-1].item())
        except Exception:
            return 0.0

    sim_metrics = {
        "mean_assets": _mean_assets,
        "mean_consumption_prop": _mean_consumption_prop,
        "inflation": _inflation,
        "unemployment": _unemployment,
    }
    sim = Simulator(env=env, learner=learner, loss=loss, metrics=sim_metrics)
    print(f"[run] starting outer loop: n_rounds={n_rounds} K={k_steps}", flush=True)
    t_loop = time.time()
    hist = sim.run(n_rounds=n_rounds, epoch_size=k_steps, seed=seed)
    print(f"[run] loop done in {time.time() - t_loop:.1f}s", flush=True)

    torch.save([dict(r) for r in hist.records], out_dir / "history.pt")

    # Post-SFT diagnostic.
    print("[run] dumping final per-profile LM recommendations...", flush=True)
    t0 = time.time()
    final_features = env.runner.state["agents"]["consumers"]["age"].float()
    with torch.no_grad():
        final_preds = model(final_features).squeeze().detach().cpu()
    print(f"[run] final-pass forward in {time.time() - t0:.1f}s", flush=True)

    print(
        f"[diag] final preds stats: min={float(final_preds.min()):.4f} "
        f"max={float(final_preds.max()):.4f} mean={float(final_preds.mean()):.4f} "
        f"std={float(final_preds.std()):.4f}",
        flush=True,
    )
    _p = final_preds.clamp(min=0.01, max=0.99)
    print(
        f"[diag] consumption_propensity (= preds clamped): min={float(_p.min()):.4f} "
        f"max={float(_p.max()):.4f} mean={float(_p.mean()):.4f}",
        flush=True,
    )

    _sample_texts_post = model._generate([model.build_prompt(model.profile_at(i)) for i in _sample_idx])
    _sample_log_post = []
    print("[diag] sample LM outputs (post-SFT):", flush=True)
    for _idx, _txt in zip(_sample_idx, _sample_texts_post):
        _parsed = model._parse(_txt)
        _sample_log_post.append({"agent_idx": int(_idx), "raw_text": _txt, "parsed": float(_parsed)})
        print(f"  agent {_idx}: text={_txt!r}  parsed={_parsed:.3f}", flush=True)
    (out_dir / "diagnostic_post_sft.json").write_text(json.dumps(_sample_log_post, indent=2))

    trajectory = []
    for r in hist.records:
        theta = r.get("theta")
        row = {
            "round": int(r["round"]),
            "theta_norm": float(theta.norm().item()) if hasattr(theta, "norm") else None,
            "mean_assets": float(r["mean_assets"]),
            "mean_consumption_prop": float(r["mean_consumption_prop"]),
            "inflation": float(r["inflation"]),
            "unemployment": float(r["unemployment"]),
        }
        gap = r.get("stability_gap")
        if hasattr(gap, "item"):
            row["stability_gap"] = float(gap.item())
        trajectory.append(row)
        if wandb is not None:
            wandb.log(row)
        print(f"[round {row['round']}] {row}", flush=True)

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
