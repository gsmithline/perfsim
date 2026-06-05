"""Three different small LLMs compete over the Pokec FJ population; each
fine-tunes only on the agents it served. sft_kl anchors each to its own base.
"""

from __future__ import annotations

import importlib.util
import json
import os
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

from perfsim.core.model import Model
from perfsim.environments.dynamics import FJWorld
from perfsim.learners.lm.kl_sft import KLSFTLearner
from perfsim.learners.lm.sft import SFTLearner
from perfsim.losses import MSELoss
from perfsim.models.hf_causal_lm import HFCausalLMModel

_BASE_PATH = Path(__file__).resolve().parent / "run_pokec_fj_lm.py"
_spec = importlib.util.spec_from_file_location("run_pokec_fj_lm", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


class FixedPredictions(Model):
    """Per-agent prediction vector assembled from the assigned platforms."""

    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.values = values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values


def main() -> int:
    run_tag = base._env_or("RUN_TAG")
    training_style = base._env_or("TRAINING_STYLE", "sft")
    kl_beta = base._env_float("KL_BETA", 0.0)
    tau = base._env_float("TAU", 0.05)
    base_models = os.environ.get(
        "BASE_MODELS",
        "Qwen/Qwen2.5-1.5B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct,"
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ).split(",")
    placements_env = os.environ.get("PLACEMENTS", "")
    placements = [float(v) for v in placements_env.split(",")] if placements_env else []
    place_passes = base._env_int("PLACE_PASSES", 0)
    n_rounds = base._env_int("N_ROUNDS", 12)
    epoch_size = base._env_int("EPOCH_SIZE", 100)
    seed = base._env_int("SEED", 0)
    n_labeled = base._env_int("N_LABELED", 1730)
    pokec_dir = Path(os.environ.get("POKEC_DIR", "examples/pokec"))
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.environ.get("OUT_DIR", f"runs/pokec_fj_competition/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT")
    max_steps = base._env_int("SFT_MAX_STEPS", 1)
    sft_epochs = base._env_int("SFT_EPOCHS", 1)
    gen_batch_size = base._env_int("GEN_BATCH_SIZE", 32)
    sft_batch_size = base._env_int("SFT_BATCH_SIZE", 2)
    lora_r = base._env_int("LORA_R", 8)
    use_lora = base._env_int("USE_LORA", 1) == 1
    sft_lr = base._env_float("SFT_LR", 5e-5)
    max_new_tokens = base._env_int("MAX_NEW_TOKENS", 6)
    n_bins = base._env_int("HIST_BINS", 50)

    n_platforms = len(base_models)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag, "training_style": training_style, "kl_beta": kl_beta,
        "tau": tau, "base_models": base_models, "placements": placements,
        "place_passes": place_passes, "n_rounds": n_rounds, "epoch_size": epoch_size,
        "seed": seed, "n_labeled": n_labeled, "max_steps": max_steps,
        "sft_epochs": sft_epochs, "lora_r": lora_r, "use_lora": use_lora,
        "sft_lr": sft_lr, "host": os.uname().nodename,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[run] {json.dumps(config)}", flush=True)

    wandb = None
    if wandb_project and _HAS_WANDB:
        wandb = _wandb
        wandb.init(project=wandb_project, name=run_tag, config=config)

    torch.manual_seed(seed)
    setup = base.load_pokec_setup(pokec_dir)
    n = setup["n"]
    innate = setup["innate"]
    print(f"[run] pokec ready: N={n}", flush=True)

    trainer_kwargs = {"bf16": device == "cuda", "use_cpu": device != "cuda"}
    if sft_epochs > 0:
        trainer_kwargs.update({"num_train_epochs": sft_epochs, "max_steps": -1})

    def _prompt_builder(profile: pd.Series, tokenizer) -> str:
        lines = []
        for col in base.PROMPT_COLS:
            val = profile.get(col, "")
            if pd.isna(val) or val == "" or str(val) == "nan":
                continue
            if col == "age":
                if float(val) == 0.0:
                    continue
                val = int(val)
            elif col == "gender":
                val = {0.0: "female", 1.0: "male"}.get(float(val), "unknown")
            elif col == "relation_to_alcohol":
                val = base.translate_alcohol(val)
            lines.append(f"- {col}: {val}")
        profile_str = "\n".join(lines) if lines else "- (no profile info)"
        user_msg = (
            "Estimate this user's attitude toward smoking based on their profile.\n"
            "Profile:\n"
            f"{profile_str}\n\n"
            "Output a single number in [0, 1] (1 = very positive, 0 = very negative). "
            "Respond with only the number, e.g. 0.42."
        )
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def format_number(y) -> str:
        return f"{float(y):.2f}"

    def response_template(name: str) -> str:
        low = name.lower()
        if "tinyllama" in low or "phi" in low or "zephyr" in low:
            return "<|assistant|>\n"
        if "llama-3" in low:
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return "<|im_start|>assistant\n"

    def make_lm(name: str):
        return HFCausalLMModel(
            base_model_name=name,
            profiles=setup["profiles"],
            prompt_builder=_prompt_builder,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=2 * lora_r,
            device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            max_new_tokens=max_new_tokens,
            gen_batch_size=gen_batch_size,
            load_now=True,
        )

    lms, learners = [], []
    for p, name in enumerate(base_models):
        t0 = time.time()
        lm = make_lm(name)
        kwargs = dict(
            model=lm, loss=MSELoss(), max_steps=max_steps,
            per_device_batch_size=sft_batch_size,
            output_dir=str(out_dir / f"trl_p{p}"),
            response_template=response_template(name), learning_rate=sft_lr,
            target_formatter=format_number, trainer_kwargs=trainer_kwargs,
        )
        if training_style == "sft":
            learner = SFTLearner(**kwargs)
        elif training_style == "sft_kl":
            learner = KLSFTLearner(**kwargs, ref_model_name=name, kl_beta=kl_beta)
        else:
            raise ValueError(f"unknown TRAINING_STYLE: {training_style!r}")
        lms.append(lm)
        learners.append(learner)
        print(f"[run] platform {p} = {name} loaded in {time.time() - t0:.1f}s", flush=True)

    world = FJWorld(
        innate=innate, graph=setup["W"], peer_sus=setup["peer_sus"],
        platform_sus=setup["platform_sus"], features=innate, profiles=setup["profiles"],
    )
    world.reset(seed=seed)

    mask = torch.zeros(n, dtype=torch.bool)
    mask[:n_labeled] = True
    idx_all = torch.arange(n)

    if placements and place_passes > 0:
        for p, anchor in enumerate(placements):
            place_data = {
                "x": innate[mask].unsqueeze(-1),
                "y": torch.full((int(mask.sum()), 1), float(anchor)),
                "agent_idx": idx_all[mask],
            }
            for _ in range(place_passes):
                learners[p].train(place_data)
            print(f"[run] platform {p} placed at {anchor}", flush=True)

    gen = torch.Generator()
    gen.manual_seed(seed + 1000)
    trajectory = []
    preds_raw, op_raw, assign_raw = [], [], []

    t_loop = time.time()
    for t in range(n_rounds):
        preds = torch.stack([
            lm(innate.unsqueeze(-1)).detach().squeeze(-1).float() for lm in lms
        ])                                                          # (P, N)
        x = world.state["opinion"].float()
        probs = torch.softmax(-(preds - x.unsqueeze(0)).abs().t() / tau, dim=1)
        assign = torch.multinomial(probs, 1, generator=gen).squeeze(1)   # (N,)
        combined = preds[assign, idx_all]

        world.run(FixedPredictions(combined.unsqueeze(-1)), n_steps=epoch_size)
        op = world.state["opinion"].float()

        for p in range(n_platforms):
            served = (assign == p) & mask
            if int(served.sum()) > 0:
                learners[p].train({
                    "x": innate[served].unsqueeze(-1),
                    "y": op[served].detach().unsqueeze(-1),
                    "agent_idx": idx_all[served],
                })

        means = preds.mean(dim=1)
        div = torch.zeros(n_platforms, n_platforms)
        for a in range(n_platforms):
            for b_ in range(n_platforms):
                div[a, b_] = (preds[a] - preds[b_]).abs().mean()
        row = {
            "round": t,
            "op_mean": float(op.mean()), "op_std": float(op.std()),
            "op_eff_support": base.cm.summary(op, bins=n_bins)["eff_support"],
            "platform_means": [float(v) for v in means],
            "position_gap": float(means.max() - means.min()),
            "pred_divergence": float(div.sum() / (n_platforms * (n_platforms - 1))),
            "shares": [float((assign == p).float().mean()) for p in range(n_platforms)],
        }
        trajectory.append(row)
        (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
        if wandb is not None:
            wandb.log({k: v for k, v in row.items() if not isinstance(v, list)})
        print(f"[round {t}] means={[f'{v:.3f}' for v in means]} "
              f"gap={row['position_gap']:.3f} div={row['pred_divergence']:.3f} "
              f"op_std={row['op_std']:.4f} shares={row['shares']}", flush=True)

        preds_raw.append(preds.cpu())
        op_raw.append(op.cpu())
        assign_raw.append(assign.cpu())

    print(f"[run] loop done in {time.time() - t_loop:.1f}s", flush=True)
    torch.save(
        {
            "trajectory": trajectory, "config": config,
            "preds_raw": torch.stack(preds_raw),
            "op_raw": torch.stack(op_raw),
            "assign_raw": torch.stack(assign_raw),
            "innate": innate.cpu(),
        },
        out_dir / "trajectory.pt",
    )
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
