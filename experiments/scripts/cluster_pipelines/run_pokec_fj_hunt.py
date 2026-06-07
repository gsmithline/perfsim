"""LLM platforms that take gradient steps to grab population share (hunters)
vs SFT refitters (describers), over the Pokec FJ population. Positions come
from a differentiable expected-value readout over digit tokens (0-9 scale).
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

from perfsim.environments.dynamics import FJWorld
from perfsim.learners.lm.sft import SFTLearner
from perfsim.losses import MSELoss
from perfsim.models.hf_causal_lm import HFCausalLMModel
from perfsim.core.model import Model

_BASE_PATH = Path(__file__).resolve().parent / "run_pokec_fj_lm.py"
_spec = importlib.util.spec_from_file_location("run_pokec_fj_lm", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


class FixedPredictions(Model):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.values = values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values


def digit_ids(tokenizer) -> torch.Tensor:
    ids = []
    for d in range(10):
        enc = tokenizer.encode(str(d), add_special_tokens=False)
        ids.append(enc[-1])
    return torch.tensor(ids)


def readout(lm, prompts: list[str], dids: torch.Tensor, *, grad: bool,
            chunk: int = 64) -> torch.Tensor:
    """Expected value over digit tokens at the next position, in [0, 1]."""
    dev = lm.inner_model.device
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, len(prompts), chunk):
            batch = prompts[i:i + chunk]
            inputs = lm.tokenizer(batch, return_tensors="pt", padding=True,
                                  truncation=True).to(dev)
            logits = lm.inner_model(**inputs).logits
            last = inputs["attention_mask"].sum(dim=1) - 1
            row = torch.arange(logits.shape[0], device=dev)
            dlog = logits[row, last][:, dids.to(dev)]
            probs = torch.softmax(dlog.float(), dim=1)
            vals = torch.arange(10, device=dev, dtype=torch.float32) / 9.0
            outs.append(probs @ vals)
    return torch.cat(outs)


def main() -> int:
    run_tag = base._env_or("RUN_TAG")
    types = base._env_or("PLATFORM_TYPES", "hunt+hunt+hunt").split("+")
    tau = base._env_float("TAU", 0.05)
    hunt_lr = base._env_float("HUNT_LR", 1e-5)
    hunt_steps = base._env_int("HUNT_STEPS", 8)
    hunt_batch = base._env_int("HUNT_BATCH", 64)
    base_models = os.environ.get(
        "BASE_MODELS",
        "Qwen/Qwen2.5-1.5B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct,"
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ).split(",")
    n_rounds = base._env_int("N_ROUNDS", 12)
    epoch_size = base._env_int("EPOCH_SIZE", 100)
    seed = base._env_int("SEED", 0)
    n_labeled = base._env_int("N_LABELED", 1730)
    pokec_dir = Path(os.environ.get("POKEC_DIR", "examples/pokec"))
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.environ.get("OUT_DIR", f"runs/pokec_fj_hunt/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT")
    sft_epochs = base._env_int("SFT_EPOCHS", 1)
    sft_batch_size = base._env_int("SFT_BATCH_SIZE", 2)
    lora_r = base._env_int("LORA_R", 8)
    sft_lr = base._env_float("SFT_LR", 5e-5)

    n_platforms = len(base_models)
    assert len(types) == n_platforms
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag, "types": types, "tau": tau, "hunt_lr": hunt_lr,
        "hunt_steps": hunt_steps, "hunt_batch": hunt_batch,
        "base_models": base_models, "n_rounds": n_rounds, "epoch_size": epoch_size,
        "seed": seed, "n_labeled": n_labeled, "sft_epochs": sft_epochs,
        "lora_r": lora_r, "sft_lr": sft_lr, "host": os.uname().nodename,
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
            "Output a single integer from 0 (very negative) to 9 (very positive). "
            "Respond with only the integer, e.g. 4."
        )
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def format_digit(y) -> str:
        return str(int(round(float(y) * 9)))

    def response_template(name: str) -> str:
        low = name.lower()
        if "tinyllama" in low or "phi" in low or "zephyr" in low:
            return "<|assistant|>\n"
        if "llama-3" in low:
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return "<|im_start|>assistant\n"

    trainer_kwargs = {"bf16": device == "cuda", "use_cpu": device != "cuda"}
    if sft_epochs > 0:
        trainer_kwargs.update({"num_train_epochs": sft_epochs, "max_steps": -1})

    lms, learners, opts, dids_all, prompts_all = [], [], [], [], []
    for p, name in enumerate(base_models):
        t0 = time.time()
        lm = HFCausalLMModel(
            base_model_name=name, profiles=setup["profiles"],
            prompt_builder=_prompt_builder, use_lora=True, lora_r=lora_r,
            lora_alpha=2 * lora_r, device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            max_new_tokens=4, gen_batch_size=32, load_now=True,
        )
        if lm.tokenizer.pad_token_id is None:
            lm.tokenizer.pad_token = lm.tokenizer.eos_token
        lms.append(lm)
        dids_all.append(digit_ids(lm.tokenizer))
        prompts_all.append([lm.build_prompt(lm.profile_at(i)) for i in range(n)])
        if types[p] == "desc":
            learners.append(SFTLearner(
                model=lm, loss=MSELoss(), max_steps=1,
                per_device_batch_size=sft_batch_size,
                output_dir=str(out_dir / f"trl_p{p}"),
                response_template=response_template(name), learning_rate=sft_lr,
                target_formatter=format_digit, trainer_kwargs=trainer_kwargs,
            ))
            opts.append(None)
        else:
            learners.append(None)
            trainable = [q for q in lm.inner_model.parameters() if q.requires_grad]
            opts.append(torch.optim.AdamW(trainable, lr=hunt_lr))
        print(f"[run] platform {p} = {name} ({types[p]}) in {time.time() - t0:.1f}s", flush=True)

    world = FJWorld(
        innate=innate, graph=setup["W"], peer_sus=setup["peer_sus"],
        platform_sus=setup["platform_sus"], features=innate, profiles=setup["profiles"],
    )
    world.reset(seed=seed)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:n_labeled] = True
    idx_all = torch.arange(n)
    gen = torch.Generator()
    gen.manual_seed(seed + 1000)

    trajectory, preds_raw, op_raw, assign_raw = [], [], [], []
    t_loop = time.time()
    for t in range(n_rounds):
        preds = torch.stack([
            readout(lms[p], prompts_all[p], dids_all[p], grad=False).cpu()
            for p in range(n_platforms)
        ])                                                          # (P, N)
        x = world.state["opinion"].float()
        probs = torch.softmax(-(preds - x.unsqueeze(0)).abs().t() / tau, dim=1)
        assign = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        world.run(FixedPredictions(preds[assign, idx_all].unsqueeze(-1)), n_steps=epoch_size)
        op = world.state["opinion"].float()

        for p in range(n_platforms):
            if types[p] == "desc":
                served = (assign == p) & mask
                if int(served.sum()):
                    learners[p].train({
                        "x": innate[served].unsqueeze(-1),
                        "y": op[served].detach().unsqueeze(-1),
                        "agent_idx": idx_all[served],
                    })
            else:
                # gradient steps on expected capture, rivals fixed
                rivals = torch.stack([preds[q] for q in range(n_platforms) if q != p])
                rival_w = torch.exp(-(rivals - op.unsqueeze(0)).abs() / tau).sum(dim=0)
                for _ in range(hunt_steps):
                    sel = torch.randperm(n, generator=gen)[:hunt_batch]
                    f = readout(lms[p], [prompts_all[p][i] for i in sel],
                                dids_all[p], grad=True)
                    own = torch.exp(-(f - op[sel].to(f.device)).abs() / tau)
                    capture = own / (own + rival_w[sel].to(f.device))
                    loss = -capture.mean()
                    opts[p].zero_grad()
                    loss.backward()
                    opts[p].step()

        means = preds.mean(dim=1)
        div = sum(float((preds[a] - preds[b_]).abs().mean())
                  for a in range(n_platforms) for b_ in range(n_platforms) if a != b_)
        row = {
            "round": t,
            "op_mean": float(op.mean()), "op_std": float(op.std()),
            "platform_means": [float(v) for v in means],
            "position_gap": float(means.max() - means.min()),
            "pred_divergence": div / (n_platforms * (n_platforms - 1)),
            "shares": [float((assign == p).float().mean()) for p in range(n_platforms)],
        }
        trajectory.append(row)
        (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
        if wandb is not None:
            wandb.log({k: v for k, v in row.items() if not isinstance(v, list)})
        print(f"[round {t}] means={[f'{v:.3f}' for v in means]} "
              f"gap={row['position_gap']:.3f} div={row['pred_divergence']:.3f} "
              f"op_std={row['op_std']:.4f} shares={row['shares']}", flush=True)
        preds_raw.append(preds)
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
