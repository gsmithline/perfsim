"""Single-config Schelling + LM + KL-SFT run for the condor sweep."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
from perfsim.scenarios.at_schelling import make_schelling_env
from perfsim.scenarios.at_schelling.model_scoring import BinaryLMScorer
from perfsim.simulator import Simulator


TYPE_NAMES = ["White", "Black", "Hispanic", "Asian"]
# values: -1=empty, 0=White, 1=Black, 2=Hispanic, 3=Asian (shifted +1 for indexing)
CELL_CMAP = ListedColormap(["#dddddd", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])


def _render_grid(grid_type: torch.Tensor, title: str) -> "plt.Figure":
    arr = (grid_type.cpu().numpy() + 1).astype("int32")
    fig, ax = plt.subplots(figsize=(4, 4), dpi=90)
    ax.imshow(arr, cmap=CELL_CMAP, vmin=0, vmax=4, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def _env_or(name, default=None):
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var {name!r} not set")
    return val


def _env_int(name, default): return int(os.environ.get(name, str(default)))
def _env_float(name, default): return float(os.environ.get(name, str(default)))


def main() -> int:
    run_tag = _env_or("RUN_TAG")
    kl_beta = _env_float("KL_BETA", 0.0)
    training_style = _env_or("TRAINING_STYLE", "sft_kl")
    base_model = _env_or("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    n_rounds = _env_int("N_ROUNDS", 10)
    k_steps = _env_int("K_STEPS", 3)
    seed = _env_int("SEED", 0)
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.environ.get("OUT_DIR", f"runs/at_schelling_lm/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT")
    max_steps = _env_int("SFT_MAX_STEPS", 50)
    gen_batch_size = _env_int("GEN_BATCH_SIZE", 32)
    sft_batch_size = _env_int("SFT_BATCH_SIZE", 16)
    sft_full_epoch = os.environ.get("SFT_FULL_EPOCH", "0").lower() in ("1", "true", "yes")
    lora_r = _env_int("LORA_R", 32)
    use_lora = _env_int("USE_LORA", 1) == 1
    sft_lr = _env_float("SFT_LR", 1e-5)
    num_agents = _env_int("NUM_AGENTS", 200)
    grid_size = _env_int("GRID_SIZE", 20)
    baseline_threshold = _env_float("BASELINE_THRESHOLD", 0.30)
    lambda_ = _env_float("LAMBDA", 0.15)
    neighborhood_radius = _env_int("NEIGHBORHOOD_RADIUS", 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag,
        "kl_beta": kl_beta,
        "training_style": training_style,
        "base_model": base_model,
        "n_rounds": n_rounds,
        "k_steps": k_steps,
        "seed": seed,
        "num_agents": num_agents,
        "grid_size": grid_size,
        "baseline_threshold": baseline_threshold,
        "lambda": lambda_,
        "neighborhood_radius": neighborhood_radius,
        "max_steps": max_steps,
        "sft_batch_size": sft_batch_size,
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

    torch.manual_seed(seed)
    print("[run] building Schelling env", flush=True)
    t0 = time.time()
    env = make_schelling_env(
        init_seed=seed,
        num_agents=num_agents,
        grid_height=grid_size,
        grid_width=grid_size,
        baseline_threshold=baseline_threshold,
        lambda_=lambda_,
        neighborhood_radius=neighborhood_radius,
        num_steps_per_episode=k_steps,
        device=device,
        keep_trajectory=True,
    )
    n = env.runner.state["agents"]["residents"]["type"].shape[0]
    print(f"[run] env ready: {n} agents on {grid_size}x{grid_size} grid in {time.time() - t0:.1f}s", flush=True)

    residents = env.runner.state["agents"]["residents"]
    types_ = residents["type"].long().cpu().tolist()
    profiles = pd.DataFrame({
        "agent_id": list(range(n)),
        "type_name": [TYPE_NAMES[min(int(t), 3)] for t in types_],
        "type_idx": types_,
    })

    SYSTEM_MSG = (
        "You are predicting whether a household will be HAPPY or UNHAPPY "
        "with their neighborhood next round in a Schelling segregation model. "
        "Output ONE token: HAPPY or UNHAPPY."
    )

    def prompt_builder(profile_row, tokenizer):
        i = int(profile_row["agent_id"])
        r = env.runner.state["agents"]["residents"]
        same_frac = float(r["same_frac"][i].item())
        opp_frac = float(r["opp_frac"][i].item())
        empty_frac = float(r["empty_frac"][i].item())
        prev = float(r["previous_state"][i].item())
        prev_str = "happy" if prev >= 0.5 else ("unhappy" if prev < 0.5 else "unknown")
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": (
                    f"Agent type: {profile_row['type_name']}.\n"
                    f"Same-type neighbor fraction: {same_frac:.2f}.\n"
                    f"Opposite-type neighbor fraction: {opp_frac:.2f}.\n"
                    f"Empty-neighbor fraction: {empty_frac:.2f}.\n"
                    f"Baseline happiness threshold: {baseline_threshold:.2f}.\n"
                    f"Previous state: {prev_str}.\n"
                    f"After one Schelling update, will this agent be happy?\n"
                    f"Answer with one token: HAPPY or UNHAPPY."
                ),
            },
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def happy_or_unhappy(y):
        return "HAPPY" if float(y) >= 0.5 else "UNHAPPY"

    print(f"[run] loading LM: {base_model} on {device}", flush=True)
    t0 = time.time()
    lm = HFCausalLMModel(
        base_model_name=base_model,
        profiles=profiles,
        prompt_builder=prompt_builder,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=2 * lora_r,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        max_new_tokens=4,
        gen_batch_size=gen_batch_size,
        load_now=True,
    )
    print(f"[run] LM loaded in {time.time() - t0:.1f}s", flush=True)

    scorer = BinaryLMScorer(lm, yes_token="HAPPY", no_token="UNHAPPY", batch_size=gen_batch_size)

    if sft_full_epoch:
        effective_max_steps = -(-n // sft_batch_size)
    else:
        effective_max_steps = max_steps

    learner_kwargs = dict(
        model=lm,
        loss=MSELoss(),
        max_steps=effective_max_steps,
        per_device_batch_size=sft_batch_size,
        output_dir=str(out_dir / "trl"),
        response_template="<|im_start|>assistant\n",
        learning_rate=sft_lr,
        target_formatter=happy_or_unhappy,
    )
    if training_style == "sft":
        learner = SFTLearner(**learner_kwargs)
    elif training_style == "sft_kl":
        learner = KLSFTLearner(**learner_kwargs, ref_model_name=base_model, kl_beta=kl_beta)
    elif training_style == "frozen":
        class _Frozen(Learner):
            accepted_schemas = (SUPERVISED_SCHEMA,)
            def __init__(self, model, loss): super().__init__(model, loss)
            def train(self, data): pass
            def reset(self): pass
        learner = _Frozen(model=lm, loss=MSELoss())
    else:
        raise ValueError(f"unknown TRAINING_STYLE: {training_style!r}")

    def metrics(sim):
        r = sim.env.runner.state["agents"]["residents"]
        env_state = sim.env.runner.state["environment"]
        same_frac = r["same_frac"].float()
        realized = r["realized_happiness"].float()
        types_ = r["type"].long()
        out = {
            "mean_same_frac": float(same_frac.mean().item()),
            "mean_realized_happy": float(realized.mean().item()),
            "n_movers": float(r["move_decision"].sum().item()),
            "mean_p_pred": float(r["p_pred"].float().mean().item()),
            "pred_std": float(r["p_pred"].float().std().item()),
        }
        grid_type = env_state["grid_type"].cpu()
        for ta, na in enumerate(TYPE_NAMES):
            mask = types_ == ta
            n_t = int(mask.sum().item())
            if n_t > 0:
                out[f"same_frac_{na}"] = float(same_frac[mask].mean().item())
                out[f"happy_{na}"] = float(realized[mask].mean().item())
        from experiments.scripts.calibrate_schelling import dissimilarity_index, TYPE_IDX
        for pair_name, (a, b, _tgt) in [
            ("D_BW", ("black", "white", 0.79)),
            ("D_HW", ("hispanic", "white", 0.62)),
            ("D_AW", ("asian", "white", 0.51)),
        ]:
            out[pair_name] = dissimilarity_index(grid_type, TYPE_IDX[a], TYPE_IDX[b])
        return out

    trajectory = []
    grids_dir = out_dir / "grids"
    grids_dir.mkdir(exist_ok=True)

    # Log the initial-placement grid (pre-loop). This is identical across
    # jobs at the same seed, so the t=-1 frame is the shared starting state.
    init_grid = env.runner.state["environment"]["grid_type"].detach().cpu().clone()
    torch.save(init_grid, grids_dir / "grid_init.pt")
    if wandb is not None:
        fig0 = _render_grid(init_grid, title=f"{run_tag}  initial")
        wandb.log({"round": -1, "grid": wandb.Image(fig0)})
        plt.close(fig0)

    def on_round(t, record):
        row = {"round": t}
        for k, v in record.items():
            if k == "round":
                continue
            if isinstance(v, dict):
                row.update({f"{k}.{kk}": float(vv) if hasattr(vv, "__float__") else vv for kk, vv in v.items()})
            elif hasattr(v, "item"):
                try: row[k] = float(v.item())
                except Exception: pass
            elif isinstance(v, (int, float)):
                row[k] = float(v)
        trajectory.append(row)
        grid = sim.env.runner.state["environment"]["grid_type"].detach().cpu().clone()
        torch.save(grid, grids_dir / f"grid_t{t:03d}.pt")
        if wandb is not None:
            fig = _render_grid(grid, title=f"{run_tag}  round {t}")
            wandb.log({**row, "grid": wandb.Image(fig)})
            plt.close(fig)
        print(f"[round {t}] {row}", flush=True)

    sim = Simulator(env=env, learner=learner, loss=MSELoss(), metrics={"m": metrics})
    sim.predictor._model = scorer  # env queries scorer; learner still trains lm
    print(f"[run] starting outer loop: n_rounds={n_rounds} K={k_steps}", flush=True)
    t_loop = time.time()
    hist = sim.run(n_rounds=n_rounds, epoch_size=k_steps, seed=seed, on_round=on_round)
    print(f"[run] loop done in {time.time() - t_loop:.1f}s", flush=True)

    torch.save([dict(r) for r in hist.records], out_dir / "history.pt")
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
