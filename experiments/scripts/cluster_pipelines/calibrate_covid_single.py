"""Single (seed_frac, R2) calibration job for Condor."""

from __future__ import annotations

import copy
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

try:
    import agent_torch
except ImportError:
    agent_torch = None

from perfsim.scenarios.at_covid import build_covid_runner


def _env(name, default=None):
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var {name!r} not set")
    return val


def snapshot_state(state):
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().clone()
        elif isinstance(v, dict):
            out[k] = snapshot_state(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def restore_state(target, snap):
    for k, v in snap.items():
        if isinstance(v, dict):
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            restore_state(target[k], v)
        elif isinstance(v, torch.Tensor):
            target[k] = v.detach().clone()
        else:
            target[k] = copy.deepcopy(v)


def main() -> int:
    seed_frac = float(_env("SEED_FRAC"))
    run_tag = _env("RUN_TAG")
    start_week = int(_env("START_WEEK", "17"))
    n_weeks = int(_env("N_WEEKS", "3"))
    n_iters = int(_env("N_ITERS", "40"))
    lr = float(_env("LR", "0.5"))
    seed = int(_env("SEED", "0"))
    out_dir = Path(_env("OUT_DIR", f"runs/calibration/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT", "")

    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_tag": run_tag,
        "seed_frac": seed_frac,
        "start_week": start_week,
        "n_weeks": n_weeks,
        "n_iters": n_iters,
        "lr": lr,
        "seed": seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[calibrate] {json.dumps(config)}", flush=True)

    wandb = None
    if wandb_project and _HAS_WANDB:
        wandb = _wandb
        wandb.init(project=wandb_project, name=run_tag, config=config)

    data_path = (
        Path(agent_torch.__file__).parent
        / "models" / "covid" / "data" / "county_data.csv"
    )
    df = pd.read_csv(data_path)
    astoria = df[df["neighborhood"].str.contains("Astoria", case=False, na=False)]
    astoria = astoria.sort_values("epiweek").reset_index(drop=True)
    target_rows = astoria.iloc[start_week : start_week + n_weeks]
    target_cases = target_rows["cases_week"].tolist()
    target_total = sum(target_cases)

    print(f"[calibrate] target: weeks {start_week}-{start_week + n_weeks - 1}, "
          f"total={target_total:.0f} cases", flush=True)

    runner = build_covid_runner(seed=seed)
    n_agents = runner.state["agents"]["citizens"]["age"].shape[0]
    runner.state["agents"]["citizens"]["platform_signal"] = torch.zeros(n_agents)

    # Seed infections
    citizens = runner.state["agents"]["citizens"]
    n = citizens["disease_stage"].shape[0]
    gen = torch.Generator().manual_seed(seed)
    mask = torch.rand(n, 1, generator=gen) < seed_frac
    ds = citizens["disease_stage"].clone()
    ds[mask] = 2.0
    citizens["disease_stage"] = ds
    if "infected_time" in citizens:
        it = citizens["infected_time"].clone()
        it[mask] = 0
        citizens["infected_time"] = it
    n_seeded = int(mask.sum().item())
    print(f"[calibrate] {n_agents} agents, {n_seeded} seeded infected ({seed_frac:.1%})", flush=True)

    initial_snap = snapshot_state(runner.state)

    # Locate R2
    transmission = runner.initializer.transition_function["0"].new_transmission
    r2_param = transmission.calibrate_R2
    r2_default = float(r2_param.detach().flatten()[0])
    print(f"[calibrate] R2 default: {r2_default}", flush=True)

    # Optimize
    total_steps = 7 * n_weeks
    optimizer = torch.optim.Adam([r2_param], lr=lr)
    target_tensor = torch.tensor(float(target_total))
    history = []
    best_loss = float("inf")
    best_r2 = r2_default
    best_pred = 0.0

    print(f"[calibrate] starting: {n_iters} iters, {total_steps} steps/iter", flush=True)
    t_start = time.time()

    for it in range(n_iters):
        t_iter = time.time()
        restore_state(runner.state, initial_snap)
        runner.reset_state_before_episode()
        optimizer.zero_grad()

        runner.step(num_steps=total_steps)

        di = runner.state["environment"]["daily_infected"]
        pred_total = di.sum()
        loss = ((pred_total - target_tensor) / (target_tensor + 1.0)) ** 2

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            r2_param.clamp_(min=0.5, max=15.0)

        r2_val = float(r2_param.detach().flatten()[0])
        pred_val = float(pred_total.detach())
        loss_val = float(loss.detach())
        iter_time = time.time() - t_iter

        if loss_val < best_loss:
            best_loss = loss_val
            best_r2 = r2_val
            best_pred = pred_val

        rec = {
            "iter": it,
            "loss": loss_val,
            "R2": r2_val,
            "pred_total": pred_val,
            "target_total": target_total,
            "ratio": pred_val / target_total if target_total > 0 else 0,
            "best_loss": best_loss,
            "best_R2": best_r2,
            "iter_seconds": iter_time,
        }
        history.append(rec)

        if wandb is not None:
            wandb.log({
                "iter": it,
                "loss": loss_val,
                "R2": r2_val,
                "pred_total": pred_val,
                "ratio": rec["ratio"],
                "best_loss": best_loss,
            })

        if it % 5 == 0 or it == n_iters - 1:
            print(
                f"  iter {it:3d}  loss={loss_val:.6f}  R2={r2_val:.4f}  "
                f"pred={pred_val:.0f}  ratio={rec['ratio']:.3f}  ({iter_time:.1f}s)",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(f"\n[calibrate] done in {elapsed:.0f}s", flush=True)
    print(f"[calibrate] best: R2={best_r2:.4f}  pred={best_pred:.0f}  "
          f"target={target_total:.0f}  ratio={best_pred/target_total:.3f}", flush=True)

    # Save
    torch.save(r2_param.detach().clone(), out_dir / "calibrated_R2.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({
        "R2": best_r2,
        "seed_frac": seed_frac,
        "n_seeded": n_seeded,
        "target_total": target_total,
        "best_pred": best_pred,
        "best_loss": best_loss,
    }, out_dir / "result.pt")

    if wandb is not None:
        wandb.summary["best_R2"] = best_r2
        wandb.summary["best_pred"] = best_pred
        wandb.summary["best_loss"] = best_loss
        wandb.summary["ratio"] = best_pred / target_total
        wandb.finish()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
