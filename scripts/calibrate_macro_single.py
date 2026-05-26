"""Single macro calibration job for Condor.

Optimizes UAC (unemployment_adaptation_coefficient) to match real Queens
County monthly unemployment data. Logs to W&B if WANDB_PROJECT is set.

Environment variables:
  RUN_TAG         required. Job identifier.
  START_MONTH     Index into Queens data (0=Jan 2019). Default 15 (Apr 2020).
  N_STEPS         Monthly timesteps to fit. Default 9 (max 10).
  N_ITERS         Optimization iterations. Default 50.
  LR              Adam learning rate. Default 0.1.
  SEED            RNG seed. Default 0.
  OUT_DIR         Output directory. Default runs/calibration_macro/$RUN_TAG.
  WANDB_PROJECT   W&B project name. Empty to disable.
"""

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


def load_queens_unemployment() -> pd.DataFrame:
    import agent_torch
    data_path = (
        Path(agent_torch.__file__).parent
        / "models" / "macro_economics" / "data"
        / "unemployment_rate_csvs" / "Queens-Table.csv"
    )
    df = pd.read_csv(data_path, skiprows=2)
    df.columns = [
        "area", "year", "month", "labor_force", "employed", "unemployed", "unemp_rate",
    ]
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace("\t", "").str.strip()
    df = df[df["month"] != "Avg"]
    df = df[df["year"].str.match(r"^\d")]
    df["year"] = df["year"].astype(float).astype(int)
    df["unemp_rate"] = pd.to_numeric(df["unemp_rate"], errors="coerce")
    df = df.dropna(subset=["unemp_rate"])
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df["m"] = df["month"].map({m: i for i, m in enumerate(MONTHS)})
    df = df.sort_values(["year", "m"]).reset_index(drop=True)
    return df


def main() -> int:
    run_tag = _env("RUN_TAG")
    start_month = int(_env("START_MONTH", "15"))
    n_steps = int(_env("N_STEPS", "9"))
    n_iters = int(_env("N_ITERS", "50"))
    lr = float(_env("LR", "0.1"))
    seed = int(_env("SEED", "0"))
    out_dir = Path(_env("OUT_DIR", f"runs/calibration_macro/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT", "")

    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_tag": run_tag,
        "start_month": start_month,
        "n_steps": n_steps,
        "n_iters": n_iters,
        "lr": lr,
        "seed": seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[calibrate_macro] {json.dumps(config)}", flush=True)

    wandb = None
    if wandb_project:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=wandb_project, name=run_tag, config=config)

    # Load real data
    queens = load_queens_unemployment()
    n_months = min(n_steps, len(queens) - start_month)
    target_rows = queens.iloc[start_month : start_month + n_months]
    target_rates = target_rows["unemp_rate"].tolist()
    target_labels = [
        f"{r['year']}-{r['month']}" for _, r in target_rows.iterrows()
    ]
    target_tensor = torch.tensor(target_rates, dtype=torch.float32)

    print(f"[calibrate_macro] target: {target_labels[0]} to {target_labels[-1]}", flush=True)
    for lbl, rate in zip(target_labels, target_rates):
        print(f"  {lbl}: {rate:.1f}%", flush=True)

    (out_dir / "target_data.json").write_text(json.dumps({
        "start_month": start_month,
        "n_months": n_months,
        "labels": target_labels,
        "unemp_rates": target_rates,
    }, indent=2))

    # Build macro env
    n_agents_cfg = int(_env("N_AGENTS", "100"))
    macro_yaml = _env("MACRO_YAML", "config_100_agents.yaml")
    from perfsim.scenarios.at_macro import make_macro_env
    env = make_macro_env(
        init_seed=seed,
        yaml_name=macro_yaml,
        n_agents=n_agents_cfg,
    )
    n_agents = env.runner.state["agents"]["consumers"]["age"].shape[0]
    print(f"[calibrate_macro] {n_agents} agents (config={macro_yaml})", flush=True)

    # Find UAC parameter
    uac_param = None
    for tf_key in env.runner.initializer.transition_function:
        tf = env.runner.initializer.transition_function[tf_key]
        for name, module in tf.named_modules():
            if hasattr(module, "external_UAC"):
                uac_param = module.external_UAC
                break
        if uac_param is not None:
            break

    if uac_param is None:
        print("ERROR: Could not find external_UAC parameter", flush=True)
        return 1

    print(f"[calibrate_macro] UAC shape: {tuple(uac_param.shape)}", flush=True)

    # Policy model (dummy — just need something for env.run)
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.5)

    initial_snap = snapshot_state(env.runner.state)
    optimizer = torch.optim.Adam([uac_param], lr=lr)

    history = []
    best_loss = float("inf")
    best_uac = uac_param.detach().clone()

    print(f"[calibrate_macro] starting: {n_iters} iters", flush=True)
    t_start = time.time()

    for it in range(n_iters):
        t_iter = time.time()
        restore_state(env.runner.state, initial_snap)
        env.runner.reset_state_before_episode()
        optimizer.zero_grad()

        env.run(model, n_steps=n_months)

        U = env.runner.state["environment"]["U"].flatten()[:n_months]
        loss = ((U - target_tensor) ** 2).mean()
        loss.backward()
        optimizer.step()

        pred_rates = U.detach().tolist()
        loss_val = float(loss.detach())
        iter_time = time.time() - t_iter

        if loss_val < best_loss:
            best_loss = loss_val
            best_uac = uac_param.detach().clone()

        pred_mean = sum(pred_rates) / len(pred_rates)
        target_mean = sum(target_rates[:n_months]) / n_months

        rec = {
            "iter": it,
            "loss": loss_val,
            "pred_mean_unemp": pred_mean,
            "target_mean_unemp": target_mean,
            "best_loss": best_loss,
            "iter_seconds": iter_time,
        }
        history.append(rec)

        if wandb is not None:
            log_data = {
                "iter": it,
                "loss": loss_val,
                "pred_mean_unemp": pred_mean,
                "best_loss": best_loss,
            }
            for i, (p, t_val) in enumerate(zip(pred_rates, target_rates)):
                log_data[f"pred_month_{i}"] = p
                log_data[f"target_month_{i}"] = t_val
            wandb.log(log_data)

        if it % 5 == 0 or it == n_iters - 1:
            print(
                f"  iter {it:3d}  loss={loss_val:.4f}  "
                f"pred_mean={pred_mean:.2f}%  target_mean={target_mean:.2f}%  "
                f"({iter_time:.1f}s)",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(f"\n[calibrate_macro] done in {elapsed:.0f}s", flush=True)

    # Save
    torch.save(best_uac, out_dir / "calibrated_UAC.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({
        "UAC": best_uac.tolist(),
        "best_loss": best_loss,
        "target_rates": target_rates,
        "start_month": start_month,
        "n_months": n_months,
    }, out_dir / "result.pt")

    if wandb is not None:
        wandb.summary["best_loss"] = best_loss
        wandb.finish()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
