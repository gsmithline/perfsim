"""Calibrate AT macro_economics ABM against Queens County unemployment data."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import pandas as pd
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    matplotlib = None
    plt = None
    _HAS_MPL = False

try:
    import agent_torch
except ImportError:
    agent_torch = None

from perfsim.scenarios._deprecated.at_macro import make_macro_env


def load_queens_unemployment() -> pd.DataFrame:
    """Load Queens County monthly unemployment rates from bundled AT data."""
    data_path = (
        Path(agent_torch.__file__).parent
        / "models"
        / "macro_economics"
        / "data"
        / "unemployment_rate_csvs"
        / "Queens-Table.csv"
    )
    df = pd.read_csv(data_path, skiprows=2)
    df.columns = [
        "area", "year", "month", "labor_force", "employed", "unemployed", "unemp_rate",
    ]
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace("\t", "").str.strip()

    df = df.dropna(subset=["year"])
    df = df[df["month"] != "Avg"]
    df["year"] = df["year"].astype(float).astype(int)
    df["unemp_rate"] = pd.to_numeric(df["unemp_rate"], errors="coerce")
    df = df.dropna(subset=["unemp_rate"])
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df



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



def main():
    parser = argparse.ArgumentParser(description="Calibrate AT macro to Queens unemployment")
    parser.add_argument("--out-dir", type=str, default="runs/calibrated_macro")
    parser.add_argument("--n-iters", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=10, help="AT timesteps (months)")
    parser.add_argument("--start-month", type=int, default=0,
                        help="Index into Queens data to start from")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load real unemployment data
    print("Loading Queens unemployment data...", flush=True)
    queens = load_queens_unemployment()
    print(f"  {len(queens)} monthly records, {queens['year'].min()}-{queens['year'].max()}")

    # Select target window
    n_months = min(args.n_steps, len(queens) - args.start_month)
    target_rows = queens.iloc[args.start_month : args.start_month + n_months]
    target_rates = target_rows["unemp_rate"].tolist()
    target_dates = [f"{r['year']}-{r['month']}" for _, r in target_rows.iterrows()]

    print(f"\nTarget: {n_months} months of unemployment")
    for d, r in zip(target_dates, target_rates):
        print(f"  {d}: {r:.1f}%")

    target_tensor = torch.tensor(target_rates, dtype=torch.float32)

    (out_dir / "target_data.json").write_text(json.dumps({
        "start_month": args.start_month,
        "n_months": n_months,
        "dates": target_dates,
        "unemp_rates": target_rates,
    }, indent=2))

    print("\nBuilding macro env...", flush=True)
    env = make_macro_env(init_seed=args.seed)
    n_agents = env.runner.state["agents"]["consumers"]["age"].shape[0]
    print(f"  {n_agents} agents")

    # Locate the UAC parameter to optimize
    macro_rates = None
    for name, module in env.runner.initializer.transition_function["0"].named_modules():
        if hasattr(module, "external_UAC"):
            macro_rates = module
            break

    if macro_rates is None:
        for tf_key in env.runner.initializer.transition_function:
            tf = env.runner.initializer.transition_function[tf_key]
            for name, module in tf.named_modules():
                if hasattr(module, "external_UAC"):
                    macro_rates = module
                    break

    if macro_rates is None:
        print("ERROR: Could not find _PatchedMacroRates with external_UAC")
        print("Named modules:")
        for tf_key in env.runner.initializer.transition_function:
            tf = env.runner.initializer.transition_function[tf_key]
            for name, module in tf.named_modules():
                print(f"  [{tf_key}] {name}: {type(module).__name__}")
        return

    uac_param = macro_rates.external_UAC
    print(f"  UAC shape: {tuple(uac_param.shape)}")
    print(f"  UAC initial: {uac_param.detach().flatten()[:6].tolist()}")

    # Also create a simple policy model
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.5)

    initial_snap = snapshot_state(env.runner.state)

    # Optimize UAC to match unemployment trajectory
    optimizer = torch.optim.Adam([uac_param], lr=args.lr)
    history = []

    print(f"\nCalibrating: {args.n_iters} iters, {n_months} steps/iter")
    print(f"{'='*70}")

    for it in range(args.n_iters):
        t_iter = time.time()
        restore_state(env.runner.state, initial_snap)
        env.runner.reset_state_before_episode()
        optimizer.zero_grad()

        env.run(model, n_steps=n_months)

        # Extract unemployment trajectory from the (1, num_timesteps) tensor
        U = env.runner.state["environment"]["U"].flatten()[:n_months]

        # Compare against real rates
        loss = ((U - target_tensor[:n_months]) ** 2).mean()
        loss.backward()
        optimizer.step()

        pred_rates = U.detach().tolist()
        loss_val = float(loss.detach())
        iter_time = time.time() - t_iter

        history.append({
            "iter": it,
            "loss": loss_val,
            "pred_rates": pred_rates,
            "target_rates": target_rates[:n_months],
            "uac_sample": uac_param.detach().flatten()[:6].tolist(),
            "iter_seconds": iter_time,
        })

        if it % 5 == 0 or it == args.n_iters - 1:
            print(
                f"  iter {it:3d}  loss={loss_val:.4f}  "
                f"pred_mean={sum(pred_rates)/len(pred_rates):.2f}  "
                f"target_mean={sum(target_rates[:n_months])/n_months:.2f}  "
                f"({iter_time:.1f}s)",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"{'='*70}")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    torch.save(uac_param.detach().clone(), out_dir / "calibrated_UAC.pt")
    (out_dir / "calibration_history.json").write_text(json.dumps(history, indent=2))

    # Plot
    print("\nGenerating plot...", flush=True)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        iters = [h["iter"] for h in history]
        losses = [h["loss"] for h in history]

        axes[0].plot(iters, losses, "o-", color="#2196F3", linewidth=2, markersize=3)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("MSE")
        axes[0].set_title("Calibration Loss")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)

        # Final predicted vs target trajectory
        final = history[-1]
        months = list(range(n_months))
        axes[1].plot(months, final["target_rates"], "o--", color="#E91E63", label="Real Queens", linewidth=2)
        axes[1].plot(months, final["pred_rates"], "s-", color="#4CAF50", label="ABM predicted", linewidth=2)
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Unemployment rate (%)")
        axes[1].set_title("Final: Predicted vs Real")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Trajectory of predictions over iterations (first and last month)
        first_month = [h["pred_rates"][0] for h in history]
        last_month = [h["pred_rates"][-1] for h in history]
        axes[2].plot(iters, first_month, "-", color="#FF9800", label="Month 0 pred", linewidth=2)
        axes[2].plot(iters, last_month, "-", color="#9C27B0", label=f"Month {n_months-1} pred", linewidth=2)
        axes[2].axhline(target_rates[0], color="#FF9800", linestyle="--", alpha=0.5)
        axes[2].axhline(target_rates[-1], color="#9C27B0", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Unemployment rate (%)")
        axes[2].set_title("Convergence by Month")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            f"Macro Calibration to Queens Unemployment ({target_dates[0]} - {target_dates[-1]})",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "calibration_plot.png", dpi=150, bbox_inches="tight")
        print(f"  Plot: {out_dir / 'calibration_plot.png'}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()
