"""Calibrate AT covid ABM against real Astoria winter surge data."""

from __future__ import annotations

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

from perfsim.scenarios.at_covid import build_covid_runner


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


def load_astoria_cases():
    data_path = (
        Path(agent_torch.__file__).parent
        / "models" / "covid" / "data" / "county_data.csv"
    )
    df = pd.read_csv(data_path)
    astoria = df[df["neighborhood"].str.contains("Astoria", case=False, na=False)]
    return astoria.sort_values("epiweek").reset_index(drop=True)


def seed_infections_differentiable(runner, fraction: float, seed: int = 0):
    """Seed a fraction of agents as INFECTED (disease_stage=2).

    Non-differentiable but deterministic. Called before each forward pass
    with the current fraction value.
    """
    citizens = runner.state["agents"]["citizens"]
    n = citizens["disease_stage"].shape[0]
    gen = torch.Generator().manual_seed(seed)
    mask = torch.rand(n, 1, generator=gen) < fraction
    ds = citizens["disease_stage"].clone()
    ds[mask] = 2.0  # INFECTED
    citizens["disease_stage"] = ds
    if "infected_time" in citizens:
        it = citizens["infected_time"].clone()
        it[mask] = 0
        citizens["infected_time"] = it
    return int(mask.sum().item())



def main():
    out_dir = Path("runs/calibrated_covid_surge")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load target: weeks 17-19 (Dec 2020 surge, 3 weeks = 21 days)
    astoria = load_astoria_cases()
    start_week = 17
    n_weeks = 3
    target_rows = astoria.iloc[start_week : start_week + n_weeks]
    target_cases = target_rows["cases_week"].tolist()
    target_dates = target_rows["date"].tolist()
    target_total = sum(target_cases)

    print("Target: Astoria winter surge")
    for d, c in zip(target_dates, target_cases):
        print(f"  {d}: {c:.0f} cases")
    print(f"  Total: {target_total:.0f} cases over {n_weeks} weeks")

    (out_dir / "target_data.json").write_text(json.dumps({
        "start_week": start_week, "n_weeks": n_weeks,
        "dates": target_dates, "cases_week": target_cases,
        "total": target_total,
    }, indent=2))

    # Build runner
    print("\nBuilding AT covid runner...", flush=True)
    t0 = time.time()
    runner = build_covid_runner(seed=0)
    n_agents = runner.state["agents"]["citizens"]["age"].shape[0]
    print(f"  {n_agents} agents, built in {time.time() - t0:.1f}s")

    runner.state["agents"]["citizens"]["platform_signal"] = torch.zeros(n_agents)

    # Locate R2
    transmission = runner.initializer.transition_function["0"].new_transmission
    r2_param = transmission.calibrate_R2
    r2_default = float(r2_param.detach().flatten()[0])
    print(f"  R2 default: {r2_default}")

    # Grid search over seed fractions, optimizing R2 for each.
    # The seed fraction is not easily differentiable (it's a discrete mask),
    # so we grid-search it and optimize R2 within each.
    total_steps = 7 * n_weeks  # 21 days

    seed_fractions = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    n_iters_per = 30
    lr = 0.5

    print(f"\nGrid search: {len(seed_fractions)} seed fractions x {n_iters_per} R2 iters each")
    print(f"{'='*70}")

    all_results = []

    for sf in seed_fractions:
        # Reset R2 to default
        with torch.no_grad():
            r2_param.fill_(r2_default)

        # Build fresh initial state with this seed fraction
        base_snap = snapshot_state(runner.state)
        # Seed infections on top of the base state
        restore_state(runner.state, base_snap)
        n_seeded = seed_infections_differentiable(runner, fraction=sf, seed=0)
        seeded_snap = snapshot_state(runner.state)

        optimizer = torch.optim.Adam([r2_param], lr=lr)
        best_loss = float("inf")
        best_r2 = r2_default
        best_pred = 0.0

        for it in range(n_iters_per):
            restore_state(runner.state, seeded_snap)
            runner.reset_state_before_episode()
            optimizer.zero_grad()

            runner.step(num_steps=total_steps)

            di = runner.state["environment"]["daily_infected"]
            pred_total = di.sum()
            target_t = torch.tensor(float(target_total))
            loss = ((pred_total - target_t) / (target_t + 1.0)) ** 2

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                r2_param.clamp_(min=0.5, max=15.0)

            l = float(loss.detach())
            if l < best_loss:
                best_loss = l
                best_r2 = float(r2_param.detach().flatten()[0])
                best_pred = float(pred_total.detach())

        ratio = best_pred / target_total
        result = {
            "seed_frac": sf,
            "n_seeded": n_seeded,
            "best_loss": best_loss,
            "best_R2": best_r2,
            "best_pred": best_pred,
            "target": target_total,
            "ratio": ratio,
        }
        all_results.append(result)
        print(
            f"  frac={sf:.3f}  seeded={n_seeded:5d}  "
            f"R2={best_r2:.3f}  pred={best_pred:.0f}  "
            f"target={target_total:.0f}  ratio={ratio:.3f}  loss={best_loss:.6f}"
        )

    # Find best overall
    best = min(all_results, key=lambda r: r["best_loss"])
    print(f"\n{'='*70}")
    print(f"Best fit: seed_frac={best['seed_frac']:.3f}  R2={best['best_R2']:.3f}  "
          f"pred={best['best_pred']:.0f}  target={target_total:.0f}  "
          f"ratio={best['ratio']:.3f}")

    # Save
    (out_dir / "grid_results.json").write_text(json.dumps(all_results, indent=2))

    # Re-run the best configuration and save calibrated state
    with torch.no_grad():
        r2_param.fill_(best["best_R2"])
    torch.save(r2_param.detach().clone(), out_dir / "calibrated_R2.pt")
    torch.save({
        "R2": best["best_R2"],
        "seed_frac": best["seed_frac"],
        "target_total": target_total,
        "pred_total": best["best_pred"],
    }, out_dir / "calibrated_params.pt")

    # Plot
    print("\nGenerating plot...", flush=True)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        fracs = [r["seed_frac"] for r in all_results]
        preds = [r["best_pred"] for r in all_results]
        r2s = [r["best_R2"] for r in all_results]
        losses = [r["best_loss"] for r in all_results]

        # Predicted vs target across seed fractions
        ax = axes[0]
        ax.bar(range(len(fracs)), preds, color="#4CAF50", alpha=0.8)
        ax.axhline(target_total, color="#E91E63", linestyle="--", linewidth=2, label=f"target={target_total:.0f}")
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels([f"{f:.1%}" for f in fracs], rotation=45)
        ax.set_xlabel("Initial infection fraction")
        ax.set_ylabel("Total infections (best R2)")
        ax.set_title("Fit quality by seed fraction")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Best R2 per fraction
        ax = axes[1]
        ax.plot(fracs, r2s, "o-", color="#FF5722", linewidth=2, markersize=6)
        ax.axhline(r2_default, color="gray", linestyle="--", alpha=0.5, label=f"default={r2_default}")
        ax.set_xlabel("Initial infection fraction")
        ax.set_ylabel("Best R2")
        ax.set_title("Calibrated R2 by seed fraction")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss
        ax = axes[2]
        ax.plot(fracs, losses, "o-", color="#2196F3", linewidth=2, markersize=6)
        ax.set_xlabel("Initial infection fraction")
        ax.set_ylabel("Relative MSE (best)")
        ax.set_title("Calibration loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        best_idx = losses.index(min(losses))
        for ax_i in axes:
            ax_i.axvline(fracs[best_idx], color="gold", linestyle=":", alpha=0.5)

        fig.suptitle(
            f"Joint Calibration: Seed Fraction × R2 → Astoria Surge ({target_dates[0]}-{target_dates[-1]})\n"
            f"Best: frac={best['seed_frac']:.1%}, R2={best['best_R2']:.2f}, "
            f"pred={best['best_pred']:.0f} vs target={target_total:.0f} "
            f"(ratio={best['ratio']:.2f})",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "calibration_plot.png", dpi=150, bbox_inches="tight")
        print(f"  Plot: {out_dir / 'calibration_plot.png'}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
