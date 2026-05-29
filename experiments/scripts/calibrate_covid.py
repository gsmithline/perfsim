"""Calibrate AT covid ABM against real Astoria case data.

Subcommands: r2 (gradient R2 + optional policy fit), surge (seed-fraction x R2
grid search against the winter surge), pick-best (best params per season).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from perfsim.scenarios.at_covid import (
    build_covid_runner,
    default_signal_writer_grad,
    make_covid_env,
    seed_initial_infections,
)

try:
    import agent_torch
except ImportError:
    agent_torch = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    plt = None
    _HAS_MPL = False


def load_astoria_cases() -> pd.DataFrame:
    """Load the bundled Astoria weekly case data from agent_torch."""
    data_path = (
        Path(agent_torch.__file__).parent
        / "models"
        / "covid"
        / "data"
        / "county_data.csv"
    )
    df = pd.read_csv(data_path)
    astoria = df[df["neighborhood"].str.contains("Astoria", case=False, na=False)].copy()
    astoria = astoria.sort_values("epiweek").reset_index(drop=True)
    return astoria


def cases_to_daily_target(
    cases_week: list[float],
    n_weeks: int,
    n_agents: int = 37518,
) -> torch.Tensor:
    """Convert weekly case counts to a per-timestep target for daily_infected.

    The AT covid sim tracks `daily_infected` as cumulative new infections per
    substep. One AT timestep ~ one day. We divide weekly cases by 7 to get a
    rough daily rate, then scale by (sim_population / real_population) if
    needed.

    Returns a tensor of shape (n_weeks * 7,) with daily targets, or
    (n_weeks,) if we compare at weekly granularity.
    """
    weekly = torch.tensor(cases_week[:n_weeks], dtype=torch.float32)
    return weekly



def snapshot_state(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().clone()
        elif isinstance(v, dict):
            out[k] = snapshot_state(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def restore_state(target: dict, snap: dict) -> None:
    for k, v in snap.items():
        if isinstance(v, dict):
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            restore_state(target[k], v)
        elif isinstance(v, torch.Tensor):
            target[k] = v.detach().clone()
        else:
            target[k] = copy.deepcopy(v)



def calibrate_r2(
    runner,
    target: torch.Tensor,
    *,
    steps_per_week: int = 7,
    n_weeks: int = 5,
    n_iters: int = 30,
    lr: float = 0.3,
    seed_frac: float = 0.05,
    seed: int = 0,
) -> dict:
    """Fit R2 to match real weekly case data.

    Each iteration: reset state -> seed infections -> roll out
    n_weeks * steps_per_week AT steps -> compare weekly aggregated
    daily_infected against target -> backward -> step R2.
    """
    seed_initial_infections(runner, fraction=seed_frac, seed=seed)
    n_agents = runner.state["agents"]["citizens"]["age"].shape[0]

    # Zero out platform_signal (no policy model in phase 1)
    runner.state["agents"]["citizens"]["platform_signal"] = torch.zeros(n_agents)

    initial_snap = snapshot_state(runner.state)

    # Locate R2 parameter
    transmission = runner.initializer.transition_function["0"].new_transmission
    r2_param = transmission.calibrate_R2
    r2_default = r2_param.detach().clone()

    print(f"  R2 initial value: {r2_param.detach().flatten()[:3].tolist()}")
    print(f"  target (weekly cases): {target.tolist()}")

    optimizer = torch.optim.Adam([r2_param], lr=lr)
    total_steps = steps_per_week * n_weeks
    history = []

    for it in range(n_iters):
        restore_state(runner.state, initial_snap)
        runner.reset_state_before_episode()
        optimizer.zero_grad()

        runner.step(num_steps=total_steps)

        # Aggregate daily_infected into weekly buckets
        # daily_infected is cumulative across the rollout; take the final value
        # and compare against total cases over the period
        di = runner.state["environment"]["daily_infected"]
        pred_total = di.sum()
        target_total = target[:n_weeks].sum()

        loss = (pred_total - target_total) ** 2 / (target_total ** 2 + 1e-8)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            r2_param.clamp_(min=0.1)

        rec = {
            "iter": it,
            "loss": float(loss.detach()),
            "R2": float(r2_param.detach().flatten()[0]),
            "pred_total": float(pred_total.detach()),
            "target_total": float(target_total),
        }
        history.append(rec)

        if it % 5 == 0 or it == n_iters - 1:
            print(
                f"  iter {it:3d}  loss={rec['loss']:.6f}  "
                f"R2={rec['R2']:.4f}  "
                f"pred={rec['pred_total']:.0f}  target={rec['target_total']:.0f}"
            )

    return {
        "history": history,
        "R2_calibrated": r2_param.detach().clone(),
        "R2_default": r2_default,
    }



def calibrate_policy(
    *,
    r2_value: torch.Tensor,
    target: torch.Tensor,
    steps_per_week: int = 7,
    n_weeks: int = 5,
    n_iters: int = 30,
    lr: float = 0.01,
    seed_frac: float = 0.05,
    seed: int = 0,
) -> dict:
    """With R2 calibrated, fit a linear policy model (w, b) so the ABM +
    policy produces the target epidemic trajectory.

    Uses grad_run to get gradients through the full pipeline:
    model params -> isolation policy -> epidemic dynamics -> daily_infected.
    """
    env = make_covid_env(
        init_seed=seed,
        signal_writer=default_signal_writer_grad,
        initial_infections_fraction=seed_frac,
    )

    # Set calibrated R2
    transmission = env.runner.initializer.transition_function["0"].new_transmission
    with torch.no_grad():
        transmission.calibrate_R2.copy_(r2_value)
    print(f"  R2 set to calibrated value: {r2_value.flatten()[:3].tolist()}")

    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)

    initial_snap = snapshot_state(env.runner.state)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = steps_per_week * n_weeks
    history = []

    for it in range(n_iters):
        restore_state(env.runner.state, initial_snap)
        env.runner.reset_state_before_episode()
        optimizer.zero_grad()

        env.grad_run(model, n_steps=total_steps)

        di = env.runner.state["environment"]["daily_infected"]
        pred_total = di.sum()
        target_total = target[:n_weeks].sum()

        loss = (pred_total - target_total) ** 2 / (target_total ** 2 + 1e-8)

        loss.backward()
        optimizer.step()

        rec = {
            "iter": it,
            "loss": float(loss.detach()),
            "w": float(model.weight.detach()),
            "b": float(model.bias.detach()),
            "pred_total": float(pred_total.detach()),
            "target_total": float(target_total),
        }
        history.append(rec)

        if it % 5 == 0 or it == n_iters - 1:
            print(
                f"  iter {it:3d}  loss={rec['loss']:.6f}  "
                f"w={rec['w']:.4f}  b={rec['b']:.4f}  "
                f"pred={rec['pred_total']:.0f}  target={rec['target_total']:.0f}"
            )

    return {
        "history": history,
        "model_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }



def run_surge(out_dir: Path, *, start_week: int = 17, n_weeks: int = 3,
              n_iters_per: int = 30, lr: float = 0.5, seed: int = 0) -> None:
    """Grid-search seed fraction x gradient-R2 against the Astoria winter surge."""
    out_dir.mkdir(parents=True, exist_ok=True)
    astoria = load_astoria_cases()
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
        "dates": target_dates, "cases_week": target_cases, "total": target_total,
    }, indent=2))

    print("\nBuilding AT covid runner...", flush=True)
    t0 = time.time()
    runner = build_covid_runner(seed=seed)
    n_agents = runner.state["agents"]["citizens"]["age"].shape[0]
    runner.state["agents"]["citizens"]["platform_signal"] = torch.zeros(n_agents)
    print(f"  {n_agents} agents, built in {time.time() - t0:.1f}s")

    transmission = runner.initializer.transition_function["0"].new_transmission
    r2_param = transmission.calibrate_R2
    r2_default = float(r2_param.detach().flatten()[0])
    total_steps = 7 * n_weeks
    seed_fractions = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

    all_results = []
    for sf in seed_fractions:
        with torch.no_grad():
            r2_param.fill_(r2_default)
        seed_initial_infections(runner, fraction=sf, seed=seed)
        seeded_snap = snapshot_state(runner.state)
        optimizer = torch.optim.Adam([r2_param], lr=lr)
        best_loss, best_r2, best_pred = float("inf"), r2_default, 0.0
        for _ in range(n_iters_per):
            restore_state(runner.state, seeded_snap)
            runner.reset_state_before_episode()
            optimizer.zero_grad()
            runner.step(num_steps=total_steps)
            pred_total = runner.state["environment"]["daily_infected"].sum()
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
        n_seeded = int((runner.state["agents"]["citizens"]["disease_stage"].squeeze() == 2.0).sum())
        all_results.append({
            "seed_frac": sf, "n_seeded": n_seeded, "best_loss": best_loss,
            "best_R2": best_r2, "best_pred": best_pred, "target": target_total,
            "ratio": best_pred / target_total,
        })
        print(f"  frac={sf:.3f}  seeded={n_seeded:5d}  R2={best_r2:.3f}  "
              f"pred={best_pred:.0f}  target={target_total:.0f}  "
              f"ratio={best_pred / target_total:.3f}  loss={best_loss:.6f}")

    best = min(all_results, key=lambda r: r["best_loss"])
    print(f"\nBest fit: seed_frac={best['seed_frac']:.3f}  R2={best['best_R2']:.3f}  "
          f"pred={best['best_pred']:.0f}  target={target_total:.0f}  ratio={best['ratio']:.3f}")
    (out_dir / "grid_results.json").write_text(json.dumps(all_results, indent=2))
    with torch.no_grad():
        r2_param.fill_(best["best_R2"])
    torch.save(r2_param.detach().clone(), out_dir / "calibrated_R2.pt")
    torch.save({
        "R2": best["best_R2"], "seed_frac": best["seed_frac"],
        "target_total": target_total, "pred_total": best["best_pred"],
    }, out_dir / "calibrated_params.pt")
    _plot_surge(all_results, target_total, target_dates, best, r2_default, out_dir)
    print(f"\nAll outputs in {out_dir}")


def _plot_surge(all_results, target_total, target_dates, best, r2_default, out_dir):
    if not _HAS_MPL:
        print("matplotlib unavailable; skipping plot")
        return
    fracs = [r["seed_frac"] for r in all_results]
    preds = [r["best_pred"] for r in all_results]
    r2s = [r["best_R2"] for r in all_results]
    losses = [r["best_loss"] for r in all_results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(range(len(fracs)), preds, color="#4CAF50", alpha=0.8)
    axes[0].axhline(target_total, color="#E91E63", linestyle="--", linewidth=2, label=f"target={target_total:.0f}")
    axes[0].set_xticks(range(len(fracs)))
    axes[0].set_xticklabels([f"{f:.1%}" for f in fracs], rotation=45)
    axes[0].set(xlabel="Initial infection fraction", ylabel="Total infections", title="Fit by seed fraction")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(fracs, r2s, "o-", color="#FF5722", linewidth=2)
    axes[1].axhline(r2_default, color="gray", linestyle="--", alpha=0.5, label=f"default={r2_default}")
    axes[1].set(xlabel="Initial infection fraction", ylabel="Best R2", title="Calibrated R2")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(fracs, losses, "o-", color="#2196F3", linewidth=2)
    axes[2].set(xlabel="Initial infection fraction", ylabel="Relative MSE", title="Loss")
    axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_plot.png", dpi=150, bbox_inches="tight")
    print(f"  Plot: {out_dir / 'calibration_plot.png'}")


def run_pick_best(runs_dir: Path) -> None:
    """Read all calibration result.pt files and print the best params per season."""
    if not runs_dir.exists():
        print(f"No results yet at {runs_dir}")
        return
    seasons: dict[str, list] = defaultdict(list)
    for result_path in sorted(runs_dir.glob("*/result.pt")):
        result = torch.load(result_path, weights_only=False)
        result["tag"] = result_path.parent.name
        result["ratio"] = result["best_pred"] / result["target_total"]
        seasons[result["tag"].split("_f")[0]].append(result)
    if not seasons:
        print(f"No result.pt files found in {runs_dir}/*/")
        return
    print(f"{'Season':<12} {'Best tag':<22} {'Frac':>6} {'R2':>7} "
          f"{'Pred':>7} {'Target':>7} {'Ratio':>7} {'Loss':>10}")
    print("-" * 90)
    best_per_season = {}
    for season in sorted(seasons):
        best = min(seasons[season], key=lambda r: r["best_loss"])
        best_per_season[season] = best
        print(f"{season:<12} {best['tag']:<22} {best['seed_frac']:>6.3f} "
              f"{best['R2']:>7.3f} {best['best_pred']:>7.0f} "
              f"{best['target_total']:>7.0f} {best['ratio']:>7.3f} {best['best_loss']:>10.6f}")
    out_path = runs_dir / "best_per_season.pt"
    torch.save(best_per_season, out_path)
    print(f"\nSaved to {out_path}")


def run_r2(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load real data
    print("Loading Astoria case data...", flush=True)
    astoria = load_astoria_cases()
    cases = astoria["cases_week"].tolist()
    dates = astoria["date"].tolist()

    # Select the window to calibrate against
    start = args.start_week
    end = start + args.n_weeks
    target_cases = cases[start:end]
    target_dates = dates[start:end]
    target = cases_to_daily_target(target_cases, n_weeks=args.n_weeks)

    print(f"Target window: weeks {start}-{end-1}")
    for i, (d, c) in enumerate(zip(target_dates, target_cases)):
        print(f"  week {i}: {d}  cases={c:.0f}")

    target_info = {
        "start_week": start,
        "n_weeks": args.n_weeks,
        "dates": target_dates,
        "cases_week": target_cases,
    }
    (out_dir / "target_data.json").write_text(json.dumps(target_info, indent=2))

    # Phase 1: Calibrate R2
    print(f"\n{'='*60}", flush=True)
    print("Phase 1: Calibrating R2 to real Astoria data", flush=True)
    print(f"{'='*60}", flush=True)

    runner = build_covid_runner(seed=args.seed)
    r2_result = calibrate_r2(
        runner,
        target,
        n_weeks=args.n_weeks,
        n_iters=args.n_iters_r2,
        lr=args.lr_r2,
        seed_frac=args.seed_frac,
        seed=args.seed,
    )

    torch.save(r2_result["R2_calibrated"], out_dir / "calibrated_R2.pt")
    (out_dir / "r2_history.json").write_text(
        json.dumps(r2_result["history"], indent=2)
    )
    print(f"\nR2 calibrated: {r2_result['R2_default'].flatten()[0]:.4f} -> "
          f"{r2_result['R2_calibrated'].flatten()[0]:.4f}")

    # Phase 2: Fit policy model (optional)
    if args.fit_policy:
        print(f"\n{'='*60}", flush=True)
        print("Phase 2: Fitting policy model with calibrated R2", flush=True)
        print(f"{'='*60}", flush=True)

        policy_result = calibrate_policy(
            r2_value=r2_result["R2_calibrated"],
            target=target,
            n_weeks=args.n_weeks,
            n_iters=args.n_iters_policy,
            lr=args.lr_policy,
            seed_frac=args.seed_frac,
            seed=args.seed,
        )

        torch.save(policy_result["model_state_dict"], out_dir / "calibrated_model.pt")
        (out_dir / "policy_history.json").write_text(
            json.dumps(policy_result["history"], indent=2)
        )
        w = policy_result["model_state_dict"]["weight"].item()
        b = policy_result["model_state_dict"]["bias"].item()
        print(f"\nPolicy calibrated: w={w:.4f}, b={b:.4f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Outputs in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate AT covid to real Astoria data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_r2 = sub.add_parser("r2", help="Gradient R2 calibration (+ optional policy fit)")
    p_r2.add_argument("--out-dir", type=str, default="runs/calibrated_covid")
    p_r2.add_argument("--fit-policy", action="store_true", help="Also fit policy model (phase 2)")
    p_r2.add_argument("--n-weeks", type=int, default=5, help="Weeks of real data to fit")
    p_r2.add_argument("--n-iters-r2", type=int, default=30, help="R2 calibration iterations")
    p_r2.add_argument("--n-iters-policy", type=int, default=30, help="Policy calibration iterations")
    p_r2.add_argument("--lr-r2", type=float, default=0.3, help="R2 learning rate")
    p_r2.add_argument("--lr-policy", type=float, default=0.01, help="Policy model learning rate")
    p_r2.add_argument("--seed-frac", type=float, default=0.05, help="Initial infection fraction")
    p_r2.add_argument("--seed", type=int, default=0)
    p_r2.add_argument("--start-week", type=int, default=0,
                      help="Index into Astoria data to start from (0=Aug 2020)")

    p_surge = sub.add_parser("surge", help="Seed-fraction x R2 grid search vs the winter surge")
    p_surge.add_argument("--out-dir", type=str, default="runs/calibrated_covid_surge")
    p_surge.add_argument("--start-week", type=int, default=17)
    p_surge.add_argument("--n-weeks", type=int, default=3)
    p_surge.add_argument("--n-iters", type=int, default=30)
    p_surge.add_argument("--lr", type=float, default=0.5)
    p_surge.add_argument("--seed", type=int, default=0)

    p_pick = sub.add_parser("pick-best", help="Best calibration params per season")
    p_pick.add_argument("--runs-dir", type=str, default="runs/calibration")

    args = parser.parse_args()
    if args.cmd == "r2":
        run_r2(args)
    elif args.cmd == "surge":
        run_surge(Path(args.out_dir), start_week=args.start_week, n_weeks=args.n_weeks,
                  n_iters_per=args.n_iters, lr=args.lr, seed=args.seed)
    elif args.cmd == "pick-best":
        run_pick_best(Path(args.runs_dir))


if __name__ == "__main__":
    main()
