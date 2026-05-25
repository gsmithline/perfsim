"""Calibrate AT covid ABM to real Astoria weekly case data.

Fits the transmission parameter R2 (and optionally the policy model) so the
ABM's daily_infected trajectory matches real Astoria COVID case counts bundled
in agent_torch.

Two phases:
  Phase 1: Calibrate R2 via gradient descent through the ABM (AT's own
           learnable parameter). No perfsim predictor involved.
  Phase 2 (optional): With R2 calibrated, fit a linear policy model's (w, b)
           so the ABM + policy produces a target epidemic trajectory.

Outputs:
  $OUT_DIR/calibrated_R2.pt         calibrated R2 tensor
  $OUT_DIR/calibrated_model.pt      calibrated policy model state_dict (phase 2)
  $OUT_DIR/calibration_history.json  per-iteration loss and parameter values
  $OUT_DIR/target_data.json          the real Astoria case data used as target

Usage:
  python scripts/calibrate_covid.py                    # phase 1 only
  python scripts/calibrate_covid.py --fit-policy       # phase 1 + phase 2
  python scripts/calibrate_covid.py --out-dir runs/calibrated
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import pandas as pd
import torch

from perfsim.scenarios.at_covid import (
    build_covid_runner,
    default_signal_writer_grad,
    make_covid_env,
    seed_initial_infections,
)


# ---- Load real Astoria data -------------------------------------------------


def load_astoria_cases() -> pd.DataFrame:
    """Load the bundled Astoria weekly case data from agent_torch."""
    import agent_torch

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


# ---- State snapshot/restore -------------------------------------------------


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


# ---- Phase 1: Calibrate R2 -------------------------------------------------


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


# ---- Phase 2: Fit policy model ---------------------------------------------


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


# ---- Main -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Calibrate AT covid to real Astoria data")
    parser.add_argument("--out-dir", type=str, default="runs/calibrated_covid")
    parser.add_argument("--fit-policy", action="store_true", help="Also fit policy model (phase 2)")
    parser.add_argument("--n-weeks", type=int, default=5, help="Weeks of real data to fit")
    parser.add_argument("--n-iters-r2", type=int, default=30, help="R2 calibration iterations")
    parser.add_argument("--n-iters-policy", type=int, default=30, help="Policy calibration iterations")
    parser.add_argument("--lr-r2", type=float, default=0.3, help="R2 learning rate")
    parser.add_argument("--lr-policy", type=float, default=0.01, help="Policy model learning rate")
    parser.add_argument("--seed-frac", type=float, default=0.05, help="Initial infection fraction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-week", type=int, default=0,
                        help="Index into Astoria data to start from (0=Aug 2020)")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
