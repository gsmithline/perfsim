"""Sweep uniform isolation levels on the COVID ABM to verify policy sensitivity."""

from __future__ import annotations

import os
import json
from pathlib import Path

import torch

from perfsim.scenarios.at_covid.env import build_covid_runner, seed_initial_infections


def main() -> int:
    seed_frac = float(os.environ.get("SEED_FRAC", "0.05"))
    calibrated_r2 = os.environ.get("CALIBRATED_R2")
    k_steps = int(os.environ.get("K_STEPS", "21"))
    seed = int(os.environ.get("SEED", "0"))

    isolation_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    results = []

    print(f"Sensitivity sweep: seed_frac={seed_frac}, R2={calibrated_r2 or 'default'}, "
          f"k_steps={k_steps}, seed={seed}")
    print(f"{'isolation':>10} | {'di_sum':>8} | {'frac_nonS':>10} | {'S':>6} | {'E':>5} | {'I':>5} | {'R':>5} | {'D':>5}")
    print("-" * 75)

    for iso in isolation_levels:
        torch.manual_seed(seed)
        runner = build_covid_runner(seed)
        seed_initial_infections(runner, fraction=seed_frac, seed=seed)

        if calibrated_r2:
            runner.state["disease_stages"]["R2"] = torch.tensor([float(calibrated_r2)])

        p = max(0.01, min(0.99, iso))
        logit_val = torch.log(torch.tensor(p / (1.0 - p)))
        n = runner.state["agents"]["citizens"]["age"].shape[0]
        runner.state["agents"]["citizens"]["platform_signal"] = torch.full((n,), logit_val.item())

        runner.step(num_steps=k_steps)

        ds = runner.state["agents"]["citizens"]["disease_stage"].squeeze()
        di = runner.state["environment"]["daily_infected"]
        di_sum = float(di.sum().item())
        frac_nonS = float((ds > 0).float().mean().item())
        sc = {
            "S": int((ds == 0).sum().item()),
            "E": int((ds == 1).sum().item()),
            "I": int((ds == 2).sum().item()),
            "R": int((ds == 3).sum().item()),
            "D": int((ds == 4).sum().item()),
        }

        print(f"{iso:>10.2f} | {di_sum:>8.0f} | {frac_nonS:>10.4f} | "
              f"{sc['S']:>6} | {sc['E']:>5} | {sc['I']:>5} | {sc['R']:>5} | {sc['D']:>5}")

        results.append({"isolation": iso, "di_sum": di_sum, "frac_nonS": frac_nonS, **sc})

    out = Path("runs/sensitivity_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
