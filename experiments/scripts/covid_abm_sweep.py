"""Diagnostic: sweep fixed isolation values through AT covid ABM, no LLM."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from perfsim.scenarios.at_covid import (
    make_covid_env,
    seed_initial_infections,
)


def run_fixed_isolation(isolation: float, *, n_rounds: int = 5, k_steps: int = 3, seed: int = 0) -> list[dict]:
    torch.manual_seed(seed)

    def fixed_signal_writer(runner, preds):
        n = runner.state["agents"]["citizens"]["age"].shape[0]
        p = torch.full((n,), isolation).clamp(min=0.01, max=0.99)
        logit_p = torch.log(p / (1.0 - p))
        runner.state["agents"]["citizens"]["platform_signal"] = logit_p

    env = make_covid_env(
        init_seed=seed,
        signal_writer=fixed_signal_writer,
        initial_infections_fraction=0.05,
    )

    class _Dummy(torch.nn.Module):
        def __init__(self): super().__init__(); self._p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x): return torch.full((x.shape[0], 1), isolation, dtype=torch.float32)

    model = _Dummy()
    rows = []
    for t in range(n_rounds):
        env.run(model, n_steps=k_steps)
        citizens = env.runner.state["agents"]["citizens"]
        env_state = env.runner.state["environment"]
        ds = citizens["disease_stage"].squeeze()
        n_susceptible = int((ds == 0).sum().item())
        n_exposed = int((ds == 1).sum().item())
        n_infected = int((ds == 2).sum().item())
        n_recovered = int((ds == 3).sum().item())
        n_total = int(ds.shape[0])
        di = env_state.get("daily_infected")
        di_sum = float(di.sum().item()) if di is not None else 0.0
        rows.append({
            "round": t,
            "isolation_in": isolation,
            "n_susceptible": n_susceptible,
            "n_exposed": n_exposed,
            "n_infected": n_infected,
            "n_recovered": n_recovered,
            "fraction_non_S": (n_total - n_susceptible) / n_total,
            "daily_infected_sum": di_sum,
        })
    return rows


def main() -> int:
    isolations = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_rounds = 5
    print(f"\nSweeping isolation across {isolations} for {n_rounds} rounds each\n")
    print(f"{'iso':>5} {'round':>5} {'S':>6} {'E':>6} {'I':>6} {'R':>6} {'frac_nonS':>10} {'di_sum':>10}")
    print("-" * 70)

    all_results = []
    for iso in isolations:
        t0 = time.time()
        rows = run_fixed_isolation(iso, n_rounds=n_rounds, seed=0)
        elapsed = time.time() - t0
        for r in rows:
            print(
                f"{r['isolation_in']:>5.2f} {r['round']:>5} {r['n_susceptible']:>6} "
                f"{r['n_exposed']:>6} {r['n_infected']:>6} {r['n_recovered']:>6} "
                f"{r['fraction_non_S']:>10.4f} {r['daily_infected_sum']:>10.1f}"
            )
        print(f"  -> {elapsed:.1f}s\n")
        all_results.append({"isolation": iso, "trajectory": rows})

    out_path = Path("experiments/runs/covid_abm_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))

    print("--- DIAGNOSTIC ---")
    final = [r["trajectory"][-1] for r in all_results]
    di_range = max(r["daily_infected_sum"] for r in final) - min(r["daily_infected_sum"] for r in final)
    nonS_range = max(r["fraction_non_S"] for r in final) - min(r["fraction_non_S"] for r in final)
    print(f"After {n_rounds} rounds, outcomes across isolation 0->1:")
    print(f"  daily_infected_sum range: {di_range:.1f}")
    print(f"  fraction_non_S range:     {nonS_range:.4f}")
    if di_range < 10 and nonS_range < 0.01:
        print("  ABM SATURATED")
    else:
        print("  ABM DIFFERENTIATES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
