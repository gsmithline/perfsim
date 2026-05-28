"""Diagnostic: sweep fixed consumption values through AT macro ABM, no LLM.

Tests whether the ABM differentiates outcomes given different policy inputs,
or whether dynamics saturate to the same equilibrium regardless of policy.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from perfsim.scenarios._deprecated.at_macro import make_macro_env


def run_fixed_policy(
    consumption: float,
    *,
    n_rounds: int = 5,
    k_steps: int = 3,
    n_agents: int = 100,
    seed: int = 0,
    calibrated_uac_path: str | None = None,
) -> list[dict]:
    """Run the macro ABM with a fixed per-agent consumption value each round."""
    torch.manual_seed(seed)

    def fixed_signal_writer(runner, preds):
        n = runner.state["agents"]["consumers"]["age"].shape[0]
        p = torch.full((n,), consumption).clamp(min=0.01, max=0.99)
        logit_p = torch.log(p / (1.0 - p))
        runner.state["agents"]["consumers"]["platform_signal"] = logit_p

    env = make_macro_env(
        init_seed=seed,
        yaml_name="config_100_agents.yaml",
        n_agents=n_agents,
        signal_writer=fixed_signal_writer,
        keep_trajectory=True,
    )

    if calibrated_uac_path:
        uac_data = torch.load(calibrated_uac_path, weights_only=False)
        for tf_key in env.runner.initializer.transition_function:
            tf = env.runner.initializer.transition_function[tf_key]
            for _, module in tf.named_modules():
                if hasattr(module, "external_UAC"):
                    n_steps_local = module.external_UAC.shape[0]
                    with torch.no_grad():
                        module.external_UAC.copy_(uac_data[:n_steps_local])
                    break

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            n = x.shape[0]
            return torch.full((n, 1), consumption, dtype=torch.float32)

    model = _DummyModel()
    rows = []
    for t in range(n_rounds):
        env.run(model, n_steps=k_steps)
        state = env.runner.state
        consumers = state["agents"]["consumers"]
        env_state = state["environment"]

        assets = consumers["assets"].float().mean().item()
        cons_prop = consumers.get("consumption_propensity")
        cons_actual = float(cons_prop.float().mean().item()) if cons_prop is not None else 0.0

        pi = env_state.get("P_i")
        inflation = 0.0
        if pi is not None:
            try:
                val = float(pi[-1][-1].item())
                if abs(val) < 1.0:
                    inflation = val
            except Exception:
                pass

        u = env_state.get("U")
        unemployment = 0.0
        if u is not None:
            try:
                row = u[-1]
                nz = row.nonzero(as_tuple=True)[0]
                if len(nz):
                    unemployment = float(row[nz[-1]].item())
            except Exception:
                pass

        p = env_state.get("P")
        price = 0.0
        if p is not None:
            try:
                price = float(p[-1][-1].item())
            except Exception:
                pass

        rows.append({
            "round": t,
            "consumption_in": consumption,
            "consumption_actual": cons_actual,
            "mean_assets": assets,
            "inflation": inflation,
            "unemployment": unemployment,
            "price": price,
        })
    return rows


def main() -> int:
    consumption_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_rounds = 5

    print(f"\nSweeping consumption across {consumption_values} for {n_rounds} rounds each\n")
    print(f"{'cons':>5} {'round':>5} {'assets':>10} {'cons_act':>9} {'inflation':>10} {'unemp':>8} {'price':>8}")
    print("-" * 70)

    all_results = []
    for c in consumption_values:
        t0 = time.time()
        rows = run_fixed_policy(c, n_rounds=n_rounds, n_agents=100, seed=0)
        elapsed = time.time() - t0
        for r in rows:
            print(
                f"{r['consumption_in']:>5.2f} {r['round']:>5} {r['mean_assets']:>10.1f} "
                f"{r['consumption_actual']:>9.3f} {r['inflation']:>10.4f} "
                f"{r['unemployment']:>8.2f} {r['price']:>8.2f}"
            )
        print(f"  -> {elapsed:.1f}s\n")
        all_results.append({"consumption": c, "trajectory": rows})

    out_path = Path("experiments/runs/macro_abm_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"saved to {out_path}")

    print("\n--- DIAGNOSTIC ---")
    final_rounds = [r["trajectory"][-1] for r in all_results]
    asset_range = max(r["mean_assets"] for r in final_rounds) - min(r["mean_assets"] for r in final_rounds)
    inflation_range = max(r["inflation"] for r in final_rounds) - min(r["inflation"] for r in final_rounds)
    unemp_range = max(r["unemployment"] for r in final_rounds) - min(r["unemployment"] for r in final_rounds)
    print(f"After {n_rounds} rounds, outcomes across consumption 0.1->0.9:")
    print(f"  assets range:       {asset_range:.1f}")
    print(f"  inflation range:    {inflation_range:.4f}")
    print(f"  unemployment range: {unemp_range:.2f}")
    if asset_range < 100 and inflation_range < 0.05 and unemp_range < 1.0:
        print("  ABM SATURATED: outcomes converged regardless of policy input")
    else:
        print("  ABM DIFFERENTIATES: outcomes depend on policy input")
    return 0


if __name__ == "__main__":
    sys.exit(main())
