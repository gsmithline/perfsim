"""Calibration of Schelling parameters against NYC dissimilarity indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from perfsim.scenarios.at_schelling import make_schelling_env


NYC_DISSIMILARITY = {
    "black_white": 0.79,
    "hispanic_white": 0.62,
    "asian_white": 0.51,
}

TYPE_IDX = {"white": 0, "black": 1, "hispanic": 2, "asian": 3}

PAIR_TARGETS = {
    "black_white": ("black", "white", 0.79),
    "hispanic_white": ("hispanic", "white", 0.62),
    "asian_white": ("asian", "white", 0.51),
}


def _auto_blocks(grid_size: int, cells_per_block: int = 5) -> int:
    return max(2, grid_size // cells_per_block)


def dissimilarity_index(grid_type: torch.Tensor, type_a: int, type_b: int, n_blocks: int | None = None) -> float:
    H, W = grid_type.shape
    if n_blocks is None:
        n_blocks = _auto_blocks(min(H, W))
    bh = H // n_blocks
    bw = W // n_blocks
    total_a = int((grid_type == type_a).sum().item())
    total_b = int((grid_type == type_b).sum().item())
    if total_a == 0 or total_b == 0:
        return 0.0
    s = 0.0
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = grid_type[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            a_i = int((block == type_a).sum().item())
            b_i = int((block == type_b).sum().item())
            s += abs(a_i / total_a - b_i / total_b)
    return 0.5 * s


def run_pure_schelling(
    H_0_per_type: list[float],
    n_rounds: int,
    num_agents: int = 100,
    grid: int = 12,
    seed: int = 0,
) -> torch.Tensor:
    env = make_schelling_env(
        init_seed=seed,
        num_agents=num_agents,
        grid_height=grid,
        grid_width=grid,
        baseline_threshold=float(np.mean(H_0_per_type)),
        baseline_threshold_per_type=list(H_0_per_type),
        lambda_=0.0,
        num_steps_per_episode=n_rounds,
    )
    runner = env.runner
    n = num_agents
    runner.state["agents"]["residents"]["platform_signal"] = torch.full((n,), 0.0)
    runner.step(num_steps=n_rounds)
    return runner.state["environment"]["grid_type"]


def compute_all_d(grid_type: torch.Tensor) -> dict[str, float]:
    out = {}
    for pair_name, (a, b, _target) in PAIR_TARGETS.items():
        out[pair_name] = dissimilarity_index(grid_type, type_a=TYPE_IDX[a], type_b=TYPE_IDX[b])
    return out


def evaluate_H_0(
    H_0_per_type: list[float],
    n_rounds: int,
    num_agents: int,
    grid_size: int,
    seeds: list[int],
) -> dict:
    targets = {k: v[2] for k, v in PAIR_TARGETS.items()}
    d_per_seed: dict[str, list[float]] = {k: [] for k in PAIR_TARGETS}
    for s in seeds:
        grid_type = run_pure_schelling(H_0_per_type, n_rounds, num_agents=num_agents, grid=grid_size, seed=s)
        d_dict = compute_all_d(grid_type)
        for k, v in d_dict.items():
            d_per_seed[k].append(v)
    d_means = {k: sum(v) / len(v) for k, v in d_per_seed.items()}
    per_pair_loss = {k: (d_means[k] - targets[k]) ** 2 for k in targets}
    loss = sum(per_pair_loss.values())
    return {
        "H_0_per_type": list(H_0_per_type),
        "d_means": d_means,
        "targets": targets,
        "per_pair_loss": per_pair_loss,
        "loss": loss,
    }


def nelder_mead_optimize(
    init_H_0: list[float],
    n_rounds: int,
    num_agents: int,
    grid_size: int,
    seeds: list[int],
    max_iter: int = 50,
) -> dict:
    trace: list[dict] = []
    type_names = ["White", "Black", "Hispanic", "Asian"]

    def objective(H_vec: np.ndarray) -> float:
        H_clamped = list(np.clip(H_vec, 0.05, 0.95))
        result = evaluate_H_0(H_clamped, n_rounds, num_agents, grid_size, seeds)
        trace.append(result)
        line = "  ".join(f"{n}={h:.2f}" for n, h in zip(type_names, H_clamped))
        print(f"  iter {len(trace):2d}  loss={result['loss']:.4f}  H=[{line}]")
        return result["loss"]

    print(f"Nelder-Mead from H_0={init_H_0}")
    res = minimize(
        objective,
        x0=np.array(init_H_0, dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 0.02, "fatol": 0.005, "adaptive": True},
    )

    best_H = list(np.clip(res.x, 0.05, 0.95))
    best_result = min(trace, key=lambda r: r["loss"])
    print(f"\nNelder-Mead finished: success={res.success}, n_evals={len(trace)}")
    return {"best": best_result, "trace": trace, "final_x": best_H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-H0", type=str, default="0.3,0.3,0.3,0.3", help="initial H_0 per type (4 comma-separated floats: White,Black,Hispanic,Asian)")
    ap.add_argument("--n-rounds", type=int, default=25)
    ap.add_argument("--n-agents", type=int, default=100)
    ap.add_argument("--grid-size", type=int, default=15)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--out", type=str, default="experiments/runs/schelling_calibration.json")
    args = ap.parse_args()

    init_H_0 = [float(x) for x in args.init_H0.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"Per-type Nelder-Mead calibration against 3 NYC targets")
    print(f"{args.n_agents} agents, {args.grid_size}x{args.grid_size} grid, {args.n_rounds} rounds, {len(seeds)} seeds, max_iter={args.max_iter}")
    print(f"Types: 0=White 1=Black 2=Hispanic 3=Asian")
    print(f"Targets: Black-White=0.79, Hispanic-White=0.62, Asian-White=0.51")
    print()
    out = nelder_mead_optimize(init_H_0, args.n_rounds, args.n_agents, args.grid_size, seeds, args.max_iter)

    best = out["best"]
    type_names = ["White", "Black", "Hispanic", "Asian"]
    print(f"\nBEST: loss={best['loss']:.4f}")
    print(f"  H_0_per_type:")
    for n, h in zip(type_names, best["H_0_per_type"]):
        print(f"    {n}: {h:.3f}")
    print(f"  Pair Ds:")
    for k in best["targets"]:
        sim = best["d_means"][k]; tgt = best["targets"][k]
        print(f"    {k}: simulated={sim:.3f}  target={tgt:.3f}  diff={sim - tgt:+.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
