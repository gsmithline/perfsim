"""Read all calibration results and print the best (R2, seed_frac) per season.

Usage: python scripts/pick_best_calibration.py [--runs-dir runs/calibration]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs/calibration")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"No results yet at {runs_dir}")
        return

    seasons = defaultdict(list)

    for result_path in sorted(runs_dir.glob("*/result.pt")):
        tag = result_path.parent.name
        result = torch.load(result_path, weights_only=False)

        season = tag.split("_f")[0]
        result["tag"] = tag
        result["ratio"] = result["best_pred"] / result["target_total"]
        seasons[season].append(result)

    if not seasons:
        print(f"No result.pt files found in {runs_dir}/*/")
        return

    print(f"{'Season':<12} {'Best tag':<22} {'Frac':>6} {'R2':>7} "
          f"{'Pred':>7} {'Target':>7} {'Ratio':>7} {'Loss':>10}")
    print("-" * 90)

    best_per_season = {}

    for season in sorted(seasons):
        results = seasons[season]
        best = min(results, key=lambda r: r["best_loss"])
        best_per_season[season] = best

        print(f"{season:<12} {best['tag']:<22} {best['seed_frac']:>6.3f} "
              f"{best['R2']:>7.3f} {best['best_pred']:>7.0f} "
              f"{best['target_total']:>7.0f} {best['ratio']:>7.3f} "
              f"{best['best_loss']:>10.6f}")

        if len(results) > 1:
            for r in sorted(results, key=lambda r: r["best_loss"]):
                marker = " <-- best" if r["tag"] == best["tag"] else ""
                print(f"  {r['tag']:<20} frac={r['seed_frac']:.3f}  "
                      f"R2={r['R2']:.3f}  pred={r['best_pred']:.0f}  "
                      f"ratio={r['ratio']:.3f}  loss={r['best_loss']:.6f}{marker}")
            print()

    # Save best params per season
    out_path = runs_dir / "best_per_season.pt"
    torch.save(best_per_season, out_path)
    print(f"\nSaved to {out_path}")
    print("\nTo load in run_covid_lm.py:")
    print("  best = torch.load('runs/calibration/best_per_season.pt')")
    print("  best['alpha']['R2'], best['alpha']['seed_frac']")


if __name__ == "__main__":
    main()
