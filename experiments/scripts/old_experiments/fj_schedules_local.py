"""Deployment-schedule factorial, locally with a non-LLM predictor (no GPU).

Mirrors experiments/condor/configs_pokec_fj_schedules.txt but runs on CPU with
a chosen baseline predictor, so the cadence + data-aggregation effects can be
inspected fast before launching the LLM sweep:

  1. deploy every round   -> replace, accumulate
  2. deploy every K=3      -> replace, accumulate, deployed_into, not_deployed_into

Tracks the full collapse suite (mean/var/entropy/eff_support/mode_mass/gini/
Jaccard) for both the population and the predictions, writes JSON + a figure.

Usage: python experiments/scripts/fj_schedules_local.py [predictor] [K]
  predictor in {ridge, mlp, mean, perfect}  (default ridge)
  K = deploy-every for the cadence cells     (default 3)
"""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BASE = Path(__file__).resolve().parent / "fj_baselines_local.py"
_spec = importlib.util.spec_from_file_location("fj_baselines_local", _BASE)
bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bl)


def main():
    predictor = sys.argv[1] if len(sys.argv) > 1 else "ridge"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    data = bl.load_pokec()
    innate = data["innate"]
    out = Path(f"runs/fj_schedules_local/{predictor}_K{K}")
    out.mkdir(parents=True, exist_ok=True)

    cells = [
        ("K1_replace", 1, "replace"),
        ("K1_accumulate", 1, "accumulate"),
        (f"K{K}_replace", K, "replace"),
        (f"K{K}_accumulate", K, "accumulate"),
        (f"K{K}_deployed_into", K, "deployed_into"),
        (f"K{K}_not_deployed_into", K, "not_deployed_into"),
    ]
    print(f"predictor={predictor}  innate mean={innate.mean():.4f} var={innate.var():.5f}\n")
    traj = {}
    for name, de, reg in cells:
        rows = bl.run_baseline(predictor, data, deploy_every=de, data_regime=reg)
        traj[name] = rows
        r = rows[-1]
        print(f"{name:24} final  op_mean={r['op_mean']:.4f} op_var={r['op_var']:.5f} "
              f"op_eff_sup={r['op_eff_support']:.2f} pred_mean={r['pred_mean']:.4f}")
    (out / "schedules.json").write_text(json.dumps(traj, indent=2))

    panels = [("op_mean", "mean opinion", float(innate.mean())),
              ("op_var", "variance", float(innate.var())),
              ("op_eff_support", "eff support", bl.cm.eff_support(innate)),
              ("jaccard_init", "Jaccard vs round 0", None)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title, ref) in zip(axes.ravel(), panels):
        for name, rows in traj.items():
            ts = [r["round"] for r in rows if key in r]
            ax.plot(ts, [r[key] for r in rows if key in r], label=name)
        if ref is not None:
            ax.axhline(ref, ls=":", c="gray", label="innate")
        ax.set(title=f"{predictor}: population {title}", xlabel="retraining step t", ylabel=title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fj_schedules.png", dpi=110)
    print(f"\nwrote {out}/schedules.json and {out}/fj_schedules.png")


if __name__ == "__main__":
    main()
