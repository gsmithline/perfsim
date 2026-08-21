#!/usr/bin/env python3
"""Analyzer for the adapter KL / soft-decode probe (2026-08-21). CPU only.

THE COMPARISON THIS EXISTS TO MAKE. Put the GREEDY distance to frozen
Qwen and the SOFT-DECODED distance to frozen Qwen on the same axis, over
the same dose dial. The dose wave found the greedy distance flat at
~0.31 at every dose. Two readings survive that:

  greedy flat AND soft flat        the entering model is not retained at
                                   any dose, in served space or in
                                   distribution -- the implicit-anchor
                                   hypothesis fails on its own terms.

  greedy flat BUT soft rising      the greedy map departs immediately
                                   while the distribution stays near the
                                   base at low dose: the flat greedy
                                   distance is an ARGMAX ARTIFACT and the
                                   implicit anchor is present internally,
                                   hidden by discretisation.

Both are real answers. Nothing here encodes an expectation of either,
and the CSV carries the raw curves so the reading is checkable.

READ KL WITH CARE. KL to the base grows with dose almost by
construction: more training moves the weights further. A monotone KL
curve is NOT evidence for the implicit anchor. It is reported because
its SHAPE against the greedy curve is informative -- specifically
whether KL is still small in a regime where the greedy map has already
fully departed. The normalized column kl_frac (this cell's KL over the
full-dose cell's) is the form in which that comparison is legible.

THE MECHANICAL PRECONDITION. base_margin is the base model's top-1 minus
top-2 probability at the decision position, over agents. The argmax
explanation is only AVAILABLE if that margin is small: a confident base
cannot be flipped by a small logit change. It is printed first for that
reason.

TAGS ARE PARSED, NOT REBUILT. Family and dial come from the run tag
itself. Re-deriving a tag from a grammar in a second place is what
produced the 5em05/5em5 mismatch in the dose analyzer.

Outputs (notes/pofd/adapter_kl_probe/):
  adapter_kl_cells.csv    one row per adapter: greedy vs soft vs KL
  adapter_kl.png / .pdf   the three families, greedy and soft together
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402

HERE = Path(os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())
REPO = HERE.parent.parent.parent
OUT_DIR = REPO / "notes" / "pofd" / "adapter_kl_probe"
DOSE_CSV = REPO / "notes" / "pofd" / "sft_dose_analysis" / "sft_dose_cells.csv"
STD_U, STD_LR, STD_RANK = 181, "5e-5", 512
TAG_RE = re.compile(r"^pofdsftdose_\w+?_u(\d+)_lr([0-9a-zA-Z]+)_rank(\d+)_")


def untok_lr(tok):
    """5em5 -> 5e-5, 1p25em5 -> 1.25e-5. Inverse of the generator's
    _lrtok, applied to the tag the job actually ran under."""
    return tok.replace("m", "-").replace("p", ".")


def parse_tag(tag):
    """(U, lr, rank, family) from the run tag. The shared full-dose cell
    sits at the top of all three families and is labelled 'endpoint'."""
    m = TAG_RE.match(tag)
    if not m:
        raise SystemExit(f"[akl] cannot parse dose tag {tag!r}")
    u, lr, rank = int(m.group(1)), untok_lr(m.group(2)), int(m.group(3))
    if u != STD_U:
        fam = "update"
    elif lr != STD_LR:
        fam = "lr"
    elif rank != STD_RANK:
        fam = "rank"
    else:
        fam = "endpoint"
    return u, lr, rank, fam


def dose_lookup(path=DOSE_CSV):
    """(U, lr, rank) -> the dose wave's greedy metrics."""
    if not Path(path).exists():
        print(f"[akl] NOTE: {path} absent -- greedy columns will be blank; "
              f"run analyze_sft_dose.py to fill them", file=sys.stderr)
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        if r["family"] == "reference":
            continue
        out[(int(r["U"]), r["lr"], int(r["rank"]))] = r
    return out


def analyse(probe_dir, out_dir):
    mf = json.load(open(Path(probe_dir) / "probe_manifest.json"))
    base = torch.load(Path(probe_dir) / "base_probe.pt", map_location="cpu",
                      weights_only=False)
    soft_base = np.asarray(base["soft_base"], dtype=np.float64)
    dose = dose_lookup()

    rows = []
    for tag in mf["tags"]:
        r = torch.load(Path(probe_dir) / f"adapter_{tag}.pt",
                       map_location="cpu", weights_only=False)
        u, lr, rank, fam = parse_tag(tag)
        d = dose.get((u, lr, rank), {})
        soft = np.asarray(r["soft_adapter"], dtype=np.float64)
        rows.append({
            "family": fam, "U": u, "lr": lr, "rank": rank, "tag": tag,
            "soft_rmse_to_base": float(np.sqrt(np.mean(
                (soft - soft_base) ** 2))),
            "soft_mean_shift": float(soft.mean() - soft_base.mean()),
            "soft_corr_to_base": float(np.corrcoef(soft, soft_base)[0, 1])
            if soft.std() > 1e-12 and soft_base.std() > 1e-12
            else float("nan"),
            "greedy_rmse_to_base": float(d["rmse_to_base"]) if d else
            float("nan"),
            "greedy_rmse_to_target": float(d["rmse_to_target"]) if d else
            float("nan"),
            "kl_fwd_per_tok": float((np.asarray(r["kl_fwd_sum"])
                                     / np.asarray(r["n_tok"])).mean()),
            "kl_fwd_tstar": float(np.mean(r["kl_fwd_tstar"])),
            "kl_rev_tstar": float(np.mean(r["kl_rev_tstar"])),
            # kl_fwd_* is the RAW model distribution; kl_served_* is after
            # the decoder's repetition penalty, the frame the served
            # opinions actually come from
            "kl_served_tstar": float(np.mean(r["kl_served_tstar"]))
            if "kl_served_tstar" in r else float("nan"),
            "flip_rate": float(np.mean(r["flip_tstar"])),
            "flip_rate_vs_generated": float(np.mean(
                r["flip_vs_generated"])) if "flip_vs_generated" in r
            else float("nan"),
            "early_div_frac": float(np.mean(
                (np.asarray(r["first_div"]) >= 0)
                & (np.asarray(r["first_div"])
                   < np.asarray(base["tstar"])))),
            "tail_adapter_max": float(np.max(r["tail_adapter"])),
        })

    # kl_frac: this cell's KL at t* over the shared full-dose cell's. The
    # normalized form is what makes "KL still small where greedy has
    # already departed" a readable statement rather than a units question.
    end = [x for x in rows if x["family"] == "endpoint"]
    ref_kl = end[0]["kl_fwd_tstar"] if end else float("nan")
    for x in rows:
        x["kl_frac_of_full_dose"] = (x["kl_fwd_tstar"] / ref_kl
                                     if ref_kl and np.isfinite(ref_kl)
                                     else float("nan"))
        x["soft_frac_of_full_dose"] = (
            x["soft_rmse_to_base"] / end[0]["soft_rmse_to_base"]
            if end and end[0]["soft_rmse_to_base"] > 0 else float("nan"))
        x["greedy_frac_of_full_dose"] = (
            x["greedy_rmse_to_base"] / end[0]["greedy_rmse_to_base"]
            if end and end[0]["greedy_rmse_to_base"] > 0 else float("nan"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out_dir / "adapter_kl_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for x in rows:
            w.writerow(x)
    print(f"[akl] wrote {out_dir / 'adapter_kl_cells.csv'} ({len(rows)} rows)")
    report(rows, mf)
    figure(rows, out_dir)
    return rows


def _dial(x):
    return {"update": x["U"], "lr": x["lr"], "rank": x["rank"],
            "endpoint": "full"}[x["family"]]


def _fam_series(rows, fam):
    """This family's cells plus the shared full-dose endpoint, ordered."""
    sel = [x for x in rows if x["family"] == fam]
    end = [x for x in rows if x["family"] == "endpoint"]
    sel = sel + end
    key = {"update": lambda x: x["U"], "lr": lambda x: float(x["lr"]),
           "rank": lambda x: x["rank"]}[fam]
    return sorted(sel, key=key), key


def report(rows, mf):
    print(f"\n[akl] base model at the decision position: mean top-1 "
          f"{mf['base_top1_mean']:.3f}, mean top1-top2 margin "
          f"{mf['base_margin_mean']:.3f}"
          + (f", median {mf['base_margin_median']:.3f}"
             if "base_margin_median" in mf else ""))
    print("[akl]   a SMALL margin is what makes the argmax-amplification "
          "reading available at all")
    tm = mf.get("tf_mismatch")
    if tm:
        # the frozen base disagreeing with ITSELF across two numerically
        # equivalent decoding paths is a direct, model-only measurement of
        # how knife-edge the served map is -- no adapter involved
        print(f"[akl] serving frame: repetition_penalty="
              f"{mf.get('repetition_penalty', 'UNRECORDED')} -- served "
              f"opinions are the argmax of the PENALIZED distribution")
        print(f"[akl] the FROZEN base flips its own argmax for "
              f"{tm['n']} agent-positions ({100 * tm['frac_of_agents']:.1f}%"
              f" of agents) between cached generation and a full forward, "
              f"at positions {tm['positions']}, all within margin "
              f"{tm['max_margin']:.2e}")
    print()
    print(f"[akl] {'family':>8} {'dial':>7} {'greedy->base':>13} "
          f"{'soft->base':>11} {'KL@t*':>8} {'KLfrac':>7} {'flip%':>7} "
          f"{'softcorr':>9}")
    for fam in ("update", "lr", "rank"):
        ser, _ = _fam_series(rows, fam)
        for x in ser:
            print(f"[akl] {fam:>8} {str(_dial(x)):>7} "
                  f"{x['greedy_rmse_to_base']:>13.4f} "
                  f"{x['soft_rmse_to_base']:>11.4f} "
                  f"{x['kl_fwd_tstar']:>8.4f} "
                  f"{x['kl_frac_of_full_dose']:>7.3f} "
                  f"{100 * x['flip_rate']:>7.1f} "
                  f"{x['soft_corr_to_base']:>9.3f}")
        print()


def figure(rows, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    for ax, fam in zip(axes, ("update", "lr", "rank")):
        ser, key = _fam_series(rows, fam)
        xs = [key(x) for x in ser]
        ax.plot(xs, [x["greedy_rmse_to_base"] for x in ser], marker="o",
                ms=5, lw=1.7, color="#c1443c", label="greedy served")
        ax.plot(xs, [x["soft_rmse_to_base"] for x in ser], marker="s",
                ms=5, lw=1.7, color="#2a6fb5", label="soft-decoded")
        ax.set_xscale("log")
        ax.set_xlabel({"update": "optimizer updates $U$",
                       "lr": "learning rate",
                       "rank": "LoRA rank $r$"}[fam])
        ax.grid(alpha=0.25, lw=0.5)
        tw = ax.twinx()
        tw.plot(xs, [x["kl_fwd_tstar"] for x in ser], marker="^", ms=4,
                lw=1.1, ls="--", color="#6a9a56")
        tw.set_yscale("log")
        if fam == "rank":
            tw.set_ylabel("$KL$(base $\\Vert$ adapter) at $t^*$  (nats)",
                          fontsize=9, color="#6a9a56")
        else:
            tw.set_yticklabels([])
        tw.tick_params(axis="y", labelsize=8, colors="#6a9a56")
    axes[0].set_ylabel("RMSE to frozen Qwen", fontsize=10)
    axes[0].legend(fontsize=9, frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = Path(out_dir) / f"adapter_kl.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[akl] wrote {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path,
                    default=REPO / "notes" / "pofd" / "cluster"
                    / "adapter_kl_probe")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    analyse(args.probe_dir, args.out_dir)
    print(f"[akl] outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
