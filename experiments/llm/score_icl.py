#!/usr/bin/env python3
"""Score the ICL (in-context adaptation) runs against the Q8 registered reads.

Reads runs/pokec_gated_lm/{icl_*,iclsmk_*}/trajectory.pt + config.json, using
the same conventions as read_atlas_slab.py (TAIL=5, clip[0,1], dr =
op_std(tail)/innate_std). No cluster access, no LLM. Run from repo root:

    python experiments/llm/score_icl.py                 # all icl_* + iclsmk_*
    python experiments/llm/score_icl.py --glob 'iclsmk_*'   # smoke gate only

Q8 predictions (QUESTIONS.md): (1) K-dose tracking toward the live population;
(2) context collapse = pred std shrinks over rounds at e040/K>0; (3) memory
echo (D15,K0); (4) K x D: own-history dominates; (5) Llama unlock (pred
non-constant, off 0.50); (6) Gemma first loop; (7) kNN protects vs echo.
"""
import argparse
import glob
import json
import os
import re
import numpy as np
import torch

RUNS = "runs/pokec_gated_lm"
TAIL = 5
POP_ALONE = "experiments/llm/figs/fig_pop_alone.json"


def model_key(base):
    b = (base or "").lower()
    for k in ("qwen", "llama", "gemma", "olmo"):
        if k in b:
            return k
    return b or "?"


def noai_baseline():
    """no-AI dr(30) by eps at gamma=0 -- the 'tracks the live population' anchor."""
    if not os.path.exists(POP_ALONE):
        return {}
    cells = json.load(open(POP_ALONE))["cells"]
    return {round(c["eps"], 3): c["dr30_mean"] for c in cells if abs(c["gamma"]) < 1e-9}


def load(tag):
    d = torch.load(f"{RUNS}/{tag}/trajectory.pt", map_location="cpu", weights_only=False)
    cfg = json.loads(open(f"{RUNS}/{tag}/config.json").read())
    op = np.clip(np.asarray(d["op_raw"], np.float32), 0, 1)      # (rounds, agents)
    pr = np.clip(np.asarray(d["pred_raw"], np.float32), 0, 1)
    inn = np.asarray(d["innate"], np.float32)
    ppl = np.asarray(d.get("ppl_raw", []), np.float32)
    inn_std = float(inn.std()) + 1e-9

    pred_std_round = np.nanstd(pr, axis=1)                       # context-collapse signal
    op_stdF = op[-TAIL:].std(1).mean()
    # memory echo: corr(pred_i(t), own opinion the prior round) over tail rounds
    mem_corr = np.nan
    icl_days = int(cfg.get("icl_days", 0) or 0)
    if icl_days > 0 and op.shape[0] >= 2:
        cs = []
        for t in range(max(1, op.shape[0] - TAIL), op.shape[0]):
            a, b = pr[t], op[t - 1]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 5 and a[m].std() > 1e-6 and b[m].std() > 1e-6:
                cs.append(np.corrcoef(a[m], b[m])[0, 1])
        mem_corr = float(np.mean(cs)) if cs else np.nan
    # frozen-prior reference: corr(pred, innate) tells sorting vs tracking
    a, b = pr[-1], inn
    m = np.isfinite(a) & np.isfinite(b)
    corr_innate = float(np.corrcoef(a[m], b[m])[0, 1]) if (m.sum() > 5 and a[m].std() > 1e-6) else np.nan

    return dict(
        tag=tag, model=model_key(cfg.get("base_model")), eps=round(float(cfg.get("eps", 0)), 3),
        K=int(cfg.get("icl_k", 0) or 0), D=icl_days, sel=cfg.get("icl_select", "?"),
        rounds=int(op.shape[0]),
        dr=float(op_stdF / inn_std),
        bias=float(op[-TAIL:].mean() - inn.mean()),
        pred_std0=float(pred_std_round[0]), pred_stdE=float(np.nanmean(pred_std_round[-TAIL:])),
        pred_meanE=float(np.nanmean(pr[-TAIL:])),
        nan_frac=float(np.isnan(pr).mean()),
        mem_corr=mem_corr, corr_innate=corr_innate,
        pmed=float(np.median(ppl[-TAIL:])) if ppl.size else np.nan,
    )


def cell_name(r):
    parts = []
    if r["D"]:
        parts.append(f"d{r['D']}")
    if r["K"]:
        parts.append(f"k{r['K']}")
    if r["sel"] == "knn":
        parts.append("knn")
    return "+".join(parts) or "frozen"


def smoke_gate(rows):
    print("\n" + "=" * 78 + "\nSMOKE GATE (iclsmk_*)\n" + "=" * 78)
    ok = True
    for r in rows:
        cfg_ok = r["K"] is not None and r["sel"] in ("random", "knn")
        nonconst = r["pred_stdE"] > 1e-3
        parse_ok = r["nan_frac"] < 0.2
        verdict = "PASS" if (cfg_ok and nonconst and parse_ok) else "FAIL"
        ok = ok and verdict == "PASS"
        print(f"  [{verdict}] {r['tag']}")
        print(f"         config: k={r['K']} d={r['D']} select={r['sel']}  (recorded {'OK' if cfg_ok else 'MISSING'})")
        print(f"         pred std end={r['pred_stdE']:.3f} ({'non-constant OK' if nonconst else 'CONSTANT -- FAIL'})"
              f"   nan_frac={r['nan_frac']:.3f} ({'OK' if parse_ok else 'high -- FAIL'})")
    print(f"\n  overall: {'ALL PASS -> clear to submit loop' if ok else 'FAILURE -- do not submit loop'}")
    print("  (prompt-block presence: check DEBUG_GEN sample in the .out log separately)")


def score_loop(rows, base):
    T = lambda x: f"{x:+.3f}" if isinstance(x, float) and np.isfinite(x) else "  n/a"
    print("\n" + "=" * 78 + "\nPER-RUN METRICS\n" + "=" * 78)
    print(f"  {'model':6} {'eps':>4} {'cell':>10} | {'dr':>5} {'bias':>6} | "
          f"{'pstd0':>6} {'pstdE':>6} {'pmeanE':>6} | {'memC':>6} {'innC':>6}")
    for r in sorted(rows, key=lambda r: (r["model"], r["eps"], r["D"], r["K"], r["sel"])):
        print(f"  {r['model']:6} {r['eps']:>4} {cell_name(r):>10} | {r['dr']:>5.2f} {T(r['bias'])} | "
              f"{r['pred_std0']:>6.3f} {r['pred_stdE']:>6.3f} {r['pred_meanE']:>6.3f} | "
              f"{T(r['mem_corr']):>6} {T(r['corr_innate']):>6}")

    def grp(**kw):
        return sorted([r for r in rows if all(r[k] == v for k, v in kw.items())],
                      key=lambda r: r["K"])

    print("\n" + "=" * 78 + "\nQ8 READS\n" + "=" * 78)
    for model in sorted({r["model"] for r in rows}):
        for eps in sorted({r["eps"] for r in rows if r["model"] == model}):
            b = base.get(eps)
            print(f"\n-- {model} eps={eps}  (no-AI dr baseline "
                  f"{b:.2f})" if b else f"\n-- {model} eps={eps}")
            # (1) K-dose at D=0 (random selection)
            kd = [r for r in grp(model=model, eps=eps, D=0, sel="random")]
            if len(kd) >= 2:
                seq = ", ".join(f"K{r['K']}:dr{r['dr']:.2f}/bias{r['bias']:+.2f}/pstdE{r['pred_stdE']:.2f}" for r in kd)
                print(f"   (1) K-dose D0: {seq}")
                print(f"       -> tracking = dr/bias move toward no-AI as K up"
                      + (f" (baseline dr {b:.2f})" if b else ""))
            # (2) context collapse: pstd0 vs pstdE at K>0
            for r in [x for x in kd if x["K"] > 0]:
                d = r["pred_std0"] - r["pred_stdE"]
                print(f"   (2) collapse K{r['K']}: pred std {r['pred_std0']:.3f}->{r['pred_stdE']:.3f} "
                      f"({'SHRINKS' if d > 0.01 else 'flat/widens'})")
            # (3) memory echo D15K0
            mem = [r for r in rows if r["model"] == model and r["eps"] == eps and r["D"] and r["K"] == 0]
            for r in mem:
                print(f"   (3) memory D{r['D']}K0: mem_corr={T(r['mem_corr'])} vs innate_corr={T(r['corr_innate'])} "
                      f"(echo if mem>>innate); dr={r['dr']:.2f}")
            # (4) K x D interaction
            d15k8 = [r for r in rows if r["model"] == model and r["eps"] == eps and r["D"] and r["K"] == 8 and r["sel"] == "random"]
            d15k0 = [r for r in rows if r["model"] == model and r["eps"] == eps and r["D"] and r["K"] == 0]
            d0k8 = [r for r in rows if r["model"] == model and r["eps"] == eps and not r["D"] and r["K"] == 8 and r["sel"] == "random"]
            if d15k8 and d15k0 and d0k8:
                print(f"   (4) KxD: d15k8 bias {d15k8[0]['bias']:+.2f} vs d15k0 {d15k0[0]['bias']:+.2f} "
                      f"vs d0k8 {d0k8[0]['bias']:+.2f} (own-history wins if closer to d15k0)")
            # (5) Llama unlock
            if model == "llama":
                for r in [x for x in rows if x["model"] == "llama" and x["eps"] == eps and x["K"] > 0]:
                    off = abs(r["pred_meanE"] - 0.5)
                    print(f"   (5) llama {cell_name(r)}: pred std {r['pred_stdE']:.3f} "
                          f"({'UNLOCKED' if r['pred_stdE'] > 0.02 else 'still degenerate'}), "
                          f"|mean-0.5|={off:.3f}")
            # (7) kNN vs random twin
            knn = [r for r in rows if r["model"] == model and r["eps"] == eps and r["sel"] == "knn"]
            for r in knn:
                twin = [x for x in rows if x["model"] == model and x["eps"] == eps and x["K"] == r["K"] and x["D"] == r["D"] and x["sel"] == "random"]
                if twin:
                    dd = r["dr"] - twin[0]["dr"]
                    print(f"   (7) kNN k{r['K']}: dr {r['dr']:.2f} vs random {twin[0]['dr']:.2f} "
                          f"({'PROTECTS' if dd > 0.03 else 'ECHO-CHAMBER' if dd < -0.03 else 'no diff'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=None, help="run-dir glob (default: icl_* and iclsmk_*)")
    args = ap.parse_args()
    pats = [args.glob] if args.glob else ["icl_*", "iclsmk_*"]
    tags = sorted({os.path.basename(p) for pat in pats
                   for p in glob.glob(f"{RUNS}/{pat}")
                   if os.path.exists(f"{p}/trajectory.pt")})
    if not tags:
        print(f"No runs with trajectory.pt under {RUNS} matching {pats}.")
        print("Pull them first (from your terminal):")
        print("  rsync -av login.cluster.is.localnet:'perfsim/runs/pokec_gated_lm/icl*' runs/pokec_gated_lm/")
        return
    rows = []
    for t in tags:
        try:
            rows.append(load(t))
        except Exception as e:
            print(f"  !! {t}: {e}")
    smk = [r for r in rows if r["tag"].startswith("iclsmk_")]
    loop = [r for r in rows if not r["tag"].startswith("iclsmk_")]
    if smk:
        smoke_gate(smk)
    if loop:
        score_loop(loop, noai_baseline())


if __name__ == "__main__":
    main()
