#!/usr/bin/env python3
"""Deterministic graph-placement masks for the innate-clamp GRAPH wave
(2026-08-17, mistral_innate_clamp_graph_s0).

Builds the two fixed cohorts BEFORE any GPU job exists, from the
MovieLens Action kNN graph and innate opinions ONLY (no LLM
predictions, no experimental outcomes):

  graph_clumped    145 agents occupying a concentrated low-cut region:
                   minimize fixed-responsive cut edges and conductance,
                   maximize the largest fixed induced component.
  graph_scattered  145 agents distributed across the graph: maximize
                   responsive-node coverage and fixed-responsive cut
                   edges, minimize fixed-fixed internal edges.

MATCHING CONSTRAINT: both masks draw IDENTICAL quotas from the joint
innate-opinion-quintile x degree-tercile strata (rank-based bins, id
tie-break, largest-remainder sizing), so they are matched on the
innate and degree distributions by construction; a repair phase then
tightens the innate match. Hard acceptance criteria (the script exits
1 and writes NOTHING on failure): |d mean| < 0.01, |d SD| < 0.01,
W1 < 0.01 between the two fixed innate distributions; scattered
exposure >= clumped exposure + 20pp; scattered cut >= 1.3x clumped
cut; identical stratum counts.

Deterministic: greedy initialization with lowest-id tie-breaks, then
strict hill-climb local search on same-stratum swaps with a dedicated
torch.Generator (SEED below) and a FIXED iteration budget.

Writes experiments/condor/clamp_graph_masks.json: mask ids + sha256
(the runner loads masks from here; the checker re-verifies), the full
innate/degree/edge arrays so every graph statistic is re-computable
offline, per-node stratum labels, quotas, all diagnostics, and the
criteria verdicts. --verify reloads the dataset and re-checks
everything instead of writing.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ARTIFACT = os.path.join(REPO, "experiments", "condor",
                        "clamp_graph_masks.json")

_spec_gp = importlib.util.spec_from_file_location(
    "gp_maskbuild", os.path.join(HERE, "_gated_pop.py"))
gp = importlib.util.module_from_spec(_spec_gp)
_spec_gp.loader.exec_module(gp)

N_FIXED = 145
SEED = 20260817
ITERS_GRAPH = 20000
ITERS_REPAIR = 8000
# scalarized objectives (documented weights; minimize both)
#   clumped:   cut + W_COND*conductance - W_LCC*largest_component
#   scattered: -W_EXP*exposed_count - cut + W_FF*ff_edges
W_COND, W_LCC = 300.0, 3.0
W_EXP, W_FF = 3.0, 4.0
CRIT = {"d_mean": 0.01, "d_sd": 0.01, "w1": 0.01,
        "exposure_gap_pp": 0.20, "cut_ratio": 1.3}


def load_graph():
    """innate (float64 [n]), adj (bool [n,n]) via the runner's exact
    MovieLens construction (imported, never duplicated)."""
    _spec_rm = importlib.util.spec_from_file_location(
        "runner_for_maskbuild", os.path.join(HERE,
                                             "run_pokec_gated_lm.py"))
    RM = importlib.util.module_from_spec(_spec_rm)
    _spec_rm.loader.exec_module(RM)
    ml_dir = Path(os.environ.get(
        "ML_DIR", os.path.join(REPO, "experiments", "data", "movielens",
                               "ml-100k")))
    setup = RM.load_movielens_setup(ml_dir, target="Action")
    innate = setup["innate"].double().numpy()
    adj = (setup["adj"].numpy() > 0)
    return innate, adj


def rank_bins(values, n_bins):
    """Per-node bin labels: rank by (value, id), split into n_bins
    near-even rank bins (largest-remainder sizing)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: (float(values[i]), i))
    sizes = gp._largest_remainder([1] * n_bins, n)
    lab = np.empty(n, dtype=np.int64)
    lo = 0
    for b, sz in enumerate(sizes):
        for i in order[lo:lo + sz]:
            lab[i] = b
        lo += sz
    return lab


def mask_stats(f, adj, deg, innate):
    """All graph + innate diagnostics for a bool mask f."""
    r = ~f
    cut = int(adj[f][:, r].sum())
    ff = int(adj[f][:, f].sum()) // 2
    vol_f = int(deg[f].sum())
    vol_r = int(deg[r].sum())
    cond = cut / max(1, min(vol_f, vol_r))
    exposed = int((adj[r][:, f].sum(1) > 0).sum())
    n_resp = int(r.sum())
    # largest fixed induced component (BFS over members)
    ids = np.where(f)[0]
    pos = {int(v): k for k, v in enumerate(ids)}
    seen = np.zeros(len(ids), dtype=bool)
    best = 0
    sub = adj[np.ix_(ids, ids)]
    for s in range(len(ids)):
        if seen[s]:
            continue
        stack, seen[s], size = [s], True, 1
        while stack:
            u = stack.pop()
            for v in np.where(sub[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    size += 1
                    stack.append(int(v))
        best = max(best, size)
    iv = innate[f]
    return {"cut_edges": cut, "ff_edges": ff, "conductance": round(cond, 6),
            "largest_fixed_component": best,
            "exposure": round(exposed / n_resp, 6),
            "exposed_responsive": exposed, "n_responsive": n_resp,
            "vol_fixed": vol_f,
            "innate_mean": round(float(iv.mean()), 6),
            "innate_sd": round(float(iv.std()), 6),
            "innate_quantiles": {q: round(float(np.quantile(iv, qq)), 6)
                                 for q, qq in (("q00", 0), ("q05", .05),
                                               ("q25", .25), ("q50", .5),
                                               ("q75", .75), ("q95", .95),
                                               ("q100", 1))},
            "degree_mean": round(float(deg[f].mean()), 4),
            "degree_min": int(deg[f].min()),
            "degree_max": int(deg[f].max())}


def score_clumped(f, adj, deg, innate):
    s = mask_stats(f, adj, deg, innate)
    return (s["cut_edges"] + W_COND * s["conductance"]
            - W_LCC * s["largest_fixed_component"])


def score_scattered(f, adj, deg, innate):
    s = mask_stats(f, adj, deg, innate)
    return (-W_EXP * s["exposed_responsive"] - s["cut_edges"]
            + W_FF * s["ff_edges"])


def greedy_init(adj, deg, stratum, quotas, kind):
    n = adj.shape[0]
    remaining = list(quotas)
    f = np.zeros(n, dtype=bool)
    if kind == "clumped":
        # seed: highest-degree node with quota, lowest id on ties; grow
        # by max neighbors-in-F (ties: higher degree, then lowest id)
        cand = sorted(range(n), key=lambda i: (-deg[i], i))
        seed = next(i for i in cand if remaining[stratum[i]] > 0)
        f[seed] = True
        remaining[stratum[seed]] -= 1
        for _ in range(N_FIXED - 1):
            best_i, best_key = None, None
            for i in range(n):
                if f[i] or remaining[stratum[i]] == 0:
                    continue
                key = (-int(adj[i][f].sum()), -deg[i], i)
                if best_key is None or key < best_key:
                    best_i, best_key = i, key
            f[best_i] = True
            remaining[stratum[best_i]] -= 1
    else:
        # scattered: max fresh coverage, penalize F-neighbors; ties
        # broken by lowest id
        covered = np.zeros(n, dtype=bool)
        for _ in range(N_FIXED):
            best_i, best_key = None, None
            for i in range(n):
                if f[i] or remaining[stratum[i]] == 0:
                    continue
                fresh = int((adj[i] & ~covered & ~f).sum())
                key = (-(fresh - 2 * int(adj[i][f].sum())), i)
                if best_key is None or key < best_key:
                    best_i, best_key = i, key
            f[best_i] = True
            remaining[stratum[best_i]] -= 1
            covered |= adj[best_i]
    assert int(f.sum()) == N_FIXED and all(v == 0 for v in remaining)
    return f


def local_search(f, adj, deg, innate, stratum, score_fn, iters, gen):
    cur = score_fn(f, adj, deg, innate)
    for _ in range(iters):
        members = np.where(f)[0]
        i = int(members[int(torch.randint(len(members), (1,),
                                          generator=gen))])
        pool = np.where(~f & (stratum == stratum[i]))[0]
        if len(pool) == 0:
            continue
        j = int(pool[int(torch.randint(len(pool), (1,), generator=gen))])
        f[i], f[j] = False, True
        new = score_fn(f, adj, deg, innate)
        if new < cur:
            cur = new
        else:
            f[i], f[j] = True, False
    return f, cur


def match_penalty(fc, fs, innate):
    a, b = np.sort(innate[fc]), np.sort(innate[fs])
    dm = abs(a.mean() - b.mean())
    dsd = abs(a.std() - b.std())
    w1 = float(np.abs(a - b).mean())
    return dm, dsd, w1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="reload the dataset and re-check the committed "
                         "artifact instead of writing")
    args = ap.parse_args()

    innate, adj = load_graph()
    n = adj.shape[0]
    deg = adj.sum(1).astype(np.int64)
    q_lab = rank_bins(innate, 5)
    d_lab = rank_bins(deg, 3)
    stratum = q_lab * 3 + d_lab
    sizes = [int((stratum == s).sum()) for s in range(15)]
    quotas = gp._largest_remainder(sizes, N_FIXED)
    assert sum(quotas) == N_FIXED and all(q <= s for q, s
                                          in zip(quotas, sizes))

    if args.verify:
        art = json.load(open(ARTIFACT))
        errs = []
        if art["n"] != n or art["n_fixed"] != N_FIXED:
            errs.append("n/n_fixed mismatch")
        if np.abs(np.array(art["innate"]) - innate).max() > 1e-6:
            errs.append("innate vector differs from the dataset")
        if art["degree"] != deg.tolist():
            errs.append("degree vector differs from the graph")
        edges = {(min(a, b), max(a, b)) for a, b in art["edges"]}
        want_edges = {(i, int(j)) for i in range(n)
                      for j in np.where(adj[i])[0] if i < j}
        if edges != want_edges:
            errs.append("edge list differs from the graph")
        if art["quotas"] != quotas:
            errs.append("stratum quotas differ")
        for name in ("graph_clumped", "graph_scattered"):
            m = art["masks"][name]
            f = np.zeros(n, dtype=bool)
            f[m["ids"]] = True
            got = gp.innate_clamp_hash(torch.tensor(f))
            if got != m["hash"]:
                errs.append(f"{name} hash mismatch")
            st = mask_stats(f, adj, deg, innate)
            for k, v in m["stats"].items():
                if isinstance(v, dict):
                    continue
                if abs(float(st[k]) - float(v)) > 1e-6:
                    errs.append(f"{name} stat {k}: artifact {v} != "
                                f"recomputed {st[k]}")
            cnt = [int((stratum[m['ids']] == s).sum()) for s in range(15)]
            if cnt != quotas:
                errs.append(f"{name} stratum counts != quotas")
        if errs:
            print("VERIFY FAILED:\n  " + "\n  ".join(errs))
            sys.exit(1)
        print("[masks] artifact verifies against the dataset: OK")
        return

    gen_c = torch.Generator().manual_seed(SEED)
    gen_s = torch.Generator().manual_seed(SEED + 1)
    print(f"[masks] optimizing graph_clumped ({ITERS_GRAPH} iters)...")
    fc = greedy_init(adj, deg, stratum, quotas, "clumped")
    fc, sc = local_search(fc, adj, deg, innate, stratum, score_clumped,
                          ITERS_GRAPH, gen_c)
    print(f"[masks] optimizing graph_scattered ({ITERS_GRAPH} iters)...")
    fs = greedy_init(adj, deg, stratum, quotas, "scattered")
    fs, ss = local_search(fs, adj, deg, innate, stratum,
                          score_scattered, ITERS_GRAPH, gen_s)

    # innate-match repair on the scattered mask: accept same-stratum
    # swaps that reduce the match penalty without eroding the graph
    # objective past a fixed global budget (2% of |optimized score|)
    budget = ss + 0.02 * abs(ss)
    dm, dsd, w1 = match_penalty(fc, fs, innate)
    pen = dm + dsd + w1
    gen_r = torch.Generator().manual_seed(SEED + 2)
    for _ in range(ITERS_REPAIR):
        members = np.where(fs)[0]
        i = int(members[int(torch.randint(len(members), (1,),
                                          generator=gen_r))])
        pool = np.where(~fs & (stratum == stratum[i]))[0]
        if len(pool) == 0:
            continue
        j = int(pool[int(torch.randint(len(pool), (1,),
                                       generator=gen_r))])
        fs[i], fs[j] = False, True
        ndm, ndsd, nw1 = match_penalty(fc, fs, innate)
        npen = ndm + ndsd + nw1
        nscore = score_scattered(fs, adj, deg, innate)
        if npen < pen and nscore <= budget:
            pen, ss = npen, nscore
        else:
            fs[i], fs[j] = True, False
    dm, dsd, w1 = match_penalty(fc, fs, innate)

    st_c = mask_stats(fc, adj, deg, innate)
    st_s = mask_stats(fs, adj, deg, innate)
    checks = {
        "identical_stratum_counts": bool(
            [int((stratum[np.where(fc)[0]] == s).sum())
             for s in range(15)] ==
            [int((stratum[np.where(fs)[0]] == s).sum())
             for s in range(15)] == quotas),
        "d_mean_lt": bool(dm < CRIT["d_mean"]),
        "d_sd_lt": bool(dsd < CRIT["d_sd"]),
        "w1_lt": bool(w1 < CRIT["w1"]),
        "exposure_gap_ok": bool(st_s["exposure"]
                                >= st_c["exposure"]
                                + CRIT["exposure_gap_pp"]),
        "cut_ratio_ok": bool(st_c["cut_edges"] > 0 and
                             st_s["cut_edges"]
                             >= CRIT["cut_ratio"] * st_c["cut_edges"]),
    }

    print("\n== graph_clumped ==")
    print(json.dumps(st_c, indent=1))
    print("== graph_scattered ==")
    print(json.dumps(st_s, indent=1))
    print(f"== match ==  d_mean={dm:.5f}  d_sd={dsd:.5f}  w1={w1:.5f}")
    print(f"== criteria == {json.dumps(checks)}")
    if not all(checks.values()):
        print("\nCRITERIA NOT MET -- best candidates reported above; "
              "NO artifact written, NO jobs may be generated.")
        sys.exit(1)

    art = {
        "version": "2026-08-17",
        "built_by": "build_clamp_graph_masks.py (deterministic: greedy "
                    "init + strict hill-climb, torch.Generator seeds "
                    f"{SEED}/{SEED + 1}/{SEED + 2}, {ITERS_GRAPH} graph "
                    f"iters + {ITERS_REPAIR} repair iters; weights "
                    f"cond={W_COND} lcc={W_LCC} exp={W_EXP} ff={W_FF})",
        "n": n, "n_fixed": N_FIXED, "quotas": quotas,
        "stratum": stratum.tolist(),
        "innate": [round(float(v), 8) for v in innate],
        "degree": deg.tolist(),
        "edges": sorted((i, int(j)) for i in range(n)
                        for j in np.where(adj[i])[0] if i < j),
        "masks": {
            "graph_clumped": {
                "ids": sorted(int(i) for i in np.where(fc)[0]),
                "hash": gp.innate_clamp_hash(torch.tensor(fc)),
                "stats": st_c},
            "graph_scattered": {
                "ids": sorted(int(i) for i in np.where(fs)[0]),
                "hash": gp.innate_clamp_hash(torch.tensor(fs)),
                "stats": st_s},
        },
        "match": {"d_mean": round(dm, 6), "d_sd": round(dsd, 6),
                  "w1": round(w1, 6)},
        "criteria": checks,
        "criteria_thresholds": CRIT,
    }
    with open(ARTIFACT, "w") as fh:
        json.dump(art, fh)
        fh.write("\n")
    print(f"\n[masks] wrote {ARTIFACT}")
    print(f"[masks] clumped ids: {art['masks']['graph_clumped']['ids']}")
    print(f"[masks] clumped hash: "
          f"{art['masks']['graph_clumped']['hash']}")
    print(f"[masks] scattered ids: "
          f"{art['masks']['graph_scattered']['ids']}")
    print(f"[masks] scattered hash: "
          f"{art['masks']['graph_scattered']['hash']}")


if __name__ == "__main__":
    main()
