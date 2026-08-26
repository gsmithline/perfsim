"""END-TO-END exercise of the Figure-4 anchor-tradeoff analyzer and
preview (analyze_fig4_anchor.py / plot_fig4_anchor.py) on a SYNTHETIC
80-cell grid: 60 trained run dirs (trajectory.pt + telemetry.json with
the training-witness fields), 60 frozen replays (one per trained cell,
beta = 0 named "shared"), two zero-shot vectors (one per model, constant
across every frozen replay's rounds), and a gate verdict in the
checker's shape.

The grid API comes from a FAKE generator module implementing the F4A
contract verbatim (f4a_cells / f4a_source / f4a_tag / f4a_frozen_name /
F4A_* constants / f4a_ext_requests), written into tmp and passed to the
scripts with --gen, so these tests pass whether or not the real
generator's F4A block exists yet.  No models, no cluster, no GPU: every
artifact is a small torch tensor.

    USE_TF=0 python3 -m pytest tests/test_fig4_anchor_analysis.py -q
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "experiments", "scripts", "cluster_pipelines")
ANALYZE = os.path.join(PIPE, "analyze_fig4_anchor.py")
PLOT = os.path.join(PIPE, "plot_fig4_anchor.py")

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "perfsim-f4a-test-mpl"))

N = 723
ROUNDS = 30
MODELS = ("qwen7b", "qwen3_8b")
BASE_MODEL = {"qwen7b": "Qwen/Qwen2.5-7B-Instruct", "qwen3_8b": "Qwen/Qwen3-8B"}
GIT_SHA = "0123abcd0123abcd0123abcd0123abcd0123abcd"

# planted cells (model, es, beta, gamma), all TRAINED
CYCLE_CELL = ("qwen3_8b", 0.2, 0.5, 0.5)     # 2-cycle: small drift, range > .01
DRIFT_CELL = ("qwen7b", 0.05, 0.75, 0.2)      # monotone drifter -> extend_to_60
DRIFT_SRC = ("qwen7b", 0.2, 1.0, 1.0)         # drifting SOURCE of 3 dups
EXT60_CELL = ("qwen3_8b", 0.05, 0.25, 1.0)    # has a still-drifting _r60 -> 100
PLANTED_DUP = ("qwen7b", 0.2, 1.0, 0.0)       # dup of DRIFT_SRC: never in manifest

FAKE_GEN = '''
"""Stand-in for gen_pofd_sweep.py's F4A block -- the contract API verbatim."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
F4A_KEY = "fig4_anchor_tradeoff"
F4A_SMOKE_KEY = "fig4_anchor_tradeoff_smoke"
F4A_EXT_KEY = "fig4_anchor_tradeoff_ext"
F4A_MODELS = ("qwen7b", "qwen3_8b")
F4A_ES = (0.05, 0.2)
F4A_BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
F4A_GAMMAS = (1.0, 0.5, 0.2, 0.0)
F4A_SEED = 0
F4A_ROUNDS = 30
F4A_SMOKE_ROUNDS = 3
F4A_SWEEPS = 1
F4A_LAMBDA = 2.0
F4A_ALPHA = 0.5
F4A_REUSED = {}
F4A_BETA0_MODEL = "qwen3_8b"
F4A_EXT_REQUEST_PATH = os.path.join(HERE, "fig4_anchor_extension_request.json")
F4A_ZSPRIOR = {"qwen3_8b": "pofdzsprior_qwen3_8b_w0p5_l0p2_es0_s0",
               "qwen7b": "pofdzsprior_qwen7b_w0p5_l0p2_es0_s0"}
F4A_ZSPRIOR_SHA = {"qwen3_8b": None, "qwen7b": None}
FAM_MODELS = {"qwen7b": {"base_model": "Qwen/Qwen2.5-7B-Instruct", "chatthink": "default"},
              "qwen3_8b": {"base_model": "Qwen/Qwen3-8B", "chatthink": "0"}}


def _num(v):
    return f"{v:g}".replace(".", "p")


def f4a_source(model, es, beta, gamma):
    if float(beta) == 1.0:
        return (model, float(es), 1.0, 1.0)
    if float(beta) == 0.0:
        return (F4A_BETA0_MODEL, float(es), 0.0, float(gamma))
    return (model, float(es), float(beta), float(gamma))


def f4a_cells():
    out = []
    for model in F4A_MODELS:
        for es in F4A_ES:
            for beta in F4A_BETAS:
                for gamma in F4A_GAMMAS:
                    src = f4a_source(model, es, beta, gamma)
                    kind = "gpu" if src == (model, es, beta, gamma) else "dup"
                    out.append((model, es, beta, gamma, kind, src))
    assert len(out) == 80
    assert sum(1 for c in out if c[4] == "gpu") == 60
    return out


def f4a_tag(model, es, beta, gamma, rounds=F4A_ROUNDS, smoke=False):
    pre = "pofdf4asmk" if smoke else "pofdf4a"
    return (f"{pre}_{model}_fwdlam2_sw100_eaopen_w{_num(beta)}_k{_num(gamma)}"
            f"_es{_num(es)}_anch2_s0_r{rounds}")


def f4a_frozen_name(model, es, beta, gamma, rounds=30):
    return (f"frozen_f4a_{model}_w{_num(beta)}_k{_num(gamma)}_es{_num(es)}"
            f"_sw100_r{rounds}.pt")


def f4a_ext_requests():
    if not os.path.exists(F4A_EXT_REQUEST_PATH):
        return []
    req = json.load(open(F4A_EXT_REQUEST_PATH))
    assert isinstance(req, list), "manifest must be a JSON list"
    trained = {c[:4] for c in f4a_cells() if c[4] == "gpu"}
    out = []
    for e in req:
        key = (e["model"], float(e["es"]), float(e["beta"]), float(e["gamma"]))
        rounds = int(e["rounds"])
        assert rounds in (60, 100), e
        assert key in trained, f"extension names a dup or off-grid cell: {e}"
        out.append(key + (rounds,))
    assert len(set(out)) == len(out), "duplicate extension requests"
    return out
'''


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(argv, cwd=ROOT):
    env = dict(os.environ, USE_TF="0", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    return subprocess.run([sys.executable] + argv, cwd=cwd, env=env,
                          capture_output=True, text=True)


# ============================================================== fixture
def _innate(rng):
    return (rng.beta(2.0, 2.0, N) * 0.9 + 0.05).astype(np.float32)


def _zsprior(rng):
    """Two zero-shot vectors, .2f-quantized: Qwen3 spread over ~11
    values, Qwen2.5 essentially binary (0.25 / 0.65)."""
    q3 = np.round(np.clip(rng.normal(0.36, 0.09, N), 0.15, 0.85), 2)
    q7 = np.where(rng.random(N) < 0.6, 0.25, 0.65)
    return {"qwen3_8b": q3.astype(np.float32), "qwen7b": q7.astype(np.float32)}


def fixed_point(innate, m, beta, gamma):
    """Analytic fixed point of z = (1-w) h + w m with h = gamma innate +
    (1-gamma) x under a constant served m (beta = 0 -> innate)."""
    if beta == 0.0:
        return innate.copy()
    return ((1.0 - beta) * gamma * innate + beta * m) / (beta + gamma * (1.0 - beta))


def _relax(innate, target, rounds):
    xs = []
    for t in range(rounds):
        s = 1.0 - np.exp(-(t + 1) / 3.0)
        xs.append(innate + (target - innate) * s)
    return np.stack(xs)


def _config(model, es, beta, gamma, rounds):
    return {
        "run_tag": None, "w_plat": float(beta), "innate_lambda": float(gamma),
        "eps": float(es), "eps_ai": 1.0, "kl_beta": 2.0,
        "kl_direction": "forward", "training_style": "sft_kl",
        "kl_ref_adapter": "", "ab_sweeps": 100, "deffuant_alpha": 0.5,
        "n_rounds": rounds, "seed": 0, "dataset": "movielens",
        "ml_target": "Action", "base_model": BASE_MODEL[model],
        "ai_gate_mode": "all_open", "peer_gate_mode": "threshold",
        "ai_gate_reference": "anchor",
        "population_update": "nested_ai_anchored_then_social_v2",
        "pop_model": "ab", "icl_k": 0, "train_cap": N, "n_labeled": N,
        "lora_r": 512, "use_lora": True, "fresh_each_round": True,
        "sft_epochs": 1, "sft_batch_size": 4, "sft_lr": 5e-5,
        "save_raw_gen": True, "serve_eval_mode": True, "do_sample": False,
        "parse_mode": "strict", "train_witness": True, "witness_probe_n": 64,
        "homophily_gamma": 0.0, "chat_thinking": False, "git_sha": GIT_SHA,
    }


def make_trained(model, es, beta, gamma, innate, zs, rounds=ROUNDS,
                 cycle=False, drift=False):
    """A plausible trained-cell artifact: the population relaxes from
    innate toward the analytic fixed point of an ADAPTED served vector
    (between the zero-shot prior and the data); pred_raw moves a little
    and is .2f-quantized; the twin never sees the model."""
    m_fin = 0.75 * zs[model] + 0.25 * innate
    target = fixed_point(innate, m_fin, beta, gamma)
    op = _relax(innate, target, rounds)
    if cycle:
        for t in range(rounds):
            op[t] += 0.011 * (1.0 if t % 2 == 0 else -1.0)
    if drift:
        for t in range(rounds):
            op[t] += 0.003 * t
    op = np.clip(op, 0.0, 1.0)
    if beta == 0.0:
        op = np.stack([innate.copy() for _ in range(rounds)])
    twin = np.stack([innate.copy() for _ in range(rounds)])
    pred = []
    for t in range(rounds):
        prev = innate if t == 0 else op[t - 1]
        pred.append(np.round(m_fin + (prev - m_fin) * 0.2 * min(t, 10) / 10.0, 2))
    pred = np.clip(np.stack(pred), 0.0, 1.0)
    cfg = _config(model, es, beta, gamma, rounds)
    return {
        "config": cfg,
        "op_raw": torch.as_tensor(op, dtype=torch.float32),
        "twin_raw": torch.as_tensor(twin, dtype=torch.float32),
        "pred_raw": torch.as_tensor(pred, dtype=torch.float32),
        "innate": torch.as_tensor(innate, dtype=torch.float32),
        "trajectory": [{"round": t, "contact": 1.0,
                        "peer_gate_mode": "threshold", "peer_pairs": N * 100,
                        "accepted": N - 50} for t in range(rounds)],
        "sft_dose": [{"global_step": 181 * (t + 1), "trainer_seed": 0,
                      "n_rows": N} for t in range(rounds)],
    }


def make_frozen(model_or_shared, es, beta, gamma, innate, zs, rounds=ROUNDS):
    m = "qwen3_8b" if model_or_shared == "shared" else model_or_shared
    target = fixed_point(innate, zs[m], beta, gamma)
    op = np.clip(_relax(innate, target, rounds), 0.0, 1.0)
    if beta == 0.0:
        op = np.stack([innate.copy() for _ in range(rounds)])
    twin = np.stack([innate.copy() for _ in range(rounds)])
    pred = np.stack([zs[m].copy() for _ in range(rounds)])
    cfg = {
        "platform": "frozen_offline_replay",
        "population_update": "nested_ai_anchored_then_social_v2",
        "innate_k": float(gamma), "w_plat": float(beta),
        "eps_social": float(es), "eps_ai": 1.0, "ai_gate_mode": "all_open",
        "peer_gate_mode": "threshold", "ab_sweeps": 100,
        "deffuant_alpha": 0.5, "gamma_bias": 0.0, "rounds": rounds,
        "seed": 0, "dataset": "movielens", "ml_target": "Action",
        "n_agents": N, "base_model": BASE_MODEL[m],
        "source_run_tag": f"pofdzsprior_{m}_w0p5_l0p2_es0_s0",
    }
    return {"config": cfg,
            "op_raw": torch.as_tensor(op, dtype=torch.float32),
            "twin_raw": torch.as_tensor(twin, dtype=torch.float32),
            "pred_raw": torch.as_tensor(pred, dtype=torch.float32),
            "innate": torch.as_tensor(innate, dtype=torch.float32)}


def write_telemetry(path, rounds):
    rows = []
    for t in range(rounds):
        rows.append(json.dumps({
            "round": t, "l_init": 1.2 - 0.01 * t, "n_train": N,
            "grad_norm0": 0.5, "grad_kl_norm0": 0.02 + 0.001 * t,
            "contact": 1.0,
            "witness_steps": 181, "witness_steps_requested": 181,
            "witness_n_rows": N,
            "witness_lora_b_norm": 3.0 + 0.1 * (t % 4),
            "witness_lora_ab_norm": 1.5 + 0.05 * t,
            "witness_data_loss_last": 0.9 - 0.005 * t,
            "witness_kl_last": 0.05,
            "witness_probe_kl_fwd": 0.004 * (1 + t % 3),
            "witness_probe_kl_rev": 0.005,
            "witness_probe_argmax_agree": 0.90 + 0.002 * (t % 5),
            "witness_probe_n": 64,
            "witness_probe_sha": "a" * 64}))
    path.write_text("\n".join(rows) + "\n")


def build_fixture(base, zs_override=None, innate_override=None):
    """Populate `base` with gen.py (the fake generator), runs/, frozen/,
    gate.json and gate_smoke.json.  Returns (gen_path, runs, frozen, base).
    zs_override: {model: float32[723]} replaces a model's synthetic
    zero-shot vector (e.g. with the real archived one, so the sha pinned
    in the real generator's F4A_ZSPRIOR_SHA is satisfied);
    innate_override: float32[723] replaces the synthetic innate (e.g. the
    real MovieLens one, so real replay artifacts can be mixed in)."""
    base.mkdir(parents=True, exist_ok=True)
    gen_path = base / "gen_fake_f4a.py"
    gen_path.write_text(FAKE_GEN)
    g = _load("_fake_gen_f4a_" + str(abs(hash(str(base))) % 10 ** 6),
              str(gen_path))
    runs = base / "runs"
    frozen = base / "frozen"
    runs.mkdir(exist_ok=True)
    frozen.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    innate = _innate(rng)
    if innate_override is not None:
        innate = np.asarray(innate_override, dtype=np.float32).reshape(-1)
    zs = _zsprior(rng)
    for m, v in (zs_override or {}).items():
        zs[m] = np.asarray(v, dtype=np.float32).reshape(-1)
    cells = g.f4a_cells()
    gate_cells = []
    for (model, es, beta, gamma, kind, src) in cells:
        if kind == "dup":
            gate_cells.append({"tag": g.f4a_tag(*src), "status": "PASS",
                               "kind": "dup", "model": model, "es": es,
                               "beta": beta, "gamma": gamma,
                               "source_tag": g.f4a_tag(*src)})
            continue
        key = (model, es, beta, gamma)
        tag = g.f4a_tag(model, es, beta, gamma)
        d = make_trained(model, es, beta, gamma, innate, zs,
                         cycle=(key == CYCLE_CELL),
                         drift=(key in (DRIFT_CELL, DRIFT_SRC)))
        d["config"]["run_tag"] = tag
        (runs / tag).mkdir(parents=True, exist_ok=True)
        torch.save(d, runs / tag / "trajectory.pt")
        write_telemetry(runs / tag / "telemetry.json", ROUNDS)
        gate_cells.append({"tag": tag, "status": "PASS", "kind": "gpu",
                           "model": model, "es": es, "beta": beta,
                           "gamma": gamma, "git_sha": GIT_SHA})
        fm = "shared" if beta == 0.0 else model
        torch.save(make_frozen(fm, es, beta, gamma, innate, zs),
                   frozen / g.f4a_frozen_name(fm, es, beta, gamma))
    # one cell already extended to 60 rounds and STILL drifting
    tag60 = g.f4a_tag(*EXT60_CELL, rounds=60)
    d = make_trained(*EXT60_CELL, innate, zs, rounds=60, drift=True)
    d["config"]["run_tag"] = tag60
    (runs / tag60).mkdir(parents=True, exist_ok=True)
    torch.save(d, runs / tag60 / "trajectory.pt")
    write_telemetry(runs / tag60 / "telemetry.json", 60)
    assert len(os.listdir(frozen)) == 60
    assert len(os.listdir(runs)) == 61
    verdict = {"ok": True, "wave": g.F4A_KEY, "n_cells": 80, "n_trained": 60,
               "n_dup": 20, "cells": gate_cells,
               "zsprior": {m: {"status": "PASS"} for m in MODELS},
               "git_sha": [GIT_SHA]}
    (base / "gate.json").write_text(json.dumps(verdict, indent=1))
    smoke = {"ok": True, "wave": g.F4A_SMOKE_KEY, "n_cells": 3,
             "cells": [{"tag": g.f4a_tag("qwen7b", 0.2, 0.5, 0.5, rounds=3,
                                         smoke=True), "status": "PASS"},
                       {"tag": g.f4a_tag("qwen3_8b", 0.2, 0.5, 0.5, rounds=3,
                                         smoke=True), "status": "PASS"},
                       {"tag": g.F4A_ZSPRIOR["qwen7b"], "status": "PASS"}],
             "git_sha": [GIT_SHA]}
    (base / "gate_smoke.json").write_text(json.dumps(smoke, indent=1))
    return gen_path, runs, frozen, base


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("f4a"))


@pytest.fixture(scope="module")
def analysis(grid):
    gen_path, runs, frozen, base = grid
    out = base / "analysis"
    r = _run([ANALYZE, "--gen", str(gen_path), "--run-root", str(runs),
              "--frozen-dir", str(frozen), "--gate-json",
              str(base / "gate.json"), "--out-dir", str(out)])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-3000:]
    rows = list(csv.DictReader((out / "fig4_anchor_cells.csv").open()))
    summary = json.loads((out / "fig4_anchor_summary.json").read_text())
    manifest = json.loads(
        (out / "fig4_anchor_extension_request.json").read_text())
    return out, rows, summary, manifest, r


def _row(rows, model, es, beta, gamma):
    sel = [x for x in rows if x["model"] == model and x["es"] == f"{es:g}"
           and x["beta"] == f"{beta:g}" and x["gamma"] == f"{gamma:g}"]
    assert len(sel) == 1, (model, es, beta, gamma, len(sel))
    return sel[0]


# ============================================================= analyzer
def test_analyzer_writes_80_rows_60_gpu_20_dup_with_source_tags(analysis,
                                                                grid):
    gen_path, runs, frozen, base = grid
    g = _load("_fake_gen_f4a_t1", str(gen_path))
    out, rows, summary, manifest, r = analysis
    assert len(rows) == 80
    assert sum(1 for x in rows if x["kind"] == "gpu") == 60
    dups = [x for x in rows if x["kind"] == "dup"]
    assert len(dups) == 20
    for d in dups:
        src = g.f4a_source(d["model"], float(d["es"]), float(d["beta"]),
                           float(d["gamma"]))
        assert d["source_tag"] == g.f4a_tag(*src), d
        # a dup's statistics are its source's statistics
        s = _row(rows, *src)
        assert s["kind"] == "gpu"
        for col in ("final_mean", "final_sd", "drift10", "late5_range",
                    "settled", "frozen_final_mean", "l1_final_vs_frozen"):
            assert d[col] == s[col], (col, d, s)
    # beta = 0 dups resolve to qwen3_8b for qwen7b; beta = 1 dups to gamma 1
    d0 = _row(rows, "qwen7b", 0.05, 0.0, 0.5)
    assert d0["kind"] == "dup" and "qwen3_8b" in d0["source_tag"]
    d1 = _row(rows, "qwen3_8b", 0.2, 1.0, 0.2)
    assert d1["kind"] == "dup" and "_w1_k1_" in d1["source_tag"]
    assert summary["n_cells"] == 80 and summary["n_trained"] == 60
    assert summary["n_dup"] == 20


def test_summary_names_tensor_window_tolerances_and_quantization(analysis):
    out, rows, summary, manifest, r = analysis
    assert summary["tensor"] == "op_raw (end-of-round, post-peer)"
    assert summary["postpeer"] is True
    assert summary["window_rounds"] == [21, 30]
    assert summary["window"] == 10
    assert summary["drift_tol"] == 0.005
    assert summary["late_range_tol"] == pytest.approx(0.01)
    assert summary["served_quantization"] == 0.01
    assert summary["gated"] is True and summary["gate_n_cells"] == 80
    assert summary["git_sha"] == [GIT_SHA]
    assert set(summary["zsprior"]) == set(MODELS)
    for m in MODELS:
        z = summary["zsprior"][m]
        assert len(z["sha256"]) == 64 and z["n_frozen_artifacts"] == 26
    assert summary["wave"] == "fig4_anchor_tradeoff"


def test_settled_and_cyclic_semantics_on_the_planted_cells(analysis):
    out, rows, summary, manifest, r = analysis
    cyc = _row(rows, *CYCLE_CELL)
    assert abs(float(cyc["drift5"])) <= 0.005, cyc["drift5"]
    assert abs(float(cyc["drift10"])) <= 0.005, cyc["drift10"]
    assert float(cyc["late5_range"]) > 0.01, cyc["late5_range"]
    assert cyc["settled"] == "False" and cyc["cyclic"] == "True", cyc
    assert cyc["outcome"] == "extend_to_60"
    drf = _row(rows, *DRIFT_CELL)
    assert drf["settled"] == "False" and drf["cyclic"] == "False", drf
    assert abs(float(drf["drift10"])) > 0.005
    assert drf["outcome"] == "extend_to_60"
    # everything not planted is settled: the planted set is CYCLE_CELL,
    # DRIFT_CELL, DRIFT_SRC + its 3 dups, EXT60_CELL (at 60 rounds)
    unsettled = {(x["model"], x["es"], x["beta"], x["gamma"])
                 for x in rows if x["settled"] != "True"}
    want = {("qwen3_8b", "0.2", "0.5", "0.5"), ("qwen7b", "0.05", "0.75", "0.2"),
            ("qwen7b", "0.2", "1", "1"), ("qwen7b", "0.2", "1", "0.5"),
            ("qwen7b", "0.2", "1", "0.2"), ("qwen7b", "0.2", "1", "0"),
            ("qwen3_8b", "0.05", "0.25", "1")}
    assert unsettled == want, unsettled ^ want
    assert summary["n_unsettled"] == 7 and summary["n_cyclic"] == 1


def test_extension_manifest_is_the_generators_format_trained_only(analysis,
                                                                  grid):
    gen_path, runs, frozen, base = grid
    out, rows, summary, manifest, r = analysis
    assert isinstance(manifest, list)
    for e in manifest:
        assert set(e) == {"model", "es", "beta", "gamma", "rounds"}, e
    keys = {(e["model"], e["es"], e["beta"], e["gamma"], e["rounds"])
            for e in manifest}
    assert (DRIFT_CELL + (60,)) in keys
    assert (CYCLE_CELL + (60,)) in keys
    assert (DRIFT_SRC + (60,)) in keys
    # the cell already at 60 rounds asks for 100
    assert (EXT60_CELL + (100,)) in keys
    assert len(keys) == 4 and len(manifest) == 4
    # the planted dup (and every other dup) never appears
    assert not any((e["model"], e["es"], e["beta"], e["gamma"]) == PLANTED_DUP
                   for e in manifest)
    trained = {(m, es, b, gm) for (m, es, b, gm, k, s) in
               _load("_fake_gen_f4a_t2", str(gen_path)).f4a_cells()
               if k == "gpu"}
    for e in manifest:
        assert (e["model"], e["es"], e["beta"], e["gamma"]) in trained
    # round-trip through the generator's own reader
    g = _load("_fake_gen_f4a_t3", str(gen_path))
    g.F4A_EXT_REQUEST_PATH = str(out / "fig4_anchor_extension_request.json")
    got = g.f4a_ext_requests()
    assert len(got) == 4 and (DRIFT_CELL + (60,)) in got
    assert (EXT60_CELL + (100,)) in got


def test_extended_cell_is_analysed_at_its_longest_horizon(analysis, grid):
    gen_path, runs, frozen, base = grid
    out, rows, summary, manifest, r = analysis
    g = _load("_fake_gen_f4a_t4", str(gen_path))
    x = _row(rows, *EXT60_CELL)
    assert x["horizon"] == "60" and x["window_rounds"] == "51-60"
    assert x["source_tag"] == g.f4a_tag(*EXT60_CELL, rounds=60)
    assert x["frozen_rounds"] == "30"     # only the 30-round replay exists
    assert summary["horizons"] == [30, 60]


def test_frozen_comparison_witness_and_cardinality_columns(analysis):
    out, rows, summary, manifest, r = analysis
    for col in ("frozen_name", "frozen_rounds", "frozen_final_mean",
                "frozen_final_sd", "abs_final_mean_diff_vs_frozen",
                "ks_final_vs_frozen", "hist_l1_final_vs_frozen",
                "l1_final_vs_frozen",
                "mean_abs_final_vs_frozen", "twin_final_mean",
                "innate_mean", "entering_mean", "served_distinct_final",
                "served_distinct_min", "served_modal_share_final",
                "witness_lora_b_norm_min", "witness_probe_kl_fwd_min",
                "witness_probe_argmax_agree_mean", "path", "source_tag"):
        assert col in rows[0], col
    # the frozen comparison is distributional (the replay is not
    # RNG-matched to the runner) and the summary says so
    assert summary["frozen_comparison"] == "distributional only"
    assert "NOT RNG-matched" in summary["frozen_comparison_note"]
    assert "ks_final_vs_frozen" in summary["frozen_comparison_columns"]
    for x in rows:
        assert x["frozen_name"].startswith("frozen_f4a_")
        assert float(x["l1_final_vs_frozen"]) >= 0.0
        assert float(x["abs_final_mean_diff_vs_frozen"]) >= 0.0
        assert 0.0 <= float(x["ks_final_vs_frozen"]) <= 1.0
        assert 0.0 <= float(x["hist_l1_final_vs_frozen"]) <= 2.0
        assert float(x["witness_lora_b_norm_min"]) == pytest.approx(3.0)
        assert float(x["witness_probe_kl_fwd_min"]) == pytest.approx(0.004)
        assert 0.9 <= float(x["witness_probe_argmax_agree_mean"]) <= 0.91
        assert int(x["served_distinct_min"]) >= 1
        assert x["innate_mean"] == rows[0]["innate_mean"]
        if float(x["beta"]) == 0.0:
            assert x["frozen_name"].startswith("frozen_f4a_shared_")
            assert float(x["final_mean"]) == pytest.approx(
                float(x["innate_mean"]), abs=1e-6)
            assert float(x["final_mean"]) == pytest.approx(
                float(x["twin_final_mean"]), abs=1e-6)
    # the entering-model mean is per model and differs between them
    e7 = {x["entering_mean"] for x in rows if x["model"] == "qwen7b"}
    e3 = {x["entering_mean"] for x in rows if x["model"] == "qwen3_8b"}
    assert len(e7) == 1 and len(e3) == 1 and e7 != e3
    # the anchor trade-off at gamma = 1: final moves from innate toward
    # the entering model as beta grows, and the frozen replay sits at
    # the analytic fixed point of the zero-shot vector
    # (on an es whose gamma = 1 row carries no planted cell)
    for model, es in (("qwen7b", 0.05), ("qwen3_8b", 0.2)):
        inn = float(rows[0]["innate_mean"])
        ent = float(next(iter(e7 if model == "qwen7b" else e3)))
        gaps = [abs(float(_row(rows, model, es, b, 1.0)["final_mean"]) - ent)
                for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert gaps == sorted(gaps, reverse=True), gaps
        assert gaps[0] == pytest.approx(abs(inn - ent), abs=1e-6)


def test_paper_mode_exits_4_while_cells_are_unsettled(grid):
    gen_path, runs, frozen, base = grid
    r = _run([ANALYZE, "--gen", str(gen_path), "--run-root", str(runs),
              "--frozen-dir", str(frozen), "--gate-json",
              str(base / "gate.json"), "--out-dir", str(base / "paper"),
              "--paper"])
    assert r.returncode == 4, r.stdout[-1500:] + r.stderr[-1500:]
    assert "PAPER GATE FAIL" in r.stderr


def test_refuses_a_smoke_shaped_gate_verdict(grid):
    gen_path, runs, frozen, base = grid
    r = _run([ANALYZE, "--gen", str(gen_path), "--run-root", str(runs),
              "--frozen-dir", str(frozen), "--gate-json",
              str(base / "gate_smoke.json"), "--out-dir", str(base / "x")])
    assert r.returncode == 1, r.stdout[-1500:] + r.stderr[-1500:]
    assert "REFUSING" in r.stderr and "80" in r.stderr
    assert not (base / "x" / "fig4_anchor_cells.csv").exists()


def test_refuses_a_gate_with_a_failing_trained_tag(grid, tmp_path):
    gen_path, runs, frozen, base = grid
    v = json.loads((base / "gate.json").read_text())
    g = _load("_fake_gen_f4a_t5", str(gen_path))
    bad = g.f4a_tag("qwen3_8b", 0.2, 0.5, 0.2)
    for c in v["cells"]:
        if c["tag"] == bad and c["kind"] == "gpu":
            c["status"] = "FAIL"
    p = tmp_path / "gate_fail.json"
    p.write_text(json.dumps(v))
    r = _run([ANALYZE, "--gen", str(gen_path), "--run-root", str(runs),
              "--frozen-dir", str(frozen), "--gate-json", str(p),
              "--out-dir", str(tmp_path / "out")])
    assert r.returncode == 1
    assert "not PASS" in r.stderr


def test_refuses_a_missing_frozen_artifact(grid, tmp_path):
    gen_path, runs, frozen, base = grid
    g = _load("_fake_gen_f4a_t6", str(gen_path))
    sandbox = tmp_path / "frozen"
    sandbox.mkdir()
    drop = g.f4a_frozen_name("qwen7b", 0.2, 0.5, 0.2)
    for f in os.listdir(frozen):
        if f != drop:
            os.symlink(frozen / f, sandbox / f)
    r = _run([ANALYZE, "--gen", str(gen_path), "--run-root", str(runs),
              "--frozen-dir", str(sandbox), "--gate-json",
              str(base / "gate.json"), "--out-dir", str(tmp_path / "out")])
    assert r.returncode == 3, r.stdout[-1500:] + r.stderr[-1500:]
    assert "REFUSING" in r.stderr and drop in r.stderr


def test_refuses_a_missing_trained_cell(grid, tmp_path):
    gen_path, runs, frozen, base = grid
    g = _load("_fake_gen_f4a_t7", str(gen_path))
    sandbox = tmp_path / "runs"
    sandbox.mkdir()
    drop = g.f4a_tag("qwen3_8b", 0.05, 0.0, 0.2)   # a beta=0 source cell
    for d in os.listdir(runs):
        if d != drop:
            os.symlink(runs / d, sandbox / d)
    r = _run([ANALYZE, "--gen", str(gen_path), "--run-root", str(sandbox),
              "--frozen-dir", str(frozen), "--gate-json",
              str(base / "gate.json"), "--out-dir", str(tmp_path / "out")])
    assert r.returncode == 2, r.stdout[-1500:] + r.stderr[-1500:]
    assert drop in r.stderr


def test_refuses_a_frozen_replay_with_the_wrong_dials(grid, tmp_path):
    gen_path, runs, frozen, base = grid
    g = _load("_fake_gen_f4a_t8", str(gen_path))
    sandbox = tmp_path / "frozen"
    sandbox.mkdir()
    name = g.f4a_frozen_name("qwen3_8b", 0.05, 0.75, 0.5)
    for f in os.listdir(frozen):
        if f != name:
            os.symlink(frozen / f, sandbox / f)
    d = torch.load(frozen / name, weights_only=False)
    d["config"] = dict(d["config"], w_plat=0.5, platform="perfect_prediction")
    torch.save(d, sandbox / name)
    r = _run([ANALYZE, "--gen", str(gen_path), "--run-root", str(runs),
              "--frozen-dir", str(sandbox), "--gate-json",
              str(base / "gate.json"), "--out-dir", str(tmp_path / "out")])
    assert r.returncode == 3
    assert "platform" in r.stderr and "w_plat" in r.stderr


# ================================================================= plot
def test_plot_renders_four_figures_with_twenty_untitled_shared_axes(
        analysis, grid):
    gen_path, runs, frozen, base = grid
    out, rows, summary, manifest, r = analysis
    P = _load("_plot_f4a_t", PLOT)
    g = P.AN._grid(gen=_load("_fake_gen_f4a_t9", str(gen_path)))
    for with_frozen in (False, True):
        figs = list(P.build_figures(out, [runs], frozen, g, with_frozen))
        assert [(m, es) for (m, es, _, _) in figs] == [
            ("qwen7b", 0.05), ("qwen7b", 0.2), ("qwen3_8b", 0.05),
            ("qwen3_8b", 0.2)]
        for model, es, fig, cap in figs:
            axes = fig.axes
            assert len(axes) == 20, len(axes)
            assert fig._suptitle is None
            ylims = {tuple(np.round(ax.get_ylim(), 9)) for ax in axes}
            assert len(ylims) == 1, ylims
            for ax in axes:
                assert ax.get_title() == ""
                assert ax.get_xlim() == (0.0, 1.0)
                assert len(ax.patches) == (4 if with_frozen else 3), \
                    len(ax.patches)
                assert len(ax.texts) == 0        # no per-panel statistics
            assert len(fig.legends) == 1
            assert any("no title text" in l for l in cap)
            import matplotlib.pyplot as plt
            plt.close(fig)


def test_plot_cli_writes_pdf_and_png_for_both_models_and_both_es(analysis,
                                                                 grid):
    gen_path, runs, frozen, base = grid
    out, rows, summary, manifest, r = analysis
    prev = base / "previews"
    r = _run([PLOT, "--gen", str(gen_path), "--analysis-dir", str(out),
              "--run-root", str(runs), "--frozen-dir", str(frozen),
              "--out-dir", str(prev), "--with-frozen"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for model in MODELS:
        for tok in ("0p05", "0p2"):
            for ext in ("pdf", "png"):
                assert (prev / f"fig4_anchor_{model}_es{tok}.{ext}").exists()
            assert (prev / f"fig4_anchor_{model}_es{tok}_caption.txt").exists()
    assert "ALGEBRAIC DUPS" in r.stdout and "UNSETTLED" in r.stdout


def test_plot_refuses_without_the_analysis_summary(grid, tmp_path):
    gen_path, runs, frozen, base = grid
    r = _run([PLOT, "--gen", str(gen_path), "--analysis-dir", str(tmp_path),
              "--run-root", str(runs), "--frozen-dir", str(frozen),
              "--out-dir", str(tmp_path / "prev")])
    assert r.returncode == 1
    assert "REFUSING" in r.stderr


def test_plot_refuses_a_missing_frozen_artifact(analysis, grid, tmp_path):
    gen_path, runs, frozen, base = grid
    out, rows, summary, manifest, r = analysis
    g = _load("_fake_gen_f4a_t10", str(gen_path))
    sandbox = tmp_path / "frozen"
    sandbox.mkdir()
    drop = g.f4a_frozen_name("qwen3_8b", 0.2, 0.5, 0.2)
    for f in os.listdir(frozen):
        if f != drop:
            os.symlink(frozen / f, sandbox / f)
    r = _run([PLOT, "--gen", str(gen_path), "--analysis-dir", str(out),
              "--run-root", str(runs), "--frozen-dir", str(sandbox),
              "--out-dir", str(tmp_path / "prev"), "--with-frozen"])
    assert r.returncode == 2
    assert drop in r.stderr


def test_plot_source_carries_no_title_or_kde_calls():
    src = open(PLOT).read()
    for bad in ("set_title(", "plt.title(", "suptitle(", "gaussian_kde",
                "kdeplot", "sns."):
        assert bad not in src, f"paper figures carry no title text / KDE: {bad}"
    assert "np.linspace(0.0, 1.0, 51)" in src
    assert "np.histogram(" in src


def test_plot_refuses_to_write_under_paper(analysis, grid):
    gen_path, runs, frozen, base = grid
    out, rows, summary, manifest, r = analysis
    r = _run([PLOT, "--gen", str(gen_path), "--analysis-dir", str(out),
              "--run-root", str(runs), "--frozen-dir", str(frozen),
              "--out-dir", os.path.join(ROOT, "paper", "figures")])
    assert r.returncode != 0
    assert "never go under paper/" in (r.stderr + r.stdout)
