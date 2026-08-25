"""Tests for check_fig4_anchor.py -- the Figure-4 anchor trade-off gate.

A synthetic wave (60 trained run dirs + 2 zero-shot priors, pure torch,
no HF models) is built once per module at the production horizon; every
hard gate is then exercised by rebuilding ONE run dir with a single
defect and asserting the checker fails BY NAME, then restoring it.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "experiments" / "condor" / "gen_pofd_sweep.py"
CHECK = (ROOT / "experiments" / "scripts" / "cluster_pipelines" /
         "check_fig4_anchor.py")
N = 723
VALS = torch.tensor([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
GIT = "0123abcd0123abcd0123abcd0123abcd0123abcd"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha_vec(t):
    return hashlib.sha256(t.detach().contiguous().float().numpy().tobytes()).hexdigest()


@pytest.fixture(scope="module")
def g():
    """The real generator, with the zero-shot prior hashes re-pinned to
    the SYNTHETIC vectors this module serves (the real pins name the
    archived cluster artifacts, which the fixture cannot reproduce)."""
    mod = _load(GEN, "_gen_f4a_chk_test")
    assert mod.F4A_ZSPRIOR_SHA["qwen3_8b"] == \
        "fdfdeab7466345159cd7ae16ee487d4982d686cfdb93287780ae4d109ccba3f7"
    assert mod.F4A_ZSPRIOR_WARN_SHA["qwen7b"] == \
        "1674ee5f8d833f46de672791d933e1d3bdeefb07484c2d110dec84ce71da30bb"
    mod.F4A_ZSPRIOR_SHA = {"qwen3_8b": _sha_vec(_zs_pred("qwen3_8b")[0]),
                           "qwen7b": None}
    mod.F4A_ZSPRIOR_WARN_SHA = {"qwen7b": _sha_vec(_zs_pred("qwen7b")[0])}
    return mod


@pytest.fixture(scope="module")
def c(g):
    mod = _load(CHECK, "_check_f4a_test")
    real = mod._load_gen()                       # the checker finds the checkout
    assert real.F4A_KEY == "fig4_anchor_tradeoff" and len(real.f4a_cells()) == 80
    mod._load_gen = lambda path=None: g          # ...then gates against the re-pinned one
    return mod


# ------------------------------------------------------------ synthetic data
def _seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4],
                          "little")


def _rand(rows, *parts):
    gen = torch.Generator().manual_seed(_seed(*parts))
    return torch.rand(rows, N, generator=gen)


def _innate():
    gen = torch.Generator().manual_seed(12345)
    return torch.rand(N, generator=gen)


def _twin(es, gamma, rounds, shift=0):
    # the twin depends on (seed, es, gamma) only; the first rows agree
    # across horizons so an extension shares its base cell's twin
    return _rand(120, "twin", es, gamma, shift)[:rounds].clone()


def _op(model, es, beta, gamma, rounds):
    if beta == 0.0:
        return _twin(es, gamma, rounds)
    return _rand(120, "op", model, es, beta, gamma)[:rounds].clone()


def _pred(model, es, beta, gamma, rounds):
    gen = torch.Generator().manual_seed(_seed("pred", model, es, beta, gamma))
    idx = torch.randint(0, len(VALS), (120, N), generator=gen)
    return VALS[idx][:rounds].clone()


def _zs_pred(model):
    gen = torch.Generator().manual_seed(_seed("zs", model))
    idx = torch.randint(0, len(VALS), (N,), generator=gen)
    return VALS[idx].unsqueeze(0).clone()


def _witness(model, **over):
    w = {"witness_steps": 181, "witness_steps_requested": 181,
         "witness_n_rows": N, "witness_lora_b_norm": 3.25,
         "witness_lora_ab_norm": 1.75, "witness_data_loss_last": 0.91,
         "witness_kl_last": 0.021, "witness_probe_kl_fwd": 0.0123,
         "witness_probe_kl_rev": 0.0117, "witness_probe_argmax_agree": 0.97,
         "witness_probe_n": 64, "witness_probe_sha": f"probe-{model}"}
    w.update(over)
    return w


def _config(g, model, es, beta, gamma, rounds, tag, git_sha):
    m = g.FAM_MODELS[model]
    cfg = {"run_tag": tag, "w_plat": beta, "innate_lambda": gamma, "eps": es,
           "eps_ai": 1.0, "kl_beta": 2.0, "kl_direction": "forward",
           "training_style": "sft_kl", "kl_ref_adapter": "", "ab_sweeps": 1,
           "deffuant_alpha": 0.5, "n_rounds": rounds, "seed": 0,
           "dataset": "movielens", "ml_target": "Action",
           "base_model": m["base_model"], "ai_gate_mode": "all_open",
           "peer_gate_mode": "threshold", "ai_gate_reference": "anchor",
           "population_update": "nested_ai_anchored_then_social_v2",
           "pop_model": "ab", "icl_k": 0, "train_cap": N, "n_labeled": N,
           "lora_r": 512, "use_lora": True, "fresh_each_round": True,
           "sft_epochs": 1, "sft_batch_size": 4, "sft_lr": 5e-5,
           "save_raw_gen": True, "serve_eval_mode": True, "do_sample": False,
           "parse_mode": "strict", "train_witness": True, "witness_probe_n": 64,
           "gamma_bias": 0.0, "git_sha": git_sha, "seed_base_data": True,
           "hardware": {"gpu_name": "NVIDIA H100 80GB HBM3"}}
    if model == "qwen3_8b":
        cfg["chat_thinking"] = False
    return cfg


def _write_raw_log(run_dir, rows):
    with gzip.open(run_dir / "raw_gen_log.json.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def build_cell(root, g, model, es, beta, gamma, rounds, smoke=False,
               tag=None, git_sha=GIT, no_raw_log=False, parse_fail_frac=None,
               misparse=False, no_witness=False, witness=None,
               op_ne_twin=False, twin_shift=0, contact=1.0,
               peer_gate_mode="threshold", cfg=None, tag_rounds=None,
               no_peer_fields=False):
    """Write one synthetic trained run dir; keyword defects break exactly
    one gate each."""
    tag = tag or g.f4a_tag(model, es, beta, gamma,
                           rounds=(tag_rounds or rounds), smoke=smoke)
    d = Path(root) / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    innate = _innate()
    tw = _twin(es, gamma, rounds, shift=twin_shift)
    op = _op(model, es, beta, gamma, rounds) if twin_shift == 0 else tw.clone()
    if beta == 0.0 and twin_shift:
        op = tw.clone()
    pred = _pred(model, es, beta, gamma, rounds)
    if op_ne_twin:
        op = op.clone()
        op[1, 0] = float(min(1.0, op[1, 0] + 1e-3))
    if misparse:
        pred = pred.clone()
        pred[0] = 1.0
    traj = [{"round": t, "contact": contact, "accepted": 700 - t,
             "peer_gate_mode": peer_gate_mode, "peer_pairs": N,
             "s_tag": tag, "op_mean": float(op[t].mean())}
            for t in range(rounds)]
    if no_peer_fields:
        # what a threshold-gate run WITHOUT TRAIN_WITNESS=1 records
        for row in traj:
            del row["peer_gate_mode"], row["peer_pairs"]
    config = _config(g, model, es, beta, gamma, rounds, tag, git_sha)
    if cfg:
        config.update(cfg)
    torch.save({"config": config, "op_raw": op, "twin_raw": tw,
                "pred_raw": pred, "innate": innate, "trajectory": traj,
                "sft_dose": [{"round": t, "global_step": 181, "trainer_seed": 0,
                              "n_rows": N} for t in range(rounds)]},
               d / "trajectory.pt")
    tel = []
    for t in range(rounds):
        row = {"round": t, "deployment": t, "is_deploy": 1, "l_init": 1.1,
               "n_train": N, "grad_norm0": 1.0, "grad_kl_norm0": 0.05,
               "contact": contact}
        if not no_witness:
            row.update(_witness(model, **(witness or {})))
        tel.append(row)
    (d / "telemetry.json").write_text("\n".join(json.dumps(r) for r in tel) + "\n")
    if not no_raw_log:
        rows = []
        for t in range(rounds):
            parsed = [float(v) for v in pred[t].tolist()]
            raw = [f"{v:g}" for v in parsed]
            if misparse and t == 0:
                raw = [".64 (\n"] * N
                parsed = [1.0] * N
            pff = 0.0
            if parse_fail_frac is not None and t == 1:
                pff = parse_fail_frac
            rows.append({"round": t, "parse_fail_frac": pff, "parsed": parsed,
                         "raw": raw})
        _write_raw_log(d, rows)
    return tag


def build_zsprior(root, g, model, tag=None, pred=None):
    """An ARCHIVED-style zero-shot prior artifact: the v1 operator, no
    serve_eval_mode / parse_mode / git_sha (exactly what the real Qwen3
    artifact records) -- the wave's config pins must not apply here."""
    tag = tag or g.F4A_ZSPRIOR[model]
    d = Path(root) / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    pred = _zs_pred(model) if pred is None else pred
    innate = _innate()
    cfg = {"run_tag": tag, "base_model": g.FAM_MODELS[model]["base_model"],
           "training_style": "frozen", "use_lora": False, "icl_k": 0,
           "icl_days": 0, "dataset": "movielens", "ml_target": "Action",
           "seed": 0, "do_sample": False, "save_raw_gen": True, "n_rounds": 1,
           "eps": 0.0, "eps_ai": 0.0, "ai_gate_mode": "threshold",
           "population_update": "nested_ai_then_social_v1"}
    if model == "qwen3_8b":
        cfg["chat_thinking"] = False
    torch.save({"config": cfg, "op_raw": innate.unsqueeze(0).clone(),
                "twin_raw": torch.empty(0), "pred_raw": pred,
                "innate": innate, "trajectory": [{"round": 0, "contact": 0.0}]},
               d / "trajectory.pt")
    parsed = [float(v) for v in pred[0].tolist()]
    _write_raw_log(d, [{"round": 0, "parse_fail_frac": 0.0, "parsed": parsed,
                        "raw": [f"{v:g}" for v in parsed]}])
    return tag


def build_wave(root, g, smoke=False, rounds=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if smoke:
        rounds = g.F4A_SMOKE_ROUNDS if rounds is None else rounds
        for (m, e, b, gm) in g.F4A_SMOKE_CELLS:
            build_cell(root, g, m, e, b, gm, rounds, smoke=True)
        build_zsprior(root, g, "qwen7b")
    else:
        rounds = g.F4A_ROUNDS if rounds is None else rounds
        for (m, e, b, gm, kind, _s) in g.f4a_cells():
            if kind == "gpu":
                # a short wave keeps the PRODUCTION tags (_r30) so the
                # horizon gate, not the coverage gate, is what fails
                build_cell(root, g, m, e, b, gm, rounds, tag_rounds=g.F4A_ROUNDS)
        for m in g.F4A_ZSPRIOR:
            build_zsprior(root, g, m)
    return root


def run_checker(c, root, *extra):
    out = Path(root).parent / f"verdict_{Path(root).name}.json"
    if out.exists():
        out.unlink()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = c.main(["--run-root", str(root), "--json", str(out), *extra])
    return rc, json.loads(out.read_text()), buf.getvalue()


def _cell_errors(verdict, tag):
    for r in verdict["cells"] + verdict["extensions"]:
        if r.get("tag") == tag:
            return r["status"], r["errors"]
    raise AssertionError(f"{tag} not in the verdict")


def _wave_errors(verdict):
    return [r["errors"][0] for r in verdict["wave_failures"]]


@pytest.fixture(scope="module")
def prod(tmp_path_factory, g):
    return build_wave(tmp_path_factory.mktemp("f4a_prod") / "runs", g)


@contextlib.contextmanager
def defect(prod, g, cell, **mut):
    """Rebuild one production cell with a defect, then restore it."""
    tag = build_cell(prod, g, *cell, g.F4A_ROUNDS, **mut)
    try:
        yield tag
    finally:
        build_cell(prod, g, *cell, g.F4A_ROUNDS)


CELL_A = ("qwen7b", 0.05, 0.25, 0.2)         # trained, beta > 0
CELL_B0 = ("qwen3_8b", 0.2, 0.0, 0.5)        # trained beta = 0 (twin == pop)
CELL_Q3 = ("qwen3_8b", 0.05, 1.0, 1.0)      # beta = 1 source of 3 gamma dups


# ================================================================== PASS
def test_complete_production_wave_passes(c, g, prod):
    rc, v, out = run_checker(c, prod)
    assert rc == 0, out
    assert v["ok"] is True and v["wave"] == "fig4_anchor_tradeoff"
    assert (v["n_cells"], v["n_trained"], v["n_dup"]) == (80, 60, 20)
    assert len(v["cells"]) == 80 and v["n_ext"] == 0
    kinds = {}
    for r in v["cells"]:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        assert r["status"] == "PASS" and r["path"]
        assert r["greedy"]["distinct_min"] >= 1 and 0 < r["greedy"]["final_modal_share"] <= 1
        assert r["witness"]["min_probe_kl_fwd"] > 1e-9
    assert kinds == {"gpu": 60, "dup": 20}
    dups = [r for r in v["cells"] if r["kind"] == "dup"]
    for r in dups:
        src = g.f4a_source(r["model"], r["es"], r["beta"], r["gamma"])
        assert r["source_tag"] == g.f4a_tag(*src)
        assert r["path"].endswith(r["source_tag"])
    assert set(v["zsprior"]) == {"qwen3_8b", "qwen7b"}
    for m, z in v["zsprior"].items():
        assert z["status"] == "PASS" and len(z["sha256_pred0"]) == 64
        assert z["tag"] == g.F4A_ZSPRIOR[m] and z["warnings"] == []
    assert v["zsprior"]["qwen3_8b"]["sha256_pred0"] == g.F4A_ZSPRIOR_SHA["qwen3_8b"]
    assert v["zsprior"]["qwen3_8b"]["expected_sha256"] == g.F4A_ZSPRIOR_SHA["qwen3_8b"]
    assert v["zsprior"]["qwen7b"]["expected_sha256"] is None
    assert v["zsprior"]["qwen7b"]["archived_sha256"] == g.F4A_ZSPRIOR_WARN_SHA["qwen7b"]
    assert v["warnings"] == []
    assert v["git_sha"] == [GIT] and len(v["innate_sha"]) == 1
    assert "PROVENANCE" in out and "[check_f4a] PASS" in out
    assert "distinctServed" in out and "probeKL" in out


def test_smoke_wave_passes_under_smoke_only(c, g, tmp_path):
    root = build_wave(tmp_path / "smoke", g, smoke=True)
    rc, v, out = run_checker(c, root, "--smoke")
    assert rc == 0, out
    assert v["smoke"] is True and (v["n_cells"], v["n_trained"], v["n_dup"]) == (2, 2, 0)
    assert set(v["zsprior"]) == {"qwen7b"}
    assert all(r["rounds"] == 3 for r in v["cells"])
    assert all(r["witness"]["steps"] == 181 for r in v["cells"])
    # production mode on the smoke root is 60 absent cells, never a pass
    rc, v, _ = run_checker(c, root)
    assert rc == 1 and sum(r["status"] == "ABSENT" for r in v["cells"]) >= 60


def test_production_gate_requires_the_30_round_horizon(c, g, tmp_path):
    root = build_wave(tmp_path / "short", g, rounds=3)
    rc, v, _ = run_checker(c, root)
    assert rc == 1
    status, errs = _cell_errors(v, g.f4a_tag(*CELL_A))
    assert status == "FAIL"
    assert any("CONFIG n_rounds=3" in e for e in errs)
    assert any("ARTIFACT" in e and "3 rounds" in e for e in errs)
    assert all(r["status"] == "FAIL" for r in v["cells"])
    assert not _wave_errors(v)                   # nothing EXTRA, nothing absent


# ================================================================== FAIL
def test_missing_raw_log_fails(c, g, prod):
    with defect(prod, g, CELL_A, no_raw_log=True) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("raw_gen_log.json.gz ABSENT" in e for e in _cell_errors(v, tag)[1])


def test_parse_failure_fraction_fails(c, g, prod):
    with defect(prod, g, CELL_A, parse_fail_frac=1 / N) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("parse_fail_frac must be exactly 0" in e
               for e in _cell_errors(v, tag)[1])


def test_misparsed_leading_dot_string_fails(c, g, prod):
    with defect(prod, g, CELL_A, misparse=True) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    errs = _cell_errors(v, tag)[1]
    assert any("served value" in e for e in errs)


def test_missing_witness_fields_fail_by_name(c, g, prod):
    with defect(prod, g, CELL_A, no_witness=True) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("finite-lambda run skipped the training witness" in e
               for e in _cell_errors(v, tag)[1])


def test_witness_steps_180_fails(c, g, prod):
    with defect(prod, g, CELL_A, witness={"witness_steps": 180}) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("witness_steps == witness_steps_requested == 181" in e
               for e in _cell_errors(v, tag)[1])


def test_lora_b_norm_zero_fails(c, g, prod):
    with defect(prod, g, CELL_A, witness={"witness_lora_b_norm": 0.0}) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("witness_lora_b_norm must be > 0" in e
               for e in _cell_errors(v, tag)[1])


def test_probe_kl_zero_is_identical_to_frozen(c, g, prod):
    with defect(prod, g, CELL_A, witness={"witness_probe_kl_fwd": 0.0}) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("IDENTICAL" in e for e in _cell_errors(v, tag)[1])


def test_probe_sha_differing_across_runs_fails(c, g, prod):
    with defect(prod, g, CELL_A, witness={"witness_probe_sha": "other"}) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert _cell_errors(v, tag)[0] == "PASS"       # the run itself is fine
    assert any("witness_probe_sha differs across the qwen7b runs" in e
               for e in _wave_errors(v))


def test_beta0_population_must_equal_its_twin(c, g, prod):
    with defect(prod, g, CELL_B0, op_ne_twin=True) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("BETA0 op_raw != twin_raw" in e for e in _cell_errors(v, tag)[1])


def test_twin_differing_across_models_fails(c, g, prod):
    # a qwen7b cell at (es=.05, gamma=.2) carrying a different twin than
    # every other run at that (es, gamma), including qwen3_8b's
    with defect(prod, g, CELL_A, twin_shift=1):
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("distinct twin_raw hashes at (es=0.05, gamma=0.2)" in e
               for e in _wave_errors(v))


def test_contact_below_one_fails_the_runtime_gate(c, g, prod):
    with defect(prod, g, CELL_A, contact=0.99) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("GATE-RUNTIME" in e for e in _cell_errors(v, tag)[1])


def test_all_open_peer_gate_fails_the_runtime_gate(c, g, prod):
    with defect(prod, g, CELL_A, peer_gate_mode="all_open") as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("GATE-RUNTIME" in e and "'threshold'" in e
               for e in _cell_errors(v, tag)[1])


def test_rows_without_peer_fields_fail_naming_train_witness(c, g, prod):
    # under the threshold peer gate the runner writes peer_gate_mode /
    # peer_pairs only with TRAIN_WITNESS=1; their absence is a failure
    with defect(prod, g, CELL_A, no_peer_fields=True) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    errs = _cell_errors(v, tag)[1]
    assert any("GATE-RUNTIME" in e and "TRAIN_WITNESS" in e for e in errs)


def test_qwen3_zero_shot_prior_sha_is_a_hard_pin(c, g, prod):
    other = _zs_pred("qwen3_8b").clone()
    other[0, 0] = 0.5 if float(other[0, 0]) != 0.5 else 0.25
    build_zsprior(prod, g, "qwen3_8b", pred=other)
    try:
        rc, v, out = run_checker(c, prod)
    finally:
        build_zsprior(prod, g, "qwen3_8b")
    assert rc == 1
    z = v["zsprior"]["qwen3_8b"]
    assert z["status"] == "FAIL"
    assert any("sha256(pred_raw[0])" in e and "pinned" in e for e in z["errors"])
    assert all(r["status"] == "PASS" for r in v["cells"])


def test_qwen7b_zero_shot_prior_sha_only_warns(c, g, prod):
    other = _zs_pred("qwen7b").clone()
    other[0, 0] = 0.5 if float(other[0, 0]) != 0.5 else 0.25
    build_zsprior(prod, g, "qwen7b", pred=other)
    try:
        rc, v, out = run_checker(c, prod)
    finally:
        build_zsprior(prod, g, "qwen7b")
    assert rc == 0, out
    z = v["zsprior"]["qwen7b"]
    assert z["status"] == "PASS" and z["errors"] == []
    assert len(z["warnings"]) == 1 and "archived" in z["warnings"][0]
    assert z["sha256_pred0"] in z["warnings"][0]
    assert g.F4A_ZSPRIOR_WARN_SHA["qwen7b"] in z["warnings"][0]
    assert v["warnings"] == z["warnings"] and v["ok"] is True
    assert "WARN ZSPRIOR" in out and "1 WARN line(s)" in out


def test_zero_shot_prior_is_not_subject_to_the_wave_pins(c, g, prod):
    # the archived-style artifact (v1 operator, no serve_eval_mode /
    # parse_mode / git_sha) passes; a LoRA-trained or context-fed serve
    # is not a zero-shot prior and fails
    rc, v, _ = run_checker(c, prod)
    assert rc == 0 and v["zsprior"]["qwen3_8b"]["status"] == "PASS"
    tag = g.F4A_ZSPRIOR["qwen7b"]
    d = torch.load(prod / tag / "trajectory.pt", weights_only=False)
    assert d["config"]["population_update"] == "nested_ai_then_social_v1"
    assert "serve_eval_mode" not in d["config"] and "git_sha" not in d["config"]
    d["config"]["use_lora"] = True
    d["config"]["icl_k"] = 8
    torch.save(d, prod / tag / "trajectory.pt")
    try:
        rc, v, _ = run_checker(c, prod)
    finally:
        build_zsprior(prod, g, "qwen7b")
    assert rc == 1
    errs = v["zsprior"]["qwen7b"]["errors"]
    assert any("use_lora" in e for e in errs) and any("icl_k" in e for e in errs)


def test_dup_cell_present_as_a_run_dir_is_extra(c, g, prod):
    dup = ("qwen7b", 0.05, 0.0, 0.2)                     # beta=0 qwen7b dup
    assert g.f4a_source(*dup) != dup
    dtag = g.f4a_tag(*dup)
    shutil.copytree(prod / g.f4a_tag(*g.f4a_source(*dup)), prod / dtag)
    try:
        rc, v, _ = run_checker(c, prod)
    finally:
        shutil.rmtree(prod / dtag)
    assert rc == 1
    assert any(e.startswith("EXTRA") and "algebraic dup" in e and dtag in e
               for e in _wave_errors(v))
    # a foreign pofdf4a_ dir is EXTRA too
    foreign = prod / "pofdf4a_qwen7b_fwdlam2_sw1_eaopen_w0p5_k0p5_es0p1_anch2_s0_r30"
    foreign.mkdir()
    try:
        rc, v, _ = run_checker(c, prod)
    finally:
        foreign.rmdir()
    assert rc == 1 and any("EXTRA run dir" in e for e in _wave_errors(v))


def test_absent_trained_cell_fails(c, g, prod):
    tag = g.f4a_tag(*CELL_Q3)
    os.rename(prod / tag, prod / (tag + ".off"))
    try:
        rc, v, _ = run_checker(c, prod)
    finally:
        os.rename(prod / (tag + ".off"), prod / tag)
    assert rc == 1
    status, errs = _cell_errors(v, tag)
    assert status == "ABSENT" and any("absent" in e for e in errs)
    # the dups that resolve through it are ABSENT as well
    dep = [r for r in v["cells"] if r["kind"] == "dup" and r["source_tag"] == tag]
    assert dep and all(r["status"] == "ABSENT" for r in dep)


def test_wrong_n_rounds_fails(c, g, prod):
    with defect(prod, g, CELL_A, cfg={"n_rounds": 31}) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("CONFIG n_rounds=31" in e for e in _cell_errors(v, tag)[1])


def test_git_sha_mismatch_across_runs_fails(c, g, prod):
    with defect(prod, g, CELL_A, git_sha="feedfacefeedfacefeedfacefeedfacefeedface"):
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    assert any("distinct git SHAs" in e for e in _wave_errors(v))
    assert len(v["git_sha"]) == 2


def test_missing_zero_shot_prior_fails(c, g, prod):
    tag = g.F4A_ZSPRIOR["qwen7b"]
    os.rename(prod / tag, prod / (tag + ".off"))
    try:
        rc, v, _ = run_checker(c, prod)
    finally:
        os.rename(prod / (tag + ".off"), prod / tag)
    assert rc == 1
    assert v["zsprior"]["qwen7b"]["status"] == "ABSENT"
    assert all(r["status"] == "PASS" for r in v["cells"])


def test_config_pins_fail_by_name(c, g, prod):
    with defect(prod, g, CELL_A, cfg={"population_update": "nested_ai_then_social_v1",
                                      "peer_gate_mode": "all_open",
                                      "train_witness": False,
                                      "parse_mode": "legacy"}) as tag:
        rc, v, _ = run_checker(c, prod)
    assert rc == 1
    errs = _cell_errors(v, tag)[1]
    for needle in ("CONFIG population_update", "CONFIG peer_gate_mode",
                   "CONFIG train_witness", "CONFIG parse_mode"):
        assert any(needle in e for e in errs), needle


# ============================================================ extensions
def test_extensions_are_gated_at_their_horizon(c, g, prod, tmp_path):
    manifest = tmp_path / "ext.json"
    ext = ("qwen7b", 0.2, 0.75, 0.5)
    manifest.write_text(json.dumps([{"model": ext[0], "es": ext[1],
                                     "beta": ext[2], "gamma": ext[3],
                                     "rounds": 60}]))
    # requested but absent -> the gate refuses
    rc, v, _ = run_checker(c, prod, "--ext-manifest", str(manifest))
    assert rc == 1 and v["n_ext"] == 1
    assert v["extensions"][0]["status"] == "ABSENT"
    # present at 60 rounds with the base twin -> passes
    tag = build_cell(prod, g, *ext, 60)
    try:
        rc, v, _ = run_checker(c, prod, "--ext-manifest", str(manifest))
        assert rc == 0, [r["errors"] for r in v["extensions"]] + _wave_errors(v)
        assert v["extensions"][0]["rounds"] == 60 and v["extensions"][0]["kind"] == "ext"
        # without the manifest the same dir is EXTRA
        rc, v, _ = run_checker(c, prod, "--ext-manifest", str(tmp_path / "none.json"))
        assert rc == 1 and any("EXTRA run dir" in e for e in _wave_errors(v))
    finally:
        shutil.rmtree(prod / tag)


# ================================================================ parser
STRICT_CASES = [
    (".64 (\n", (0.64, True)), ("0.64", (0.64, True)), ("0.61 (", (0.61, True)),
    ("58 (58", (None, False)), ("0 (0 (0", (0.0, True)), ("0.00", (0.0, True)),
    ("1.0", (1.0, True)), ("1", (1.0, True)), (" 0.35\n", (0.35, True)),
    ("Answer: 0.4", (None, False)), ("", (None, False)), ("abc", (None, False)),
    ("1.2", (None, False)), (".5", (0.5, True)),
]


def test_strict_parse_mirrors_the_wrapper(c):
    for text, (want_v, want_ok) in STRICT_CASES:
        v, ok = c.strict_parse(text)
        assert ok == want_ok and (v is None or abs(v - want_v) < 1e-12), text
    pytest.importorskip("transformers")
    os.environ.setdefault("USE_TF", "0")
    from perfsim.models.hf_causal_lm import HFCausalLMModel as H
    for text, (want_v, want_ok) in STRICT_CASES:
        v, ok = H._parse_strict(text)
        assert ok == want_ok, text
        if want_ok:
            assert abs(v - want_v) < 1e-12, text
