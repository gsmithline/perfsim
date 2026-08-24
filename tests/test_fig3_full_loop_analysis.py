"""END-TO-END exercise of the Figure-3 gate, analyzer and preview on a
SYNTHETIC complete 108-cell grid (2026-08-24 v2).

The real grid needs 63 GPU jobs that have not run yet, so without this
the analyzer and the preview would ship unexecuted. Here the whole
figure is fabricated in tmp_path -- 91 GPU trajectories WITH per-round
telemetry, 13 frozen replays, and the twins that ride inside the GPU
artifacts -- and the three tools run over it for real.

The two semantic points of the 2026-08-24 audit are pinned here:
  * per-round training telemetry is the HARD retraining witness --
    absent, empty, or missing a round FAILS the gate;
  * a CONSTANT served map with full telemetry PASSES (a valid loop can
    settle), where the first draft wrongly hard-failed it.

No models, no cluster, no GPU: every artifact is a small torch tensor.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, "experiments", "scripts", "cluster_pipelines")
GEN = os.path.join(ROOT, "experiments", "condor", "gen_pofd_sweep.py")
CHECK = os.path.join(PIPE, "check_fig3_full_loop.py")
ANALYZE = os.path.join(PIPE, "analyze_fig3_full_loop.py")
PREVIEW = os.path.join(PIPE, "plot_fig3_previews.py")

N = 723
ROUNDS = 30
SWEEPS = 100
MODEL_ID = "Qwen/Qwen3-8B"
# one cell drifts on purpose, to exercise outcome classification and the
# --paper hard-fail: its tail keeps moving past the 0.005 tolerance
DRIFT_CELL = (0.75, 0.5, 2.0)


def _gen():
    spec = importlib.util.spec_from_file_location("_g_f3an", GEN)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_g_f3an"] = m
    spec.loader.exec_module(m)
    return m


def _num(v):
    return f"{v:g}".replace(".", "p")


def _innate():
    g = torch.Generator().manual_seed(11)
    return torch.rand(N, generator=g).double()


def _telemetry(path, rounds, lam):
    rows = []
    for t in range(rounds):
        rec = {"round": t, "l_init": 1.0 - 0.01 * t, "n_train": N,
               "grad_norm0": 0.5}
        if lam and lam > 0.0 and not math.isinf(lam):
            # round 0 exempt: a fresh LoRA IS the reference there
            rec["grad_kl_norm0"] = 0.0 if t == 0 else 0.02
        rows.append(json.dumps(rec))
    path.write_text("\n".join(rows) + "\n")


def _traj(beta, gamma, lam, innate, rounds=ROUNDS, drift=False):
    """A plausible full-loop artifact. The twin is a pure function of
    (gamma, round) so every cell at one gamma carries the SAME twin over
    any shared prefix -- the invariant the checker enforces for the
    beta = 0 column."""
    target = 0.63 - 0.05 * min(lam, 8.0) if not math.isinf(lam) else 0.36
    op, pred, twin = [], [], []
    x = innate.clone()
    tw = innate.clone()
    for t in range(rounds):
        # the served map moves early then settles: at beta = 1 the
        # population IS the served map, so an uncapped linear drift here
        # would (correctly!) fail the convergence test for every beta=1
        # cell -- the fixture must settle for the default cells to be
        # equilibria
        pdrift = 0.0 if math.isinf(lam) else 0.001 * min(t + 1, 10)
        pred.append(torch.full((N,), float(target + pdrift)).double())
        h = gamma * innate + (1.0 - gamma) * x
        x = (1.0 - beta) * h + beta * pred[-1]
        if drift:
            x = x + 0.004 * (t + 1)          # keeps moving through the tail
        else:
            x = x + 1e-6 * (t + 1)
        x = x.clamp(0.0, 1.0)
        op.append(x.clone())
        tw = gamma * innate + (1.0 - gamma) * tw
        tw = tw + 1e-7 * (t + 1)
        twin.append(tw.clone())
    return {
        "op_raw": torch.stack(op).float(),
        "twin_raw": torch.stack(twin).float(),
        "pred_raw": torch.stack(pred).float(),
        "innate": innate.float(),
        "trajectory": [{"round": t} for t in range(rounds)],
        "config": {
            "w_plat": float(beta), "innate_lambda": float(gamma),
            "kl_beta": 0.0 if math.isinf(lam) else float(lam),
            "ab_sweeps": SWEEPS,
            "n_rounds": rounds, "seed": 0, "dataset": "movielens",
            "ml_target": "Action", "base_model": MODEL_ID,
            "kl_direction": "forward", "ai_gate_mode": "all_open",
            "peer_gate_mode": "all_open", "icl_k": 0, "train_cap": N,
            "n_labeled": N, "lora_r": 512, "use_lora": 1,
            "fresh_each_round": 1, "homophily_gamma": 0.0,
            "training_style": "sft" if lam == 0.0 else "sft_kl",
            "population_update": "nested_ai_anchored_then_social_v2",
            "ai_gate_reference": "anchor",
        },
    }


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    """A complete synthetic Figure 3: run root + frozen dir + gate json."""
    g = _gen()
    base = tmp_path_factory.mktemp("f3grid")
    runs = base / "runs"
    frozen = base / "frozen"
    runs.mkdir()
    frozen.mkdir()
    innate = _innate()
    for (beta, gamma, lam, kind) in g.f3_cells():
        gam = 1.0 if gamma is None else gamma
        if kind == "gpu":
            tag = g.F3_REUSED.get((beta, gamma, lam)) or \
                g.f3_tag(beta, gam, lam)
            rounds = 60 if tag.endswith("_r60") else ROUNDS
            d = _traj(beta, gam, lam, innate, rounds=rounds,
                      drift=(beta, gam, lam) == DRIFT_CELL)
            d["config"]["n_rounds"] = rounds
            (runs / tag).mkdir(parents=True, exist_ok=True)
            torch.save(d, runs / tag / "trajectory.pt")
            _telemetry(runs / tag / "telemetry.json", rounds, lam)
        elif kind == "frozen":
            d = _traj(beta, gam, math.inf, innate)
            d["config"].update({"platform": "frozen_offline_replay",
                                "innate_k": float(gam)})
            name = (f"frz_k{_num(gam)}_w{_num(beta)}_eaopen_esopen"
                    f"_sw{SWEEPS}_s0_r{ROUNDS}.pt")
            torch.save(d, frozen / name)
    return g, runs, frozen, base


def _run(argv, cwd=ROOT):
    env = dict(os.environ, USE_TF="0", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    return subprocess.run([sys.executable] + argv, cwd=cwd, env=env,
                          capture_output=True, text=True)


def _sandbox_with(runs, tmp_path, tag, mutate):
    """Symlink the grid, replacing one run dir with a mutated copy."""
    sandbox = tmp_path / "runs"
    (sandbox / tag).mkdir(parents=True)
    src = runs / tag
    d = torch.load(src / "trajectory.pt", weights_only=False)
    tel = (src / "telemetry.json").read_text()
    d, tel = mutate(d, tel)
    torch.save(d, sandbox / tag / "trajectory.pt")
    if tel is not None:
        (sandbox / tag / "telemetry.json").write_text(tel)
    for other in os.listdir(runs):
        if other != tag:
            os.symlink(runs / other, sandbox / other)
    return sandbox


# ================================================================= gate
def test_gate_passes_on_a_complete_synthetic_grid(grid):
    g, runs, frozen, base = grid
    r = _run([CHECK, "--run-root", str(runs), "--frozen-dir", str(frozen),
              "--json", str(base / "gate.json")])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    v = json.loads((base / "gate.json").read_text())
    assert v["ok"] is True and v["n_failing"] == 0
    # 91 gpu + 13 frozen + 4 twin verdicts (+ any PENDING-EXT rows from
    # the committed extension request, which never fail the base gate)
    assert v["n_cells"] >= 108, v["n_cells"]


def test_gate_hard_fails_when_telemetry_is_absent(grid, tmp_path):
    g, runs, frozen, base = grid
    tag = g.f3_tag(0.25, 0.0, 1.0)
    sandbox = _sandbox_with(runs, tmp_path, tag,
                            lambda d, tel: (d, None))   # drop telemetry
    r = _run([CHECK, "--run-root", str(sandbox), "--frozen-dir", str(frozen)])
    assert r.returncode == 1
    assert "telemetry.json ABSENT" in r.stdout


def test_gate_hard_fails_when_telemetry_misses_a_round(grid, tmp_path):
    g, runs, frozen, base = grid
    tag = g.f3_tag(0.25, 0.2, 4.0)

    def cut(d, tel):
        lines = [l for l in tel.splitlines()
                 if json.loads(l).get("round") != 17]
        return d, "\n".join(lines) + "\n"
    sandbox = _sandbox_with(runs, tmp_path, tag, cut)
    r = _run([CHECK, "--run-root", str(sandbox), "--frozen-dir", str(frozen)])
    assert r.returncode == 1
    assert "missing rounds [17]" in r.stdout


def test_constant_served_map_with_full_telemetry_passes(grid, tmp_path):
    """The 2026-08-24 semantic fix: a settled loop may serve the same map
    every round. With the telemetry witness intact that is a NOTE, never
    a failure."""
    g, runs, frozen, base = grid
    tag = g.f3_tag(0.25, 0.0, 1.0)

    def freeze_pred(d, tel):
        d = dict(d)
        d["pred_raw"] = d["pred_raw"][0:1].repeat(d["pred_raw"].shape[0], 1)
        return d, tel
    sandbox = _sandbox_with(runs, tmp_path, tag, freeze_pred)
    r = _run([CHECK, "--run-root", str(sandbox), "--frozen-dir", str(frozen)])
    assert r.returncode == 0, r.stdout[-3000:]
    assert "NOTE served map constant" in r.stdout


def test_gate_fails_the_old_v1_operator(grid, tmp_path):
    g, runs, frozen, base = grid
    tag = g.f3_tag(0.75, 1.0, 4.0)

    def v1(d, tel):
        d = dict(d)
        d["config"] = dict(d["config"],
                           population_update="nested_ai_then_social_v1",
                           ai_gate_reference="x0")
        return d, tel
    sandbox = _sandbox_with(runs, tmp_path, tag, v1)
    r = _run([CHECK, "--run-root", str(sandbox), "--frozen-dir", str(frozen)])
    assert r.returncode == 1
    assert "nested_ai_then_social_v1" in r.stdout


def test_gate_fails_a_frozen_replay_with_wrong_model_provenance(grid,
                                                               tmp_path):
    """The Qwen2.5-vs-Qwen3-8B mixup, as a permanent regression check."""
    g, runs, frozen, base = grid
    bad_dir = tmp_path / "frozen"
    bad_dir.mkdir()
    for f in os.listdir(frozen):
        os.symlink(frozen / f, bad_dir / f)
    name = f"frz_k0_w0p25_eaopen_esopen_sw{SWEEPS}_s0_r{ROUNDS}.pt"
    os.unlink(bad_dir / name)
    d = torch.load(frozen / name, weights_only=False)
    d["config"] = dict(d["config"], base_model="Qwen/Qwen2.5-7B-Instruct")
    torch.save(d, bad_dir / name)
    r = _run([CHECK, "--run-root", str(runs), "--frozen-dir", str(bad_dir)])
    assert r.returncode == 1
    assert "Qwen2.5-7B-Instruct" in r.stdout and "base_model" in r.stdout


# ============================================================== analyzer
def test_analyzer_produces_108_cells_and_refuses_without_a_gate(grid):
    g, runs, frozen, base = grid
    out = base / "analysis"
    r = _run([ANALYZE, "--run-root", str(runs), "--frozen-dir", str(frozen),
              "--out-dir", str(out)])
    assert r.returncode == 2, "no --gate-json must be a refusal"
    r = _run([ANALYZE, "--run-root", str(runs), "--frozen-dir", str(frozen),
              "--out-dir", str(out), "--gate-json", str(base / "gate.json")])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rows = list(csv.DictReader((out / "fig3_cells.csv").open()))
    assert len(rows) == 108, len(rows)
    for col in ("mean_postpeer_final", "final_window_drift", "outcome",
                "horizon_rounds", "window_rounds"):
        assert col in rows[0], rows[0].keys()
    assert sum(1 for x in rows if x["kind"] == "twin") == 4
    assert sum(1 for x in rows if x["kind"] == "frozen") == 13
    assert sum(1 for x in rows if x["gamma_innate_lambda"] == "dedup") == 8
    assert json.loads((base / "analysis" / "fig3_summary.json").read_text()
                      )["innate_mean"] is not None


def test_analyzer_classifies_the_drifting_cell_and_requests_extension(grid):
    g, runs, frozen, base = grid
    out = base / "analysis"
    rows = list(csv.DictReader((out / "fig3_cells.csv").open()))
    b, gam, lam = DRIFT_CELL
    cell = [x for x in rows
            if x["beta_w_plat"] == f"{b:g}"
            and x["gamma_innate_lambda"] == f"{gam:g}"
            and x["lambda_kl_beta"] == f"{lam:g}"]
    assert len(cell) == 1
    assert cell[0]["outcome"] == "extend_to_60", cell[0]
    req = json.loads((out / "fig3_extension_request.json").read_text())
    assert any(e["beta"] == b and e["gamma"] == gam and e["lam"] == lam
               and e["rounds"] == 60 for e in req["cells"])
    # everything else settled
    assert sum(1 for x in rows if x["outcome"] != "equilibrium") == 1


def test_paper_mode_hard_fails_while_a_cell_needs_extension(grid):
    g, runs, frozen, base = grid
    r = _run([ANALYZE, "--run-root", str(runs), "--frozen-dir", str(frozen),
              "--out-dir", str(base / "analysis_paper"),
              "--gate-json", str(base / "gate.json"), "--paper"])
    assert r.returncode == 4, r.stdout[-1500:] + r.stderr[-1500:]
    assert "PAPER GATE FAIL" in r.stderr
    assert "EXTEND beta=0.75" in r.stderr


def test_analyzer_prefers_the_extended_horizon_when_present(grid, tmp_path):
    """Drop a settled 60-round pofdf3 _r60 artifact next to the drifting
    cell: the analyzer must analyse THAT horizon and call it settled."""
    g, runs, frozen, base = grid
    b, gam, lam = DRIFT_CELL
    sandbox = tmp_path / "runs"
    sandbox.mkdir()
    for other in os.listdir(runs):
        os.symlink(runs / other, sandbox / other)
    ext_tag = g.f3_tag(b, gam, lam, rounds=60)
    d = _traj(b, gam, lam, _innate(), rounds=60, drift=False)
    d["config"]["n_rounds"] = 60
    (sandbox / ext_tag).mkdir()
    torch.save(d, sandbox / ext_tag / "trajectory.pt")
    _telemetry(sandbox / ext_tag / "telemetry.json", 60, lam)
    out = tmp_path / "out"
    r = _run([ANALYZE, "--run-root", str(sandbox), "--frozen-dir",
              str(frozen), "--out-dir", str(out), "--allow-ungated",
              "--paper"])
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    rows = list(csv.DictReader((out / "fig3_cells.csv").open()))
    cell = [x for x in rows
            if x["beta_w_plat"] == f"{b:g}"
            and x["gamma_innate_lambda"] == f"{gam:g}"
            and x["lambda_kl_beta"] == f"{lam:g}"][0]
    assert cell["horizon_rounds"] == "60" and cell["outcome"] == "equilibrium"
    assert cell["source"] == ext_tag


def test_analyzer_refuses_a_partial_grid(grid, tmp_path):
    g, runs, frozen, base = grid
    sandbox = tmp_path / "runs"
    sandbox.mkdir()
    drop = g.f3_tag(0.5, 0.0, 1.0)
    for other in os.listdir(runs):
        if other != drop:
            os.symlink(runs / other, sandbox / other)
    r = _run([ANALYZE, "--run-root", str(sandbox), "--frozen-dir",
              str(frozen), "--out-dir", str(tmp_path / "out"),
              "--allow-ungated"])
    assert r.returncode == 3, r.stdout[-1500:] + r.stderr[-1500:]
    assert "REFUSING" in r.stderr and drop in r.stderr


# =============================================================== preview
def test_preview_renders_the_one_row_four_panel_figure(grid):
    g, runs, frozen, base = grid
    out = base / "analysis"
    r = _run([PREVIEW, "--in-dir", str(out)])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    prev = out / "previews"
    assert (prev / "fig3_preview_onerow.pdf").exists()
    assert (prev / "fig3_preview_onerow.png").exists()
    assert "CAPTION BLOCK" in r.stdout
    assert "perfect prediction" in r.stdout.lower() or \
        "perfect-prediction" in r.stdout.lower()
    # the drifting cell must be named as ringed
    assert "beta=0.75 gamma=0.5 lambda=2" in r.stdout


def test_preview_source_carries_no_title_calls():
    src = open(PREVIEW).read()
    for bad in ("ax.set_title(", "plt.title(", "fig.suptitle(",
                "plt.suptitle("):
        assert bad not in src, f"paper figures carry no title text: {bad}"


def test_previews_refuse_to_write_under_paper(grid):
    g, runs, frozen, base = grid
    r = _run([PREVIEW, "--in-dir", str(base / "analysis"),
              "--out-dir", os.path.join(ROOT, "paper", "figures")])
    assert r.returncode != 0
    assert "never go under paper/" in (r.stderr + r.stdout)
