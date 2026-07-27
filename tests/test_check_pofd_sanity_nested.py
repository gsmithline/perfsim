"""check_pofd_sanity replays the right operator for each population_update marker.

Builds synthetic no-peer runs (eps_social=0, so the saved opinion IS the
pre-social state z) and asserts the checker:
  * passes a run whose dynamics match its marker, under BOTH versions;
  * FAILS a run generated with legacy dynamics but labelled nested, and vice
    versa -- i.e. old and new runs cannot be silently interchanged.
"""
import importlib.util
import json
import os

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_CP = os.path.join(HERE, os.pardir, "experiments", "scripts", "cluster_pipelines")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_CP, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chk = _load("check_pofd_sanity", "check_pofd_sanity.py")
gp = _load("_gated_pop", "_gated_pop.py")

N, ROUNDS, EPS_AI, W, LAM = 24, 4, 0.2, 0.5, 0.2


def _roll(nested):
    """Generate a no-peer trajectory under the chosen operator."""
    g = torch.Generator().manual_seed(0)
    innate = torch.rand(N, generator=g)
    x = innate.clone()
    w_agent = torch.full((N,), W)
    ops, preds, traj = [], [], []
    for t in range(ROUNDS):
        served = torch.rand(N, generator=g)
        gate = (served - x).abs() < EPS_AI
        if nested:
            h = LAM * innate + (1.0 - LAM) * x
            x = torch.where(gate, (1.0 - w_agent) * h + w_agent * served, h)
        else:
            mid = torch.where(gate, (1.0 - w_agent) * x + w_agent * served, x)
            x = (1.0 - LAM) * mid + LAM * innate
        ops.append(x.clone())
        preds.append(served.clone())
        traj.append({"round": t, "accepted": 0, "is_deploy": True,
                     "n_train": 723, "contact": float(gate.float().mean())})
    return innate, torch.stack(ops), torch.stack(preds), traj


def _write(tmp_path, marker, nested_dynamics):
    innate, op, pred, traj = _roll(nested_dynamics)
    cfg = {"eps": 0.0, "w_plat": W, "innate_lambda": LAM, "eps_ai": EPS_AI,
           "canary_delta": 0.0, "data_regime": "replace", "pop_model": "ab",
           "run_mode": "loop", "ab_sweeps": 1, "pop_reset": False,
           "platform_sus_scale": 1.0, "dataset": "movielens",
           "pristine_frac": 0.0, "fresh_each_round": True, "train_cap": 723}
    if marker is not None:
        cfg["population_update"] = marker
    d = os.path.join(tmp_path, f"pofdw_qwen7b_b1_ea0p2_w0p5_l0p2_s0_fresh_data")
    os.makedirs(d, exist_ok=True)
    torch.save({"config": cfg, "trajectory": traj, "op_raw": op,
                "pred_raw": pred, "innate": innate},
               os.path.join(d, "trajectory.pt"))
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    return d


@pytest.mark.parametrize("marker,dynamics", [
    ("nested_ai_then_social_v1", True),   # new runs
    (None, False),                        # archived runs, marker absent
])
def test_matching_marker_and_dynamics_passes(tmp_path, marker, dynamics):
    errs = chk.check_run(_write(tmp_path, marker, dynamics))
    assert errs == [], f"expected a clean replay, got {errs}"


@pytest.mark.parametrize("marker,dynamics", [
    ("nested_ai_then_social_v1", False),  # legacy dynamics claiming the new marker
    (None, True),                         # new dynamics with no marker
])
def test_mismatched_marker_and_dynamics_fails(tmp_path, marker, dynamics):
    errs = chk.check_run(_write(tmp_path, marker, dynamics))
    assert any("EXACT-COPY" in e for e in errs), \
        f"marker/dynamics mismatch must be caught, got {errs}"


def test_gate_uses_start_of_round_opinion(tmp_path):
    """Corrupting only the logged contact (the gate fraction) must be caught."""
    d = _write(tmp_path, "nested_ai_then_social_v1", True)
    p = os.path.join(d, "trajectory.pt")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    blob["trajectory"][1]["contact"] += 0.25
    torch.save(blob, p)
    errs = chk.check_run(d)
    assert any("gate-on-x0" in e for e in errs), errs
