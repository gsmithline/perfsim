"""Laptop mock test for the gated-population helpers in _gated_pop.py.

Pure torch/numpy: fake predictor = random tensor, 20 agents, 3 rounds. Covers
the AB sweep, the gated blend, the canary pattern, the probe selection, the
snapshot/swap bookkeeping, and crash-safe telemetry appends. No model classes,
no transformers anywhere on the import path.

Run: python experiments/scripts/cluster_pipelines/_mock_gated_test.py
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import torch

_GP_PATH = Path(__file__).resolve().parent / "_gated_pop.py"
_spec = importlib.util.spec_from_file_location("_gated_pop", _GP_PATH)
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)


def main() -> int:
    torch.manual_seed(0)
    n, n_rounds = 20, 3
    out_dir = Path(tempfile.mkdtemp(prefix="pokec-gated-mock-"))
    innate = torch.rand(n)
    # ring + a few chords so every agent has neighbors
    adj = torch.zeros(n, n)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1.0
        adj[i, (i + 3) % n] = adj[(i + 3) % n, i] = 1.0
    plat_sus = torch.rand(n)
    w_agent = (0.3 * plat_sus).clamp(0.0, 1.0)
    eps, gamma = 0.3, 1.5

    canary = gp.make_canary(n, 0.02, seed=0)
    assert canary.abs().eq(0.02).all() and canary.shape == (n,)
    assert torch.equal(canary, gp.make_canary(n, 0.02, seed=0))  # seeded, fixed
    assert gp.make_canary(n, 0.0, seed=0).eq(0).all()

    probe_idx = gp.select_probe_indices(innate, 5)
    assert probe_idx.shape == (5,) and innate[probe_idx].diff().ge(0).all()
    assert torch.equal(probe_idx, gp.select_probe_indices(innate, 5))

    # snapshot/swap on a stand-in trainable module
    net = torch.nn.Linear(4, 1)
    snap0 = gp.snapshot_trainable(net)
    with torch.no_grad():
        net.weight.add_(1.0)
    moved = net.weight.detach().clone()
    with gp.swapped_params(net, snap0):
        assert torch.allclose(net.weight, snap0["weight"])
    assert torch.allclose(net.weight, moved)

    # gate semantics: an agent only moves when |served - x| < eps, and then by
    # exactly w_i of the gap
    x_pre = innate.clone()
    served0 = torch.rand(n)
    x_post, contact0 = gp.gated_blend(x_pre.clone(), served0, w_agent, eps)
    gate = (served0 - x_pre).abs() < eps
    assert torch.allclose(x_post[~gate], x_pre[~gate])
    expect = (1 - w_agent[gate]) * x_pre[gate] + w_agent[gate] * served0[gate]
    assert torch.allclose(x_post[gate], expect)
    assert abs(contact0 - float(gate.float().mean())) < 1e-6

    tel_path = out_dir / "telemetry.json"
    tel_path.write_text("")
    x = innate.clone()
    for t in range(n_rounds):
        preds = torch.rand(n)  # mock predictor
        accepted = gp.ab_sweep(x, adj, eps, gamma)
        served = (preds + canary).clamp(0.0, 1.0)
        x, contact = gp.gated_blend(x, served, w_agent, eps)
        assert x.min() >= 0.0 and x.max() <= 1.0
        gp.append_telemetry(tel_path, {
            "round": t, "l_init": float(torch.rand(1)),  # stub losses
            "batch_var": float(x.var(unbiased=False)), "grad_norm0": 0.0,
            "l_cc": 0.0, "l_c0": 0.0, "l_0c": 0.0, "l_00": 0.0,
            "probe_pred": preds[probe_idx].tolist(), "contact": contact,
            "accepted": accepted,
        })
        print(f"[mock round {t}] accepted={accepted} contact={contact:.2f} "
              f"x_mean={float(x.mean()):.4f} x_std={float(x.std()):.4f}")
    rows = [json.loads(line) for line in tel_path.read_text().splitlines()]
    assert len(rows) == n_rounds and all(len(r["probe_pred"]) == 5 for r in rows)
    print(f"[mock] PASS  telemetry rows={len(rows)}  out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
