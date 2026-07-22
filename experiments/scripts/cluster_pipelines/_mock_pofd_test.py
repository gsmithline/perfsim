#!/usr/bin/env python3
"""Laptop mock for the pofd_ design (pure torch, no transformers): exercises the
REAL population code (_gated_pop.py) at the pofd corner -- eps_social=0, W=1 --
and replicates the pipeline's per-round order (peer sweep -> platform blend,
no innate anchor) to prove the two sanity invariants before any GPU job:
  (a) eps=0  -> ab_sweep never accepts a pair, opinions untouched by peers;
  (b) W=1    -> gated agents land EXACTLY on the served prediction, ungated
               agents are EXACTLY unchanged, contact == gate fraction.
Run:  python3 experiments/scripts/cluster_pipelines/_mock_pofd_test.py
"""
import importlib.util
from pathlib import Path

import torch

_GP = Path(__file__).resolve().parent / "_gated_pop.py"
spec = importlib.util.spec_from_file_location("_gated_pop", _GP)
gp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gp)

torch.manual_seed(0)
n = 200
gen = torch.Generator().manual_seed(123)
innate = torch.rand(n, generator=gen)
# ring graph: every agent has 2 neighbours, so peer pairs WOULD form if eps > 0
adj = torch.zeros(n, n)
for i in range(n):
    adj[i, (i - 1) % n] = adj[i, (i + 1) % n] = 1.0

# (a) eps_social = 0: the peer step must never move anyone -----------------
x = innate.clone()
acc = sum(gp.ab_sweep(x, adj, eps=0.0, gamma=0.0, gen=gen) for _ in range(20))
assert acc == 0, f"eps=0 accepted {acc} peer pairs"
assert torch.equal(x, innate), "eps=0 moved opinions"
# control: the same graph DOES interact at eps > 0 (the zero above is the eps,
# not a dead graph)
x_ctl = innate.clone()
acc_ctl = gp.ab_sweep(x_ctl, adj, eps=0.5, gamma=0.0, gen=gen)
assert acc_ctl > 0, "control failed: eps=0.5 accepted nothing (dead graph?)"
print(f"PASS (a) eps_social=0: 20 sweeps, 0/{acc_ctl}+ peer moves, opinions bit-identical")

# (b) W = 1: accepted -> exact copy, rejected -> exactly unchanged ---------
w_agent = torch.ones(n)      # movielens platform_sus is all-ones; pscale=1; W_PLAT=1
for eps_ai in (0.05, 0.1, 0.2, 0.4):
    served = torch.rand(n, generator=gen)
    x2, contact = gp.gated_blend(innate.clone(), served, w_agent, eps_ai)
    gate = (served - innate).abs() < eps_ai
    assert torch.equal(x2[gate], served[gate]), f"W=1 not exact at eps_ai={eps_ai}"
    assert torch.equal(x2[~gate], innate[~gate]), f"rejected moved at eps_ai={eps_ai}"
    assert abs(contact - float(gate.float().mean())) < 1e-12
print("PASS (b) W=1: exact adoption + untouched rejects at all four eps_AI values")

# (c) full pofd round order, 5 rounds, mock 'model' = biased constant ------
# (peer sweep no-op -> gated blend -> NO innate anchor), mirrors the pipeline's
# peer_first branch at ab_sweeps=1 exactly
ab_x = innate.clone()
history = []
for t in range(5):
    served = torch.full((n,), 0.55)                       # deployed prediction
    assert gp.ab_sweep(ab_x, adj, eps=0.0, gamma=0.0, gen=gen) == 0
    x_before = ab_x.clone()
    ab_x, contact = gp.gated_blend(ab_x, served, w_agent, 0.2)
    gate = (served - x_before).abs() < 0.2
    assert torch.equal(ab_x[gate], served[gate])
    assert torch.equal(ab_x[~gate], x_before[~gate])
    history.append((t, float(contact), float(ab_x.std())))
# after round 0 every gated agent sits AT 0.55 -> the gated set is absorbed and
# stable; ungated agents can never enter (their distance to 0.55 is fixed)
assert history[0][1] == history[-1][1], "gate set changed without any dynamics to move it"
assert torch.equal(ab_x[~gate], innate[~gate]), "ungated drifted from innate over 5 rounds"
print("PASS (c) 5-round platform-only loop: gated set absorbed at the served value, "
      f"ungated pinned at innate (contact {history[0][1]:.3f}, op_std {history[-1][2]:.3f})")
print("ALL MOCK CHECKS PASS")
