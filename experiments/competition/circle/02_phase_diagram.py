"""Block 2: phase diagram. Moving population (circular FJ) vs competing platforms.

Opinions live on the circle. Each round:
  1. users pick a platform (softmax over circular distance),
  2. platforms move to maximize their own captured share (competition),
  3. the population updates by a circular Friedkin-Johnsen step: embed each
     opinion as a unit vector, blend innate + consumed-platform + neighbor
     resultant, project back to an angle (the projection is the nonlinearity).

Malleability = how far the population moves off its innate opinion each round.
malleability=0 is the no-feedback control: the population stays at its innate
uniform spread, so the K=3 platforms segment. We sweep malleability up and look
for a sharp transition where the loop drags platforms + population into one cluster.
"""

import math

import torch
import torch.nn.functional as F

from experiments.competition.circle._common import circle_dist

TWO_PI = 2 * math.pi


def to_vec(x):
    a = TWO_PI * x
    return torch.stack([torch.cos(a), torch.sin(a)], dim=-1)


def to_angle(v):
    return (torch.atan2(v[..., 1], v[..., 0]) / TWO_PI) % 1.0


def random_graph(n, degree, seed):
    g = torch.Generator().manual_seed(seed)
    rows = torch.arange(n).repeat_interleave(degree)
    cols = torch.randint(n, (n * degree,), generator=g)
    w = torch.zeros(n, n)
    w[rows, cols] = 1.0
    w.fill_diagonal_(0.0)
    w = w / w.sum(1, keepdim=True).clamp(min=1.0)
    return w


def compete(pos, x, tau, lr=0.01, steps=1200):
    pos = pos.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=lr)
    k = pos.numel()
    for _ in range(steps):
        p = pos % 1.0
        total = 0.0
        for j in range(k):
            stacked = torch.stack([p[q] if q == j else p[q].detach() for q in range(k)])
            d = circle_dist(stacked[:, None], x[None, :])
            s = F.softmax(-d / tau, dim=0)
            total = total - s[j].mean()
        opt.zero_grad()
        total.backward()
        opt.step()
    return (pos % 1.0).detach()


def fj_step(x, innate, w_graph, pos, tau, malleability, peer_share, inner=3):
    w_innate = 1.0 - malleability
    w_peer = malleability * peer_share
    w_plat = malleability * (1.0 - peer_share)
    z0 = to_vec(innate)
    for _ in range(inner):
        d = circle_dist(pos[:, None], x[None, :])
        sel = F.softmax(-d / tau, dim=0)
        zp = (sel[:, :, None] * to_vec(pos)[:, None, :]).sum(0)
        target = w_innate * z0 + w_plat * zp + w_peer * (w_graph @ to_vec(x))
        x = to_angle(target)
    return x


def circ_var(x):
    return float(1.0 - to_vec(x).mean(0).norm())


def plat_min_gap(pos):
    p = pos.sort().values
    gaps = torch.cat([p[1:] - p[:-1], (p[0] + 1.0 - p[-1]).unsqueeze(0)])
    return float(gaps.min())


def run(malleability, k=3, tau=0.2, n=3000, degree=10, rounds=25, peer_share=0.5,
        seed=0, graph=None, innate=None):
    torch.manual_seed(seed)
    if graph is not None:
        w_graph = graph
        n = graph.shape[0]
    else:
        w_graph = random_graph(n, degree, seed)
    if innate is None:
        innate = torch.rand(n)
    x = innate.clone()
    pos = torch.tensor([(i + 0.5) / k for i in range(k)])  # start segmented
    for _ in range(rounds):
        pos = compete(pos, x, tau)
        x = fj_step(x, innate, w_graph, pos, tau, malleability, peer_share)
    return pos, x


def main():
    print("Block 2: K=3, tau=0.2, circular FJ. Sweep malleability.")
    print("segmented = platform gap near 1/3=0.333 and high population spread.")
    print("collapsed = platform gap near 0 and low population spread.\n")
    print(f"{'malleability':>12} | {'plat min-gap':>12} | {'pop circ-var':>12} | state")
    for m in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        pos, x = run(m)
        gap = plat_min_gap(pos)
        cv = circ_var(x)
        state = "segmented" if gap > 0.15 and cv > 0.3 else ("collapsed" if gap < 0.05 else "partial")
        print(f"{m:>12.2f} | {gap:>12.3f} | {cv:>12.3f} | {state}")


if __name__ == "__main__":
    main()
