"""Shared primitives for the competition experiments.

Platforms pick a position; users pick a platform by proximity (softmax). On the
circle, position 0 and 1 are the same point. opt_run is simultaneous gradient
play: each platform ascends its own captured share with rivals held fixed in
its gradient.
"""

import torch
import torch.nn.functional as F


def circle_dist(a, b):
    raw = (a - b).abs()
    return torch.minimum(raw, 1.0 - raw)


def line_dist(a, b):
    return (a - b).abs()


def dist_fn(geometry):
    return circle_dist if geometry == "circle" else line_dist


def opt_run(k, inits, geometry="circle", tau=0.2, lr=0.02, steps=6000, n=8000, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(n)
    dist = dist_fn(geometry)
    raw = torch.tensor([float(v) for v in inits], requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    for _ in range(steps):
        pos = raw % 1.0 if geometry == "circle" else raw.clamp(0.0, 1.0)
        total = 0.0
        for p in range(k):
            stacked = torch.stack([pos[q] if q == p else pos[q].detach() for q in range(k)])
            d = dist(stacked[:, None], x[None, :])
            s = F.softmax(-d / tau, dim=0)
            total = total - s[p].mean()
        opt.zero_grad()
        total.backward()
        opt.step()
    pos = (raw % 1.0 if geometry == "circle" else raw.clamp(0.0, 1.0)).detach().sort().values
    if geometry == "circle":
        gaps = torch.cat([pos[1:] - pos[:-1], (pos[0] + 1.0 - pos[-1]).unsqueeze(0)])
    else:
        gaps = pos[1:] - pos[:-1]
    return pos, gaps


def classify(k, gaps, geometry="circle"):
    if geometry == "circle":
        if float(gaps.min()) < 0.03:
            return "clumped"
        if float((gaps - 1.0 / k).abs().max()) < 0.10:
            return "equidistant"
        return "other"
    return "clumped" if float(gaps.sum()) < 0.05 else "spread"


def softmax_share_landscape(rivals, tau, geometry="circle", cost="linear", grid=1000, n=40000):
    """Fix the rivals, sweep a candidate platform, return its captured share."""
    x = torch.linspace(0, 1, n + 1)[:-1]
    dist = dist_fn(geometry)
    cand = torch.linspace(0, 1, grid + 1)[:-1]
    r = torch.tensor([float(v) for v in rivals])

    def c(d):
        return d if cost == "linear" else d * d

    u_r = torch.exp(-c(dist(r[:, None], x[None, :])) / tau).sum(0)
    u_c = torch.exp(-c(dist(cand[:, None], x[None, :])) / tau)
    share = (u_c / (u_c + u_r[None, :])).mean(1)
    return cand, share


def hard_share_landscape(rivals, geometry="circle", grid=1000, n=40000):
    x = torch.linspace(0, 1, n + 1)[:-1]
    dist = dist_fn(geometry)
    cand = torch.linspace(0, 1, grid + 1)[:-1]
    r = torch.tensor([float(v) for v in rivals])
    rival_min = dist(r[:, None], x[None, :]).min(0).values
    dc = dist(cand[:, None], x[None, :])
    share = (dc < rival_min[None, :]).float().mean(1)
    return cand, share
