"""Block 8: geometry. Same competition on a line vs a circle.

Line has a privileged center (minimum differentiation for K=2 -> collapse).
Circle has no center. Runs both a tiny MLP and a plain scalar position; if they
agree the MLP adds no artifact. Also a hard-assignment best-response control.

Finding: location-only competition does not rescue diversity by geometry.
Line K=2 collapses to center; the circle has segmented and paired equilibria
(see 01_baseline). Spreading needs softmax choice, not hard nearest.
"""

import math

import torch
import torch.nn.functional as F

from experiments.competition.circle._common import circle_dist, dist_fn


class PositionMLP(torch.nn.Module):
    def __init__(self, geometry, hidden=8):
        super().__init__()
        self.geometry = geometry
        out = 2 if geometry == "circle" else 1
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, out),
        )

    def position(self):
        z = self.net(torch.ones(1))
        if self.geometry == "circle":
            return (torch.atan2(z[1], z[0]) / (2 * math.pi)) % 1.0
        return torch.sigmoid(z[0])


class PositionParam(torch.nn.Module):
    def __init__(self, geometry, init):
        super().__init__()
        self.geometry = geometry
        self.raw = torch.nn.Parameter(torch.tensor(float(init)))

    def position(self):
        if self.geometry == "circle":
            return self.raw % 1.0
        return torch.sigmoid(self.raw)


def gradient_run(geometry, k, kind, n=4000, tau=0.02, lr=0.02, steps=4000, seed=0, inits=None):
    torch.manual_seed(seed)
    x = torch.rand(n)
    dist = dist_fn(geometry)
    if kind == "mlp":
        plats = [PositionMLP(geometry) for _ in range(k)]
    else:
        if inits is None:
            inits = [0.5 + 0.01 * i for i in range(k)]
        plats = [PositionParam(geometry, v) for v in inits]
    opt = torch.optim.Adam([p for m in plats for p in m.parameters()], lr=lr)
    init_pos = torch.stack([m.position() for m in plats]).detach().clone()
    for _ in range(steps):
        positions = [m.position() for m in plats]
        total = 0.0
        for p in range(k):
            stacked = torch.stack(
                [positions[q] if q == p else positions[q].detach() for q in range(k)]
            )
            d = dist(stacked[:, None], x[None, :])
            s = F.softmax(-d / tau, dim=0)
            total = total - s[p].mean()
        opt.zero_grad()
        total.backward()
        opt.step()
    return init_pos, torch.stack([m.position() for m in plats]).detach()


def best_response_run(geometry, k, inits, n=4000, sweeps=100, grid=400, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(n)
    dist = dist_fn(geometry)
    cand = torch.linspace(0, 1, grid + 1)[:-1] if geometry == "circle" else torch.linspace(0, 1, grid)
    pos = torch.tensor([float(v) for v in inits])
    init = pos.clone()
    for _ in range(sweeps):
        moved = False
        for p in range(k):
            others = torch.cat([pos[:p], pos[p + 1:]])
            rival_min = dist(others[:, None], x[None, :]).min(0).values
            dc = dist(cand[:, None], x[None, :])
            share = (dc < rival_min[None, :]).float().mean(1)
            best = cand[int(share.argmax())]
            if abs(float(best) - float(pos[p])) > 1e-9:
                pos[p] = best
                moved = True
        if not moved:
            break
    return init, pos


def summarize(geometry, k, final):
    sf = final.sort().values
    if geometry == "circle":
        gaps = torch.cat([sf[1:] - sf[:-1], (sf[0] + 1.0 - sf[-1]).unsqueeze(0)])
        return (
            f"  positions {[round(float(p), 3) for p in sf]}  "
            f"gaps {[round(float(g), 3) for g in gaps]}  target={1.0 / k:.3f}"
        )
    return (
        f"  positions {[round(float(p), 3) for p in sf]}  "
        f"spread={float(final.max() - final.min()):.4f}"
    )


def main():
    for kind in ["param", "mlp"]:
        print(f"\n==================== {kind} platforms (gradient) ====================")
        for geometry in ["line", "circle"]:
            for k in [2, 4]:
                _, final = gradient_run(geometry, k, kind)
                print(f"\n[{geometry}] K={k}")
                print(summarize(geometry, k, final))

    print("\n==================== best response, hard assignment ====================")
    for geometry in ["line", "circle"]:
        for k in [2, 3, 4]:
            clustered = [0.5 + 0.01 * i for i in range(k)]
            _, final = best_response_run(geometry, k, clustered)
            print(f"\n[{geometry}] K={k}")
            print(summarize(geometry, k, final))


if __name__ == "__main__":
    main()
