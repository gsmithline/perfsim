"""Block 3: prior pull on platforms (shared vs distinct base models).

Each platform's objective gains a fixed attractor: share minus
prior_w * circular distance to its prior point. Shared prior = K LoRA copies
of one base model; distinct priors = different base models with their own
preferred regions. Three questions:
  1. segmented start, shared prior, frozen population: at what prior_w does
     the prior crush segmentation (prior-driven clumping, no feedback)?
  2. monoculture start (all platforms at the shared prior + tiny asymmetry),
     frozen population: can competition pull differentiation out of
     near-identical platforms?
  3. segmented start, moving population: do distinct priors delay the
     malleability collapse, and does a shared prior accelerate it?
"""

import importlib.util

import torch
import torch.nn.functional as F

from experiments.competition.circle._common import circle_dist

_spec = importlib.util.spec_from_file_location("b2", "experiments/competition/circle/02_phase_diagram.py")
b2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2)


def compete_prior(pos, x, tau, prior_pts, prior_w, lr=0.01, steps=1200):
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
            total = total - s[j].mean() + prior_w * circle_dist(p[j], prior_pts[j])
        opt.zero_grad()
        total.backward()
        opt.step()
    return (pos % 1.0).detach()


def run_prior(malleability, prior_pts, prior_w, start, k=3, tau=0.2, n=3000,
              degree=10, rounds=25, peer_share=0.5, seed=0):
    torch.manual_seed(seed)
    w_graph = b2.random_graph(n, degree, seed)
    innate = torch.rand(n)
    x = innate.clone()
    pos = start.clone()
    for _ in range(rounds):
        pos = compete_prior(pos, x, tau, prior_pts, prior_w)
        if malleability > 0:
            x = b2.fj_step(x, innate, w_graph, pos, tau, malleability, peer_share)
    return pos, x


def row(pos, x, prior_pts, k):
    sep = b2.plat_min_gap(pos) * k
    to_prior = float(circle_dist(pos, prior_pts).mean())
    return sep, to_prior, b2.circ_var(x)


def main():
    k, tau = 3, 0.2
    seg = torch.tensor([(i + 0.5) / k for i in range(k)])
    shared = torch.full((k,), 0.5)
    seeds = [0, 1]

    print("Exp 1: segmented start, shared prior at 0.5, frozen population.")
    print("sep=1 segmented, sep=0 clumped; to_prior = mean platform dist to 0.5")
    print(f"{'prior_w':>8} | {'sep':>6} | {'to_prior':>8}")
    for w in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]:
        res = [row(*run_prior(0.0, shared, w, seg, rounds=4, seed=s), shared, k)
               for s in seeds]
        sep = sum(r[0] for r in res) / len(res)
        dp = sum(r[1] for r in res) / len(res)
        print(f"{w:>8.2f} | {sep:>6.3f} | {dp:>8.3f}", flush=True)

    print("\nExp 2: monoculture start (all at 0.5 + [-0.01,0,0.01]), shared prior, frozen pop.")
    mono = shared + torch.tensor([-0.01, 0.0, 0.01])
    print(f"{'prior_w':>8} | {'sep':>6} | {'to_prior':>8}")
    for w in [0.0, 0.05, 0.1, 0.2]:
        res = [row(*run_prior(0.0, shared, w, mono, rounds=4, seed=s), shared, k)
               for s in seeds]
        sep = sum(r[0] for r in res) / len(res)
        dp = sum(r[1] for r in res) / len(res)
        print(f"{w:>8.2f} | {sep:>6.3f} | {dp:>8.3f}", flush=True)

    print("\nExp 3: segmented start, moving population, 25 rounds.")
    print("distinct priors = one per platform at its segmented spot; shared = all at 0.5")
    print(f"{'priors':>8} {'mall':>5} {'prior_w':>8} | {'sep':>6} | {'pop_cv':>7}")
    for label, pts in [("distinct", seg), ("shared", shared)]:
        malls = [0.4, 0.6] if label == "distinct" else [0.6]
        for m in malls:
            for w in [0.0, 0.1, 0.3]:
                if label == "shared" and w == 0.0:
                    continue
                res = [row(*run_prior(m, pts, w, seg, seed=s), pts, k) for s in seeds]
                sep = sum(r[0] for r in res) / len(res)
                cv = sum(r[2] for r in res) / len(res)
                print(f"{label:>8} {m:>5.1f} {w:>8.2f} | {sep:>6.3f} | {cv:>7.3f}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
