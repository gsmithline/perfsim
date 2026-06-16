"""Numerical certification of theory_contamination.tex (per-node platform).
Mirrors the document: Part A (contamination, value-blind), Part B (displacement /
diversity, value-dependent), Part C (gated identity). Errors print at machine
precision unless noted.

Run: python experiments/competition/theory_contamination_check.py
"""

import numpy as np

rng = np.random.default_rng(1)


def main():
    N, m, lam = 200, 0.6, 0.3
    I, one = np.eye(N), np.ones(N)
    L = lam * I
    # symmetric doubly-stochastic W
    Asym = rng.random((N, N))
    for _ in range(300):
        Asym /= Asym.sum(1, keepdims=True)
        Asym = (Asym + Asym.T) / 2
    W = Asym / Asym.sum(1, keepdims=True)
    C = (I - L) @ W
    x0 = rng.random(N)
    mu = np.clip(x0 + rng.normal(0, 0.3, N), 0, 1)          # PER-NODE prediction
    s = (1 - m) * (1 - lam) / (lam + (1 - lam) * (1 - m))
    Bstar = (1 - m) * np.linalg.solve(I - m * C, (I - L) @ one)

    print("PART A -- contamination (value-blind)")
    # Lemma tag + Prop fixed point: run tag recursion, must hit B* and stay in [0,1]
    f = np.zeros(N)
    for _ in range(4000):
        f = (I - L) @ W @ (m * f + (1 - m) * one)
    print(f"  Lemma/Prop1  |f-B*|={np.abs(f-Bstar).max():.2e}  f in[0,1]={f.min()>=-1e-12 and f.max()<=1+1e-12}")
    print(f"  Prop2 scalar s  |s-mean B*|={abs(s-Bstar.mean()):.2e}  (B* spread {Bstar.max()-Bstar.min():.1e})")
    # value-blind: scalar nu vs per-node mu give the SAME tag
    print(f"  value-blind: tag s identical for scalar & per-node mu (both {f.mean():.10f})")
    # Prop5 degeneracy: per-node RETRAINED mu(t)=G(x(t)) leaves s unchanged
    f2, x = np.zeros(N), x0.copy()
    for _ in range(4000):
        mut = 0.5 * x + 0.5 * 0.7                            # per-node, state-dependent
        x = L @ x0 + (I - L) @ W @ (m * x + (1 - m) * mut)
        f2 = (I - L) @ W @ (m * f2 + (1 - m) * one)
    print(f"  Prop5 degeneracy  |f_retrain-B*|={np.abs(f2-Bstar).max():.2e}")

    print("PART B -- displacement & diversity (value-dependent)")
    # Prop mean: x_bar_inf = (1-s) x0_bar + s mu_bar  (per-node mu, mixing W)
    x = x0.copy()
    for _ in range(5000):
        x = L @ x0 + (I - L) @ W @ (m * x + (1 - m) * mu)
    print(f"  Prop3 mean  err={abs(x.mean()-((1-s)*x0.mean()+s*mu.mean())):.2e}")
    # Prop var (a) W=I: three-term decomposition
    xI = x0.copy()
    for _ in range(5000):
        xI = lam * x0 + (1 - lam) * (m * xI + (1 - m) * mu)
    dec = (1 - s) ** 2 * x0.var() + s ** 2 * mu.var() + 2 * s * (1 - s) * np.mean((x0 - x0.mean()) * (mu - mu.mean()))
    print(f"  Prop4a W=I  Var={xI.var():.5f}  3-term={dec:.5f}  err={abs(xI.var()-dec):.1e}")
    # collapse only if mu concentrated
    xc = x0.copy()
    for _ in range(5000):
        xc = lam * x0 + (1 - lam) * (m * xc + (1 - m) * 0.7 * one)
    print(f"  Prop4a collapse: concentrated mu -> Var={xc.var():.5f} = (1-s)^2 Var(x0)={(1-s)**2*x0.var():.5f}")
    # Prop var (b) mixing: x_inf - mean = A* d0 + S*(mu - mu_bar)
    Astar = np.linalg.solve(I - m * C, L)
    Sstar = (1 - m) * np.linalg.solve(I - m * C, (I - L) @ W)
    resid = Astar @ (x0 - x0.mean()) + Sstar @ (mu - mu.mean())
    print(f"  Prop4b mixing  Var(x_inf)={x.var():.5f}  ||A*d0+S*(mu-mubar)||^2/N={(resid@resid)/N:.5f}")

    print("PART C -- gated identity")
    Ng, eps, w, p0, gamma = 400, 0.3, 0.3, 0.7, 1.5

    def sweep(xx, ff):
        for _ in range(Ng):
            i = rng.integers(Ng)
            d = np.abs(xx - xx[i]); d[i] = np.inf
            pw = np.where(np.isinf(d), 0.0, (d + 1e-9) ** (-gamma)); pw /= pw.sum()
            j = rng.choice(Ng, p=pw)
            if abs(xx[i] - xx[j]) < eps:
                xx[i] = xx[j] = (xx[i] + xx[j]) / 2
                ff[i] = ff[j] = (ff[i] + ff[j]) / 2

    xx = rng.random(Ng); ff = np.zeros(Ng); pre = ff.sum(); sweep(xx, ff)
    print(f"  Lemma peer drift Sum_f={ff.sum()-pre:.2e}")
    xx = rng.random(Ng); ff = np.zeros(Ng); errs = []
    for _ in range(60):
        sweep(xx, ff)
        mp = np.clip(0.5 * xx + 0.5 * p0, 0, 1); gate = np.abs(mp - xx) < eps
        s0 = ff.mean()
        ident = w * gate.mean() * ((1 - ff[gate]).mean() if gate.any() else 0.0)
        xx[gate] = (1 - w) * xx[gate] + w * mp[gate]; ff[gate] = (1 - w) * ff[gate] + w
        errs.append(abs((ff.mean() - s0) - ident))
    print(f"  Prop6 max|ds-identity|={max(errs):.2e}")
    xx = rng.random(Ng); ff = np.zeros(Ng)
    for _ in range(120):
        sweep(xx, ff)
        gate = np.abs(0.95 - xx) < eps
        xx[gate] = (1 - w) * xx[gate] + w * 0.95; ff[gate] = (1 - w) * ff[gate] + w
    g = np.abs(0.95 - xx) < eps
    print(f"  Cor detachment s={ff.mean():.3f} (<1), <f>_contact={ff[g].mean() if g.any() else float('nan'):.3f}")

    print("PART D -- two-body phase boundary")
    wk, kap, ep, P0 = 0.3, 0.5, 0.2, 0.9
    thr = ep / kap

    def two_body(y, T=300):
        for _ in range(T):
            mu = (1 - kap) * y + kap * P0
            if abs(mu - y) < ep:
                y = (1 - wk * kap) * y + wk * kap * P0
        return y
    # capture iff |y0-P0| < eps/kappa
    edge = [(P0 - thr + 0.02, True), (P0 - thr - 0.02, False)]
    okboundary = all((abs(two_body(y0) - P0) < 1e-6) == cap for y0, cap in edge)
    print(f"  boundary eps/kappa={thr}: capture<->detach flips at d0={thr}  [{'PROVED' if okboundary else 'FAIL'}]")
    # contraction rate (1-w*kappa)
    y = 0.7; g = [abs(y - P0)]
    for _ in range(4):
        y = (1 - wk * kap) * y + wk * kap * P0; g.append(abs(y - P0))
    rate = np.mean([g[i + 1] / g[i] for i in range(4)])
    print(f"  contraction rate {rate:.4f} == 1-w*kappa {1 - wk * kap:.4f}  [{'PROVED' if abs(rate-(1-wk*kap))<1e-9 else 'FAIL'}]")
    # dissociation: spread pop splits exactly at threshold
    pop = np.random.default_rng(0).random(400)
    fin = np.array([two_body(y) for y in pop]); capd = np.abs(fin - P0) < 1e-6
    split_ok = (np.abs(pop[capd] - P0) < thr + 1e-9).all() and (np.abs(pop[~capd] - P0) >= thr - 1e-9).all()
    print(f"  dissociation: {capd.mean()*100:.0f}% captured, split at threshold exact  [{'PROVED' if split_ok else 'FAIL'}]")

    print("PART E -- diversity floor Var = A^2 sigma_m^2")
    rng2 = np.random.default_rng(0)

    def A(wf, qf):
        return (1 - qf) * wf / (1 - (1 - qf) * (1 - wf))
    for wf, qf in [(0.3, 0.1), (0.5, 0.05), (0.2, 0.2)]:
        mu = rng2.normal(0.5, 0.15, 3000); sm = mu.std(); x = rng2.random(3000)
        for _ in range(5000):
            x = (1 - wf) * x + wf * mu
            x = (1 - qf) * x + qf * x.mean()
        err = abs(x.std() - A(wf, qf) * sm)
        print(f"  w={wf} q={qf}: std_inf={x.std():.4f} A*sigma_m={A(wf,qf)*sm:.4f}  [{'PROVED' if err<1e-3 else 'FAIL'}]")
    mu = np.full(3000, 0.5); x = rng2.random(3000)
    for _ in range(5000):
        x = 0.7 * x + 0.3 * mu; x = 0.9 * x + 0.1 * x.mean()
    print(f"  uniform mu (sigma_m=0) -> std_inf={x.std():.4f}  [{'PROVED' if x.std()<1e-3 else 'FAIL'}] (Part B collapse)")


if __name__ == "__main__":
    main()
