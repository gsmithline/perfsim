"""Gradient diagnostics for AT covid autodiff.

Three tests that determine whether grad_run produces *meaningful* gradients,
not just non-zero ones:

1. Sign check: aggressive-isolation vs no-isolation models should have
   opposite gradient directions.
2. Finite-difference validation: autograd gradient vs numerical perturbation.
3. Variance across seeds: how noisy is the gradient estimate?

Run: python scripts/grad_diagnostics.py
"""

from __future__ import annotations

import time

import torch

from perfsim.scenarios.at_covid import (
    default_signal_writer_grad,
    make_covid_env,
    seed_initial_infections,
)


INFECTION_FRACTION = 0.05
N_STEPS = 5


def _build_env(seed: int = 0) -> object:
    env = make_covid_env(
        init_seed=seed,
        signal_writer=default_signal_writer_grad,
        initial_infections_fraction=INFECTION_FRACTION,
    )
    return env


def _total_infected(env) -> torch.Tensor:
    return env.runner.state["environment"]["daily_infected"].sum()


# ---- Test 1: Sign check ----------------------------------------------------


def test_sign_check():
    """Two models with opposite isolation policies should produce gradients
    that point in opposite directions."""
    print("=" * 60)
    print("TEST 1: Sign check (aggressive vs passive isolation)")
    print("=" * 60)

    results = {}
    for label, weight_val, bias_val in [
        ("aggressive (high isolation)", 2.0, 2.0),
        ("passive (no isolation)", -2.0, -5.0),
    ]:
        env = _build_env(seed=0)
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(weight_val)
            model.bias.fill_(bias_val)

        env.grad_run(model, n_steps=N_STEPS)
        loss = _total_infected(env)
        loss.backward()

        gw = model.weight.grad.item()
        gb = model.bias.grad.item()
        results[label] = {"loss": loss.item(), "grad_w": gw, "grad_b": gb}
        print(f"\n  {label}:")
        print(f"    total_infected = {loss.item():.2f}")
        print(f"    dL/dw = {gw:.6f}")
        print(f"    dL/db = {gb:.6f}")

    agg = results["aggressive (high isolation)"]
    pas = results["passive (no isolation)"]

    w_signs_differ = (agg["grad_w"] * pas["grad_w"]) < 0
    b_signs_differ = (agg["grad_b"] * pas["grad_b"]) < 0

    print(f"\n  Weight gradient signs differ: {w_signs_differ}")
    print(f"  Bias gradient signs differ:   {b_signs_differ}")

    if agg["loss"] < pas["loss"]:
        print("  Aggressive isolation -> fewer infections (expected)")
    else:
        print("  WARNING: aggressive isolation did NOT reduce infections")

    return w_signs_differ or b_signs_differ


# ---- Test 2: Finite-difference validation -----------------------------------


def test_finite_difference(eps: float = 0.5, n_fd_seeds: int = 5):
    """Compare autograd gradient to averaged finite-difference estimate.

    Because the ABM is stochastic, we average the finite-difference over
    multiple seeds to reduce noise."""
    print("\n" + "=" * 60)
    print("TEST 2: Finite-difference validation")
    print("=" * 60)

    w0 = 0.05
    b0 = -1.0

    # Autograd gradient (single seed)
    env = _build_env(seed=0)
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(w0)
        model.bias.fill_(b0)

    env.grad_run(model, n_steps=N_STEPS)
    loss_ag = _total_infected(env)
    loss_ag.backward()
    autograd_dw = model.weight.grad.item()
    autograd_db = model.bias.grad.item()

    print(f"\n  Autograd (seed=0):")
    print(f"    dL/dw = {autograd_dw:.6f}")
    print(f"    dL/db = {autograd_db:.6f}")

    # Finite-difference averaged over seeds
    fd_dw_list = []
    fd_db_list = []

    for seed in range(n_fd_seeds):
        # f(w + eps)
        env_plus = _build_env(seed=seed)
        m_plus = torch.nn.Linear(1, 1)
        with torch.no_grad():
            m_plus.weight.fill_(w0 + eps)
            m_plus.bias.fill_(b0)
        env_plus.run(m_plus, n_steps=N_STEPS)
        loss_plus = _total_infected(env_plus).item()

        # f(w - eps)
        env_minus = _build_env(seed=seed)
        m_minus = torch.nn.Linear(1, 1)
        with torch.no_grad():
            m_minus.weight.fill_(w0 - eps)
            m_minus.bias.fill_(b0)
        env_minus.run(m_minus, n_steps=N_STEPS)
        loss_minus = _total_infected(env_minus).item()

        fd_dw = (loss_plus - loss_minus) / (2 * eps)
        fd_dw_list.append(fd_dw)

        # f(b + eps)
        env_plus_b = _build_env(seed=seed)
        m_plus_b = torch.nn.Linear(1, 1)
        with torch.no_grad():
            m_plus_b.weight.fill_(w0)
            m_plus_b.bias.fill_(b0 + eps)
        env_plus_b.run(m_plus_b, n_steps=N_STEPS)
        loss_plus_b = _total_infected(env_plus_b).item()

        env_minus_b = _build_env(seed=seed)
        m_minus_b = torch.nn.Linear(1, 1)
        with torch.no_grad():
            m_minus_b.weight.fill_(w0)
            m_minus_b.bias.fill_(b0 - eps)
        env_minus_b.run(m_minus_b, n_steps=N_STEPS)
        loss_minus_b = _total_infected(env_minus_b).item()

        fd_db = (loss_plus_b - loss_minus_b) / (2 * eps)
        fd_db_list.append(fd_db)

    fd_dw_mean = sum(fd_dw_list) / len(fd_dw_list)
    fd_db_mean = sum(fd_db_list) / len(fd_db_list)
    fd_dw_std = (sum((x - fd_dw_mean) ** 2 for x in fd_dw_list) / len(fd_dw_list)) ** 0.5
    fd_db_std = (sum((x - fd_db_mean) ** 2 for x in fd_db_list) / len(fd_db_list)) ** 0.5

    print(f"\n  Finite-difference (eps={eps}, averaged over {n_fd_seeds} seeds):")
    print(f"    dL/dw = {fd_dw_mean:.6f}  (std={fd_dw_std:.6f})")
    print(f"    dL/db = {fd_db_mean:.6f}  (std={fd_db_std:.6f})")

    # Compare
    print(f"\n  Comparison:")
    w_sign_match = (autograd_dw * fd_dw_mean) > 0 if abs(fd_dw_mean) > 1e-10 else None
    b_sign_match = (autograd_db * fd_db_mean) > 0 if abs(fd_db_mean) > 1e-10 else None
    print(f"    Weight: autograd={autograd_dw:.6f}  fd={fd_dw_mean:.6f}  sign_match={w_sign_match}")
    print(f"    Bias:   autograd={autograd_db:.6f}  fd={fd_db_mean:.6f}  sign_match={b_sign_match}")

    if abs(fd_dw_mean) > 1e-10:
        ratio_w = autograd_dw / fd_dw_mean
        print(f"    Weight ratio (autograd/fd): {ratio_w:.2f}")
    if abs(fd_db_mean) > 1e-10:
        ratio_b = autograd_db / fd_db_mean
        print(f"    Bias ratio (autograd/fd):   {ratio_b:.2f}")

    return w_sign_match, b_sign_match


# ---- Test 3: Variance across seeds -----------------------------------------


def test_variance(n_seeds: int = 10):
    """Run grad_run with the same model across different seeds.
    Report mean, std, and coefficient of variation of the gradient."""
    print("\n" + "=" * 60)
    print(f"TEST 3: Gradient variance across {n_seeds} seeds")
    print("=" * 60)

    w0 = 0.05
    b0 = -1.0

    grad_w_list = []
    grad_b_list = []
    loss_list = []

    for seed in range(n_seeds):
        env = _build_env(seed=seed)
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(w0)
            model.bias.fill_(b0)

        env.grad_run(model, n_steps=N_STEPS)
        loss = _total_infected(env)
        loss.backward()

        grad_w_list.append(model.weight.grad.item())
        grad_b_list.append(model.bias.grad.item())
        loss_list.append(loss.item())

    def _stats(vals, name):
        mean = sum(vals) / len(vals)
        std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
        cv = abs(std / mean) if abs(mean) > 1e-12 else float("inf")
        signs = sum(1 for x in vals if x > 0)
        print(f"\n  {name}:")
        print(f"    mean = {mean:.6f}")
        print(f"    std  = {std:.6f}")
        print(f"    CV   = {cv:.2f}")
        print(f"    sign consistency: {max(signs, len(vals) - signs)}/{len(vals)} same sign")
        print(f"    range: [{min(vals):.6f}, {max(vals):.6f}]")
        return mean, std, cv

    _stats(loss_list, "total_infected (loss)")
    _, _, cv_w = _stats(grad_w_list, "dL/dw")
    _, _, cv_b = _stats(grad_b_list, "dL/db")

    if cv_w < 1.0 and cv_b < 1.0:
        print("\n  RESULT: Low variance (CV < 1). Individual gradient estimates are informative.")
    elif cv_w < 2.0 and cv_b < 2.0:
        print("\n  RESULT: Moderate variance (CV 1-2). Average over ~5 seeds for reliable signal.")
    else:
        print("\n  RESULT: High variance (CV > 2). Need many seeds or a surrogate.")


# ---- Main -------------------------------------------------------------------


if __name__ == "__main__":
    t0 = time.time()

    print("AT Covid Gradient Diagnostics")
    print(f"infection_fraction={INFECTION_FRACTION}, n_steps={N_STEPS}")
    print()

    test_sign_check()
    test_finite_difference()
    test_variance()

    print(f"\n{'=' * 60}")
    print(f"Total wall time: {time.time() - t0:.1f}s")
