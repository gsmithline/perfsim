"""ABM policy-sensitivity probe.

Skip the LM. Drive AT covid directly with a chosen `will_isolate` policy
and measure how `daily_infected` responds. If the env is insensitive to
policy at this regime, no LM training will produce signal. If it IS
sensitive, we know what range of variation we need the LM to express.

Probes three things in sequence:
    1. Uniform isolation level across all agents, swept over [0, 1].
       Determines: does the env respond AT ALL to policy magnitude?
    2. Age-targeted (elderly isolate more). Does targeting beat uniform?
    3. Vary ABM regime (R2, K_STEPS) to find a sensitive operating point.

Each probe restores env state from a snapshot before stepping, so they
all start from the same seeded initial conditions.

Run locally:  python examples/at_covid_policy_sensitivity.py
Run as marimo: marimo edit examples/at_covid_policy_sensitivity.py
"""

import marimo

__generated_with = "0.23.7"
app = marimo.App()


@app.cell
def imports():
    import copy
    import time

    import torch

    from perfsim.scenarios.at_covid import build_covid_runner, seed_initial_infections
    return build_covid_runner, copy, seed_initial_infections, time, torch


@app.cell
def _intro():
    import marimo as mo
    mo.md(
        """
        # AT covid policy-sensitivity probe

        Question: does the bundled covid sim's `daily_infected` respond to
        per-agent `will_isolate` policy at the current parameter regime?

        Drive the env directly with chosen policies. No LM, no training.
        Measure response.
        """
    )
    return (mo,)


@app.cell
def build_baseline_env(build_covid_runner, seed_initial_infections, torch):
    # Build one runner. Seed infections. We will reuse this runner across
    # all policy probes by snapshotting state.
    runner = build_covid_runner(seed=0)
    seed_initial_infections(runner, fraction=0.20, seed=0)
    runner.state["agents"]["citizens"]["platform_signal"] = torch.zeros(
        runner.state["agents"]["citizens"]["age"].shape[0]
    )
    n_agents = runner.state["agents"]["citizens"]["age"].shape[0]
    n_seeded = int((runner.state["agents"]["citizens"]["disease_stage"].squeeze() == 2.0).sum().item())
    print(f"runner built: {n_agents} agents, {n_seeded} initially infected")
    return n_agents, n_seeded, runner


@app.cell
def snapshot(copy, runner, torch):
    # Take a state snapshot we can restore between probes.
    def snapshot_state(state):
        out = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().clone()
            elif isinstance(v, dict):
                out[k] = snapshot_state(v)
            else:
                out[k] = copy.deepcopy(v)
        return out

    def restore_state(target, snap):
        for k, v in snap.items():
            if isinstance(v, dict):
                if k not in target or not isinstance(target[k], dict):
                    target[k] = {}
                restore_state(target[k], v)
            elif isinstance(v, torch.Tensor):
                target[k] = v.detach().clone()
            else:
                target[k] = copy.deepcopy(v)

    initial_snap = snapshot_state(runner.state)
    print("snapshot saved")
    return initial_snap, restore_state, snapshot_state


@app.cell
def _probe_helper(initial_snap, restore_state, runner, torch):
    # Helper: install a per-agent will_isolate policy, run K substeps,
    # report (daily_infected_sum, fraction_non_S).
    # The signal -> will_isolate path goes:
    #   platform_signal -> PerfsimIsolationDecision -> sigmoid -> will_isolate
    # so to get effective isolation_prob = p in (0,1) we need to write
    # platform_signal = logit(p) = log(p / (1 - p)).
    def run_probe(per_agent_isolation_prob, K=3):
        restore_state(runner.state, initial_snap)
        runner.reset_state_before_episode()
        p = per_agent_isolation_prob.clamp(min=1e-6, max=1 - 1e-6)
        platform_signal = torch.log(p / (1 - p))
        runner.state["agents"]["citizens"]["platform_signal"] = platform_signal
        with torch.no_grad():
            runner.step(num_steps=K)
        di = runner.state["environment"]["daily_infected"].sum().item()
        ds = runner.state["agents"]["citizens"]["disease_stage"].squeeze()
        non_s = (ds > 0).float().mean().item()
        return {"daily_infected_sum": di, "fraction_non_S": non_s}
    return (run_probe,)


@app.cell
def _md_uniform(mo):
    mo.md(
        """
        ## Probe 1: uniform isolation level

        Set every agent to the same isolation probability. Sweep across
        [0, 1]. If the env is responsive at all, `daily_infected` should
        decrease as isolation rises.
        """
    )
    return


@app.cell
def probe_uniform(n_agents, run_probe, time, torch):
    levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    results_uniform = []
    _t0 = time.time()
    for _lvl in levels:
        _policy = torch.full((n_agents,), _lvl, dtype=torch.float32)
        _r = run_probe(_policy, K=3)
        results_uniform.append({"isolation_level": _lvl, **_r})
        print(f"  isolation={_lvl:.2f}  di={_r['daily_infected_sum']:.1f}  nS={_r['fraction_non_S']:.4f}")
    print(f"total time: {time.time() - _t0:.1f}s")
    return levels, results_uniform


@app.cell
def _md_age(mo):
    mo.md(
        """
        ## Probe 2: age-targeted isolation

        Vary isolation by age bucket. Does targeting elderly help vs.
        uniform at the same total isolation budget?
        """
    )
    return


@app.cell
def probe_age_targeted(n_agents, run_probe, runner, time, torch):
    age = runner.state["agents"]["citizens"]["age"].squeeze().long()
    # Five policies, all averaging roughly the same isolation but with
    # different distributions over age buckets.
    age_policies = {
        "uniform_0.5": torch.full((n_agents,), 0.5),
        "elderly_high": torch.where(age >= 4, torch.tensor(0.9), torch.tensor(0.3)),
        "elderly_low":  torch.where(age >= 4, torch.tensor(0.3), torch.tensor(0.5)),
        "young_high":   torch.where(age <= 1, torch.tensor(0.9), torch.tensor(0.4)),
        "all_high":     torch.full((n_agents,), 0.9),
        "all_low":      torch.full((n_agents,), 0.1),
    }
    results_age = []
    _t0 = time.time()
    for _name, _policy in age_policies.items():
        _r = run_probe(_policy.float(), K=3)
        _avg = float(_policy.float().mean().item())
        results_age.append({"policy": _name, "avg_iso": _avg, **_r})
        print(f"  {_name:>15s}  avg={_avg:.3f}  di={_r['daily_infected_sum']:.1f}  nS={_r['fraction_non_S']:.4f}")
    print(f"total time: {time.time() - _t0:.1f}s")
    return age, age_policies, results_age


@app.cell
def _md_interaction(mo):
    mo.md(
        """
        ## Probe 3: mean_interactions-targeted

        High isolation for high-contact agents. Does this beat both uniform
        and age-targeted?
        """
    )
    return


@app.cell
def probe_interaction_targeted(n_agents, run_probe, runner, time, torch):
    mi = runner.state["environment"]["mean_interactions"].squeeze()
    mi_policies = {
        "high_int_high": torch.where(mi >= 3.5, torch.tensor(0.9), torch.tensor(0.3)),
        "high_int_low":  torch.where(mi >= 3.5, torch.tensor(0.3), torch.tensor(0.5)),
    }
    results_mi = []
    _t0 = time.time()
    for _name, _policy in mi_policies.items():
        _r = run_probe(_policy.float(), K=3)
        _avg = float(_policy.float().mean().item())
        results_mi.append({"policy": _name, "avg_iso": _avg, **_r})
        print(f"  {_name:>20s}  avg={_avg:.3f}  di={_r['daily_infected_sum']:.1f}  nS={_r['fraction_non_S']:.4f}")
    print(f"total time: {time.time() - _t0:.1f}s")
    return mi, mi_policies, results_mi


@app.cell
def _md_K(mo):
    mo.md(
        """
        ## Probe 4: vary K_STEPS at uniform=0.5

        Longer inner loop should amplify the difference (or settle to an
        absorbing state).
        """
    )
    return


@app.cell
def probe_K_sweep(n_agents, run_probe, time, torch):
    Ks = [1, 3, 5, 10, 21]
    results_K = []
    _t0 = time.time()
    for _K in Ks:
        for _lvl in [0.0, 0.5, 1.0]:
            _policy = torch.full((n_agents,), _lvl, dtype=torch.float32)
            _r = run_probe(_policy, K=_K)
            results_K.append({"K": _K, "isolation": _lvl, **_r})
        print(f"  K={_K} done")
    print(f"total time: {time.time() - _t0:.1f}s")
    return Ks, results_K


@app.cell
def _md_summary(mo):
    mo.md(
        """
        ## Read the results

        - If probe 1 shows no movement in `daily_infected` across
          isolation in [0, 1], the env is **completely insensitive** at
          this regime. Need to change R2 or seed_frac.
        - If probe 2/3 movements are similar to probe 1 at matched
          avg_iso, targeting does not buy anything beyond uniform level
          control. The LM has nothing to differentiate on.
        - If probe 4 shows env response at larger K, increase K_STEPS in
          the cluster sweep. If still flat at K=21, the parameter regime
          is wrong.
        """
    )
    return


if __name__ == "__main__":
    app.run()
