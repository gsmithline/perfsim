"""Fixture tests for the check_pofd_sanity CTXGRID branch (2026-08-15).

Synthesizes physically-consistent runs for the pofdctxgrid_ family (six
adaptation channels x numeric eps_AI x numeric eps_social, seed 0). The
family uniquely spans BOTH peer regimes, so the builder covers both:

  es = 0    no peer step; the twin sits on innate (lam*v + (1-lam)*v is
            within 1 float32 ulp of v) and the checker's exact-copy
            replay of the nested operator applies. Static-serving
            channels (k0/fz0/f32) also keep a frozen gate mask: an
            initially-gated agent moves toward a fixed served value so
            its gate stays open, and an initially-rejected agent sits on
            the innate anchor so its gate stays shut.
  es > 0    mean-preserving Deffuant peer pass (confident pairs ->
            midpoint, accepted > 0) on both the population and the twin.

Healthy (must PASS): b0/k0 at es=0, fz0 at es=0.2, f32 at es=0.4, d32 at
es=1, and the 3-round seed-991 K=32 smoke.
Sabotage (must FAIL): K=32 dial rendering only 8 exemplars, fixed K=32
with a live snapshot round, live K=32 never refreshing, k0 carrying
context, es=0 twin drifting off innate, es=0 static gate churn, static
serving drifting, corrupted gate replay, unknown arm token, off-grid
social dose, missing twin, inactive peer step, and b0 with a KL term.

Run with USE_TF=0 (the transformers TF probe deadlocks on this Mac).
"""
import gzip
import json
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CHECKER = os.path.join(REPO, "experiments", "scripts",
                       "cluster_pipelines", "check_pofd_sanity.py")
N = 60
_G0 = torch.Generator().manual_seed(20260815)
INNATE = torch.rand(N, generator=_G0)
BASES = {"qwen7b": "Qwen/Qwen2.5-7B-Instruct",
         "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
         "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3"}
ARM_K = {"b0": 0, "k0": 0, "fz0": 8, "dyn": 8, "f32": 32, "d32": 32}
ARM_SNAP = {"fz0": 0, "f32": 0, "dyn": -1, "d32": -1}
LIVE = ("dyn", "d32")


def _num(v):
    return f"{v:g}".replace(".", "p")


def cfg_for(tag, model, arm, gate, es, seed, nrounds):
    c = {"run_tag": tag, "base_model": BASES[model], "n_rounds": nrounds,
         "seed": seed, "eps": es, "gamma_bias": 0.0, "w_plat": 0.5,
         "innate_lambda": 0.2,
         "population_update": "nested_ai_then_social_v1",
         "data_regime": "replace", "deploy_every": 1, "pop_model": "ab",
         "run_mode": "loop", "canary_delta": 0.0, "pop_reset": False,
         "ab_sweeps": 1, "platform_sus_scale": 1.0,
         "anchor_mode": "fixed", "dataset": "movielens",
         "ml_target": "Action", "do_sample": False, "n_labeled": 723,
         "seed_base_data": True, "pristine_frac": 0.0,
         "replay_frac": 0.0, "teacher_label_delta": 0.0,
         "kl_ref_adapter": "", "icrh": False, "feedback_mode": "none",
         "profile_shuffle_p": 0.0, "profile_sort_q": 0.0,
         "profile_drop_cols": [], "profile_permute_cols": [],
         "eps_ai": gate, "ai_gate_mode": "threshold", "host": "g204",
         "hardware": {"hostname": "g204", "gpu_name": "A100",
                      "gpu_cc": "8.0", "cuda_version": "12.4",
                      "torch_version": "2.5.1",
                      "transformers_version": "4.46.0"}}
    if arm == "b0":
        c.update({"training_style": "sft", "kl_beta": 0.0,
                  "use_lora": True, "lora_r": 512, "sft_lr": 5e-5,
                  "sft_epochs": 1, "sft_batch_size": 4,
                  "fresh_each_round": True, "train_cap": 723,
                  "icl_k": 0, "icl_days": 0, "icl_select": "random",
                  "icl_ctx_source": "live"})
    else:
        c.update({"training_style": "frozen", "kl_beta": 0.0,
                  "use_lora": False, "fresh_each_round": False,
                  "train_cap": 723, "icl_k": ARM_K[arm], "icl_days": 0,
                  "icl_select": "random", "icl_ctx_source": "live"})
        if arm in ARM_SNAP:
            c["icl_snapshot_round"] = ARM_SNAP[arm]
    return c


def build(parent, model, arm, gate, es, seed=0, nrounds=30,
          prefix="pofdctxgrid", tag=None, cfg_mut=None, post=None,
          peers=None, render_k=None, flat_pred=False):
    if tag is None:
        tag = (f"{prefix}_{model}_{arm}_ea{_num(gate)}_w0p5_l0p2"
               f"_es{_num(es)}_s{seed}")
    cfg = cfg_for(tag, model, arm, gate, es, seed, nrounds)
    if cfg_mut:
        cfg_mut(cfg)
    peers = (es > 0) if peers is None else peers
    k_render = ARM_K[arm] if render_k is None else render_k
    g = torch.Generator().manual_seed(3000 + seed)
    x = INNATE.clone()
    tw = INNATE.clone()
    lam, w = 0.2, 0.5
    eps_ai = cfg["eps_ai"]
    frozen_pred = (torch.full((N,), 0.5) if flat_pred
                   else INNATE.mul(0.3).add(0.35))
    rows, op_raw, pred_raw, gate_raw, twin_raw = [], [], [], [], []
    icl_idx, icl_val, ctx_rows = [], [], []
    def exemplar_ids(t):
        # offsets live in [1, N-1], so (i + offset) % N is never i --
        # self-exclusion holds by construction at any K < N, and the
        # round rotation keeps the live channels genuinely re-drawing
        return torch.stack([torch.tensor(
            [(i + 1 + ((j + t) % (N - 1))) % N
             for j in range(k_render)]) for i in range(N)])

    if k_render:
        idx0 = exemplar_ids(0)
        val0 = torch.rand(N, k_render, generator=g)

    def peer_pass(v):
        acc = 0
        for _ in range(5):
            i = int(torch.randint(0, N, (1,), generator=g))
            j = int(torch.randint(0, N, (1,), generator=g))
            if i != j and abs(float(v[i] - v[j])) < es:
                mid = (v[i] + v[j]) / 2
                v[i] = mid
                v[j] = mid
                acc += 1
        return acc

    for t in range(nrounds):
        # static-serving channels hold their prediction; b0 retrains and
        # the live-context channels re-render, so both move each round
        pred = (frozen_pred.clone() if arm in ("k0", "fz0", "f32")
                else torch.rand(N, generator=g))
        served = pred.clamp(0.0, 1.0)
        gate_t = (served - x).abs() < eps_ai
        h = lam * INNATE + (1.0 - lam) * x
        x = torch.where(gate_t, (1.0 - w) * h + w * served, h)
        accepted = peer_pass(x) if peers else 0
        tw = lam * INNATE + (1.0 - lam) * tw
        if peers:
            peer_pass(tw)
        row = {"round": t, "deployment": t, "is_deploy": 1,
               "accepted": accepted,
               "contact": float(gate_t.float().mean()),
               "twin_mean": float(tw.mean()), "twin_std": float(tw.std()),
               "twin_bias": float(tw.mean() - INNATE.mean())}
        if arm == "b0":
            row["n_train"] = 723
        else:
            row["perplexity"] = 7.77
        if k_render:
            live = arm in LIVE
            if live:
                ii = exemplar_ids(t)
                vv = val0 if t == 0 else torch.rand(N, k_render,
                                                    generator=g)
            else:
                ii, vv = idx0, val0
            icl_idx.append(ii.clone())
            icl_val.append(vv.clone())
            ctx_rows.append({"round": t,
                             "ctx": [f"ctx-{arm}-{t if live else 0}-{i}"
                                     for i in range(N)]})
        rows.append(row)
        op_raw.append(x.clone())
        pred_raw.append(pred.clone())
        gate_raw.append(gate_t.clone())
        twin_raw.append(tw.clone())
    d = {"trajectory": rows, "config": cfg,
         "op_raw": torch.stack(op_raw), "pred_raw": torch.stack(pred_raw),
         "gate_raw": torch.stack(gate_raw),
         "twin_raw": torch.stack(twin_raw), "innate": INNATE.clone()}
    if icl_idx:
        d["icl_idx_raw"] = torch.stack(icl_idx)
        d["icl_val_raw"] = torch.stack(icl_val)
    if post:
        post(d)
    rd = parent / d["config"]["run_tag"]
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(json.dumps(d["config"]))
    torch.save(d, rd / "trajectory.pt")
    if ctx_rows:
        with gzip.open(rd / "icl_ctx_log.json.gz", "wt") as fh:
            for r in ctx_rows:
                fh.write(json.dumps(r) + "\n")
    return rd


def run_check(rd):
    p = subprocess.run([sys.executable, CHECKER, str(rd)],
                       capture_output=True, text=True,
                       env={**os.environ, "USE_TF": "0"})
    return p.returncode, p.stdout + p.stderr


def edit(rd, fn):
    t = torch.load(rd / "trajectory.pt", weights_only=False)
    fn(t)
    torch.save(t, rd / "trajectory.pt")
    (rd / "config.json").write_text(json.dumps(t["config"]))


def assert_verdict(rd, want_pass, want_str=None):
    rc, out = run_check(rd)
    if want_pass:
        assert rc == 0, f"expected PASS, got exit {rc}:\n{out[-2500:]}"
    else:
        assert rc != 0, f"expected FAIL, checker passed:\n{out[-2500:]}"
        if want_str is not None:
            assert want_str in out, \
                f"expected {want_str!r} in output:\n{out[-2500:]}"


# healthy ---------------------------------------------------------------

def test_healthy_b0_no_peers(tmp_path):
    assert_verdict(build(tmp_path, "qwen7b", "b0", 0.4, 0.0), True)


def test_healthy_k0_no_peers(tmp_path):
    assert_verdict(build(tmp_path, "olmo7b", "k0", 0.1, 0.0), True)


def test_healthy_fz0_peers(tmp_path):
    assert_verdict(build(tmp_path, "mistral7b", "fz0", 0.2, 0.2), True)


def test_healthy_f32_peers(tmp_path):
    assert_verdict(build(tmp_path, "qwen7b", "f32", 0.05, 0.4), True)


def test_healthy_d32_full_open(tmp_path):
    assert_verdict(build(tmp_path, "mistral7b", "d32", 1.0, 1.0), True)


def test_healthy_k32_smoke(tmp_path):
    assert_verdict(build(tmp_path, "mistral7b", "f32", 1.0, 1.0,
                         seed=991, nrounds=3,
                         prefix="pofdctxgridsmk"), True)


# sabotage --------------------------------------------------------------

def test_sabotage_k32_dial_renders_only_8(tmp_path):
    """config says K=32 but the rendered context holds 8 exemplars."""
    rd = build(tmp_path, "qwen7b", "f32", 0.2, 0.2, render_k=8)
    assert_verdict(rd, False, "exemplars per agent")


def test_sabotage_fixed_k32_with_live_snapshot(tmp_path):
    rd = build(tmp_path, "qwen7b", "f32", 0.2, 0.2,
               cfg_mut=lambda c: c.update(icl_snapshot_round=-1))
    assert_verdict(rd, False, "icl_snapshot_round")


def test_sabotage_live_k32_never_refreshed(tmp_path):
    rd = build(tmp_path, "mistral7b", "d32", 0.4, 0.2)

    def fn(t):
        for tt in range(1, t["icl_idx_raw"].shape[0]):
            t["icl_idx_raw"][tt] = t["icl_idx_raw"][0]
    edit(rd, fn)
    assert_verdict(rd, False, "never refreshed")


def test_sabotage_k0_carrying_context(tmp_path):
    rd = build(tmp_path, "olmo7b", "k0", 0.1, 0.2)

    def fn(t):
        t["icl_idx_raw"] = torch.ones(30, N, 8, dtype=torch.long)
        t["icl_val_raw"] = torch.rand(30, N, 8)
    edit(rd, fn)
    assert_verdict(rd, False, "CTXGRID k0")


def test_sabotage_es0_twin_drifts(tmp_path):
    rd = build(tmp_path, "qwen7b", "k0", 0.2, 0.0)

    def fn(t):
        t["twin_raw"][5] = t["twin_raw"][5] + 0.01
    edit(rd, fn)
    assert_verdict(rd, False, "twin drifts off innate")


def test_sabotage_es0_static_gate_churn(tmp_path):
    rd = build(tmp_path, "mistral7b", "f32", 0.2, 0.0)

    def fn(t):
        # flip an agent sitting far from the gate boundary
        margin = ((t["pred_raw"][0].clamp(0, 1) - t["innate"]).abs()
                  - 0.2).abs()
        far = int(torch.argmax(margin))
        for tt in range(1, t["gate_raw"].shape[0]):
            t["gate_raw"][tt][far] = ~t["gate_raw"][tt][far]
    edit(rd, fn)
    assert_verdict(rd, False, "freeze the gate set")


def test_sabotage_static_serving_drifts(tmp_path):
    rd = build(tmp_path, "qwen7b", "fz0", 0.4, 0.2)

    def fn(t):
        t["pred_raw"][7] = t["pred_raw"][7] + 0.05
    edit(rd, fn)
    assert_verdict(rd, False, "static serving must be constant")


def test_sabotage_gate_replay_flip(tmp_path):
    rd = build(tmp_path, "qwen7b", "b0", 0.4, 0.2)

    def fn(t):
        t["gate_raw"][5][7] = ~t["gate_raw"][5][7]
    edit(rd, fn)
    assert_verdict(rd, False, "GATE")


def test_sabotage_unknown_arm_token(tmp_path):
    rd = build(tmp_path, "qwen7b", "dyn", 0.4, 0.2,
               tag="pofdctxgrid_qwen7b_k8_ea0p4_w0p5_l0p2_es0p2_s0")
    assert_verdict(rd, False, "unknown ctxgrid arm")


def test_sabotage_off_grid_social_dose(tmp_path):
    rd = build(tmp_path, "qwen7b", "dyn", 0.4, 0.3)
    assert_verdict(rd, False, "_es token in")


def test_sabotage_missing_twin(tmp_path):
    rd = build(tmp_path, "qwen7b", "b0", 0.4, 0.2)

    def fn(t):
        t["twin_raw"] = torch.empty(0)
    edit(rd, fn)
    assert_verdict(rd, False, "twin_raw")


def test_sabotage_inactive_peer_step(tmp_path):
    rd = build(tmp_path, "mistral7b", "d32", 0.4, 0.4, peers=False)
    assert_verdict(rd, False, "PEER-ALIVE")


def test_sabotage_total_silent_parse_failure(tmp_path):
    """Every agent every round at exactly 0.5 = the runner's silent
    parse-failure default. Observed for real on mistral7b at K=32."""
    rd = build(tmp_path, "mistral7b", "f32", 1.0, 1.0, flat_pred=True)
    assert_verdict(rd, False, "silent parse-failure default")


def test_sabotage_b0_with_kl_term(tmp_path):
    rd = build(tmp_path, "qwen7b", "b0", 0.4, 0.2,
               cfg_mut=lambda c: c.update(kl_beta=1.0))
    assert_verdict(rd, False, "kl_beta")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
