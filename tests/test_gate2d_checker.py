"""Fixture tests for the check_pofd_sanity GATE2D branch (2026-08-15).

Synthesizes physically-consistent 2-D-gate runs for the pofdgate2d_
family (mistral7b, arms b0/dyn, numeric eps_AI x numeric eps_social):
nested platform update on the start-of-round opinion, then a
mean-preserving Deffuant-style peer pass (pairs -> midpoint at
confidence < eps_social, accepted > 0), twin simulated on the h-only
path at the SAME peer setting.

Healthy (must PASS): b0 mid-grid, dyn mid-grid, the ea1/es1 full-open
corner (both arms), and the seed-991 3-round smoke.
Sabotage (must FAIL): corrupted gate replay, wrong social dose vs tag,
inactive peer step, wrong arm semantics (b0 with a KL term, dyn with a
LoRA), dyn context never refreshed, AI-gate tag mismatch, missing twin,
an es=0 tag (excluded from this family), and a non-mistral slug.

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
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"


def _num(v):
    return f"{v:g}".replace(".", "p")


def cfg_for(tag, arm, gate, es, seed, nrounds):
    c = {"run_tag": tag, "base_model": MISTRAL, "n_rounds": nrounds,
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
                  "train_cap": 723, "icl_k": 8, "icl_days": 0,
                  "icl_select": "random", "icl_ctx_source": "live",
                  "icl_snapshot_round": -1})
    return c


def build(parent, arm, gate, es, seed=0, nrounds=30,
          prefix="pofdgate2d", tag=None, cfg_mut=None, post=None,
          peers=True):
    if tag is None:
        tag = (f"{prefix}_mistral7b_{arm}_ea{_num(gate)}_w0p5_l0p2"
               f"_es{_num(es)}_s{seed}")
    cfg = cfg_for(tag, arm, gate, es, seed, nrounds)
    if cfg_mut:
        cfg_mut(cfg)
    g = torch.Generator().manual_seed(2000 + seed)
    x = INNATE.clone()
    tw = INNATE.clone()
    lam, w = 0.2, 0.5
    eps_ai = cfg["eps_ai"]
    rows, op_raw, pred_raw, gate_raw, twin_raw = [], [], [], [], []
    icl_idx, icl_val, ctx_rows = [], [], []
    if arm == "dyn":
        idx0 = torch.stack([torch.tensor(
            [(i + 7 * j + 1) % N for j in range(8)]) for i in range(N)])
        val0 = torch.rand(N, 8, generator=g)

    def peer_pass(v):
        # Deffuant-style: confident pairs to their midpoint -- conserves
        # the population mean exactly (the SOCIAL section's invariant)
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
        pred = torch.rand(N, generator=g)
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
        if arm == "dyn":
            ii = (idx0 + t) % N
            ii = torch.where(ii == torch.arange(N).unsqueeze(1),
                             (ii + 1) % N, ii)
            ii = torch.where(ii == torch.arange(N).unsqueeze(1),
                             (ii + 1) % N, ii)
            vv = val0 if t == 0 else torch.rand(N, 8, generator=g)
            icl_idx.append(ii.clone())
            icl_val.append(vv.clone())
            ctx_rows.append({"round": t,
                             "ctx": [f"ctx-dyn-{t}-{i}"
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
        assert rc == 0, f"expected PASS, got exit {rc}:\n{out[-2000:]}"
    else:
        assert rc != 0, f"expected FAIL, checker passed:\n{out[-2000:]}"
        if want_str is not None:
            assert want_str in out, \
                f"expected {want_str!r} in output:\n{out[-2000:]}"


# healthy ---------------------------------------------------------------

def test_healthy_b0_mid_grid(tmp_path):
    assert_verdict(build(tmp_path, "b0", 0.4, 0.2), True)


def test_healthy_dyn_mid_grid(tmp_path):
    assert_verdict(build(tmp_path, "dyn", 0.1, 0.4), True)


def test_healthy_b0_full_open_corner(tmp_path):
    assert_verdict(build(tmp_path, "b0", 1.0, 1.0), True)


def test_healthy_dyn_smoke(tmp_path):
    assert_verdict(build(tmp_path, "dyn", 1.0, 1.0, seed=991, nrounds=3,
                         prefix="pofdgate2dsmk"), True)


# sabotage --------------------------------------------------------------

def test_sabotage_gate_replay_flip(tmp_path):
    rd = build(tmp_path, "b0", 0.4, 0.2)

    def fn(t):
        t["gate_raw"][5][7] = ~t["gate_raw"][5][7]
    edit(rd, fn)
    assert_verdict(rd, False, "GATE")


def test_sabotage_social_dose_vs_tag(tmp_path):
    rd = build(tmp_path, "b0", 0.4, 0.4,
               cfg_mut=lambda c: c.update(eps=0.3))
    assert_verdict(rd, False, "CONFIG eps")


def test_sabotage_inactive_peer_step(tmp_path):
    rd = build(tmp_path, "dyn", 0.4, 0.2, peers=False)
    assert_verdict(rd, False, "PEER-ALIVE")


def test_sabotage_b0_with_kl_term(tmp_path):
    rd = build(tmp_path, "b0", 0.4, 0.2,
               cfg_mut=lambda c: c.update(kl_beta=1.0))
    assert_verdict(rd, False, "kl_beta")


def test_sabotage_dyn_with_lora(tmp_path):
    rd = build(tmp_path, "dyn", 0.4, 0.2,
               cfg_mut=lambda c: c.update(use_lora=1))
    assert_verdict(rd, False, "use_lora")


def test_sabotage_dyn_context_never_refreshed(tmp_path):
    rd = build(tmp_path, "dyn", 0.4, 0.2)

    def fn(t):
        for tt in range(1, t["icl_idx_raw"].shape[0]):
            t["icl_idx_raw"][tt] = t["icl_idx_raw"][0]
    edit(rd, fn)
    assert_verdict(rd, False, "never refreshed")


def test_sabotage_ai_gate_tag_mismatch(tmp_path):
    rd = build(tmp_path, "b0", 0.1, 0.2,
               tag="pofdgate2d_mistral7b_b0_ea0p4_w0p5_l0p2_es0p2_s0")
    assert_verdict(rd, False, "tag says")


def test_sabotage_missing_twin(tmp_path):
    rd = build(tmp_path, "b0", 0.4, 0.2)

    def fn(t):
        t["twin_raw"] = torch.empty(0)
    edit(rd, fn)
    assert_verdict(rd, False, "twin_raw")


def test_sabotage_es0_tag_excluded(tmp_path):
    rd = build(tmp_path, "b0", 0.4, 0.0, peers=False)
    assert_verdict(rd, False, "_es token in")


def test_sabotage_non_mistral_slug(tmp_path):
    rd = build(tmp_path, "b0", 0.4, 0.2,
               tag="pofdgate2d_qwen7b_b0_ea0p4_w0p5_l0p2_es0p2_s0",
               cfg_mut=lambda c: c.update(
                   base_model="Qwen/Qwen2.5-7B-Instruct"))
    assert_verdict(rd, False, "mistral")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
