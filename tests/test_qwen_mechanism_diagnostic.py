"""Tests for the QWEN2.5 MECHANISM DIAGNOSTIC (2026-08-20).

Two production keys (qwen_mechanism_frozen, qwen_wu_limit) plus a smoke,
a tracked perfect-prediction oracle, an offline frozen replay, and a
hard-gated analyzer.

THE SABOTAGE FIXTURES ARE THE POINT. A checker that only ever sees good
data proves nothing, and this project has already been bitten once: an
earlier sabotage fixture "passed" because it patched config.json, while
the checker actually reads the config EMBEDDED in trajectory.pt. So the
fixtures here are built by running the REAL population operators, are
verified to pass the REAL checker first, and every mutation is applied
to the embedded config or the tensors themselves.

Run with USE_TF=0.
"""
import copy
import importlib.util
import json
import os
import tempfile

import pytest
import torch

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
CONDOR = os.path.join(REPO, "experiments", "condor")
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
CANON_SRC = os.path.join(
    REPO, "notes", "pofd", "cluster",
    "pofdfam_qwen7b_k0_ea1_w0p5_l0p2_es0p05_s0")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load("gen_qmech", os.path.join(CONDOR, "gen_pofd_sweep.py"))
CHK = _load("chk_qmech", os.path.join(PIPE, "check_pofd_sanity.py"))
PP = _load("sim_pp_t", os.path.join(PIPE, "sim_perfect_predictor.py"))
AN = _load("an_qmech",
           os.path.join(PIPE, "analyze_qwen_mechanism_diagnostic.py"))
gp = _load("gp_qmech", os.path.join(PIPE, "_gated_pop.py"))

MANIFEST = json.load(open(
    os.path.join(CONDOR, "manifest_qwen_mechanism.json")))


# ======================================================================
# generator: counts, tags, and the surface each key pins
# ======================================================================

def test_frozen_key_is_exactly_the_audited_new_cells():
    rows = GEN.qmech_rows()
    assert len(rows) == MANIFEST["n_new"] == 5, len(rows)
    assert MANIFEST["n_reused"] == 19
    assert MANIFEST["n_gpu_cells"] == 24
    assert MANIFEST["n_conceptual_cells"] == 32
    assert MANIFEST["n_perfect_prediction_cells"] == 8
    tags = {r.split(",")[0] for r in rows}
    assert len(tags) == 5
    # every new cell is the FROZEN arm -- all 16 SFT cells already exist
    assert all("_k0_" in t for t in tags)


def test_frozen_new_cells_are_the_a100_replacement_plus_four_k1():
    tags = {r.split(",")[0] for r in GEN.qmech_rows()}
    assert GEN.qmech_tag("k0", 0.2, 0.0) in tags      # hardware-matched
    for es in (0.0, 0.05, 0.2, 1.0):
        assert GEN.qmech_tag("k0", 1.0, es) in tags
    # and NOTHING at k=.2 beyond the es=0 replacement
    assert sum(1 for t in tags if "_l0p2_" in t) == 1


def test_the_a100_frozen_cell_is_refused_for_hardware_not_reused():
    c = next(c for c in MANIFEST["cells"] if c["arm"] == "k0"
             and c["innate_k"] == 0.2 and c["eps_social"] == 0.0)
    assert c["status"] == "new"
    whys = " ".join(r["why"] for r in c["rejected_matches"])
    assert "A100" in whys and "H100" in whys
    assert "pofdreach_qwen7b_k0_ea1_w0p5_l0p2_es0_s0" in \
        " ".join(r["run_tag"] for r in c["rejected_matches"])


def test_canonical_frozen_hash_agrees_in_all_three_places():
    """The audit DERIVES it, the generator pins it, the checker pins it.
    A silent disagreement would let a different frozen prior into the
    grid, so all three must be the same string."""
    assert MANIFEST["canonical_frozen_pred_sha256"] == \
        GEN.QMECH_CANONICAL_PRED_SHA == CHK.QMECH_CANONICAL_PRED_SHA


def test_frozen_sub_pins_the_exact_h100_sku_and_queues_k():
    sub = GEN.qmech_sub()
    assert f'CUDADeviceName == "{GEN.QMECH_H100}"' in sub
    # the exact SKU, not a family: the pool also reports a bare
    # "NVIDIA H100" for a different card
    assert GEN.QMECH_H100 == "NVIDIA H100 80GB HBM3"
    env = next(ln for ln in sub.splitlines() if ln.startswith("environment"))
    assert "INNATE_LAMBDA=$(lam)" in env       # k spans .2 AND 1
    assert "PEER_GATE_MODE=threshold" in env
    assert "WITH_TWIN=1" in env
    assert "POP_RESET" not in env


def test_wu_limit_is_exactly_four_jobs_smoke_excluded():
    rows = GEN.qwu_rows()
    assert len(rows) == 4, len(rows)
    tags = {r.split(",")[0] for r in rows}
    assert len(tags) == 4
    for w in (0.5, 1.0):
        assert sum(1 for t in tags if f"_w{GEN._num(w)}_" in t) == 2
    smk = GEN.qwu_smoke_rows()
    assert len(smk) == 1
    assert not (tags & {r.split(",")[0] for r in smk})


def test_wu_tags_never_spell_an_open_gate_as_the_number_one():
    """Both gates are strict inequalities, so eps=1 still REJECTS a
    distance-1 pair. If 'open' could be written as 1, the boundary
    experiment would silently exclude its own subject matter."""
    for r in GEN.qwu_rows() + GEN.qwu_smoke_rows():
        t = r.split(",")[0]
        assert "_eaopen_" in t and "_esopen_" in t, t
        assert "_ea1_" not in t and "_es1_" not in t, t


def test_wu_sub_opens_both_gates_as_modes():
    for smoke in (False, True):
        env = next(ln for ln in GEN.qwu_sub(smoke=smoke).splitlines()
                   if ln.startswith("environment"))
        assert "AI_GATE_MODE=all_open" in env
        assert "PEER_GATE_MODE=all_open" in env
        assert "INNATE_LAMBDA=$(lam)" in env
        assert "KL_DIRECTION=forward" in env


def test_wu_rows_carry_a_positive_eps_even_though_it_is_inert():
    """eps_social=0 is how the NO-PEER condition is spelled everywhere
    else, so an open-peer run must not reuse that value for something
    different."""
    for r in GEN.qwu_rows():
        assert [c.strip() for c in r.split(",")][9] == "0.2"


def test_wu_production_is_100_rounds_and_the_smoke_is_3():
    for r in GEN.qwu_rows():
        assert [c.strip() for c in r.split(",")][21] == "100"
    for r in GEN.qwu_smoke_rows():
        assert [c.strip() for c in r.split(",")][21] == "3"


def test_new_keys_collide_with_nothing_else_in_the_generator():
    mine = ({r.split(",")[0] for r in GEN.qmech_rows()}
            | {r.split(",")[0] for r in GEN.qwu_rows()}
            | {r.split(",")[0] for r in GEN.qwu_smoke_rows()})
    for other in (GEN.qk1_rows(), GEN.qgs_rows(), GEN.fam_rows(),
                  GEN.fl1_rows(), GEN.flm_rows()):
        assert not (mine & {r.split(",")[0] for r in other})


def test_submit_script_registers_all_three_keys():
    src = open(os.path.join(CONDOR, "submit_pofd_sweep.sh")).read()
    assert "qwen_mechanism_frozen) TARGETS=" in src
    assert "qwen_wu_limit|qwen_wu_limit_smoke) TARGETS=" in src


# ======================================================================
# beta_eff: the theory mapping the whole diagnostic rests on
# ======================================================================

@pytest.mark.parametrize("k,w,want", [
    (0.2, 0.5, 0.9),     # the paper regime
    (1.0, 0.5, 0.5),     # k=1 ALONE is NOT a consensus limit
    (1.0, 0.9, 0.9),     # identity partner of the paper regime
    (0.0, 0.5, 1.0),
    (1.0, 1.0, 1.0),     # the actual boundary
])
def test_beta_eff_values(k, w, want):
    assert abs(PP.beta_eff(k, w) - want) < 1e-12


def test_k_equals_one_at_w_half_is_less_anchored_not_more():
    """The single most misreadable point in the mapping: at fixed W=.5,
    moving k from .2 to 1 DROPS beta_eff from .9 to .5. It is not a step
    toward consensus."""
    assert PP.beta_eff(1.0, 0.5) < PP.beta_eff(0.2, 0.5)


def test_at_w_one_k_is_algebraically_irrelevant():
    vals = {PP.beta_eff(k, 1.0) for k in (0.0, 0.2, 0.5, 1.0)}
    assert vals == {1.0}


# ======================================================================
# the perfect-prediction oracle
# ======================================================================

@pytest.fixture(scope="module")
def setup():
    from pathlib import Path
    return PP.extract_loader()(
        Path(REPO) / "experiments/data/movielens/ml-100k", "Action")


def test_served_equals_state_exactly(setup):
    op, _, pred = PP.simulate(setup, innate_k=0.2, w_plat=0.5,
                              eps_social=0.05, eps_ai=1.0, rounds=5, seed=0)
    assert torch.equal(pred[0], setup["innate"])
    for t in range(1, 5):
        assert torch.equal(pred[t], op[t - 1])


def test_innate_k_is_the_flag_and_lam_is_deprecated():
    src = open(os.path.join(PIPE, "sim_perfect_predictor.py")).read()
    assert '"--innate-k"' in src
    assert "DEPRECATED alias" in src


def test_artifact_name_encodes_every_trajectory_changing_dial():
    base = {"innate_k": 1.0, "w_plat": 0.5, "eps_social": 0.2,
            "eps_ai": 1.0, "ai_gate_mode": "threshold",
            "peer_gate_mode": "threshold", "ab_sweeps": 1, "seed": 0,
            "rounds": 30}
    n0 = PP.artifact_name(base)
    for field, val in (("innate_k", 0.2), ("w_plat", 1.0),
                       ("eps_social", 0.05), ("seed", 42), ("rounds", 300),
                       ("ai_gate_mode", "all_open"),
                       ("peer_gate_mode", "all_open")):
        cfg = dict(base, **{field: val})
        assert PP.artifact_name(cfg) != n0, field
    # open gates are spelled as modes in the filename, never as "1"
    openc = dict(base, ai_gate_mode="all_open", peer_gate_mode="all_open")
    assert "eaopen" in PP.artifact_name(openc)
    assert "esopen" in PP.artifact_name(openc)


def test_all_open_peer_with_zero_eps_is_refused(setup):
    """eps_social=0 is the NO-PEER condition; it must not double as an
    open peer channel."""
    import subprocess
    r = subprocess.run(
        ["python3", os.path.join(PIPE, "sim_perfect_predictor.py"),
         "--innate-k", "1", "--w-plat", "1", "--eps-social", "0",
         "--peer-gate-mode", "all_open", "--rounds", "3"],
        capture_output=True, text=True,
        env=dict(os.environ, USE_TF="0"))
    assert r.returncode != 0
    assert "contradictory" in (r.stderr + r.stdout)


# -- the two structural identity pairs ---------------------------------

def test_beta_eff_one_identity_is_byte_identical(setup):
    """(k=0, W=.5) and (k=1, W=1) both reduce z to EXACTLY x, so there is
    no rounding to disagree about and the trajectories are bit-equal at
    every peer gate."""
    for es in (0.0, 0.05, 0.2, 1.0):
        a, _, _ = PP.simulate(setup, innate_k=0.0, w_plat=0.5,
                              eps_social=es, eps_ai=1.0, rounds=30, seed=0)
        b, _, _ = PP.simulate(setup, innate_k=1.0, w_plat=1.0,
                              eps_social=es, eps_ai=1.0, rounds=30, seed=0)
        assert torch.equal(a, b), f"es={es}"


def test_beta_eff_point_nine_identity_holds_on_the_pre_peer_map(setup):
    """(k=.2, W=.5) and (k=1, W=.9) are the same map ALGEBRAICALLY, but
    they evaluate it through different products -- 0.5*(0.2 i + 0.8 x) +
    0.5 x versus 0.1 i + 0.9 x -- and those constants are not
    binary-exact. So the claim that is true is about the arithmetic: the
    pre-peer maps agree to within a few ulp at every state the
    trajectory visits. Byte-identity is FALSE here and the checker does
    not assert it."""
    innate = setup["innate"]
    w1 = setup["platform_sus"]
    a, _, _ = PP.simulate(setup, innate_k=0.2, w_plat=0.5, eps_social=0.05,
                          eps_ai=1.0, rounds=30, seed=0)
    worst = 0.0
    for t in range(a.shape[0]):
        x = innate if t == 0 else a[t - 1]
        z1, _ = gp.nested_presocial_update(x, x.clone(), innate, 0.2,
                                           (0.5 * w1).clamp(0, 1), 1.0)
        z2, _ = gp.nested_presocial_update(x, x.clone(), innate, 1.0,
                                           (0.9 * w1).clamp(0, 1), 1.0)
        worst = max(worst, float((z1 - z2).abs().max()))
    assert 0 < worst < 1e-6, worst


def test_bounded_confidence_amplifies_one_ulp_into_a_macroscopic_gap(setup):
    """The flip side, and a Part-E result in miniature: that same 1-ulp
    difference is amplified by the bounded-confidence gate. At es=.05 it
    reaches ~5e-2 within the FIRST round; at es=.2 it stays at ulp
    scale, because a wide gate has almost no pairs sitting on its
    boundary."""
    def gap(es):
        a, _, _ = PP.simulate(setup, innate_k=0.2, w_plat=0.5,
                              eps_social=es, eps_ai=1.0, rounds=30, seed=0)
        b, _, _ = PP.simulate(setup, innate_k=1.0, w_plat=0.9,
                              eps_social=es, eps_ai=1.0, rounds=30, seed=0)
        return float((a - b).abs().max())
    assert gap(0.0) < 1e-6          # no peers: stays at rounding scale
    assert gap(0.2) < 1e-5
    assert gap(0.05) > 1e-2         # narrow gate: amplified


def test_consensus_at_the_wu_boundary(setup):
    """k=1, W=1, both gates genuinely open: perfect prediction must
    preserve the mean and drive SD to zero. This is the limiting-case
    correspondence with Wu et al. -- a randomized-gossip analogue of
    their result, not a replication of their theorem."""
    op, _, _ = PP.simulate(setup, innate_k=1.0, w_plat=1.0, eps_social=0.2,
                           eps_ai=1.0, rounds=300, seed=0,
                           ai_gate_mode="all_open",
                           peer_gate_mode="all_open")
    fin = op[-1]
    assert abs(float(fin.mean()) - float(setup["innate"].mean())) < 1e-6
    assert float(fin.std()) < 1e-5
    assert float(fin.max() - fin.min()) < 1e-4
    assert AN.cluster_count(fin.numpy()) == 1


def test_perfect_prediction_at_finite_susceptibility_stays_heterogeneous(setup):
    """The other half of the mapping: perfect prediction is NOT enough.
    At W=.5 with a bounded peer gate the oracle keeps real dispersion,
    which is why k=1 alone is not a consensus limit."""
    op, _, _ = PP.simulate(setup, innate_k=1.0, w_plat=0.5, eps_social=0.05,
                           eps_ai=1.0, rounds=300, seed=0,
                           ai_gate_mode="all_open")
    assert float(op[-1].std()) > 1e-3


# ======================================================================
# checker sabotage
# ======================================================================

def _canonical_vec():
    if not os.path.exists(os.path.join(CANON_SRC, "trajectory.pt")):
        pytest.skip("canonical H100 frozen run not pulled locally")
    d = torch.load(os.path.join(CANON_SRC, "trajectory.pt"),
                   map_location="cpu", weights_only=False)
    return d["pred_raw"][0].clone()


def _build_qmech(setup, tmp, k=1.0, es=0.05, rounds=30,
                 peer_mode="threshold", w=0.5):
    """A genuine frozen qmech run: REAL operators, real canonical served
    vector, embedded config. Verified to pass the real checker before any
    sabotage is applied."""
    const = _canonical_vec()
    acc = []
    op, tw, pred = PP.simulate(
        setup, innate_k=k, w_plat=w, eps_social=es, eps_ai=1.0,
        rounds=rounds, seed=0, peer_gate_mode=peer_mode, accepted_out=acc,
        served_fn=lambda x, t: const, require_open_gate=False)
    innate = setup["innate"]
    n = int(setup["n"])
    gates, traj = [], []
    for t in range(rounds):
        x0 = innate if t == 0 else op[t - 1]
        g = gp.ai_gate(pred[t], x0, 1.0, "threshold")
        gates.append(g)
        row = {"round": t, "deployment": 0, "is_deploy": 1,
               "contact": float(g.float().mean()), "accepted": acc[t],
               "s_tag": 0.0, "twin_mean": float(tw[t].mean()),
               "twin_std": float(tw[t].std()), "twin_bias": 0.0,
               "op_twin_l1": 0.0, "op_twin_w1": 0.0}
        if peer_mode == "all_open":
            row["peer_gate_mode"] = "all_open"
            row["peer_pairs"] = n
        traj.append(row)
    def kn(v):
        return f"{v:g}".replace(".", "p")
    cfg = {
        "run_tag": f"pofdqmech_qwen7b_k0_ea1_w0p5_l{kn(k)}_es{kn(es)}_s0",
        "kl_beta": 0.0, "kl_direction": "forward", "kl_ref_adapter": None,
        "training_style": "frozen", "rlhf_feedback": False,
        "base_model": "Qwen/Qwen2.5-7B-Instruct", "n_rounds": rounds,
        "epoch_size": 100, "deploy_every": 1, "data_regime": "replace",
        "seed": 0, "n_labeled": 723, "max_steps": 0, "sft_epochs": 1,
        "sft_batch_size": 4, "lora_r": 512, "use_lora": False,
        "sft_lr": 5e-5, "hist_bins": 50, "seed_base_data": True,
        "train_cap": 723, "platform_sus_scale": 1.0, "anchor_mode": "fixed",
        "pop_model": "ab", "eps": es, "eps_ai": 1.0, "gamma_bias": 0.0,
        "ai_gate_mode": "threshold", "peer_gate_mode": peer_mode,
        "w_plat": w, "innate_lambda": k,
        "population_update": "nested_ai_then_social_v1", "run_mode": "loop",
        "canary_delta": 0.0, "grad_decomp": 1, "save_adapter_rounds": [],
        "icl_k": 0, "icl_days": 0, "icl_select": "random",
        "icl_ctx_source": "live", "icl_snapshot_round": -1,
        "icl_ctx_donor": None, "icl_ctx_donor_tag": None,
        "icl_ctx_donor_round": None, "icl_ctx_donor_hash": None,
        "feedback_mode": "none", "icrh": False, "reward_kind": "accuracy",
        "ab_retain": False, "n_probe": 64, "tel_eval_cap": 64,
        "grad_norm_n": 8, "fresh_each_round": False, "pristine_frac": 0.0,
        "replay_frac": 0.0, "pop_reset": False, "ab_sweeps": 1,
        "pop_order": "peer_first", "profile_shuffle_p": 0.0,
        "profile_sort_q": 0.0, "profile_drop_cols": [],
        "profile_permute_cols": [], "teacher_label_delta": 0.0,
        "teacher_label_col": None, "teacher_label_fav": None,
        "teacher_group_seed": 0, "log_gender_gaps": False,
        "dataset": "movielens", "ml_target": "Action", "log_ppl_dist": True,
        "ppl_dist_cap": 0, "do_sample": False, "gen_temperature": 1.0,
        "ans_sample_k": 0, "ans_sample_n": 64, "ans_sample_t": 1.0,
        "host": "gpu-node", "save_raw_gen": True,
        "hardware": {"hostname": "g001",
                     "gpu_name": "NVIDIA H100 80GB HBM3", "gpu_cc": "9.0",
                     "cuda_version": "12.4", "torch_version": "2.5.0",
                     "transformers_version": "4.46.0"},
    }
    payload = {
        "trajectory": traj, "config": cfg, "op_raw": op, "pred_raw": pred,
        "twin_raw": tw, "gate_raw": torch.stack(gates),
        "ppl_raw": torch.empty(0), "ans_raw": torch.empty(0),
        "ans_idx": torch.tensor([], dtype=torch.long),
        "replay_raw": torch.empty(0), "train_y_raw": torch.empty(0),
        "icl_idx_raw": torch.empty(0), "icl_val_raw": torch.empty(0),
        "icl_donor_vec": torch.empty(0), "innate": innate, "profiles": {},
        "probe_idx": torch.tensor([], dtype=torch.long),
        "canary": torch.zeros(n), "gender_true": None, "gender_disp": None,
        "teacher_pred": torch.empty(0),
    }
    return payload


def _write(payload, tmp, tag=None):
    tag = tag or payload["config"]["run_tag"]
    d = os.path.join(tmp, tag)
    os.makedirs(d, exist_ok=True)
    torch.save(payload, os.path.join(d, "trajectory.pt"))
    # config.json is written too, but the checker reads the EMBEDDED
    # config -- every sabotage below mutates payload["config"]
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(payload["config"], fh)
    return d


@pytest.fixture(scope="module")
def clean_qmech(setup):
    with tempfile.TemporaryDirectory() as tmp:
        yield _build_qmech(setup, tmp)


def test_clean_qmech_fixture_passes_the_real_checker(clean_qmech):
    """If this ever fails, every sabotage test below is meaningless."""
    with tempfile.TemporaryDirectory() as tmp:
        errs = CHK.check_run(_write(copy.deepcopy(clean_qmech), tmp))
    assert errs == [], errs


def _sabotage(clean, mutate, tag=None):
    p = copy.deepcopy(clean)
    mutate(p)
    with tempfile.TemporaryDirectory() as tmp:
        return CHK.check_run(_write(p, tmp, tag))


def _hit(errs, needle):
    return [e for e in errs if needle in e]


def test_sabotage_wrong_k(clean_qmech):
    errs = _sabotage(clean_qmech,
                     lambda p: p["config"].update(innate_lambda=0.2))
    assert _hit(errs, "innate_lambda"), errs


def test_sabotage_wrong_w(clean_qmech):
    errs = _sabotage(clean_qmech, lambda p: p["config"].update(w_plat=1.0))
    assert _hit(errs, "w_plat"), errs


def test_sabotage_wrong_ai_gate_mode(clean_qmech):
    errs = _sabotage(clean_qmech,
                     lambda p: p["config"].update(ai_gate_mode="all_open"))
    assert _hit(errs, "ai_gate_mode"), errs


def test_sabotage_wrong_peer_gate_mode(clean_qmech):
    """Claiming an open peer channel while the tag says a numeric
    threshold must fail -- that is the exact disguise the mode exists to
    prevent."""
    errs = _sabotage(clean_qmech,
                     lambda p: p["config"].update(peer_gate_mode="all_open"))
    assert _hit(errs, "PEER-GATE"), errs


def test_sabotage_unknown_peer_gate_mode(clean_qmech):
    errs = _sabotage(clean_qmech,
                     lambda p: p["config"].update(peer_gate_mode="open"))
    assert _hit(errs, "unknown peer_gate_mode"), errs


def test_sabotage_altered_frozen_prediction(clean_qmech):
    def bump(p):
        p["pred_raw"] = p["pred_raw"].clone()
        p["pred_raw"][:, 0] = 0.123
    errs = _sabotage(clean_qmech, bump)
    assert _hit(errs, "canonical"), errs


def test_sabotage_nonconstant_frozen_predictions(clean_qmech):
    def drift(p):
        p["pred_raw"] = p["pred_raw"].clone()
        p["pred_raw"][5, 3] += 0.25
    errs = _sabotage(clean_qmech, drift)
    assert _hit(errs, "NOT constant"), errs


def test_sabotage_wrong_hardware(clean_qmech):
    errs = _sabotage(
        clean_qmech,
        lambda p: p["config"]["hardware"].update(
            gpu_name="NVIDIA A100-SXM4-80GB"))
    assert _hit(errs, "A100") or _hit(errs, "H100"), errs


def test_sabotage_missing_horizon(clean_qmech):
    def truncate(p):
        p["op_raw"] = p["op_raw"][:20]
        p["pred_raw"] = p["pred_raw"][:20]
        p["twin_raw"] = p["twin_raw"][:20]
        p["gate_raw"] = p["gate_raw"][:20]
        p["trajectory"] = p["trajectory"][:20]
    errs = _sabotage(clean_qmech, truncate)
    assert _hit(errs, "ROUNDS") or _hit(errs, "n_rounds"), errs


def test_sabotage_changed_twin(clean_qmech):
    def tweak(p):
        p["twin_raw"] = p["twin_raw"].clone()
        p["twin_raw"][10] += 0.05
    errs = _sabotage(clean_qmech, tweak)
    assert _hit(errs, "twin"), errs


def test_sabotage_off_grid_tag(clean_qmech):
    errs = _sabotage(clean_qmech, lambda p: None,
                     tag="pofdqmech_qwen7b_k0_ea1_w0p5_l0p7_es0p05_s0")
    assert _hit(errs, "off-grid") or _hit(errs, "innate_lambda"), errs


# -- the all-open peer path ---------------------------------------------

@pytest.fixture(scope="module")
def clean_open(setup):
    """An open-peer run needs the _esopen_ tag AND the mode; the checker
    cross-checks them against each other."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _build_qmech(setup, tmp, k=1.0, es=0.2, rounds=6,
                         peer_mode="all_open")
        p["config"]["run_tag"] = \
            "pofdqmech_qwen7b_k0_ea1_w0p5_l1_esopen_s0"
        yield p


def test_all_open_peer_run_accepts_every_pair(clean_open):
    n = clean_open["op_raw"].shape[1]
    assert all(r["accepted"] == n for r in clean_open["trajectory"])


def test_sabotage_one_rejected_all_open_pair(clean_open):
    """A single rejected pair must be caught. It cannot hide inside an
    aggregate: under all_open, accepted == n exactly, every round."""
    def reject_one(p):
        p["trajectory"] = copy.deepcopy(p["trajectory"])
        p["trajectory"][3]["accepted"] -= 1
    errs = _sabotage(clean_open, reject_one,
                     tag=clean_open["config"]["run_tag"])
    assert _hit(errs, "rejected sampled pairs"), errs


def test_sabotage_numeric_threshold_masquerading_as_open(clean_open):
    errs = _sabotage(clean_open,
                     lambda p: p["config"].update(peer_gate_mode="threshold"),
                     tag=clean_open["config"]["run_tag"])
    assert _hit(errs, "masquerade") or _hit(errs, "_esopen_"), errs


def test_sabotage_open_mode_without_the_open_tag(clean_open):
    errs = _sabotage(clean_open, lambda p: None,
                     tag="pofdqmech_qwen7b_k0_ea1_w0p5_l1_es0p2_s0")
    assert _hit(errs, "no _esopen_ token") or _hit(errs, "PEER-GATE"), errs


# ======================================================================
# perfect-prediction checker sabotage
# ======================================================================

def _pp_artifact(setup, tmp, **kw):
    args = dict(innate_k=1.0, w_plat=0.5, eps_social=0.05, eps_ai=1.0,
                rounds=8, seed=0)
    args.update(kw)
    op, tw, pred = PP.simulate(setup, **args)
    cfg = {"platform": "perfect_prediction",
           "population_update": "nested_ai_then_social_v1",
           "innate_k": args["innate_k"], "w_plat": args["w_plat"],
           "beta_eff": PP.beta_eff(args["innate_k"], args["w_plat"]),
           "eps_social": args["eps_social"], "eps_ai": args["eps_ai"],
           "ai_gate_mode": kw.get("ai_gate_mode", "threshold"),
           "peer_gate_mode": kw.get("peer_gate_mode", "threshold"),
           "ab_sweeps": 1, "gamma_bias": 0.0, "rounds": args["rounds"],
           "seed": args["seed"], "peer_seed": args["seed"] + 424243,
           "peer_rng_device": "cpu", "dataset": "movielens",
           "ml_target": "Action", "n_agents": int(setup["n"]),
           "sim_version": "test"}
    return {"config": cfg, "op_raw": op, "twin_raw": tw, "pred_raw": pred,
            "innate": setup["innate"].clone()}


@pytest.fixture(scope="module")
def CPP():
    return _load("chk_pp", os.path.join(PIPE, "check_perfect_predictor.py"))


def _pp_check(CPP, setup, art):
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.pt")
        torch.save(art, p)
        return CPP.check_artifact(p, setup)


def test_clean_pp_artifact_passes(CPP, setup):
    with tempfile.TemporaryDirectory() as tmp:
        assert _pp_check(CPP, setup, _pp_artifact(setup, tmp)) == []


def test_pp_sabotage_imperfect_prediction(CPP, setup):
    with tempfile.TemporaryDirectory() as tmp:
        a = _pp_artifact(setup, tmp)
        a["pred_raw"] = a["pred_raw"].clone()
        a["pred_raw"][2, 5] += 0.1
        errs = _pp_check(CPP, setup, a)
    assert any("not perfect prediction" in e for e in errs), errs


def test_pp_sabotage_altered_twin(CPP, setup):
    with tempfile.TemporaryDirectory() as tmp:
        a = _pp_artifact(setup, tmp)
        a["twin_raw"] = a["twin_raw"].clone()
        a["twin_raw"][1] += 0.02
        errs = _pp_check(CPP, setup, a)
    assert any("twin_raw does NOT reproduce" in e for e in errs), errs


def test_pp_sabotage_wrong_k_and_wrong_w(CPP, setup):
    for field, val in (("innate_k", 0.2), ("w_plat", 1.0)):
        with tempfile.TemporaryDirectory() as tmp:
            a = _pp_artifact(setup, tmp)
            a["config"][field] = val
            errs = _pp_check(CPP, setup, a)
        assert any("does NOT reproduce" in e or "beta_eff" in e
                   for e in errs), (field, errs)


def test_pp_sabotage_wrong_horizon(CPP, setup):
    with tempfile.TemporaryDirectory() as tmp:
        a = _pp_artifact(setup, tmp)
        a["config"]["rounds"] = 12
        errs = _pp_check(CPP, setup, a)
    assert any("rounds" in e for e in errs), errs


def test_pp_sabotage_wrong_gate_mode(CPP, setup):
    with tempfile.TemporaryDirectory() as tmp:
        a = _pp_artifact(setup, tmp)
        a["config"]["peer_gate_mode"] = "all_open"
        errs = _pp_check(CPP, setup, a)
    assert any("does NOT reproduce" in e for e in errs), errs


def test_pp_sabotage_changed_peer_rng(CPP, setup):
    with tempfile.TemporaryDirectory() as tmp:
        a = _pp_artifact(setup, tmp)
        a["config"]["seed"] = 42
        errs = _pp_check(CPP, setup, a)
    assert any("does NOT reproduce" in e for e in errs), errs


def test_pp_sabotage_broken_identity_pair(CPP, setup):
    """Break beta_eff itself and the structural check must notice.

    The perturbation has to be ASYMMETRIC in (k, W). An earlier version
    of this test scaled beta_eff by a constant, which preserves BOTH
    identities exactly -- so it caught nothing while looking like a
    sabotage. Adding a term in W breaks each pair by a different amount.
    """
    # NOTE: patch CPP.PP, not the test module's PP. check_perfect_predictor
    # loads its own instance of sim_perfect_predictor, so patching this
    # module's copy would leave the checker using the real function and
    # the sabotage would "pass" without ever being applied -- the same
    # double-load trap that once made an ABSENT sentinel stop matching.
    real = CPP.PP.beta_eff
    try:
        CPP.PP.beta_eff = lambda k, w: 1.0 - (1.0 - w) * k + 0.05 * w
        assert CPP.PP.beta_eff(0.2, 0.5) != CPP.PP.beta_eff(1.0, 0.9)
        errs = CPP.structural_checks(setup, verbose=False)
    finally:
        CPP.PP.beta_eff = real
    assert any("IDENTITY" in e and "beta_eff" in e for e in errs), errs


def test_pp_sabotage_broken_prepeer_operator(CPP, setup):
    """The other half: leave beta_eff correct but break the OPERATOR, so
    the two parameterizations stop computing the same pre-peer map. This
    is the substantive arithmetic claim, not the bookkeeping."""
    real = CPP.gp.nested_presocial_update

    def bent(x0, served, innate, k, w_agent, eps_ai, gate_mode="threshold"):
        z, g = real(x0, served, innate, k, w_agent, eps_ai, gate_mode)
        return z + 0.001 * k, g          # k-dependent, so the pair splits

    try:
        CPP.gp.nested_presocial_update = bent
        errs = CPP.structural_checks(setup, verbose=False)
    finally:
        CPP.gp.nested_presocial_update = real
    assert any("pre-peer maps differ" in e for e in errs), errs


def test_pp_sabotage_consensus_assertion(CPP, setup):
    """Raise the consensus bar to something the oracle cannot meet: the
    assertion must fail rather than quietly pass."""
    real = CPP.CONSENSUS_SD
    try:
        CPP.CONSENSUS_SD = 1e-12
        errs = CPP.structural_checks(setup, verbose=False)
    finally:
        CPP.CONSENSUS_SD = real
    assert any("CONSENSUS SD" in e for e in errs), errs


# ======================================================================
# analyzer
# ======================================================================

def test_label_timing_uses_the_preceding_population_never_the_same_round():
    """Round 0 compares against the initial training labels; later
    rounds against op_raw[t-1]. Comparing to op_raw[t] would be scoring
    predictions against opinions they themselves caused."""
    src = open(os.path.join(
        PIPE, "analyze_qwen_mechanism_diagnostic.py")).read()
    assert "labels = innate if t == 0 else op[t - 1]" in src
    assert src.count("labels = innate if t == 0 else op[t - 1]") >= 2


def test_perfect_prediction_has_zero_prediction_error_by_construction(setup):
    """A sanity anchor for the label timing: under m(t)=x(t) the served
    vector IS the label vector, so MAE must be exactly 0. If the
    analyzer compared to the SAME round's post-update opinions this
    would be non-zero."""
    op, _, pred = PP.simulate(setup, innate_k=0.2, w_plat=0.5,
                              eps_social=0.05, eps_ai=1.0, rounds=6, seed=0)
    ini = setup["innate"].numpy()
    o, p = op.numpy(), pred.numpy()
    for t in range(6):
        labels = ini if t == 0 else o[t - 1]
        m = AN.round_metrics(o[t], ini, o[t - 1] if t else None,
                             p[t], labels, o[t])
        assert m["pred_mae"] == 0.0, t
        assert m["pred_rmse"] == 0.0, t


def test_w1_helpers():
    a = torch.tensor([0.0, 1.0, 2.0]).numpy()
    b = a + 1.0
    assert abs(AN.w1(a, b) - 1.0) < 1e-9
    # centering removes a pure location shift entirely
    assert AN.w1_centered(a, b) < 1e-9


def test_centered_w1_separates_shape_from_location():
    import numpy as np
    a = np.array([0.4, 0.5, 0.6])
    shift = a + 0.2               # pure translation
    spread = np.array([0.3, 0.5, 0.7])   # same mean, wider
    assert AN.w1(a, shift) > 0.1 and AN.w1_centered(a, shift) < 1e-9
    assert abs(AN.w1(a, spread) - AN.w1_centered(a, spread)) < 1e-9
    assert AN.w1_centered(a, spread) > 0.05


def test_cluster_count_uses_a_declared_tolerance():
    import numpy as np
    assert AN.CLUSTER_TOL == 1e-4
    assert AN.cluster_count(np.array([0.1, 0.1, 0.1])) == 1
    assert AN.cluster_count(np.array([0.1, 0.5, 0.9])) == 3
    assert AN.cluster_count(np.array([0.1, 0.1 + 1e-6])) == 1


def test_shape_vector_is_seven_centered_quantiles():
    import numpy as np
    v = AN.shape_vector(np.linspace(0.0, 1.0, 101))
    assert len(v) == 7
    assert abs(v[3]) < 1e-9        # centered median of a symmetric sample


def test_contrast_readings_never_call_sft_gap_optimizer_error():
    """The causal language is fixed in the analyzer, not left to the
    write-up: ordinary SFT still carries pretrained initialization and a
    finite-rank LoRA, so its gap from perfect prediction is an
    AGGREGATE retraining effect."""
    src = open(os.path.join(
        PIPE, "analyze_qwen_mechanism_diagnostic.py")).read()
    assert "NOT optimizer error" in src
    assert "AGGREGATE parametric-retraining gap" in src
    assert "optimizer noise" not in src


def test_analyzer_hard_fails_on_missing_cells(tmp_path):
    """Hard-gated means hard-gated: a missing cell must stop the run and
    name what is missing, not silently analyze a subset."""
    with pytest.raises(SystemExit) as ei:
        AN.resolve_gpu_cells([str(tmp_path)])
    assert "HARD FAIL" in str(ei.value)
    with pytest.raises(SystemExit) as ei:
        AN.resolve_wu_cells([str(tmp_path)])
    assert "HARD FAIL" in str(ei.value)


def test_k_comparison_warning_is_carried_into_the_analysis():
    """Both warnings must reach the CSVs, not just the docstring: the
    per-cell rows carry the k warning and the oracle rows carry the W
    warning, so a reader of the data alone still sees them."""
    src = open(os.path.join(
        PIPE, "analyze_qwen_mechanism_diagnostic.py")).read()
    assert "NOT a pure memory ablation" in src
    assert "not a\nregularization comparison" in src
    assert "not a regularization dial" in src
    assert "k_comparison_warning" in src


def test_wu_reference_and_correspondence_language_is_recorded():
    """The mapping and its limits must live in the code, not only in a
    message: our operator is a randomized Deffuant process, so the
    relationship to Wu et al. is a limiting-case correspondence."""
    for f in ("analyze_qwen_mechanism_diagnostic.py",
              "sim_perfect_predictor.py"):
        src = open(os.path.join(PIPE, f)).read().lower()
        assert "2603.12137" in src
        assert "correspondence" in src
        # the limit must be stated, not merely the analogy
        assert ("not a replication" in src
                or "not a proof" in src), f

# ======================================================================
# QWU checker: arm surface and the smoke pin (2026-08-20)
# ======================================================================

def _build_qwu(setup, arm="b1", w=1.0, rounds=3, smoke=True):
    """A genuine open-gate QWU run: REAL operators, both gates all_open,
    embedded config. The served vector is arbitrary here -- the QWU
    branch gates the TRAINING surface and the gate modes, not the
    model's outputs."""
    n = int(setup["n"])
    innate = setup["innate"]
    served = (innate * 0.5 + 0.25).clone()
    acc = []
    op, tw, pred = PP.simulate(
        setup, innate_k=1.0, w_plat=w, eps_social=0.2, eps_ai=1.0,
        rounds=rounds, seed=0, ai_gate_mode="all_open",
        peer_gate_mode="all_open", accepted_out=acc,
        served_fn=lambda x, t: served, require_open_gate=False)
    gates, traj = [], []
    for t in range(rounds):
        x0 = innate if t == 0 else op[t - 1]
        g = gp.ai_gate(pred[t], x0, 1.0, "all_open")
        gates.append(g)
        traj.append({"round": t, "deployment": 0, "is_deploy": 1,
                     "n_train": 723,
                     "contact": float(g.float().mean()),
                     "accepted": acc[t], "s_tag": 0.0,
                     "peer_gate_mode": "all_open", "peer_pairs": n,
                     "twin_mean": float(tw[t].mean()),
                     "twin_std": float(tw[t].std()), "twin_bias": 0.0,
                     "op_twin_l1": 0.0, "op_twin_w1": 0.0})
    style, beta = (("sft", 0.0) if arm == "b0" else ("sft_kl", 1.0))
    tok = "smoke2" if smoke else ""
    wn = f"{w:g}".replace(".", "p")
    cfg = {
        "run_tag": f"pofdqwu_qwen7b_{arm}_eaopen_w{wn}_l1_esopen_s0"
                   f"_r{rounds}{tok}",
        "kl_beta": beta, "kl_direction": "forward", "kl_ref_adapter": None,
        "training_style": style, "rlhf_feedback": False,
        "base_model": "Qwen/Qwen2.5-7B-Instruct", "n_rounds": rounds,
        "epoch_size": 100, "deploy_every": 1, "data_regime": "replace",
        "seed": 0, "n_labeled": 723, "max_steps": 0, "sft_epochs": 1,
        "sft_batch_size": 4, "lora_r": 512, "use_lora": True,
        "sft_lr": 5e-5, "hist_bins": 50, "seed_base_data": True,
        "train_cap": 723, "platform_sus_scale": 1.0, "anchor_mode": "fixed",
        "pop_model": "ab", "eps": 0.2, "eps_ai": 1.0, "gamma_bias": 0.0,
        "ai_gate_mode": "all_open", "peer_gate_mode": "all_open",
        "w_plat": w, "innate_lambda": 1.0,
        "population_update": "nested_ai_then_social_v1", "run_mode": "loop",
        "canary_delta": 0.0, "grad_decomp": 1, "save_adapter_rounds": [],
        "icl_k": 0, "icl_days": 0, "icl_select": "random",
        "icl_ctx_source": "live", "icl_snapshot_round": -1,
        "icl_ctx_donor": None, "icl_ctx_donor_tag": None,
        "icl_ctx_donor_round": None, "icl_ctx_donor_hash": None,
        "feedback_mode": "none", "icrh": False, "reward_kind": "accuracy",
        "ab_retain": False, "n_probe": 64, "tel_eval_cap": 64,
        "grad_norm_n": 8, "fresh_each_round": True, "pristine_frac": 0.0,
        "replay_frac": 0.0, "pop_reset": False, "ab_sweeps": 1,
        "pop_order": "peer_first", "profile_shuffle_p": 0.0,
        "profile_sort_q": 0.0, "profile_drop_cols": [],
        "profile_permute_cols": [], "teacher_label_delta": 0.0,
        "teacher_label_col": None, "teacher_label_fav": None,
        "teacher_group_seed": 0, "log_gender_gaps": False,
        "dataset": "movielens", "ml_target": "Action", "log_ppl_dist": True,
        "ppl_dist_cap": 0, "do_sample": False, "gen_temperature": 1.0,
        "ans_sample_k": 16, "ans_sample_n": 64, "ans_sample_t": 1.0,
        "host": "gpu-node", "save_raw_gen": True,
        "hardware": {"hostname": "g001",
                     "gpu_name": "NVIDIA H100 80GB HBM3", "gpu_cc": "9.0",
                     "cuda_version": "12.4", "torch_version": "2.5.0",
                     "transformers_version": "4.46.0"},
    }
    return {
        "trajectory": traj, "config": cfg, "op_raw": op, "pred_raw": pred,
        "twin_raw": tw, "gate_raw": torch.stack(gates),
        "ppl_raw": torch.empty(0), "ans_raw": torch.empty(0),
        "ans_idx": torch.tensor([], dtype=torch.long),
        "replay_raw": torch.empty(0), "train_y_raw": torch.empty(0),
        "icl_idx_raw": torch.empty(0), "icl_val_raw": torch.empty(0),
        "icl_donor_vec": torch.empty(0), "innate": innate, "profiles": {},
        "probe_idx": torch.tensor([], dtype=torch.long),
        "canary": torch.zeros(n), "gender_true": None, "gender_disp": None,
        "teacher_pred": torch.empty(0),
    }


@pytest.fixture(scope="module")
def clean_qwu(setup):
    return _build_qwu(setup)


def test_clean_qwu_smoke_fixture_passes(clean_qwu):
    with tempfile.TemporaryDirectory() as tmp:
        errs = CHK.check_run(_write(copy.deepcopy(clean_qwu), tmp))
    assert errs == [], errs


def test_clean_qwu_production_arms_pass(setup):
    for arm in ("b0", "b1"):
        for w in (0.5, 1.0):
            p = _build_qwu(setup, arm=arm, w=w, rounds=100, smoke=False)
            with tempfile.TemporaryDirectory() as tmp:
                errs = CHK.check_run(_write(p, tmp))
            assert errs == [], (arm, w, errs)


def test_qwu_b0_must_have_no_kl_term(clean_qwu, setup):
    """b0 is ORDINARY fresh SFT. If it carried a KL term the
    'regularized minus ordinary' contrast would not measure forward-KL
    retention at all."""
    p = _build_qwu(setup, arm="b0", w=1.0, rounds=100, smoke=False)
    p["config"].update(training_style="sft_kl", kl_beta=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        errs = CHK.check_run(_write(p, tmp))
    assert _hit(errs, "training_style") or _hit(errs, "kl_beta"), errs


def test_qwu_b1_must_carry_kl_weight_one_forward(clean_qwu):
    for field, bad in (("kl_beta", 0.5), ("kl_direction", "reverse"),
                       ("training_style", "sft")):
        errs = _sabotage(clean_qwu, lambda p, f=field, b=bad:
                         p["config"].update({f: b}),
                         tag=clean_qwu["config"]["run_tag"])
        assert _hit(errs, field), (field, errs)


def test_qwu_b1_reference_must_be_the_fixed_pristine_base(clean_qwu):
    """kl_ref_adapter would regularize toward a teacher checkpoint
    instead of the entering model -- a different experiment wearing the
    same tag."""
    errs = _sabotage(
        clean_qwu,
        lambda p: p["config"].update(kl_ref_adapter="runs/teacher/adapter_r0"),
        tag=clean_qwu["config"]["run_tag"])
    assert _hit(errs, "PRISTINE"), errs


def test_qwu_b0_rejects_a_reference_adapter(setup):
    p = _build_qwu(setup, arm="b0", w=1.0, rounds=100, smoke=False)
    p["config"]["kl_ref_adapter"] = "runs/teacher/adapter_r0"
    with tempfile.TemporaryDirectory() as tmp:
        errs = CHK.check_run(_write(p, tmp))
    assert _hit(errs, "no reference model"), errs


def test_qwu_fresh_each_round_is_required(clean_qwu):
    errs = _sabotage(clean_qwu,
                     lambda p: p["config"].update(fresh_each_round=False),
                     tag=clean_qwu["config"]["run_tag"])
    assert _hit(errs, "fresh_each_round"), errs


# -- the smoke is ONE specific cell, not "any short run" -----------------

def test_smoke_must_be_b1(setup):
    p = _build_qwu(setup, arm="b0", w=1.0, rounds=3, smoke=True)
    with tempfile.TemporaryDirectory() as tmp:
        errs = CHK.check_run(_write(p, tmp))
    assert _hit(errs, "smoke must be the b1 arm"), errs


def test_smoke_must_be_w_one(setup):
    p = _build_qwu(setup, arm="b1", w=0.5, rounds=3, smoke=True)
    with tempfile.TemporaryDirectory() as tmp:
        errs = CHK.check_run(_write(p, tmp))
    assert _hit(errs, "smoke must be W=1"), errs


def test_smoke_must_be_three_rounds(setup):
    p = _build_qwu(setup, arm="b1", w=1.0, rounds=5, smoke=True)
    with tempfile.TemporaryDirectory() as tmp:
        errs = CHK.check_run(_write(p, tmp))
    assert _hit(errs, "smoke must be 3 rounds"), errs


def test_smoke_must_be_k_one(clean_qwu):
    errs = _sabotage(clean_qwu,
                     lambda p: p["config"].update(innate_lambda=0.2),
                     tag=clean_qwu["config"]["run_tag"])
    assert _hit(errs, "smoke must be k=1") or _hit(errs, "innate_lambda"), errs


def test_smoke_must_have_both_gates_open(clean_qwu):
    for gk in ("ai_gate_mode", "peer_gate_mode"):
        errs = _sabotage(clean_qwu,
                         lambda p, g=gk: p["config"].update({g: "threshold"}),
                         tag=clean_qwu["config"]["run_tag"])
        assert errs, gk
        assert _hit(errs, gk) or _hit(errs, "PEER-GATE"), (gk, errs)


def test_new_smoke_tag_cannot_be_satisfied_by_the_stale_run():
    """The pre-fix smoke was submitted before serving was forced into
    eval mode, so its predictions decoded with LoRA dropout active. The
    idempotent executable no-ops COMPLETED runs, so the ONLY way to
    guarantee a rerun is a tag the old directory cannot answer to."""
    assert GEN.QWU_SMOKE_TOKEN == "smoke2"
    tag = GEN.qwu_smoke_rows()[0].split(",")[0]
    stale = "pofdqwu_qwen7b_b1_eaopen_w1_l1_esopen_s0_r3smoke"
    assert tag != stale
    assert tag == stale + "2"


def test_checker_still_accepts_the_new_smoke_token():
    """A tag the generator emits must be a tag the checker can parse --
    otherwise the corrected smoke could never gate clean."""
    import re as _re
    tag = GEN.qwu_smoke_rows()[0].split(",")[0]
    assert _re.match(r"^pofdqwu_qwen7b_(b0|b1)_eaopen_w(0p5|1)_l1_"
                     r"esopen_s0_r(\d+)(smoke\d*)?$", tag), tag


def test_open_peer_pass_line_does_not_say_no_peer_updates():
    """The PASS flavour text is computed separately in main() from the
    NUMERIC _es token, which _esopen_ does not match -- so an open-peer
    run printed "no peer updates" even though every sampled pair was
    accepted. check_run was always right (an open-peer run takes the
    social branch, and the no-peer branch would have failed on
    accepted != 0); only the label lied. Caught on the real 2026-08-20
    smoke."""
    src = open(os.path.join(PIPE, "check_pofd_sanity.py")).read()
    i = src.index("n_fail = 0")
    tail = src[i:]
    assert 'elif "_esopen_" in name:' in tail
    # the open branch must come BEFORE the numeric-token branch, or it
    # can never be reached
    assert tail.index('elif "_esopen_" in name:') < \
        tail.index('m_es_p := re.search')
    assert "peer step OPEN" in tail


def test_non_finite_training_telemetry_is_caught(setup, tmp_path):
    """A NaN CE or KL loss means the round's adapter is garbage, and
    nothing else downstream would say so -- the population keeps moving
    and every structural check still passes. Absent telemetry.json is
    fine (older runs, frozen arms); a non-finite VALUE is not."""
    p = _build_qwu(setup)
    d = _write(p, str(tmp_path))
    # clean: telemetry present and finite
    with open(os.path.join(d, "telemetry.json"), "w") as fh:
        for t in range(3):
            fh.write(json.dumps({"round": t, "grad_norm0": 1.5,
                                 "grad_kl_norm0": 0.5}) + "\n")
    assert CHK.check_run(d) == []
    # sabotage: one NaN KL gradient
    with open(os.path.join(d, "telemetry.json"), "w") as fh:
        for t in range(3):
            fh.write(json.dumps({"round": t, "grad_norm0": 1.5,
                                 "grad_kl_norm0": (float("nan") if t == 1
                                                   else 0.5)}) + "\n")
    errs = CHK.check_run(d)
    assert _hit(errs, "non-finite training telemetry"), errs
    # absent telemetry is NOT an error
    os.remove(os.path.join(d, "telemetry.json"))
    assert CHK.check_run(d) == []
