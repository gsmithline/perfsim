#!/usr/bin/env python3
"""Gate for the SECTION-4 CORRECTED-GATE waves (pofds4g_).

ONE checker, TWO waves, selected by --wave:

  section4_gate_anch2       (alias v1)   the original 72-cell wave
                                         (2026-08-24): ea {0.2, 1} x
                                         es {0, 0.2, 1} x 2 arms x 2
                                         conditions x 3 seeds. DEFAULT.
  section4_gate_anch2_fig6  (alias fig6) the Figure-6 grid (2026-08-25):
                                         ea, es in {0, .1, .3, 1} -> 192
                                         cells = 144 GPU + 2 ea=0 WITNESS
                                         + 46 TWIN-DERIVED, plus optional
                                         _r60/_r100 horizon EXTENSIONS.

THE GRID IS READ FROM THE GENERATOR (experiments/condor/gen_pofd_sweep.py,
loaded via importlib exactly as check_fig3_full_loop.py does it), never
restated here: S4G_* / S4G2_* constants, s4g2_cells(), S4G2_EA0_WITNESS,
s4g_smoke_rows / s4g2_smoke_rows, s4g2_ext_requests() and s4g_tag() are
the single source of truth, and this file's parser is self-tested
against s4g_tag() on every expected cell before anything is opened.

Both waves re-run the completed Mistral bottom-20% FIXED-vs-EVOLVING
Section-4 experiment under the CORRECTED AI gate:

    |m - x'| < eps_AI   with x' = k*innate + (1-k)*x   (the ANCHORED
                                                        opinion)

instead of the raw start-of-round opinion x0. In the runner that is
AI_GATE_REFERENCE=anchor, which makes

    config["population_update"] == "nested_ai_anchored_then_social_v2"
    config["ai_gate_reference"] == "anchor"

(see _POP_UPDATE_MARKER in run_pokec_gated_lm.py). EVERY archived
Section-4 cell carries the OLD "nested_ai_then_social_v1" marker, so a
run recording v1 is a HARD FAILURE here, not a warning: it is an
archived-semantics run wearing a corrected-gate tag, and nothing about
this wave's claim survives it. That is the single most important gate in
this file.

WHAT IS CHECKED (each item HARD-FAILS; nothing in this file warns):
  1. GRAMMAR + COVERAGE -- every present run dir parses under the pinned
     tag grammar, and the parsed (arm, cond, ea, es, seed[, horizon]) set
     equals the selected wave's expected product. Missing cells are
     listed explicitly and are a hard failure; a short grid never looks
     like a complete result.
  2. OPERATOR -- population_update == nested_ai_anchored_then_social_v2
     AND ai_gate_reference == "anchor". v1 fails by name.
  3. GRID FIELDS, TAG vs CONFIG IN BOTH DIRECTIONS -- eps_ai, eps
     (social), w_plat, innate_lambda, homophily gamma, n_rounds, base
     model, dataset/target, seed, train_cap, n_labeled, kl_direction,
     the numeric threshold gates, and the arm envelope (style, kl_beta,
     icl_k, icl_days, use_lora, fresh_each_round). Both directions means
     the config value is re-rendered into the tag grammar and compared to
     the token, and the arm/condition are INFERRED BACK from the config
     and compared to the tag -- a table lookup in one direction only
     cannot catch a tag that lies about a config it also matches.
  4. CONDITION INTEGRITY --
       fixed:    innate_clamp_mode=="bottom", frac==0.2, seed==run seed,
                 peer mode=="stubborn", count==145, and the cohort is
                 RECONSTRUCTED from innate (the deterministic
                 innate-then-id ranking) rather than trusted; the clamped
                 rows equal innate BIT-EXACTLY in op_raw AND twin_raw at
                 every recorded round.
       evolving: the config carries NO innate_clamp_* key at all and the
                 trajectory carries no clamp artifact.
  5. COHORT PAIRING -- for each (arm, ea, es, seed[, horizon]) the fixed
     and evolving members of the pair must share the innate vector
     BIT-EXACTLY, hence the same reconstructed bottom-145 cohort. The
     whole wave must in fact sit on ONE innate vector: with
     PROFILE_SHUFFLE_P=0 and no routing treatment, load_movielens_setup
     is a pure function of (dataset, target) -- it takes no seed at all,
     innate is (rating-1)/4 on the LCC of a cosine 10-NN graph -- and
     gp.innate_clamp_mask("bottom") is a pure sort with no RNG draw. So
     innate, the graph and the cohort are SEED-INVARIANT.
  5b. SEED-DISTINCTNESS -- A WARNING, NEVER THE EXIT CODE. Because
     nothing about the world depends on the seed, config["seed"] is the
     ONLY per-run evidence that a seed reached the runner, and that field
     is written from the environment rather than observed from
     behaviour. So within each (arm, cond, ea, es[, horizon]) cell two
     seeds producing a BIT-IDENTICAL op_raw is flagged as a WARN line
     (and listed under "warnings" in the JSON verdict) for the analyst:
     it CAN mean the seed never reached the training/serving stream,
     which would collapse the three-seed intervals to one observation --
     but greedy serving on a quantized value grid can also legitimately
     give bit-identical outcomes across seeds, so identical trajectories
     are not proof of a lost seed and do not fail the wave. Compared by
     sha256 over op_raw, so no two runs' tensors are ever resident at
     once. A missing third seed is a COVERAGE failure (fatal), not a pass
     here.
       EXEMPTION, STATED AND PRINTED: d8 at eps_social = 0 is the
     STRUCTURAL NULL (analyze_section4_gate.build_null_rows): frozen
     weights, greedy decoding, own-history prompts and an inert
     strict-< peer step mean NO random draw reaches the population, so
     three seeds of a d8/es=0 cell are EXPECTED to be bit-identical.
     Those groups are skipped with a NOTE naming the reason.
       TWIN SEED-DISTINCTNESS (the twin-derived cells' counterpart, the
     same WARN semantics): the twin is advanced by its own seeded
     generator, so within each (cond, es > 0[, horizon]) two seeds
     sharing twin_raw is flagged. es = 0 is excluded for the same
     strict-< reason (no accepted pair, the twin is RNG-free and
     seed-invariant there).
  6. TWIN present, correctly shaped, finite, in [0,1] and non-degenerate;
     and TWIN AGREEMENT: the twin is a pure function of (cond, es,
     seed) -- the served vector never enters ab_x_cf -- so every run at
     one (cond, es, seed) must carry a BIT-IDENTICAL twin_raw over the
     base horizon, whatever its arm or eps_AI (and extension runs must
     agree with the base cells over the first S4G_ROUNDS rows).
  7. ZERO PARSE FAILURES -- raw_gen_log.json.gz is REQUIRED for EVERY
     run. SAVE_RAW_GEN=1 is pinned in every Section-4 corrected-gate sub
     template (smoke, production, extension, both waves), and
     parse_fail_frac is recorded NOWHERE else. Nothing in trajectory.pt
     can stand in for it: the runner's parser falls back to a FINITE 0.5
     on failure (run_pokec_gated_lm.py: "the parser fell back to its 0.5
     default"), so a wave with widespread parse failures looks perfectly
     finite in pred_raw. Hard failures:
       (a) the file absent;
       (b) any round with parse_fail_frac > 0 (or without the field);
       (c) any round 0..n_rounds-1 missing from the log (the log must
           carry exactly the rounds 0..n_rounds-1, in order, once);
       (d) a round parsing fewer than 723 agents.
     THE ONLY ESCAPE is --inspect-archived, which downgrades (a) to a
     loud WARN so the four archived pre-fix smoke runs under
     notes/pofd/cluster/pofds4gsmk_* can still be looked at. It is NOT A
     GATE: the banner, every affected cell, the verdict line and the JSON
     say so, and nothing produced under it may be cited as a pass.
  8. len(trajectory) >= n_rounds, with op_raw/pred_raw/twin_raw shaped
     [n_rounds, 723] (n_rounds = the horizon for an extension run).
  9. --smoke gates the 3-round pofds4gsmk_ cells of the selected wave
     (4 of them: both arms x both conditions at the wave's smoke gate --
     v1: ea=1 es=0.2; fig6: ea=0.1 es=0.3 -- seed 0) and HONOURS THE RUN
     ROOT IT IS PASSED. (check_section3.py --smoke has a bug where the
     run dir it is given is ignored; that bug is deliberately not
     copied.)
 10. d8 PERSONAL-HISTORY LOCALITY (both waves, both conditions) -- the
     rendered contexts in icl_days_log.json.gz are REPLAYED BYTE-EXACTLY
     from (innate, op_raw): agent i's round-t sentence must be exactly
     the last icl_days entries of [innate_i, op_raw[0,i], ...,
     op_raw[t-1,i]] rendered as "%.2f", oldest to newest. Byte-equality
     simultaneously proves that every value is one of that SAME agent's
     own previous post-peer opinions, that no more than icl_days (8)
     values are rendered, and that nothing from another agent entered
     the context. The first mismatch is classified (foreign value /
     another agent's sentence / too many values / wrong window) in the
     failure line. MIRRORS check_pofd_sanity.check_run's d8 replay (the
     `hist_cl` loop of its "-- 1j CLAMP" block and the `hist_e` loop of
     its `if is_evo:` block), which is the runner's own rendering.
 11. ea0-witness (any run at eps_AI = 0; in fig6 the two
     S4G2_EA0_WITNESS cells) -- op_raw == twin_raw BIT-EXACTLY over every
     round AND every telemetry.json row's `contact` (the AI-gate open
     fraction) == 0 exactly. gp.ai_gate is strict-<, so |m - x'| < 0
     never opens: eps_AI = 0 IS the twin, and this is the empirical
     proof that lets the other ea=0 cells be twin-derived.
 12. TWIN-DERIVED cells (fig6: ea = 0, non-witness; no run dir) -- drawn
     from twin_raw of the runs at the same (cond, es, seed). Hard-fail
     when no run exists there at all, or when those runs disagree on
     twin_raw (check 6). Reported as their own rows.
 13. EXTENSIONS (fig6; s4g2_ext_requests() from the committed manifest)
     -- a present _r60/_r100 run gets the full cell checks at its
     horizon; an absent one is PENDING-EXT (non-failing); a present
     extension whose fixed/evolving partner is absent is a FAILURE.

The clamp logic is not re-invented here: the cohort reconstruction, the
"bit-exact in population AND twin" assertion, the stubborn-peer
treatment of the responsive twin, and the d8 personal-history replay all
MIRROR check_pofd_sanity.check_run's "-- 1j CLAMP" block (and its "1j
EVO" counterpart at `if is_evo:`), which is itself the algorithm of
_gated_pop.innate_clamp_mask(mode="bottom"). Each mirrored piece names
its source in a comment.

--------------------------------------------------------------------
Usage
--------------------------------------------------------------------
  # the original 72-cell wave (default --wave), on the cluster login
  # node (threads are pinned inside this module, before torch is
  # imported)
  python check_section4_gate.py \\
      --run-root /home/gsmithline/perfsim/runs/pokec_gated_lm

  # the Figure-6 grid: 4-job 3-round smoke, then the full gate
  python check_section4_gate.py --wave fig6 --smoke \\
      --run-root /home/gsmithline/perfsim/runs/pokec_gated_lm
  python check_section4_gate.py --wave fig6 \\
      --run-root /home/gsmithline/perfsim/runs/pokec_gated_lm

  # gate exactly the tags in a file (a deliberately partial pull)
  python check_section4_gate.py --run-root RUNS --tags-file tags.txt

  # machine-readable verdict
  python check_section4_gate.py --run-root RUNS --json /tmp/s4g.json

  # LOOK AT (not gate) the archived pre-fix smokes, which predate
  # SAVE_RAW_GEN=1 and carry no raw_gen_log.json.gz
  python check_section4_gate.py --smoke --inspect-archived \\
      --run-root notes/pofd/cluster

Exit codes: 0 = every check passed, 1 = hard failure, 2 = usage error.
WARN lines (seed-distinctness, and the missing raw log under
--inspect-archived) never change the exit code.
"""
from __future__ import annotations

import os

# PERFORMANCE / SHARED-NODE HYGIENE: this gate may run on the cluster
# LOGIN NODE, so BLAS fan-out is pinned BEFORE torch is imported (after
# the import the env vars no longer take effect). USE_TF=0 keeps
# transformers' TensorFlow probe out of the way if anything on the path
# pulls it in.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("USE_TF", "0")

import argparse          # noqa: E402
import gzip              # noqa: E402
import hashlib           # noqa: E402
import importlib.util    # noqa: E402
import json              # noqa: E402
import re                # noqa: E402
import sys               # noqa: E402
from pathlib import Path  # noqa: E402

import torch             # noqa: E402

torch.set_num_threads(1)

# remote capture pipes checkers over ssh stdin, where __file__ is unset
HERE = (os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals() else os.getcwd())

# the ONE cohort definition (gp.innate_clamp_mask / gp.innate_clamp_hash)
# is shared with the runner via _gated_pop.py, so the deployed clamp and
# this reconstruction can never drift. Loaded exactly as the runner and
# check_pofd_sanity load it.
_GP_PATH = os.path.join(HERE, "_gated_pop.py")
_spec_gp = importlib.util.spec_from_file_location("_gated_pop_s4g", _GP_PATH)
gp = importlib.util.module_from_spec(_spec_gp)
_spec_gp.loader.exec_module(gp)

LOG = "[check_s4g]"

# ---------------------------------------------------------------- design
DEFAULT_RUN_ROOT = "/home/gsmithline/perfsim/runs/pokec_gated_lm"

# the generator is the single definition of both grids; searched for
# relative to this file first (the checkout layout), then the cwd and
# its ancestors (the ssh-stdin capture, where __file__ is unset)
GEN_REL = os.path.join("experiments", "condor", "gen_pofd_sweep.py")

WAVE_V1 = "section4_gate_anch2"
WAVE_FIG6 = "section4_gate_anch2_fig6"
WAVE_PROBE = "section4_gate_anch2_probe"
WAVE_ALIASES = {WAVE_V1: WAVE_V1, "v1": WAVE_V1,
                WAVE_FIG6: WAVE_FIG6, "fig6": WAVE_FIG6,
                WAVE_PROBE: WAVE_PROBE, "probe": WAVE_PROBE}
WAVE_CHOICES = (WAVE_V1, WAVE_FIG6, WAVE_PROBE, "v1", "fig6", "probe")

PROD_PREFIX = "pofds4g"
SMOKE_PREFIX = "pofds4gsmk"
PROBE_PREFIX = "pofds4gp"       # the 5-round beta=0.75 probe wave
# mirror of HFCausalLMModel._parse_strict's regex (never import
# transformers on the login node); tests pin the two to agree
_WELL_FORMED_RE = re.compile(r"^\s*(\d*\.\d+|\d+(?:\.\d*)?)")
# scan prefixes carry the separator: "pofds4gsmk_..." also startswith
# "pofds4g", so a bare-prefix scan in production mode would silently
# swallow the smokes.
PROD_SCAN = PROD_PREFIX + "_"
SMOKE_SCAN = SMOKE_PREFIX + "_"
PROBE_SCAN = PROBE_PREFIX + "_"

OP_TOKEN = "anch2"                 # <-> nested_ai_anchored_then_social_v2
OP_INFIX = "_" + OP_TOKEN + "_"
WANT_MARKER = "nested_ai_anchored_then_social_v2"
OLD_MARKER = "nested_ai_then_social_v1"
WANT_GATE_REF = "anchor"

N = 723
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SLUG = "mistral7b"

COND_TOK = {"fixed": "fixb20", "evolving": "evoall"}
TOK_COND = {v: k for k, v in COND_TOK.items()}
W_PLAT = 0.5                       # the paper's beta
INNATE_LAMBDA = 0.2                # the paper's gamma / innate anchor k
GAMMA_BIAS = 0.0                   # homophily selection bias: always 0
CLAMP_FRAC = 0.2
CLAMP_COUNT = 145                  # round(0.2 * 723)

# TAG GRAMMAR (pinned). PARSED, never table-looked-up: a new dose or seed
# must fail as a GRID error naming the value, not as a "malformed tag".
# The trailing horizon token is OPTIONAL in the grammar and admitted
# ONLY by a wave that declares extension horizons (fig6 _r60/_r100);
# base cells never carry one.
TAG_RE = re.compile(
    r"^(?P<pre>pofds4gsmk|pofds4gp|pofds4g)"
    r"_(?P<slug>[a-z0-9]+)"
    r"_(?P<arm>b0|d8)"
    r"_(?P<cond>fixb20|evoall)"
    r"_(?P<op>anch2)"
    r"_ea(?P<ea>[0-9p]+)"
    r"_w(?P<w>[0-9p]+)"
    r"_l(?P<l>[0-9p]+)"
    r"_es(?P<es>[0-9p]+)"
    r"_s(?P<seed>\d+)"
    r"(?:_r(?P<r>\d+))?$")

# Everything HELD FIXED across all cells, byte-matched to the completed
# Section-4 surface. Values that are true by construction of the sub
# template's env are pinned here anyway: "true by construction" is a
# claim about the generator, and this file gates the ARTIFACT.
PINS = {
    "base_model": BASE_MODEL,
    "dataset": "movielens",
    "ml_target": "Action",
    "n_labeled": N,
    "train_cap": N,
    "kl_direction": "forward",
    "ai_gate_mode": "threshold",
    "peer_gate_mode": "threshold",
    "gamma_bias": GAMMA_BIAS,
    "w_plat": W_PLAT,
    "innate_lambda": INNATE_LAMBDA,
    "pop_model": "ab",
    "run_mode": "loop",
    "anchor_mode": "fixed",
    "data_regime": "replace",
    "deploy_every": 1,
    "platform_sus_scale": 1.0,
    "canary_delta": 0.0,
    "ab_sweeps": 1,
    "epoch_size": 100,
    "sft_epochs": 1,
    "sft_batch_size": 4,
    "lora_r": 512,
    "sft_lr": 5e-5,
    "pristine_frac": 0.0,
    "replay_frac": 0.0,
    "teacher_label_delta": 0.0,
    "kl_ref_adapter": "",
    "feedback_mode": "none",
    "icrh": False,
    "do_sample": False,
    "seed_base_data": True,
    "serve_eval_mode": True,
    "fj_update_version": "legacy",
}

# The two arms, as config surfaces. b0 = ordinary SFT; d8 = frozen
# personal-history ICL (each prompt carries only that agent's OWN last 8
# recorded opinions -- ICL_K=0, so no cross-user exemplar exists).
ARM_WANT = {
    "b0": {"training_style": "sft", "kl_beta": 0.0, "icl_k": 0,
           "icl_days": 0, "use_lora": True, "fresh_each_round": True},
    "d8": {"training_style": "frozen", "kl_beta": 0.0, "icl_k": 0,
           "icl_days": 8, "use_lora": False, "fresh_each_round": False},
}
# clamp keys that must be ABSENT from an evolving config (and from an
# evolving trajectory)
CLAMP_CFG_KEYS = ("innate_clamp_mode", "innate_clamp_frac",
                  "innate_clamp_seed", "innate_clamp_peer_mode",
                  "sft_exclude_clamped")
CLAMP_TRAJ_KEYS = ("innate_clamp_mask", "innate_clamp_count",
                   "innate_clamp_mode", "innate_clamp_frac",
                   "innate_clamp_seed", "innate_clamp_hash",
                   "innate_clamp_peer_mode", "clamp_fr_touch_raw")

# the personal-history sentence, byte-for-byte as run_pokec_gated_lm
# renders it (`"This user's own opinion of " f"{ML_TARGET} movies over "
# f"the most recent days (oldest to newest): {days}."`, days =
# ", ".join(f"{v:.2f}" ...)) and as check_pofd_sanity replays it
DAYS_PREFIX = ("This user's own opinion of {target} movies over the most "
               "recent days (oldest to newest): ")


# ------------------------------------------------------------- grammar
def _num(v):
    """float -> the tag's number grammar. Mirrors gen_pofd_sweep._num:
    0.2 -> '0p2', 1.0 -> '1', 0.0 -> '0'."""
    return f"{float(v):g}".replace(".", "p")


def _unnum(tok):
    """the tag's number grammar -> float: '0p2' -> 0.2, '1' -> 1.0."""
    return float(tok.replace("p", "."))


def _raw_parse(tag):
    """(arm, cond, ea, es, seed, horizon) straight off the grammar, no
    wave validation. None when the tag is not in the grammar at all."""
    m = TAG_RE.match(tag)
    if m is None:
        return None
    r = m.group("r")
    return (m.group("arm"), TOK_COND[m.group("cond")], _unnum(m.group("ea")),
            _unnum(m.group("es")), int(m.group("seed")),
            None if r is None else int(r))


# ------------------------------------------------------------ generator
def find_generator(explicit=None):
    """Path of gen_pofd_sweep.py, or None. Checkout-relative first, then
    the cwd and its ancestors."""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    cands = [os.path.join(HERE, "..", "..", "..", GEN_REL)]
    cur = os.getcwd()
    while True:
        cands.append(os.path.join(cur, GEN_REL))
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def load_generator(path):
    """Load gen_pofd_sweep.py via importlib (the check_fig3_full_loop
    pattern). The module only defines constants/functions at import;
    main() is guarded."""
    spec = importlib.util.spec_from_file_location("_gen_s4g", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_s4g"] = mod
    spec.loader.exec_module(mod)
    return mod


class Wave:
    """The per-wave grid object: every value that used to be a module
    constant (EAS/ESS/SEEDS/smoke cell/round counts) lives here and is
    READ from the generator, so parse_tag, coverage and reporting all use
    the selected wave's values."""

    def __init__(self, name, gen, ext_manifest=None):
        self.name = WAVE_ALIASES[name]
        self.fig6 = self.name == WAVE_FIG6
        self.probe = self.name == WAVE_PROBE
        self.gen = gen
        if self.fig6:
            self.key = gen.S4G2_KEY
            self.arms = tuple(gen.S4G2_ARMS)
            self.conds = tuple(gen.S4G2_CONDS)
            self.gates = tuple(float(v) for v in gen.S4G2_GATES)
            self.ess = tuple(float(v) for v in gen.S4G2_ESS)
            self.seeds = tuple(int(s) for s in gen.S4G2_SEEDS)
            self.cells6 = [(a, c, float(ea), float(es), int(sd), k)
                           for (a, c, ea, es, sd, k) in gen.s4g2_cells()]
            self.witness = {(a, c, float(es), int(sd))
                            for (a, c, es, sd) in gen.S4G2_EA0_WITNESS}
            self.horizons_ok = tuple(int(r) for r in gen.S4G2_EXT_ROUNDS_OK)
            if ext_manifest is not None:
                gen.S4G2_EXT_REQUEST_PATH = str(ext_manifest)
            self.ext_manifest = gen.S4G2_EXT_REQUEST_PATH
            self.ext_requests = [(a, c, float(ea), float(es), int(sd), int(r))
                                 for (a, c, ea, es, sd, r)
                                 in gen.s4g2_ext_requests()]
            smoke_rows = gen.s4g2_smoke_rows
            # the generator's own declared counts must describe its cells
            kinds = [k for *_, k in self.cells6]
            declared = (gen.S4G2_N_CELLS, gen.S4G2_N_GPU,
                        gen.S4G2_N_WITNESS, gen.S4G2_N_TWIN)
            got = (len(kinds), kinds.count("gpu"), kinds.count("witness"),
                   kinds.count("twin"))
            if declared != got:
                raise ValueError(
                    f"generator declares (cells, gpu, witness, twin)="
                    f"{declared} but s4g2_cells() yields {got}")
        elif self.probe:
            # the 5-round beta=0.75 probe: 2 arms x 2 conds x es {0, 1}
            # at ea=0.7, seed 0 -- no smoke (the wave IS the probe), no
            # twin-derived cells, no witnesses, no extensions
            self.key = gen.S4GP_KEY
            self.arms = tuple(gen.S4GP_ARMS)
            self.conds = tuple(gen.S4G_CONDS)
            self.gates = tuple(float(v) for v in gen.S4GP_GATES)
            self.ess = tuple(float(v) for v in gen.S4GP_ESS)
            self.seeds = tuple(int(s) for s in gen.S4GP_SEEDS)
            self.cells6 = [(arm, cond, ea, es, seed, "gpu")
                           for seed in self.seeds for arm in self.arms
                           for cond in self.conds for ea in self.gates
                           for es in self.ess]
            self.witness = set()
            self.horizons_ok = ()
            self.ext_manifest = None
            self.ext_requests = []
            smoke_rows = lambda cond: []          # noqa: E731
            if len(self.cells6) != gen.S4GP_N_TOTAL:
                raise ValueError(f"generator declares S4GP_N_TOTAL="
                                 f"{gen.S4GP_N_TOTAL} but the product has "
                                 f"{len(self.cells6)} cells")
        else:
            self.key = gen.S4G_KEY
            self.arms = tuple(gen.S4G_ARMS)
            self.conds = tuple(gen.S4G_CONDS)
            self.gates = tuple(float(v) for v in gen.S4G_GATES)
            self.ess = tuple(float(v) for v in gen.S4G_ESS)
            self.seeds = tuple(int(s) for s in gen.S4G_SEEDS)
            # the original report order: seed-major
            self.cells6 = [(arm, cond, ea, es, seed, "gpu")
                           for seed in self.seeds for arm in self.arms
                           for cond in self.conds for ea in self.gates
                           for es in self.ess]
            self.witness = set()
            self.horizons_ok = ()
            self.ext_manifest = None
            self.ext_requests = []
            smoke_rows = gen.s4g_smoke_rows
            if len(self.cells6) != gen.S4G_N_TOTAL:
                raise ValueError(f"generator declares S4G_N_TOTAL="
                                 f"{gen.S4G_N_TOTAL} but the product has "
                                 f"{len(self.cells6)} cells")
        self.rounds = int(gen.S4GP_ROUNDS if self.probe
                          else gen.S4G_ROUNDS)
        self.smoke_rounds = int(gen.S4G_SMOKE_ROUNDS)
        self.prod_prefix = PROBE_PREFIX if self.probe else PROD_PREFIX
        self.prod_scan = self.prod_prefix + "_"
        self.tag_fn = gen.s4gp_tag if self.probe else gen.s4g_tag
        self.w_plat = float(gen.S4GP_W_PLAT if self.probe else W_PLAT)
        # config pins that hold ONLY on this wave's (fresh) runs: the
        # probe pins the strict parser and the Deffuant alpha it runs
        self.extra_pins = ({"parse_mode": "strict", "deffuant_alpha": 0.5}
                           if self.probe else {})
        # the smoke cells, parsed out of the generator's own smoke rows
        # (first CSV column is the tag)
        self.smoke_cells = []
        for cond in self.conds:
            for row in smoke_rows(cond):
                tag = row.split(",")[0].strip()
                p = _raw_parse(tag)
                if p is None or not tag.startswith(SMOKE_SCAN) or \
                        p[5] is not None:
                    raise ValueError(f"generator smoke tag {tag!r} is not in "
                                     f"this checker's smoke grammar")
                self.smoke_cells.append(p[:5])
        self.smoke_set = set(self.smoke_cells)
        self.smoke_gates = sorted({(c[2], c[3], c[4])
                                   for c in self.smoke_cells})

    # -- the conceptual grid ------------------------------------------
    def run_cells(self):
        """Cells that require a run dir (kind gpu or witness)."""
        return [c[:5] for c in self.cells6 if c[5] in ("gpu", "witness")]

    def twin_cells(self):
        """Cells with NO run dir, satisfied by twin_raw of the runs at
        the same (cond, es, seed)."""
        return [c[:5] for c in self.cells6 if c[5] == "twin"]

    def kind_of(self, cell):
        for c in self.cells6:
            if c[:5] == tuple(cell):
                return c[5]
        return None

    def is_witness(self, arm, cond, ea, es, seed):
        return float(ea) == 0.0 and (arm, cond, float(es), int(seed)) \
            in self.witness

    def render_tag(self, arm, cond, ea, es, seed, smoke=False, rounds=None):
        """THE generator's tag -- never a local re-implementation."""
        return self.tag_fn(arm, cond, float(ea), float(es), int(seed),
                           prefix=SMOKE_PREFIX if smoke
                           else self.prod_prefix,
                           rounds=rounds)

    def self_test_grammar(self, smoke):
        """parse(render(cell)) must round-trip for every expected cell:
        a checker grammar that disagrees with s4g_tag would report a
        complete wave as absent, or an absent one as complete."""
        bad = []
        if smoke:
            todo = [(c, None) for c in self.smoke_cells]
        else:
            todo = [(c, None) for c in self.run_cells()]
            todo += [(c, None) for c in self.twin_cells()]
            todo += [(e[:5], e[5]) for e in self.ext_requests]
        for cell, r in todo:
            tag = self.render_tag(*cell, smoke=smoke, rounds=r)
            info, errs = parse_tag(tag, smoke, self)
            if info is None or errs or info["key"] != tuple(cell) + (r,):
                bad.append((tag, errs or ["unparseable"]))
        return bad


def parse_tag(tag, smoke, wave):
    """(info, errs). info is None when nothing downstream can be trusted.

    Every value is PARSED out of the tag and then required to round-trip
    back through the same grammar, so a token that parses to the right
    float in the wrong spelling ('ea0p20', 'es1p0', 's042') is caught.
    Grid membership is checked against the SELECTED WAVE's values.
    """
    errs = []
    # THE operator token, checked before anything else so its absence is
    # reported as itself rather than as a generic grammar miss.
    if OP_INFIX not in tag:
        return None, [f"tag carries no {OP_INFIX!r} token -- every "
                      f"Section-4 corrected-gate tag MUST declare the "
                      f"anchored operator ({WANT_MARKER})"]
    m = TAG_RE.match(tag)
    if m is None:
        return None, [f"tag is not in the pofds4g grammar "
                      f"{wave.prod_prefix}_{MODEL_SLUG}_<b0|d8>"
                      f"_<fixb20|evoall>_{OP_TOKEN}_ea<EA>"
                      f"_w{_num(wave.w_plat)}_l0p2_es<ES>_s<SEED>"
                      f"[_r<HORIZON>]"]
    pre, slug = m.group("pre"), m.group("slug")
    arm, cond = m.group("arm"), TOK_COND[m.group("cond")]
    ea, es = _unnum(m.group("ea")), _unnum(m.group("es"))
    w, lam = _unnum(m.group("w")), _unnum(m.group("l"))
    seed = int(m.group("seed"))
    horizon = None if m.group("r") is None else int(m.group("r"))
    if (pre == SMOKE_PREFIX) != bool(smoke):
        errs.append(f"smoke/production prefix mismatch: prefix {pre!r} with "
                    f"--smoke={bool(smoke)}; a smoke cell can never stand in "
                    f"for a production cell (or the reverse)")
    elif not smoke and pre != wave.prod_prefix:
        errs.append(f"tag prefix {pre!r} does not belong to the {wave.name} "
                    f"wave (expected {wave.prod_prefix!r}) -- the probe and "
                    f"S4G production waves are gated separately")
    if slug != MODEL_SLUG:
        errs.append(f"model slug {slug!r}; this wave is {MODEL_SLUG}-only")
    rebuilt = (f"{pre}_{slug}_{arm}_{COND_TOK[cond]}_{OP_TOKEN}_ea{_num(ea)}"
               f"_w{_num(w)}_l{_num(lam)}_es{_num(es)}_s{seed}"
               + ("" if horizon is None else f"_r{horizon}"))
    if rebuilt != tag:
        errs.append(f"tag numbers do not round-trip through the pinned "
                    f"grammar (would be spelled {rebuilt!r}) -- two "
                    f"spellings of one cell make coverage unprovable")
    if w != wave.w_plat:
        errs.append(f"tag says w{m.group('w')} (= {w:g}); the {wave.name} "
                    f"wave is W_PLAT={wave.w_plat:g}")
    if lam != INNATE_LAMBDA:
        errs.append(f"tag says l{m.group('l')} (= {lam:g}); the wave is "
                    f"INNATE_LAMBDA={INNATE_LAMBDA:g}")
    if horizon is not None:
        if smoke:
            errs.append(f"smoke tag carries a horizon token _r{horizon}; "
                        f"smoke cells never do")
        elif not wave.horizons_ok:
            errs.append(f"tag carries a horizon token _r{horizon}, which is "
                        f"not part of the {wave.name} grammar (base cells "
                        f"never carry one; only the Figure-6 extensions do)")
        elif horizon not in wave.horizons_ok:
            errs.append(f"horizon _r{horizon} is not an allowed extension "
                        f"horizon {list(wave.horizons_ok)}")
    if smoke:
        if (arm, cond, ea, es, seed) not in wave.smoke_set:
            want = " or ".join(f"ea{_num(g[0])} es{_num(g[1])} s{g[2]}"
                               for g in wave.smoke_gates)
            errs.append(f"smoke cell must be {want}; got ea{_num(ea)} "
                        f"es{_num(es)} s{seed}")
    else:
        if ea not in wave.gates:
            errs.append(f"eps_ai {ea:g} is not in the {wave.name} grid "
                        f"{list(wave.gates)}")
        if es not in wave.ess:
            errs.append(f"eps_social {es:g} is not in the {wave.name} grid "
                        f"{list(wave.ess)}")
        if seed not in wave.seeds:
            errs.append(f"seed {seed} is not in the {wave.name} grid "
                        f"{list(wave.seeds)}")
    cell = (arm, cond, ea, es, seed)
    info = {"pre": pre, "arm": arm, "cond": cond, "ea": ea, "es": es,
            "seed": seed, "horizon": horizon, "cell": cell,
            "key": cell + (horizon,)}
    return info, errs


# --------------------------------------------------------------- cohort
def bottom_cohort_mask(innate, n_frozen):
    """Boolean [n] mask of the n_frozen LOWEST innate opinions, agent id
    as the deterministic tie-break.

    MIRRORED from check_pofd_sanity.check_run's "-- 1j CLAMP" block (its
    `order_cl` / `want_ids` reconstruction), which is in turn the exact
    algorithm of _gated_pop.innate_clamp_mask(mode="bottom"). Kept as an
    independent line of code on purpose: a helper bug must not be able to
    self-certify, so a fixed run is checked against BOTH this and
    gp.innate_clamp_mask.
    """
    n = int(innate.numel())
    order = sorted(range(n), key=lambda i: (float(innate[i]), i))
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.tensor(sorted(order[:n_frozen]), dtype=torch.long)] = True
    return mask


def _sha_t(t):
    """sha256 over a tensor's raw float32 bytes -- bit-identity, not
    closeness."""
    return hashlib.sha256(
        t.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


def _bit_eq_rows(block, vec):
    """Every row of `block` [T,k] bit-identical to `vec` [k]."""
    return bool((block == vec.unsqueeze(0)).all())


# ------------------------------------------------ d8 personal history
def _render_days(target, vals):
    return DAYS_PREFIX.format(target=target) + \
        ", ".join(f"{v:.2f}" for v in vals) + "."


def replay_personal_history(innate, op, rows, icl_days, target):
    """BYTE-EXACT replay of icl_days_log.json.gz from (innate, op_raw).

    MIRRORS check_pofd_sanity.check_run's d8 PERSONAL-HISTORY replay --
    the `hist_cl` loop of its "-- 1j CLAMP" block and the `hist_e` loop
    of its `if is_evo:` block -- which is the runner's own rendering
    (run_pokec_gated_lm: hist starts as [innate], op_raw[t] is appended
    after round t, and the prompt carries hist[-icl_days:] as "%.2f",
    oldest to newest).

    Schema (inspected on notes/pofd/cluster/pofdqwu_qwen7b_d8_*_r100):
    one JSON line per round, {"round": t, "ctx": [n strings]}, ctx[i]
    the exact sentence agent i's prompt carried.

    Returns None when every sentence of every round replays exactly, else
    a (round, agent, reason) triple where `reason` CLASSIFIES the first
    mismatch: a value that is not one of the agent's own previous
    opinions (and whether the sentence is ANOTHER agent's), more than
    icl_days values, or the agent's own values in the wrong window/order.
    """
    n = int(innate.numel())
    hist = [innate.tolist()]
    for t, row in enumerate(rows):
        ctxs = row.get("ctx")
        if not isinstance(ctxs, list) or len(ctxs) != n:
            return (t, None, f"{len(ctxs) if isinstance(ctxs, list) else 0} "
                             f"contexts (want {n})")
        win = hist[-icl_days:]
        for i in range(n):
            want = _render_days(target, [h[i] for h in win])
            got = ctxs[i]
            if got == want:
                continue
            # classify the first mismatch
            prefix = DAYS_PREFIX.format(target=target)
            if not isinstance(got, str) or not got.startswith(prefix) \
                    or not got.endswith("."):
                reason = "not a personal-history sentence"
            else:
                vals = got[len(prefix):-1].split(", ")
                own = {f"{h[i]:.2f}" for h in hist}
                foreign = [v for v in vals if v not in own]
                if len(vals) > icl_days:
                    reason = (f"{len(vals)} values rendered > icl_days "
                              f"{icl_days}")
                elif foreign:
                    others = [j for j in range(n) if j != i and
                              _render_days(target, [h[j] for h in win])
                              == got]
                    reason = (f"value(s) {foreign} are NOT among agent {i}'s "
                              f"own previous opinions")
                    if others:
                        reason += (f" -- the sentence is agent {others[0]}'s "
                                   f"context (ANOTHER agent's history)")
                else:
                    reason = ("own values but the wrong window/order "
                              "(want the last %d of innate + op_raw, oldest "
                              "to newest)" % icl_days)
            return (t, i, f"{reason}; got {str(got)[:90]!r} want "
                          f"{want[:90]!r}")
        if t < op.shape[0]:
            hist.append(op[t].tolist())
    return None


def _read_jsonl(path, gz=False):
    opener = gzip.open if gz else open
    with opener(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _empty_rec(run_dir, tag):
    return {"run_dir": run_dir, "tag": tag, "cell": None, "horizon": None,
            "witness": False, "errs": [], "warns": [], "notes": [],
            "parse_evidence": None, "innate_sha256": None,
            "cohort_sha256": None, "n_rounds": None, "pop_final_mean": None,
            "pop_final_sd": None, "op_twin_l1": None, "op_sha256": None,
            "twin_sha256": None, "twin_base_sha256": None,
            "twin_final_mean": None, "twin_final_sd": None,
            "contact_max": None, "gpu_name": None, "d8_replay": None}


# ------------------------------------------------------------ one cell
def check_one(run_dir, wave, smoke, inspect_archived=False):
    """Gate ONE run dir. Returns a record of scalars/hashes only: no
    tensor outlives this call, so the gate holds at most one run's
    trajectory in memory at a time (login-node budget).

    inspect_archived downgrades ONLY a missing raw_gen_log.json.gz to a
    WARN (NOT a gate; see the module docstring)."""
    run_dir = str(run_dir).rstrip("/")
    tag = os.path.basename(run_dir)
    rec = _empty_rec(run_dir, tag)
    errs = rec["errs"]

    def bad(msg):
        errs.append(msg)

    def warn(msg):
        rec["warns"].append(msg)

    info, terrs = parse_tag(tag, smoke, wave)
    errs.extend(terrs)
    if info is None:
        return rec
    rec["cell"] = info["cell"]
    rec["horizon"] = info["horizon"]
    arm, cond = info["arm"], info["cond"]
    if smoke:
        want_rounds = wave.smoke_rounds
    else:
        want_rounds = info["horizon"] or wave.rounds
    is_ea0 = (not smoke) and info["ea"] == 0.0
    rec["witness"] = bool(is_ea0 and wave.is_witness(*info["cell"]))

    tp = os.path.join(run_dir, "trajectory.pt")
    if not os.path.exists(tp):
        bad("no trajectory.pt -- trajectory.pt and raw_gen_log.json.gz are "
            "written ONLY at run completion, so this cell is incomplete "
            "(config.json alone is written at launch and proves nothing)")
        return rec
    d = torch.load(tp, map_location="cpu", weights_only=False)
    cfg = d.get("config")
    if not isinstance(cfg, dict) or not cfg:
        bad("trajectory.pt carries no config dict")
        del d
        return rec
    hw = cfg.get("hardware") if isinstance(cfg.get("hardware"), dict) else {}
    rec["gpu_name"] = hw.get("gpu_name")

    # --- 2. THE OPERATOR. The most important gate in this file. --------
    pu = cfg.get("population_update", "<absent>")
    if pu == OLD_MARKER:
        bad(f"population_update={OLD_MARKER!r} -- this is the OLD, ARCHIVED "
            f"gate reference (|m - x0| on the RAW start-of-round opinion). "
            f"The corrected-gate wave requires {WANT_MARKER!r} "
            f"(AI_GATE_REFERENCE=anchor, |m - x'| on the ANCHORED opinion). "
            f"An archived-semantics run wearing a {OP_TOKEN!r} tag cannot "
            f"stand in for a cell of this wave")
    elif pu != WANT_MARKER:
        bad(f"population_update={pu!r}, expected {WANT_MARKER!r} -- an "
            f"absent or unknown round-operator marker is a hard failure, "
            f"never a silent fallback")
    gr = cfg.get("ai_gate_reference", "<absent>")
    if gr != WANT_GATE_REF:
        bad(f"ai_gate_reference={gr!r}, expected {WANT_GATE_REF!r} (an "
            f"absent field means the run predates 2026-08-22 and gated on "
            f"x0)")
    # config.json, written at launch, must tell the same story as the
    # trajectory config written at completion
    cjp = os.path.join(run_dir, "config.json")
    if os.path.exists(cjp):
        try:
            cj = json.loads(open(cjp).read())
        except (ValueError, OSError) as e:
            bad(f"config.json unreadable: {e}")
            cj = {}
        for k in ("population_update", "ai_gate_reference"):
            if k in cj and cj.get(k) != cfg.get(k):
                bad(f"config.json {k}={cj.get(k)!r} disagrees with "
                    f"trajectory.pt {k}={cfg.get(k)!r}")
        if rec["gpu_name"] is None and isinstance(cj.get("hardware"), dict):
            rec["gpu_name"] = cj["hardware"].get("gpu_name")

    # --- 3. GRID FIELDS, TAG vs CONFIG in BOTH directions -------------
    # forward: the config value must equal what the tag says
    for key, tok, want in (("eps_ai", f"ea{_num(info['ea'])}", info["ea"]),
                           ("eps", f"es{_num(info['es'])}", info["es"])):
        got = cfg.get(key, None)
        if got is None or abs(float(got) - want) > 1e-12:
            bad(f"{key}={got!r} but the tag says {tok} (= {want:g}) -- the "
                f"queue column did not reach the runner")
        else:
            # backward: re-render the CONFIG value into the tag grammar
            # and require the token itself
            if _num(float(got)) != _num(want):
                bad(f"{key}={got!r} renders as {_num(float(got))!r}, tag "
                    f"token {tok!r}")
    if int(cfg.get("seed", -1)) != info["seed"]:
        bad(f"seed={cfg.get('seed')!r} but the tag says s{info['seed']}")
    if int(cfg.get("n_rounds", -1)) != want_rounds:
        bad(f"n_rounds={cfg.get('n_rounds')!r}, expected {want_rounds}"
            + (f" (the _r{info['horizon']} horizon of this extension)"
               if info["horizon"] else ""))
    pins = dict(PINS, w_plat=wave.w_plat, **wave.extra_pins)
    for key, want in pins.items():
        got = cfg.get(key, "<absent>")
        if isinstance(want, bool):
            if got == "<absent>" or bool(got) is not want:
                bad(f"{key}={got!r}, expected {want!r}")
        elif isinstance(want, float):
            if got == "<absent>" or not isinstance(got, (int, float)) or \
                    abs(float(got) - want) > 1e-12:
                bad(f"{key}={got!r}, expected {want!r}")
        elif got != want:
            bad(f"{key}={got!r}, expected {want!r}")
    # the ARM, forward and backward
    for key, want in ARM_WANT[arm].items():
        got = cfg.get(key, "<absent>")
        if isinstance(want, bool):
            if got == "<absent>" or bool(got) is not want:
                bad(f"arm {arm}: {key}={got!r}, expected {want!r}")
        elif isinstance(want, float):
            if got == "<absent>" or not isinstance(got, (int, float)) or \
                    abs(float(got) - want) > 1e-12:
                bad(f"arm {arm}: {key}={got!r}, expected {want!r}")
        elif got != want:
            bad(f"arm {arm}: {key}={got!r}, expected {want!r}")
    inferred = [a for a, wnt in ARM_WANT.items()
                if all(bool(cfg.get(k)) == bool(v) if isinstance(v, bool)
                       else cfg.get(k) == v for k, v in wnt.items())]
    if inferred != [arm]:
        bad(f"the config envelope reads back as arm(s) {inferred or ['?']} "
            f"while the tag claims {arm!r} -- style/kl_beta/icl_k/icl_days/"
            f"use_lora/fresh_each_round must identify exactly one arm")
    # the CONDITION, backward: a clamp key in the config IS the fixed
    # condition; its absence IS the evolving condition
    cfg_clamped = "innate_clamp_mode" in cfg
    if cfg_clamped != (cond == "fixed"):
        bad(f"tag condition {cond!r} ({COND_TOK[cond]}) but the config "
            f"{'carries' if cfg_clamped else 'carries no'} innate_clamp_mode "
            f"-- the tag and the config disagree about whether 145 agents "
            f"are pinned")

    # --- 8. shapes / horizon ------------------------------------------
    traj = d.get("trajectory")
    if not isinstance(traj, (list, tuple)):
        bad("trajectory (per-round rows) missing or not a list")
        traj = []
    if len(traj) < want_rounds:
        bad(f"trajectory holds {len(traj)} rows < n_rounds {want_rounds} "
            f"-- the run is short")
    op = d.get("op_raw")
    pr = d.get("pred_raw")
    tw = d.get("twin_raw")
    inn = d.get("innate")
    shapes_ok = True
    for nm, t in (("op_raw", op), ("pred_raw", pr), ("twin_raw", tw)):
        if not torch.is_tensor(t) or t.numel() == 0:
            bad(f"{nm} missing/empty")
            shapes_ok = False
        elif tuple(t.shape) != (want_rounds, N):
            bad(f"{nm} shape "
                f"{tuple(t.shape) if torch.is_tensor(t) else None} != "
                f"{(want_rounds, N)}")
            shapes_ok = False
    if not torch.is_tensor(inn) or tuple(inn.shape) != (N,):
        bad(f"innate shape "
            f"{tuple(inn.shape) if torch.is_tensor(inn) else None} != {(N,)}")
        shapes_ok = False
    if not shapes_ok:
        del d
        return rec
    op = op.float()
    pr = pr.float()
    tw = tw.float()
    inn = inn.float()
    rec["n_rounds"] = int(op.shape[0])

    if not torch.isfinite(op).all():
        bad("op_raw has non-finite values")
    elif float(op.min()) < -1e-6 or float(op.max()) > 1 + 1e-6:
        bad(f"op_raw outside [0,1]: [{float(op.min()):.4f}, "
            f"{float(op.max()):.4f}]")
    if not torch.isfinite(inn).all():
        bad("innate has non-finite values")
    # served-vector integrity ONLY. This is NOT parse evidence: the
    # runner's parser stores a FINITE 0.5 when a generation does not
    # parse, so pred_raw cannot reveal a parse failure (check 7 below
    # reads raw_gen_log.json.gz for that).
    if not torch.isfinite(pr).all():
        bad(f"pred_raw has {int((~torch.isfinite(pr)).sum())} non-finite "
            f"entries -- a corrupted served vector (the parser never "
            f"writes NaN; it stores 0.5 on failure)")

    # --- 6. TWIN present and non-degenerate ---------------------------
    # WITH_TWIN=1 leaves NO config field (the runner records no with_twin
    # key), so the artifact itself is the only evidence the counterfactual
    # was simulated -- hence twin_raw is gated on shape, finiteness, range
    # and dispersion rather than on a flag.
    if not torch.isfinite(tw).all():
        bad("twin_raw has non-finite values")
    elif float(tw.min()) < -1e-6 or float(tw.max()) > 1 + 1e-6:
        bad(f"twin_raw outside [0,1]: [{float(tw.min()):.4f}, "
            f"{float(tw.max()):.4f}]")
    elif float(tw.std()) == 0.0:
        bad(f"twin_raw is CONSTANT ({float(tw.reshape(-1)[0]):g}) over every "
            f"agent and round -- a degenerate twin is not a counterfactual "
            f"(innate itself is heterogeneous, so the no-platform path "
            f"cannot be constant)")
    rec["op_twin_l1"] = float((op - tw).abs().mean())
    # the deployed trajectory's fingerprint, for the wave-level
    # SEED-DISTINCTNESS check, and the twin's for TWIN AGREEMENT / the
    # twin-derived cells. Hashed here and the tensors dropped, so two runs
    # are never resident at once. twin_base is the twin over the base
    # horizon, so an extension run is comparable to its base cell.
    rec["op_sha256"] = _sha_t(op)
    rec["twin_sha256"] = _sha_t(tw)
    rec["twin_base_sha256"] = _sha_t(tw[:min(wave.rounds, int(tw.shape[0]))])
    rec["pop_final_mean"] = float(op[-1].mean())
    rec["pop_final_sd"] = float(op[-1].std())
    rec["twin_final_mean"] = float(tw[-1].mean())
    rec["twin_final_sd"] = float(tw[-1].std())

    # --- 11. ea0-witness ----------------------------------------------
    # gp.ai_gate is a STRICT inequality |m - x'| < eps_AI, so at eps_AI=0
    # the gate is closed for every agent in every round: the served
    # vector never enters, and the deployed population IS the matched
    # no-AI twin (same anchor step, same peer sweep, mirrored generator).
    # Two things must hold, both BIT-EXACT: op_raw == twin_raw over every
    # round, and the recorded AI-gate open fraction is 0 in every round.
    if is_ea0:
        label = "ea0-witness" + ("" if rec["witness"] else " (ea=0 run)")
        if not torch.equal(op, tw):
            diff = (op != tw)
            nag = int(diff.any(dim=0).sum())
            nrd = int(diff.any(dim=1).sum())
            first = torch.nonzero(diff.any(dim=1)).flatten()
            bad(f"{label}: op_raw != twin_raw -- {nag} agent(s) differ over "
                f"{nrd} round(s), first round "
                f"{int(first[0]) if first.numel() else -1}, max |diff| "
                f"{float((op - tw).abs().max()):.2e}. At eps_AI=0 the "
                f"strict-< gate never opens, so the deployed population "
                f"must BE the twin bit-exactly; a difference means the "
                f"served vector reached the population and the ea=0 "
                f"cells cannot be twin-derived")
        telp = os.path.join(run_dir, "telemetry.json")
        if not os.path.exists(telp):
            bad(f"{label}: telemetry.json missing -- the per-round AI-gate "
                f"open fraction (contact) is the second half of the ea=0 "
                f"proof and is recorded there")
        else:
            try:
                tel = _read_jsonl(telp)
            except (OSError, ValueError) as e:
                bad(f"{label}: telemetry.json unreadable: {e}")
                tel = []
            got_rounds = [r.get("round") for r in tel]
            if got_rounds != list(range(want_rounds)):
                bad(f"{label}: telemetry.json holds rounds {got_rounds[:5]}"
                    f"... (want 0..{want_rounds - 1}; the runner truncates "
                    f"it at launch and appends one row per round)")
            nonzero, cmax = [], 0.0
            for r in tel:
                c = r.get("contact", None)
                if not isinstance(c, (int, float)) or isinstance(c, bool):
                    nonzero.append((r.get("round"), c))
                    continue
                cmax = max(cmax, abs(float(c))) if c == c else float("inf")
                if float(c) != 0.0:
                    nonzero.append((r.get("round"), c))
            rec["contact_max"] = cmax
            if nonzero:
                bad(f"{label}: telemetry contact (AI-gate open fraction) is "
                    f"not exactly 0 in {len(nonzero)} round(s), e.g. round "
                    f"{nonzero[0][0]} contact={nonzero[0][1]!r} -- at "
                    f"eps_AI=0 the strict-< gate must never open")
        # the trajectory rows carry the same scalar (run_pokec_gated_lm
        # writes row["contact"] alongside tel_row["contact"])
        tnz = [(r.get("round"), r.get("contact")) for r in traj
               if isinstance(r, dict) and "contact" in r
               and (not isinstance(r.get("contact"), (int, float))
                    or float(r.get("contact")) != 0.0)]
        if tnz:
            bad(f"{label}: trajectory.pt rows record contact != 0 in "
                f"{len(tnz)} round(s), e.g. round {tnz[0][0]} contact="
                f"{tnz[0][1]!r}")

    # --- 5. the innate vector and the reconstructed cohort ------------
    rec["innate_sha256"] = _sha_t(inn)
    want_frozen = int(round(CLAMP_FRAC * int(inn.numel())))
    if want_frozen != CLAMP_COUNT:
        bad(f"frac {CLAMP_FRAC:g} of {int(inn.numel())} agents gives "
            f"{want_frozen} frozen, expected {CLAMP_COUNT}")
    rec_mask = bottom_cohort_mask(inn, want_frozen)
    rec["cohort_sha256"] = gp.innate_clamp_hash(rec_mask)

    # --- 4. CONDITION INTEGRITY ---------------------------------------
    cm_valid = None
    if cond == "fixed":
        # MIRRORS check_pofd_sanity.check_run's "-- 1j CLAMP" block.
        cl_mode = cfg.get("innate_clamp_mode", "<absent>")
        cl_frac = cfg.get("innate_clamp_frac", None)
        cl_seed = cfg.get("innate_clamp_seed", None)
        cl_peer = cfg.get("innate_clamp_peer_mode", "<absent>")
        if cl_mode != "bottom":
            bad(f"fixed: innate_clamp_mode={cl_mode!r}, expected 'bottom' "
                f"(the 145 LOWEST-innate agents)")
        if cl_frac is None or abs(float(cl_frac) - CLAMP_FRAC) > 1e-12:
            bad(f"fixed: innate_clamp_frac={cl_frac!r}, expected "
                f"{CLAMP_FRAC:g}")
        if cl_seed is None or int(cl_seed) != info["seed"]:
            bad(f"fixed: innate_clamp_seed={cl_seed!r} != run seed "
                f"{info['seed']} (the cohort seed rides the run seed)")
        if cl_peer != "stubborn":
            bad(f"fixed: innate_clamp_peer_mode={cl_peer!r}, expected "
                f"'stubborn' (the one-sided peer operator, inert at es=0)")
        if bool(cfg.get("sft_exclude_clamped")):
            bad("fixed: sft_exclude_clamped is set -- b0xa source exclusion "
                "is NOT part of this wave (SFT_EXCLUDE_CLAMPED=0)")
        cm = d.get("innate_clamp_mask")
        if not torch.is_tensor(cm) or cm.numel() == 0:
            bad("fixed: innate_clamp_mask missing/empty in trajectory.pt")
        elif cm.dtype != torch.bool or tuple(cm.shape) != (N,):
            bad(f"fixed: innate_clamp_mask dtype/shape {cm.dtype}/"
                f"{tuple(cm.shape)} (want bool [{N}])")
        else:
            cm = cm.bool()
            cm_valid = cm
            got_frozen = int(cm.sum())
            if got_frozen != CLAMP_COUNT:
                bad(f"fixed: mask pins {got_frozen} agents, expected exactly "
                    f"{CLAMP_COUNT}")
            if int(d.get("innate_clamp_count", -1)) != got_frozen:
                bad(f"fixed: innate_clamp_count="
                    f"{d.get('innate_clamp_count')!r} != mask sum "
                    f"{got_frozen}")
            for k in ("innate_clamp_mode", "innate_clamp_frac",
                      "innate_clamp_seed"):
                if d.get(k, "<absent>") != cfg.get(k, "<absent>"):
                    bad(f"fixed: trajectory {k}={d.get(k)!r} != config "
                        f"{cfg.get(k)!r}")
            want_hash = gp.innate_clamp_hash(cm)
            if d.get("innate_clamp_hash") != want_hash:
                bad(f"fixed: innate_clamp_hash="
                    f"{str(d.get('innate_clamp_hash'))[:16]!r}... does not "
                    f"match the stored mask ({want_hash[:16]!r}...) -- mask "
                    f"corrupted or tampered")
            # RECONSTRUCT the cohort; never trust the stored mask. Both
            # the shared helper and this file's independent ranking, so a
            # bug in either cannot self-certify.
            if not torch.equal(rec_mask, cm):
                bad(f"fixed: the stored mask is NOT the {CLAMP_COUNT} "
                    f"lowest-innate agents under the deterministic "
                    f"innate-then-id ranking -- "
                    f"{int((rec_mask ^ cm).sum())} agents differ")
            try:
                helper = gp.innate_clamp_mask(inn, "bottom", CLAMP_FRAC,
                                              int(cl_seed or 0))
                if not torch.equal(helper, cm):
                    bad(f"fixed: mask does not reconstruct from (innate, "
                        f"'bottom', {CLAMP_FRAC:g}, {cl_seed!r}) -- "
                        f"{int((helper ^ cm).sum())} agents differ")
            except (ValueError, TypeError) as e:
                bad(f"fixed: mask reconstruction impossible: {e}")
            # BIT-EXACT in BOTH the deployed population and the twin, at
            # every recorded round (the check_pofd_sanity CLAMP assertion)
            if not _bit_eq_rows(op[:, cm], inn[cm]):
                nbad = int((op[:, cm] != inn[cm].unsqueeze(0))
                           .any(dim=0).sum())
                bad(f"fixed: {nbad} pinned agents drift off innate in op_raw "
                    f"(max |diff| "
                    f"{float((op[:, cm] - inn[cm]).abs().max()):.2e}) -- the "
                    f"clamp must be bit-exact")
            if not _bit_eq_rows(tw[:, cm], inn[cm]):
                nbad = int((tw[:, cm] != inn[cm].unsqueeze(0))
                           .any(dim=0).sum())
                bad(f"fixed: {nbad} pinned agents drift off innate in "
                    f"twin_raw (max |diff| "
                    f"{float((tw[:, cm] - inn[cm]).abs().max()):.2e}) -- the "
                    f"clamp holds in the matched twin too")
            # STUBBORN-PEER invariants (check_pofd_sanity: with a live
            # clamp-peer operator the RESPONSIVE twin MOVES, so it is
            # never compared to innate; what must exist is the operator's
            # own per-round telemetry).
            ft = d.get("clamp_fr_touch_raw")
            if not torch.is_tensor(ft) or ft.numel() == 0:
                bad("fixed: clamp_fr_touch_raw missing/empty -- the stubborn "
                    "peer operator records per-round fixed->responsive reach")
            elif tuple(ft.shape) != (want_rounds, N):
                bad(f"fixed: clamp_fr_touch_raw shape {tuple(ft.shape)} != "
                    f"{(want_rounds, N)}")
            elif bool(ft.bool()[:, cm].any()):
                bad("fixed: clamp_fr_touch_raw marks a PINNED agent as "
                    "reached -- the reach mask lives on the responsive "
                    "subset only")
            # the responsive complement must actually be alive
            if int((~cm).sum()) and not bool(
                    (op[:, ~cm] != inn[~cm].unsqueeze(0)).any()):
                bad(f"fixed: not one of the {int((~cm).sum())} responsive "
                    f"agents ever leaves innate -- the clamp was applied "
                    f"beyond its mask")
    else:
        # EVOLVING: MIRRORS check_pofd_sanity's `if is_evo:` section --
        # no clamp, no fixed cohort, and NO clamp key anywhere.
        for k in CLAMP_CFG_KEYS:
            if k in cfg:
                bad(f"evolving: config carries {k}={cfg.get(k)!r} -- a "
                    f"fully-evolving run has no clamp and no fixed agents, "
                    f"so the key must be ABSENT (absent == off, the audit "
                    f"convention)")
        for k in CLAMP_TRAJ_KEYS:
            v = d.get(k)
            present = (v.numel() > 0) if torch.is_tensor(v) else (v is not None)
            if present:
                bad(f"evolving: trajectory carries clamp artifact {k} -- a "
                    f"fully-evolving run must not carry one")

    # --- 10. d8 personal-history LOCALITY (both conditions) -----------
    # MIRRORS the d8 PERSONAL-HISTORY replay of check_pofd_sanity's
    # CLAMP/EVO sections (see replay_personal_history): ICL_K=0 means NO
    # cross-user exemplar may exist, the rendered personal histories are
    # a mandatory artifact, and every sentence must replay BYTE-EXACTLY
    # from (innate, op_raw) -- which proves locality (own values only,
    # <= icl_days of them, nothing from another agent) in one stroke.
    icl_days = int(cfg.get("icl_days") or 0)
    if icl_days > 0:
        if int(cfg.get("icl_k") or 0) != 0:
            bad("d8: icl_k>0 -- cross-user exemplars are forbidden in the "
                "personal-history arm")
        for k in ("icl_idx_raw", "icl_val_raw"):
            v = d.get(k)
            if torch.is_tensor(v) and v.numel():
                bad(f"d8: {k} non-empty -- cross-user exemplar artifacts "
                    f"must not exist")
        if os.path.exists(os.path.join(run_dir, "icl_ctx_log.json.gz")):
            bad("d8: icl_ctx_log.json.gz present -- no cross-user context "
                "may be rendered")
        dlp = os.path.join(run_dir, "icl_days_log.json.gz")
        if not os.path.exists(dlp):
            bad("d8: icl_days_log.json.gz missing -- the rendered "
                "personal-history contexts are mandatory")
        else:
            try:
                dl_rows = _read_jsonl(dlp, gz=True)
            except (OSError, ValueError) as e:
                bad(f"d8: icl_days_log.json.gz unreadable: {e}")
                dl_rows = None
            if dl_rows is not None:
                got_rounds = [r.get("round") for r in dl_rows]
                if got_rounds != list(range(want_rounds)):
                    bad(f"d8: icl_days_log holds rounds {got_rounds[:5]}... "
                        f"(want 0..{want_rounds - 1})")
                else:
                    target = cfg.get("ml_target") or "Action"
                    fail = replay_personal_history(inn, op, dl_rows,
                                                   icl_days, target)
                    if fail is not None:
                        bad(f"d8 locality: personal-history context is OFF "
                            f"the byte-exact (innate, op_raw) replay at "
                            f"round {fail[0]} agent {fail[1]}: {fail[2]}")
                    else:
                        rec["d8_replay"] = "byte-exact"
                        # fixed agents: nothing but their own innate,
                        # stated directly on the final rendered round
                        # (check_pofd_sanity's CLAMP d8 tail check)
                        if cm_valid is not None:
                            for i in cm_valid.nonzero().flatten().tolist():
                                iv = f"{float(inn[i]):.2f}"
                                seq_s = dl_rows[-1]["ctx"][i] \
                                    .rsplit(": ", 1)[1].rstrip(".")
                                if any(v != iv for v in seq_s.split(", ")):
                                    bad(f"d8 locality: fixed agent {i} "
                                        f"history is not pure innate "
                                        f"repetition ({seq_s!r} vs {iv})")
                                    break

    # --- 7. ZERO PARSE FAILURES ---------------------------------------
    # raw_gen_log.json.gz is REQUIRED: parse_fail_frac lives nowhere
    # else, and the parser's failure value is a FINITE 0.5, so pred_raw
    # can never reveal a parse failure. SAVE_RAW_GEN=1 is pinned in every
    # Section-4 corrected-gate sub template.
    gz = os.path.join(run_dir, "raw_gen_log.json.gz")
    if not os.path.exists(gz):
        why = ("raw_gen_log.json.gz missing -- parse_fail_frac is recorded "
               "NOWHERE else and the parser stores a finite 0.5 on failure, "
               "so the parse rate of this run is NOT establishable "
               "(SAVE_RAW_GEN=1 is mandatory for every Section-4 "
               "corrected-gate run)")
        if inspect_archived:
            rec["parse_evidence"] = "missing(archived)"
            warn(why + "; admitted ONLY because --inspect-archived is set, "
                       "which is NOT a gate")
        else:
            rec["parse_evidence"] = "missing"
            bad(why)
    else:
        rec["parse_evidence"] = "raw_gen_log"
        try:
            rows = _read_jsonl(gz, gz=True)
        except (OSError, ValueError) as e:
            bad(f"raw_gen_log.json.gz unreadable: {e}")
            rows = []
        got_rounds = [r.get("round") for r in rows]
        if got_rounds != list(range(want_rounds)):
            absent = [t for t in range(want_rounds) if t not in got_rounds]
            surplus = [t for t in got_rounds
                       if not isinstance(t, int) or not 0 <= t < want_rounds]
            dup = len(got_rounds) - len(set(got_rounds))
            bad(f"raw_gen_log must carry exactly rounds 0..{want_rounds - 1} "
                f"in order, once; it holds {got_rounds[:6]}"
                f"{'...' if len(got_rounds) > 6 else ''}"
                f" -- {len(absent)} round(s) missing"
                f"{' (' + str(absent[:8]) + ')' if absent else ''}"
                f"{', ' + str(len(surplus)) + ' out-of-range' if surplus else ''}"
                f"{', ' + str(dup) + ' duplicated' if dup else ''}"
                f"; a round without a logged parse rate is an unverified "
                f"round")
        nz = []
        for r in rows:
            v = r.get("parse_fail_frac", None)
            try:
                ok = isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and float(v) == 0.0
            except (TypeError, ValueError):
                ok = False
            if not ok:
                nz.append(r)
        if nz:
            bad(f"parse failures in {len(nz)} round(s), e.g. round "
                f"{nz[0].get('round')} at parse_fail_frac="
                f"{nz[0].get('parse_fail_frac')!r} (must be exactly 0 in "
                f"every round; an absent field counts as a failure)")
        short = [r for r in rows if len(r.get("parsed") or []) != N]
        if short:
            bad(f"round {short[0].get('round')} parsed "
                f"{len(short[0].get('parsed') or [])} of {N} agents")
        # WELL-FORMED generations (2026-08-25, after the Section-3 wave):
        # parse_fail_frac only counts digit-free strings. The legacy parser
        # read Mistral-7B's ".64 (" as 64 -> clamp 1.0 and "58 (58" as 1.0
        # with NO failure flagged. Every raw string must therefore START
        # with a well-formed number in [0,1] (leading-dot allowed), and the
        # value the run served (parsed[i]) must equal that number.
        malformed, mismatched, total = [], [], 0
        for r in rows:
            raws, parsed = r.get("raw") or [], r.get("parsed") or []
            if len(raws) != len(parsed):
                bad(f"round {r.get('round')} logs {len(raws)} raw strings "
                    f"but {len(parsed)} parsed values")
                continue
            for i, (txt, pv) in enumerate(zip(raws, parsed)):
                total += 1
                m = _WELL_FORMED_RE.match(txt or "")
                v = float(m.group(1)) if m else None
                if v is None or not 0.0 <= v <= 1.0:
                    malformed.append((r.get("round"), i, str(txt)[:20]))
                elif abs(float(pv) - v) > 1e-6:
                    mismatched.append((r.get("round"), i, str(txt)[:20],
                                       float(pv)))
        rec["generations"] = {"total": total, "malformed": len(malformed),
                              "mismatched": len(mismatched)}
        if malformed:
            bad(f"{len(malformed)}/{total} generation(s) are not a "
                f"well-formed number in [0,1] at the start of the string, "
                f"e.g. {malformed[:3]} -- the served value is not what the "
                f"model wrote")
        if mismatched:
            bad(f"{len(mismatched)}/{total} served value(s) differ from the "
                f"number the model wrote, e.g. {mismatched[:3]}")

    del d, op, pr, tw, inn
    return rec


# ------------------------------------------------------------------ main
def _fmt_cell(k):
    """(arm, cond, ea, es, seed[, horizon]) -> 'b0/fixed/ea0p1/es0p3/s0'."""
    s = f"{k[0]}/{k[1]}/ea{_num(k[2])}/es{_num(k[3])}/s{k[4]}"
    if len(k) > 5 and k[5] is not None:
        s += f"/r{k[5]}"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Section-4 corrected-gate waves (pofds4g_) gate; CPU only")
    ap.add_argument("--wave", default=WAVE_V1, choices=WAVE_CHOICES,
                    help=f"which wave's grid to gate: {WAVE_V1} (alias v1; "
                         f"the original 72-cell wave, DEFAULT), "
                         f"{WAVE_FIG6} (alias fig6; the 192-cell Figure-6 "
                         f"grid with twin-derived ea=0 cells, witnesses and "
                         f"_r60/_r100 extensions) or {WAVE_PROBE} (alias "
                         f"probe; the 8-cell 5-round beta=0.75 channel "
                         f"probe, pofds4gp_ tags)")
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT,
                    help=f"directory holding the run dirs (default the "
                         f"cluster path {DEFAULT_RUN_ROOT})")
    ap.add_argument("--smoke", action="store_true",
                    help=f"gate the 3-round {SMOKE_PREFIX}_ cells of the "
                         f"selected wave (4 jobs: both arms x both "
                         f"conditions at the wave's smoke gate, seed 0) "
                         f"under --run-root")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated replication seeds: gate ONLY "
                         "those seeds' cells (the seed-staged release, e.g. "
                         "--seeds 0 before seeds 42,43 are submitted). Twin-"
                         "derived and extension cells are restricted the "
                         "same way; other seeds' run dirs are ignored, not "
                         "EXTRA. Not a full-wave verdict.")
    ap.add_argument("--tags-file", default=None,
                    help="file of tags (one per line, # comments allowed) to "
                         "gate INSTEAD of the full product; coverage is then "
                         "checked against that list")
    ap.add_argument("--inspect-archived", action="store_true",
                    help="NOT A GATE. Downgrade a MISSING raw_gen_log.json.gz "
                         "from a hard failure to a loud WARN so the four "
                         "archived pre-fix smoke runs (notes/pofd/cluster/"
                         "pofds4gsmk_*, which predate SAVE_RAW_GEN=1) can be "
                         "looked at. Every other check stays strict; the "
                         "verdict is labelled NOT A GATE and must not be "
                         "cited as a pass")
    ap.add_argument("--gen", default=None,
                    help="path of gen_pofd_sweep.py (default: found relative "
                         "to this file, then the cwd and its ancestors)")
    ap.add_argument("--ext-manifest", default=None,
                    help="fig6 only: override the committed "
                         "section4_fig6_extension_request.json the "
                         "generator's s4g2_ext_requests() reads")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the machine-readable verdict here")
    args = ap.parse_args(argv)

    root = Path(args.run_root)
    if not root.is_dir():
        print(f"{LOG} usage error: --run-root {args.run_root!r} is not a "
              f"directory", file=sys.stderr)
        return 2
    # ---- the wave, READ FROM THE GENERATOR -----------------------------
    gen_path = find_generator(args.gen)
    if gen_path is None:
        print(f"{LOG} usage error: gen_pofd_sweep.py not found "
              f"({'--gen ' + repr(args.gen) if args.gen else GEN_REL + ' relative to this file / the cwd'}); "
              f"the grid is READ from the generator, never restated here",
              file=sys.stderr)
        return 2
    try:
        gen = load_generator(gen_path)
        wave = Wave(args.wave, gen, ext_manifest=args.ext_manifest)
    except Exception as e:                      # noqa: BLE001
        print(f"{LOG} usage error: cannot build the {args.wave!r} grid from "
              f"{gen_path}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if args.ext_manifest and not wave.fig6:
        print(f"{LOG} usage error: --ext-manifest applies to --wave fig6 "
              f"only", file=sys.stderr)
        return 2
    scan = SMOKE_SCAN if args.smoke else wave.prod_scan
    if args.smoke and not wave.smoke_cells:
        print(f"{LOG} usage error: the {wave.name} wave defines no smoke "
              f"cells (the 5-round probe IS the probe run)", file=sys.stderr)
        return 2
    gbad = wave.self_test_grammar(args.smoke)
    if gbad:
        print(f"{LOG} usage error: this checker's tag grammar disagrees with "
              f"gen_pofd_sweep.s4g_tag on {len(gbad)} expected cell(s), e.g. "
              f"{gbad[0][0]!r}: {gbad[0][1][0]}", file=sys.stderr)
        return 2

    # ---- what we EXPECT (keys are (arm, cond, ea, es, seed, horizon)) -----
    if args.smoke:
        run_keys = [c + (None,) for c in wave.smoke_cells]
        twin_keys, ext_keys = [], []
    else:
        run_keys = [c + (None,) for c in wave.run_cells()]
        twin_keys = [c + (None,) for c in wave.twin_cells()]
        ext_keys = list(wave.ext_requests)
    seed_subset = None
    if args.seeds:
        seed_subset = sorted({int(x) for x in args.seeds.split(",") if x})
        run_keys = [k for k in run_keys if k[4] in seed_subset]
        twin_keys = [k for k in twin_keys if k[4] in seed_subset]
        ext_keys = [k for k in ext_keys if k[4] in seed_subset]
        if not run_keys:
            ap.error(f"--seeds {args.seeds}: no cells of this wave use "
                     f"those seeds")
    out = []
    if args.tags_file:
        try:
            raw = Path(args.tags_file).read_text().splitlines()
        except OSError as e:
            print(f"{LOG} usage error: --tags-file {args.tags_file!r}: {e}",
                  file=sys.stderr)
            return 2
        wanted = [ln.strip() for ln in raw
                  if ln.strip() and not ln.strip().startswith("#")]
        if not wanted:
            print(f"{LOG} usage error: --tags-file {args.tags_file!r} holds "
                  f"no tags. Nothing to gate is NOT a pass.", file=sys.stderr)
            return 2
        keep, keep_tags = set(), []
        for t in wanted:
            info, terrs = parse_tag(t, args.smoke, wave)
            if info is None or terrs:
                for e in (terrs or ["unparseable"]):
                    out.append(f"FAIL --tags-file {t}: {e}")
                continue
            keep.add(info["key"])
            keep_tags.append(t)
        # the tags file REPLACES the product as the coverage target, so a
        # deliberately partial pull can be gated without the full-grid
        # completeness check turning every absent cell into noise; a
        # tags file names run dirs, so no twin-derived cell is expected
        run_keys = [k for k in run_keys if k in keep]
        ext_keys = [k for k in ext_keys if k in keep]
        twin_keys = []
        run_dirs = [str(root / t) for t in keep_tags]
    elif args.smoke or seed_subset is not None:
        # Smoke: the two corrected-gate waves share the pofds4gsmk_ prefix
        # (the S4G smoke at ea=1/es=0.2 sits beside the fig6 smoke at
        # ea=0.1/es=0.3 under the same run root). Seed subset: the other
        # seeds' dirs may exist. Either way a prefix scan would report
        # foreign dirs as EXTRA, so gate exactly the selected tags: an
        # absent one is a FAIL, a foreign one is ignored. Full production
        # mode keeps the prefix scan (pofds4g_ never matches pofds4gsmk_).
        # The two corrected-gate waves share the pofds4gsmk_ prefix (the
        # S4G smoke at ea=1/es=0.2 sits beside the fig6 smoke at
        # ea=0.1/es=0.3 under the same run root), so a prefix scan would
        # report the OTHER wave's smoke dirs as EXTRA. Smoke mode gates
        # exactly the selected wave's smoke tags: an absent one is a FAIL,
        # a foreign one is ignored. Production mode keeps the prefix scan
        # (pofds4g_ never matches pofds4gsmk_).
        run_dirs = sorted(str(root / wave.render_tag(*k[:5], smoke=args.smoke,
                                                     rounds=k[5]))
                          for k in run_keys + ext_keys)
    else:
        run_dirs = sorted(str(p) for p in root.iterdir()
                          if p.is_dir() and p.name.startswith(scan))
    if not run_dirs:
        for line in out:
            print(f"{LOG} {line}")
        if out:
            print(f"{LOG} FAILED -- every requested tag was rejected before "
                  f"it could be opened.")
            return 1
        print(f"{LOG} usage error: no {scan}* run dirs under {root}. "
              f"Nothing to gate is NOT a pass.", file=sys.stderr)
        return 2
    tag_of_key = {}
    for k in run_keys + twin_keys + ext_keys:
        tag_of_key[k] = wave.render_tag(*k[:5], smoke=args.smoke, rounds=k[5])
    expected = set(run_keys) | set(ext_keys)
    witness_keys = [k for k in run_keys if wave.is_witness(*k[:5])]

    # ---- gate each run, one trajectory in memory at a time -------------
    recs = []
    for rd in run_dirs:
        if not os.path.isdir(rd):
            out.append(f"FAIL {os.path.basename(rd)}: no such run dir under "
                       f"{root}")
            r = _empty_rec(rd, os.path.basename(rd))
            r["errs"].append("run dir does not exist")
            recs.append(r)
            continue
        recs.append(check_one(rd, wave, args.smoke, args.inspect_archived))
    for r in recs:
        for e in r["errs"]:
            out.append(f"FAIL {r['tag']}: {e}")
        for w in r["warns"]:
            out.append(f"WARN {r['tag']}: {w}")
        for n in r["notes"]:
            out.append(f"NOTE {r['tag']}: {n}")

    # ---- 1. coverage ---------------------------------------------------
    by_key, dupes = {}, []
    for r in recs:
        if r["cell"] is None:
            continue
        k = tuple(r["cell"]) + (r["horizon"],)
        if k in by_key:
            dupes.append((k, by_key[k]["tag"], r["tag"]))
        by_key[k] = r
    for k, a, b in dupes:
        out.append(f"FAIL wave: two run dirs claim the same conceptual cell "
                   f"{_fmt_cell(k)}: {a} and {b}")
    extra = sorted(k for k in by_key if k not in expected)
    for k in extra:
        why = ("an ea=0 cell that is TWIN-DERIVED in this wave (no run is "
               "expected there)" if k in set(twin_keys) else
               "an extension the committed manifest does not request"
               if k[5] is not None else "NOT in the expected grid")
        out.append(f"FAIL wave: {by_key[k]['tag']} parses to cell "
                   f"{_fmt_cell(k)}, which is {why}")
    missing = [k for k in run_keys if k not in by_key]
    n_run_present = len(run_keys) - len(missing)

    # ---- 13. extensions -------------------------------------------------
    ext_pending = [k for k in ext_keys if k not in by_key]
    ext_present = [k for k in ext_keys if k in by_key]
    ext_fail = 0
    for k in ext_present:
        other = [c for c in wave.conds if c != k[1]]
        partner = (k[0], other[0], k[2], k[3], k[4], k[5]) if other else None
        if partner is not None and partner not in by_key:
            ptag = wave.render_tag(*partner[:5], smoke=args.smoke,
                                   rounds=partner[5])
            out.append(f"FAIL ext {by_key[k]['tag']}: its {other[0]} partner "
                       f"{ptag} is absent -- extensions come as matched "
                       f"fixed/evolving pairs (the paired-seed T_a must stay "
                       f"paired)")
            ext_fail += 1
    for k in ext_pending:
        out.append(f"NOTE ext {tag_of_key[k]}: PENDING-EXT (requested at "
                   f"{k[5]} rounds, not yet run; non-failing)")

    # ---- 5. cohort pairing + one-world ---------------------------------
    pair_fail = 0
    pairs = sorted({(k[0], k[2], k[3], k[4], k[5]) for k in run_keys + ext_keys},
                   key=lambda p: (p[0], p[1], p[2], p[3], p[4] or 0))
    for arm, ea, es, seed, hz in pairs:
        fx = by_key.get((arm, "fixed", ea, es, seed, hz))
        ev = by_key.get((arm, "evolving", ea, es, seed, hz))
        if fx is None or ev is None:
            continue                      # already a coverage failure
        if fx["innate_sha256"] is None or ev["innate_sha256"] is None:
            continue                      # already a per-cell failure
        pname = _fmt_cell((arm, "*", ea, es, seed, hz)).replace("/*", "")
        if fx["innate_sha256"] != ev["innate_sha256"]:
            out.append(
                f"FAIL pair {pname}: the "
                f"fixed and evolving members sit on DIFFERENT innate vectors "
                f"({fx['innate_sha256'][:16]}... vs "
                f"{ev['innate_sha256'][:16]}...) -- {fx['tag']} vs "
                f"{ev['tag']}. The comparison is between one population's "
                f"cohort A being pinned and evolving; two populations make "
                f"it meaningless")
            pair_fail += 1
        elif fx["cohort_sha256"] != ev["cohort_sha256"]:
            out.append(
                f"FAIL pair {pname}: same "
                f"innate but DIFFERENT reconstructed bottom-{CLAMP_COUNT} "
                f"cohort ({fx['cohort_sha256'][:16]}... vs "
                f"{ev['cohort_sha256'][:16]}...)")
            pair_fail += 1
    inn_shas = {r["innate_sha256"] for r in recs
                if r["innate_sha256"] is not None}
    if len(inn_shas) > 1:
        out.append(
            f"FAIL wave: {len(inn_shas)} distinct innate vectors across the "
            f"grid. load_movielens_setup is a pure function of (dataset, "
            f"target) here -- PROFILE_SHUFFLE_P=0 and no routing treatment -- "
            f"so innate does not even depend on the run seed. A difference "
            f"means a different agent set or a different agent ORDER.")
        for r in recs:
            if r["innate_sha256"]:
                out.append(f"     {r['tag']}: {r['innate_sha256'][:16]}...")

    # ---- 6. TWIN AGREEMENT + 12. twin-derived cells ---------------------
    # The twin (ab_x_cf) is advanced by the anchor step and the peer sweep
    # under its own mirrored generator; the served vector never enters
    # it. So every run at one (cond, es, seed) -- any arm, any eps_AI, any
    # horizon over the first S4G_ROUNDS rows -- must carry a BIT-IDENTICAL
    # twin_raw, and that identity is what lets an ea=0 cell be drawn from
    # its neighbours' twin_raw without a run of its own.
    twin_fail = 0
    tw_base, tw_full = {}, {}
    for r in recs:
        if r["cell"] is None or r["twin_base_sha256"] is None:
            continue
        c = r["cell"]
        tw_base.setdefault((c[1], c[3], c[4]), {}).setdefault(
            r["twin_base_sha256"], []).append(r)
        tw_full.setdefault((c[1], c[3], c[4], r["horizon"]), {}).setdefault(
            r["twin_sha256"], []).append(r)
    twin_bad_groups = set()

    def _twin_disagree(g, by_sha, what):
        nonlocal twin_fail
        twin_fail += 1
        twin_bad_groups.add(g[:3])
        members = "; ".join(
            f"{sha[:12]}...: " + ", ".join(
                f"{m['tag']}" + (f" [{m['gpu_name']}]" if m['gpu_name'] else "")
                for m in ms) for sha, ms in sorted(by_sha.items()))
        out.append(
            f"FAIL twin {g[0]}/es{_num(g[1])}/s{g[2]}"
            f"{'' if len(g) < 4 or g[3] is None else '/r' + str(g[3])}: "
            f"{len(by_sha)} distinct twin_raw ({what}) among the runs at "
            f"this (cond, es, seed) -- {members}. The twin is a pure "
            f"function of (cond, es, seed): the AI channel cannot reach it, "
            f"so a disagreement means the runs did not share one world (or "
            f"one generator), and no ea=0 cell can be drawn from them")

    for g, by_sha in sorted(tw_base.items(), key=str):
        if len(by_sha) > 1:
            _twin_disagree(g, by_sha, f"over the first {wave.rounds} rounds")
    for g, by_sha in sorted(tw_full.items(), key=str):
        if g[3] is None or len(by_sha) < 2:
            continue
        _twin_disagree(g, by_sha, f"over all {g[3]} rounds")
    twin_rows = []                        # (key, ok, reason, sample rec)
    for k in twin_keys:
        g = (k[1], k[3], k[4])
        by_sha = tw_base.get(g)
        if not by_sha:
            twin_rows.append((k, False, "no run exists at this (cond, es, "
                              "seed) -- the twin-derived cell cannot be "
                              "drawn", None))
        elif g in twin_bad_groups or len(by_sha) > 1:
            twin_rows.append((k, False, "the runs at this (cond, es, seed) "
                              "disagree on twin_raw (see FAIL twin above)",
                              None))
        else:
            twin_rows.append((k, True, None, next(iter(by_sha.values()))[0]))
    twin_cell_fail = sum(1 for t in twin_rows if not t[1])
    for k, ok, why, _ in twin_rows:
        if not ok:
            out.append(f"FAIL twin-derived {tag_of_key[k]}: {why}")
    n_twin_ok = len(twin_rows) - twin_cell_fail

    # ---- 5b. SEED-DISTINCTNESS (WARN only, never the exit code) ---------
    # innate, the 10-NN graph and the bottom-145 cohort are SEED-INVARIANT
    # for movielens (load_movielens_setup takes no seed;
    # gp.innate_clamp_mask("bottom") is a pure sort with no RNG draw), so
    # config["seed"] -- written from the environment, never observed -- is
    # the only per-run evidence that a seed reached the runner. The
    # BEHAVIOURAL evidence is whether two seeds of one cell produce the
    # same trajectory. Compared by the op_raw sha256 taken in check_one,
    # so no two runs' tensors are ever resident at once. A collision is a
    # WARNING for the analyst, not a failure: greedy serving on a
    # quantized value grid can legitimately give bit-identical outcomes
    # across seeds.
    #   EXEMPT: d8 at es=0, the structural null -- frozen weights, greedy
    # decoding, own-history prompts and an inert strict-< peer step leave
    # NO random draw on the path to the population, so identical
    # trajectories across seeds are EXPECTED there (see
    # analyze_section4_gate.build_null_rows).
    seed_warn = 0
    seed_skipped = []
    seed_groups = {}
    for k, r in by_key.items():
        if r.get("op_sha256") is None:
            continue                      # already a per-cell failure
        seed_groups.setdefault((k[0], k[1], k[2], k[3], k[5]), []).append(
            (k[4], r["tag"], r["op_sha256"]))
    for g, members in sorted(seed_groups.items(), key=str):
        if len(members) < 2:
            # a missing seed is COVERAGE (already fatal above), not a pass
            # or a failure of this check
            continue
        gname = (f"{g[0]}/{g[1]}/ea{_num(g[2])}/es{_num(g[3])}"
                 + ("" if g[4] is None else f"/r{g[4]}"))
        if g[0] == "d8" and g[3] == 0.0:
            seed_skipped.append(gname)
            out.append(
                f"NOTE seed-distinctness {gname}: skipped -- d8 at "
                f"eps_social=0 is the STRUCTURAL NULL (frozen weights, "
                f"greedy decoding, own-history prompts, strict-< peer gate "
                f"never opens): no random draw reaches the population, so "
                f"seeds {[m[0] for m in sorted(members)]} are expected to "
                f"coincide up to GPU nondeterminism and identical op_raw is "
                f"not evidence of a lost seed here")
            continue
        by_sha = {}
        for seed, tag, sha in sorted(members):
            by_sha.setdefault(sha, []).append((seed, tag))
        for sha, hits in by_sha.items():
            if len(hits) < 2:
                continue
            out.append(
                f"WARN seed-distinctness {gname}: seeds {[h[0] for h in hits]} "
                f"produced a BIT-IDENTICAL op_raw (sha256 {sha[:16]}...) -- "
                f"{', '.join(h[1] for h in hits)}. Nothing about this world "
                f"depends on the seed (innate, the 10-NN graph and the "
                f"cohort are seed-invariant), so config['seed'] -- written "
                f"from the environment, not observed -- is the only other "
                f"evidence a seed reached the runner. If it never reached "
                f"the training/serving stream the three-seed intervals "
                f"collapse to ONE observation while every per-run field "
                f"still looks correct -- but greedy serving on a quantized "
                f"value grid can also coincide legitimately, so this is a "
                f"warning for the analyst, not a failure.")
            seed_warn += 1
    # the TWIN'S seed-distinctness: the twin-derived cells' "op_raw" IS
    # the group twin, and the twin is advanced by a seeded generator, so
    # within each (cond, es > 0[, horizon]) no two seeds may share
    # twin_raw. es = 0 is excluded: the strict-< peer gate accepts no pair
    # there, the twin is RNG-free and seed-invariant by construction.
    twin_seed_groups = {}
    for k, r in by_key.items():
        if r.get("twin_sha256") is None or k[3] == 0.0:
            continue
        twin_seed_groups.setdefault((k[1], k[3], k[5]), {}).setdefault(
            k[4], set()).add(r["twin_sha256"])
    for g, per_seed in sorted(twin_seed_groups.items(), key=str):
        if len(per_seed) < 2:
            continue
        seen = {}
        for seed, shas in sorted(per_seed.items()):
            for sha in shas:
                seen.setdefault(sha, []).append(seed)
        for sha, seeds in seen.items():
            seeds = sorted(set(seeds))
            if len(seeds) < 2:
                continue
            out.append(
                f"WARN seed-distinctness(twin) {g[0]}/es{_num(g[1])}"
                f"{'' if g[2] is None else '/r' + str(g[2])}: seeds {seeds} "
                f"share a BIT-IDENTICAL twin_raw (sha256 {sha[:16]}...) -- "
                f"the twin's peer sweep draws from a generator seeded by the "
                f"run seed, so identical twins may mean the seed never "
                f"reached the population generator, in which case the "
                f"twin-derived ea=0 cells of these seeds are ONE observation "
                f"(a warning for the analyst, not a failure)")
            seed_warn += 1

    # ---- print ---------------------------------------------------------
    if args.inspect_archived:
        print(f"{LOG} ***** --inspect-archived: THIS RUN IS NOT A GATE. A "
              f"missing raw_gen_log.json.gz is downgraded to a WARN, so the "
              f"parse rate of such runs is UNVERIFIED; nothing below may be "
              f"cited as a pass. *****")
    for line in out:
        print(f"{LOG} {line}")

    hdr = (f"{'cell':<66} {'verdict':>7} {'rounds':>6} {'popMean':>8} "
           f"{'popSD':>7} {'opTwinL1':>9} {'parse':>13}")
    print("\n" + "=" * len(hdr))
    n_total = len(run_keys) + len(twin_keys)
    n_present = n_run_present + n_twin_ok
    print(f"PER-CELL REPORT -- wave {wave.name}, "
          f"{'SMOKE' if args.smoke else 'PRODUCTION'} grid, "
          f"{n_present}/{n_total} cells present")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    def _row(r, verdict):
        print(f"{r['tag']:<66} {verdict:>7} "
              f"{(r['n_rounds'] if r['n_rounds'] is not None else -1):>6} "
              f"{(r['pop_final_mean'] if r['pop_final_mean'] is not None else float('nan')):>8.4f} "
              f"{(r['pop_final_sd'] if r['pop_final_sd'] is not None else float('nan')):>7.4f} "
              f"{(r['op_twin_l1'] if r['op_twin_l1'] is not None else float('nan')):>9.4f} "
              f"{str(r['parse_evidence']):>13}")

    for k in run_keys:
        r = by_key.get(k)
        name = tag_of_key[k]
        if r is None:
            print(f"{name:<66} {'ABSENT':>7} {'-':>6} {'-':>8} {'-':>7} "
                  f"{'-':>9} {'-':>13}")
            continue
        _row(r, "PASS" if not r["errs"] else "FAIL")
    for k, ok, why, src in twin_rows:
        name = tag_of_key[k]
        if src is None:
            print(f"{name:<66} {'FAIL':>7} {'-':>6} {'-':>8} {'-':>7} "
                  f"{'-':>9} {'twin-derived':>13}")
        else:
            print(f"{name:<66} {'PASS' if ok else 'FAIL':>7} "
                  f"{wave.rounds:>6} {src['twin_final_mean']:>8.4f} "
                  f"{src['twin_final_sd']:>7.4f} {0.0:>9.4f} "
                  f"{'twin-derived':>13}")
    for k in ext_keys:
        r = by_key.get(k)
        name = tag_of_key[k]
        if r is None:
            print(f"{name:<66} {'PENDING':>7} {k[5]:>6} {'-':>8} {'-':>7} "
                  f"{'-':>9} {'PENDING-EXT':>13}")
            continue
        _row(r, "PASS" if not r["errs"] else "FAIL")
    for k in extra:
        r = by_key[k]
        print(f"{r['tag']:<66} {'EXTRA':>7} {'-':>6} {'-':>8} {'-':>7} "
              f"{'-':>9} {str(r['parse_evidence']):>13}")

    unverified = [r["tag"] for r in recs
                  if r["parse_evidence"] == "missing(archived)"]
    if unverified:
        print(f"\n{LOG} PARSE RATE UNVERIFIED (--inspect-archived, NOT A "
              f"GATE): {len(unverified)} run(s) have no raw_gen_log.json.gz. "
              f"parse_fail_frac lives nowhere else, and the parser stores a "
              f"FINITE 0.5 when a generation does not parse, so nothing in "
              f"trajectory.pt can reveal a parse failure for these runs: "
              f"{', '.join(unverified)}")

    print("\n" + "=" * len(hdr))
    twin_missing = [k for k, ok, _, _ in twin_rows if not ok]
    if missing or twin_missing:
        print(f"GRID COMPLETENESS: {n_present} of {n_total} cells present -- "
              f"{len(missing) + len(twin_missing)} ABSENT. A silently short "
              f"grid must not look like a complete result.")
        for k in missing:
            print(f"  ABSENT  arm={k[0]:<3} cond={k[1]:<8} ea={k[2]:<4g} "
                  f"es={k[3]:<4g} seed={k[4]:<3} expected tag "
                  f"{tag_of_key[k]}")
        for k in twin_missing:
            print(f"  ABSENT  arm={k[0]:<3} cond={k[1]:<8} ea={k[2]:<4g} "
                  f"es={k[3]:<4g} seed={k[4]:<3} twin-derived cell "
                  f"{tag_of_key[k]} (undrawable)")
    else:
        print(f"GRID COMPLETENESS: all {n_total} cells present"
              + (f" ({len(run_keys)} run + {len(twin_keys)} twin-derived)"
                 if twin_keys else ""))
    if wave.fig6 and not args.smoke:
        n_w_ok = sum(1 for k in witness_keys
                     if k in by_key and not by_key[k]["errs"])
        print(f"WITNESSES: {n_w_ok}/{len(witness_keys)} ea=0 witness cell(s) "
              f"pass (op_raw == twin_raw bit-exact, contact 0 every round)")
        print(f"EXTENSIONS: {len(ext_present)} present, {len(ext_pending)} "
              f"PENDING-EXT, {ext_fail} unpaired"
              + (f" (manifest {wave.ext_manifest})"
                 if ext_keys else " (no extension manifest / no requests)"))
    print("=" * len(hdr))

    n_fail_cells = sum(1 for r in recs if r["errs"])
    warnings = [l for l in out if l.startswith("WARN")]
    # WARN lines (seed-distinctness, missing raw log under
    # --inspect-archived) never reach the exit code
    allok = (n_fail_cells == 0 and not missing and not dupes and not extra
             and pair_fail == 0 and len(inn_shas) <= 1
             and twin_fail == 0 and twin_cell_fail == 0 and ext_fail == 0
             and not any(l.startswith("FAIL") for l in out))
    n_witness_ok = sum(1 for k in witness_keys
                       if k in by_key and not by_key[k]["errs"])

    def _cell_json(k):
        return {"arm": k[0], "cond": k[1], "eps_ai": k[2],
                "eps_social": k[3], "seed": k[4], "horizon": k[5],
                "tag": tag_of_key.get(k)}

    verdict = {
        "wave": wave.name,
        "seed_subset": seed_subset,
        "generator": gen_path,
        "smoke": bool(args.smoke),
        "run_root": str(root),
        "operator_required": WANT_MARKER,
        "ai_gate_reference_required": WANT_GATE_REF,
        "n_runs": len(recs),
        "n_cells_present": n_present,
        "n_cells_total": n_total,
        "n_run_cells_present": n_run_present,
        "n_run_cells_total": len(run_keys),
        "n_cells_failed": n_fail_cells,
        "n_pair_failures": pair_fail,
        "inspect_archived": bool(args.inspect_archived),
        "not_a_gate": bool(args.inspect_archived),
        "warnings": warnings,
        "n_warnings": len(warnings),
        "n_seed_distinctness_warnings": seed_warn,
        "seed_distinctness_skipped_structural_null": seed_skipped,
        "n_twin_agreement_failures": twin_fail,
        "n_twin_cells_total": len(twin_keys),
        "n_twin_cells_ok": n_twin_ok,
        "twin_cells": [dict(_cell_json(k), ok=ok, reason=why,
                            twin_sha256=(src["twin_base_sha256"]
                                         if src else None))
                       for k, ok, why, src in twin_rows],
        "n_witness_cells_total": len(witness_keys),
        "n_witness_cells_ok": n_witness_ok,
        "witness_cells": [dict(_cell_json(k),
                               ok=(k in by_key and not by_key[k]["errs"]),
                               present=k in by_key) for k in witness_keys],
        "extensions": [dict(_cell_json(k), rounds=k[5],
                            status=("PENDING-EXT" if k not in by_key else
                                    "PASS" if not by_key[k]["errs"]
                                    else "FAIL")) for k in ext_keys],
        "n_ext_present": len(ext_present),
        "n_ext_pending": len(ext_pending),
        "n_ext_unpaired": ext_fail,
        "missing": [{"arm": k[0], "cond": k[1], "eps_ai": k[2],
                     "eps_social": k[3], "seed": k[4],
                     "expected_tag": tag_of_key[k]} for k in missing],
        "duplicate_cells": [{"cell": list(k), "tags": [a, b]}
                            for k, a, b in dupes],
        "unexpected_cells": [{"cell": list(k), "tag": by_key[k]["tag"]}
                             for k in extra],
        "innate_sha256_distinct": sorted(inn_shas),
        "parse_rate_unverified": unverified,
        "pass": bool(allok),
        "cells": [{"tag": r["tag"], "run_dir": r["run_dir"],
                   "cell": list(r["cell"]) if r["cell"] else None,
                   "horizon": r["horizon"],
                   "witness": r["witness"],
                   "ok": not r["errs"], "errors": r["errs"],
                   "warnings": r["warns"],
                   "notes": r["notes"],
                   "parse_evidence": r["parse_evidence"],
                   "n_rounds": r["n_rounds"],
                   "innate_sha256": r["innate_sha256"],
                   "cohort_sha256": r["cohort_sha256"],
                   "pop_final_mean": r["pop_final_mean"],
                   "pop_final_sd": r["pop_final_sd"],
                   "op_twin_l1": r["op_twin_l1"],
                   "op_sha256": r["op_sha256"],
                   "twin_sha256": r["twin_sha256"],
                   "twin_base_sha256": r["twin_base_sha256"],
                   "contact_max": r["contact_max"],
                   "gpu_name": r["gpu_name"],
                   "d8_replay": r["d8_replay"]} for r in recs],
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(verdict, indent=2))
        print(f"{LOG} verdict -> {args.json_out}")

    warn_bit = (f"; {len(warnings)} WARN line(s) ({seed_warn} "
                f"seed-distinctness, {len(unverified)} parse rate "
                f"unverified) -- warnings never change the exit code")
    if allok:
        fig6_bit = ""
        if wave.fig6 and not args.smoke:
            fig6_bit = (f" ({n_run_present} run + {n_twin_ok} twin-derived; "
                        f"{n_witness_ok}/{len(witness_keys)} witness cell(s) "
                        f"op_raw == twin_raw bit-exact with contact 0 every "
                        f"round; {len(ext_present)} extension(s) present, "
                        f"{len(ext_pending)} PENDING-EXT)")
        head = ("PASS (--inspect-archived: NOT A GATE, parse rate "
                "unverified where WARNed)" if args.inspect_archived
                else "PASS")
        parse_bit = ("zero parse failures in every logged round"
                     if not unverified else
                     f"parse rate UNVERIFIED for {len(unverified)} run(s)")
        print(f"{LOG} {head} -- wave {wave.name}: {len(recs)} run(s), "
              f"{n_present}/{n_total} cells{fig6_bit}: every tag carries "
              f"{OP_INFIX!r} and every config the "
              f"{WANT_MARKER} operator at ai_gate_reference=anchor; grid "
              f"fields agree with the tags in both directions; the "
              f"bottom-{CLAMP_COUNT} cohort reconstructs and is bit-exact in "
              f"population and twin on every fixed cell; every evolving cell "
              f"is clamp-free; each fixed/evolving pair shares one innate "
              f"vector and one cohort; the twin agrees bit-exactly across "
              f"every run of a (cond, es, seed); d8 personal histories "
              f"replay byte-exactly (locality); twins are non-degenerate; "
              f"{parse_bit}; seed-distinctness reviewed{warn_bit}.")
        return 0
    print(f"{LOG} FAILED -- wave {wave.name}: {n_fail_cells} of {len(recs)} "
          f"run(s) failed, {len(missing)} cell(s) absent, {len(dupes)} "
          f"duplicate, {len(extra)} unexpected, {pair_fail} pair "
          f"mismatch(es), {twin_fail} twin disagreement(s), {twin_cell_fail} "
          f"undrawable twin-derived cell(s), {ext_fail} unpaired "
          f"extension(s){warn_bit}. See the FAIL lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
