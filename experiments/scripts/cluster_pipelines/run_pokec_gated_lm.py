"""Pokec gated-population + LM run: FJ or Deffuant-AB population, platform
telemetry, and control modes (sibling of run_pokec_fj_lm.py).

POP_MODEL=fj reproduces run_pokec_fj_lm.py behavior. POP_MODEL=ab swaps the
population for the torch Deffuant-with-bias sweep of
experiments/competition/16_model_mass.py, restricted to Pokec graph edges:
per round one sweep (N biased pair selections among graph neighbors, weight
|x_i-x_j|^-GAMMA_BIAS, accept if |x_i-x_j| < EPS), then a gated platform
blend x_i <- (1-w_i) x_i + w_i m_i only where |m_i - x_i| < EPS, with
w_i = W_PLAT * platform_sus_i * PLATFORM_SUS_SCALE. Opinions are already on
the AB scale: innate, model predictions, and all updates live in [0, 1].

RUN_MODE: loop = full feedback; no_feedback = platform weight forced to 0
(population evolves alone, model still trains: thermometer control); direct =
no population at all, the model trains on its OWN previous round predictions
(the Shumailov corner; op_raw rows then duplicate the served predictions so
downstream tooling keeps working). CANARY_DELTA>0 adds a fixed seeded
per-agent +/-delta pattern to served predictions before the gated blend
(ab loop only; pattern saved to canary_pattern.pt).

Platform-seat telemetry (mirrors experiments/competition/18_platform_telemetry.py),
appended one JSON line per round to telemetry.json (crash-safe):
  l_init      SFT completion CE of the current adapter on this round's batch,
              BEFORE the gradient steps, + batch_var (target variance)
  grad_norm0  trainable-grad norm of the CE objective at step 0 (no KL term)
  grad_kl_norm0 grad_ratio0 grad_cos0   (sft_kl runs, GRAD_DECOMP=1 default)
              step-0 norm of the beta*KL gradient on the same probe rows,
              ratio beta*|g_KL|/|g_CE|, and cos(g_CE, g_KL): the Q6 channel
              decomposition (domination = ratio >> 1 at negative cos;
              alignment = cos rotating positive as the corpus goes easy)
  probe_pred  predictions on a fixed innate-stratified probe set of N_PROBE
              real agents (chosen at round 0, saved to probe_set.json);
              displacement / leash are computed offline from these
  l_cc l_c0 l_0c l_00   the 2x2: {current, round-0} adapter x {current batch,
              round-0 archive batch}; round-0 adapter + batch kept on disk
              (round0_adapter/, round0_batch.pt)
  gate_raw    trajectory.pt, [rounds, n] bool: the per-agent AI gate/contact
              mask (|m_i - x_i(t)| < eps_AI on the start-of-round opinion) --
              direct platform contact per agent per round, so direct-contact
              vs peer-transmitted effects separate offline. Empty tensor in
              runs written before 2026-08-05.
  icl_idx_raw / icl_val_raw / icl_ctx_log.json.gz   ICL_K>0 runs (2026-08-07):
              per round the [n, K] exemplar agent ids and displayed opinion
              values each agent's context carried, plus the rendered exemplar
              text blocks (gzip JSONL, one line per round). With
              ICL_SNAPSHOT_ROUND=R >= 0 the context is built dynamically
              through round R and then FROZEN -- every later round replays
              each agent's round-R context verbatim (ids, order, profiles,
              displayed values; no RNG drawn on frozen rounds), so freeze-R
              and dynamic twins are bit-identical through round R.
              ICL_CTX_SOURCE=donor (2026-08-07): exemplar VALUES come from a
              donor run's saved population (ICL_CTX_DONOR dir +
              ICL_CTX_DONOR_ROUND; -1 = donor innate). config.json records
              the donor tag/round/sha256; trajectory.pt stores the full
              donor vector (icl_donor_vec) plus donor gg / incremental-R^2
              diagnostics and the realized displayed-context gap
              (icl_ctx_gg, also per-row as gg_ctx_true). Identities still
              come from the recipient's selection stream; the recipient
              population never starts from donor opinions.
  replay_raw / train_y_raw   trajectory.pt, REPLAY_FRAC>0 runs only
              (exact initial-data replay, replace regime): per deploy round
              the [n_labeled] bool label-source mask (True = row kept its
              ORIGINAL round-0 label, exactly round(REPLAY_FRAC*n) rows;
              round 0 all True) and the ACTUAL labels fed to the learner, so
              batch composition is verifiable end-to-end offline. Empty
              tensors everywhere else.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _wandb = None
    _HAS_WANDB = False

from perfsim.core.learner import Learner
from perfsim.core.types import SUPERVISED_SCHEMA
from perfsim.environments.dynamics import FJWorld, normalize_adjacency
from perfsim.learners.lm.dpo_bank import bank_from_env, state_digest
from perfsim.learners.lm.kl_sft import KLSFTLearner
from perfsim.learners.lm.sft import SFTLearner
from perfsim.losses import MSELoss
from perfsim.models.hf_causal_lm import HFCausalLMModel
from perfsim.simulator import Simulator  # noqa: F401  (kept: schema types live here)

_CM_PATH = Path(__file__).resolve().parent / "_collapse_metrics.py"
_spec = importlib.util.spec_from_file_location("_collapse_metrics", _CM_PATH)
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

# Population + telemetry helpers live in _gated_pop.py (transformers-free) so
# the laptop mock test (_mock_gated_test.py) can import them without an LLM.
_GP_PATH = Path(__file__).resolve().parent / "_gated_pop.py"
_spec_gp = importlib.util.spec_from_file_location("_gated_pop", _GP_PATH)
gp = importlib.util.module_from_spec(_spec_gp)
_spec_gp.loader.exec_module(gp)


def _env_or(name, default=None):
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required env var {name!r} not set")
    return val


def _env_int(name, default): return int(os.environ.get(name, str(default)))
def _env_float(name, default): return float(os.environ.get(name, str(default)))


def _hardware_meta():
    """Hardware/software provenance for config.json (2026-08-13): hostname,
    GPU model + compute capability, CUDA/torch/transformers versions. Pure
    metadata -- nothing in the simulation path reads it."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cc = "%d.%d" % torch.cuda.get_device_capability(0)
    else:
        gpu, cc = "cpu", ""
    try:
        import transformers
        tf_ver = transformers.__version__
    except ImportError:
        tf_ver = ""
    return {"hostname": os.uname().nodename, "gpu_name": gpu, "gpu_cc": cc,
            "cuda_version": torch.version.cuda or "",
            "torch_version": torch.__version__,
            "transformers_version": tf_ver}


def _wandb_hist(wb, values, bins):
    """wandb.Histogram with fixed [0,1] bins so shapes are comparable across rounds."""
    counts, edges = np.histogram(values.detach().cpu().numpy(), bins=bins, range=(0.0, 1.0))
    return wb.Histogram(np_histogram=(counts.tolist(), edges.tolist()))


# Byte-identical to Opinion-dynamics-post-training/llm_predictor.py so prompts
# (and therefore predictions) reproduce the original study.
PROMPT_COLS = ["age", "gender", "relation_to_alcohol"]

SK_ALCOHOL_EXACT = {
    "pijem prilezitostne": "I drink occasionally",
    "abstinent": "I abstain from alcohol",
    "uz nepijem": "I no longer drink",
    "nepijem": "I don't drink",
    "pijem pravidelne": "I drink regularly",
    "prilezitostne": "occasionally",
    "pijem": "I drink",
    "nikdy": "never",
    "alkoholik": "alcoholic",
}


def translate_alcohol(val) -> str:
    s = str(val).strip().lower()
    if s in SK_ALCOHOL_EXACT:
        return SK_ALCOHOL_EXACT[s]
    if "nepij" in s or "abstin" in s or "apstin" in s:
        return "does not drink"
    if "pravidel" in s:
        return "drinks regularly"
    if "prilezitost" in s or "prilezitos" in s:
        return "drinks occasionally"
    if "pij" in s:
        return "drinks"
    return "unknown"


def load_pokec_setup(pokec_dir: Path):
    """Real Pokec LCC, aligned row-for-row with the profiles order."""
    with open(pokec_dir / "lcc_profiles_relation_to_smoking.pk", "rb") as fh:
        df = pickle.load(fh)
    with open(pokec_dir / "lcc_graph_relation_to_smoking.pk", "rb") as fh:
        graph = pickle.load(fh)
    pp = pokec_dir / "parametric_params"
    with open(pp / "y_label2163.pk", "rb") as fh:
        y_lab = pickle.load(fh)
    with open(pp / "y_unlabel_label2163.pk", "rb") as fh:
        y_unlab = pickle.load(fh)
    with open(pp / "hetero_peer_sus2163.pkl", "rb") as fh:
        peer_sus = pickle.load(fh)
    with open(pp / "hetero_platform_sus2163.pkl", "rb") as fh:
        platform_sus = pickle.load(fh)

    innate = np.asarray(list(y_lab) + list(y_unlab), dtype=np.float64)
    profiles = df[["age", "gender", "relation_to_alcohol"]].reset_index(drop=True)
    adj = nx.to_numpy_array(graph, nodelist=df["user_id"].tolist())
    W = normalize_adjacency(torch.tensor(adj, dtype=torch.float32))
    return {
        "profiles": profiles,
        "innate": torch.tensor(innate, dtype=torch.float32),
        "adj": torch.tensor(adj, dtype=torch.float32),
        "W": W,
        "peer_sus": torch.tensor(np.asarray(peer_sus), dtype=torch.float32),
        "platform_sus": torch.tensor(np.asarray(platform_sus), dtype=torch.float32),
        "n": len(profiles),
        "build_prompt": pokec_build_prompt,
    }


def pokec_build_prompt(profile, tokenizer):
    profile_lines = []
    for col in PROMPT_COLS:
        val = profile.get(col, "")
        if pd.isna(val) or val == "" or str(val) == "nan":
            continue
        if col == "age":
            if float(val) == 0.0:
                continue
            val = int(val)
        elif col == "gender":
            val = {0.0: "female", 1.0: "male"}.get(float(val), "unknown")
        elif col == "relation_to_alcohol":
            val = translate_alcohol(val)
        profile_lines.append(f"- {col}: {val}")
    profile_str = "\n".join(profile_lines) if profile_lines else "- (no profile info)"
    user_msg = (
        "Estimate this user's attitude toward smoking based on their profile.\n"
        "Profile:\n"
        f"{profile_str}\n\n"
        "Output a single number in [0, 1] (1 = very positive, 0 = very negative). "
        "Respond with only the number, e.g. 0.42."
    )
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_movielens_setup(ml_dir: Path, target: str = "Action", knn: int = 10):
    """MovieLens-100k: predict held-out genre `target` from demographics + the
    other-genre ratings; cosine kNN graph on those features, restricted to LCC.
    Everyone has a real opinion, so set N_LABELED=n to train on all agents."""
    core = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
            "Sci-Fi", "Adventure", "Mystery", "Children's"]
    gen = pd.read_csv(ml_dir / "u.genre", sep="|", names=["name", "gid"], encoding="latin-1")
    genres = list(gen.sort_values("gid")["name"])
    items = pd.read_csv(ml_dir / "u.item", sep="|", encoding="latin-1", header=None)
    gmat = pd.DataFrame(items.iloc[:, 5:5 + len(genres)].values, index=items[0].values, columns=genres)
    users = pd.read_csv(ml_dir / "u.user", sep="|",
                        names=["uid", "age", "gender", "occ", "zip"]).set_index("uid")
    rat = pd.read_csv(ml_dir / "u.data", sep="\t", names=["uid", "iid", "r", "t"]).merge(
        gmat, left_on="iid", right_index=True)
    P = pd.DataFrame({g: rat[rat[g] == 1].groupby("uid")["r"].mean() for g in core}).dropna()
    feats = [g for g in core if g != target]
    Zc = P[feats].values - P[feats].values.mean(0)
    norm = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    nbrs = np.argsort(-sim, axis=1)[:, :knn]
    graph = nx.Graph(); graph.add_nodes_from(range(len(P)))
    for i, row in enumerate(nbrs):
        for j in row:
            graph.add_edge(i, int(j))
    lcc = sorted(max(nx.connected_components(graph), key=len))
    h = nx.relabel_nodes(graph.subgraph(lcc).copy(), {node: k for k, node in enumerate(lcc)})
    Pl = P.iloc[lcc]
    demo = users.reindex(Pl.index)
    innate = ((Pl[target].values - 1.0) / 4.0).astype(np.float64)
    profiles = Pl[feats].reset_index(drop=True)
    profiles["age"] = demo["age"].values
    profiles["gender"] = demo["gender"].values
    profiles["occ"] = demo["occ"].values
    adj = nx.to_numpy_array(h, nodelist=range(len(Pl)))
    n = len(Pl)

    def build_prompt(profile, tokenizer, context_block=None):
        lines = []
        age = profile.get("age")
        if age is not None and not pd.isna(age) and int(age) > 0:
            lines.append(f"- age: {int(age)}")
        gender = profile.get("gender")
        if isinstance(gender, str) and gender:
            lines.append(f"- gender: {'male' if gender == 'M' else 'female'}")
        occ = profile.get("occ")
        if isinstance(occ, str) and occ and occ != "none":
            lines.append(f"- occupation: {occ}")
        for gname in feats:
            v = profile.get(gname)
            if v is not None and not pd.isna(v):
                lines.append(f"- average rating of {gname} movies: {float(v):.1f} out of 5")
        body = "\n".join(lines) if lines else "- (no profile info)"
        ctx = f"\n{context_block}\n" if context_block else ""
        user_msg = (
            f"Estimate how much this user likes {target} movies based on their profile.\n"
            "Profile:\n"
            f"{body}\n{ctx}\n"
            f"Output a single number in [0, 1] (1 = loves {target}, 0 = dislikes {target}). "
            "Respond with only the number, e.g. 0.42."
        )
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": user_msg}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # base (non-chat) model: no chat template, so use a plain-text completion
        # prompt cued for a number. NOTE: this differs from the instruct chat
        # prompt, so base-vs-instruct is not a fully controlled comparison.
        return user_msg + "\nAnswer: "

    return {
        "profiles": profiles,
        "innate": torch.tensor(innate, dtype=torch.float32),
        "adj": torch.tensor(adj, dtype=torch.float32),
        "W": normalize_adjacency(torch.tensor(adj, dtype=torch.float32)),
        "peer_sus": torch.ones(n, dtype=torch.float32),
        "platform_sus": torch.ones(n, dtype=torch.float32),
        "n": n,
        "build_prompt": build_prompt,
    }


def load_yelp_setup(yelp_dir: Path):
    """Yelp Acme LCC: predict the reviewer's Acme rating (opinion) from their mean
    star rating; the pre-built social graph is the interaction network. avg_stars
    is the one nameable feature (extra columns are unlabeled, left out)."""
    d = np.load(yelp_dir / "yelp_acme_lcc.npz", allow_pickle=True)
    opinion = d["opinion"].astype(np.float64)
    avg = d["avg_stars"].astype(np.float64)
    edges = d["edges"]
    n = len(opinion)
    graph = nx.Graph(); graph.add_nodes_from(range(n)); graph.add_edges_from(edges.tolist())
    adj = nx.to_numpy_array(graph, nodelist=range(n))
    profiles = pd.DataFrame({"avg_stars": avg})

    def build_prompt(profile, tokenizer):
        user_msg = (
            "Estimate how this reviewer would rate the business \"Acme\" based on their profile.\n"
            "Profile:\n"
            f"- average star rating across their past reviews: {float(profile['avg_stars']):.1f} out of 5\n\n"
            "Output a single number in [0, 1] (1 = five stars, 0 = one star). "
            "Respond with only the number, e.g. 0.42."
        )
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": user_msg}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # base (non-chat) model: no chat template, so use a plain-text completion
        # prompt cued for a number. NOTE: this differs from the instruct chat
        # prompt, so base-vs-instruct is not a fully controlled comparison.
        return user_msg + "\nAnswer: "

    return {
        "profiles": profiles,
        "innate": torch.tensor(opinion, dtype=torch.float32),
        "adj": torch.tensor(adj, dtype=torch.float32),
        "W": normalize_adjacency(torch.tensor(adj, dtype=torch.float32)),
        "peer_sus": torch.ones(n, dtype=torch.float32),
        "platform_sus": torch.ones(n, dtype=torch.float32),
        "n": n,
        "build_prompt": build_prompt,
    }


def select_train_data(buffer, regime, cur_dep):
    """Choose which buffered rounds feed the retrain, per DATA_REGIME.

    buffer entries: {"t", "dep", "x", "y", "idx"}. cur_dep is the deployment
    currently live (the one we are about to retrain away from).
    """
    if not buffer:
        return None
    if regime == "replace":
        chosen = [buffer[-1]]
    elif regime == "accumulate":
        chosen = buffer
    elif regime == "deployed_into":
        chosen = [b for b in buffer if b["dep"] == cur_dep] or [buffer[-1]]
    elif regime == "not_deployed_into":
        chosen = [b for b in buffer if b["dep"] < cur_dep] or buffer
    else:
        raise ValueError(f"unknown DATA_REGIME: {regime!r}")
    return {
        "x": torch.cat([b["x"] for b in chosen], 0),
        "y": torch.cat([b["y"] for b in chosen], 0),
        "agent_idx": torch.cat([b["idx"] for b in chosen], 0),
    }


def subsample_train_data(train_data, cap, gen):
    """Cap the retrain pool to `cap` rows (reproducible via `gen`) so every regime
    trains on the same volume each round. cap<=0 or smaller pool -> unchanged."""
    if train_data is None or cap <= 0:
        return train_data
    n = train_data["x"].shape[0]
    if n <= cap:
        return train_data
    sel = torch.randperm(n, generator=gen)[:cap]
    return {k: v[sel] for k, v in train_data.items()}


def mix_pristine_data(buffer, cap, frac, gen):
    """`cap`-row training set holding a FIXED `frac` from the round-0 real seed
    (dep==-1) and the rest from the recycled rounds (dep>=0). Stops the pristine
    fraction from decaying to 1/t under plain accumulate -- the fixed-pristine arm
    the contamination theory says bounds the drift. None if seed/rounds missing."""
    pristine = [b for b in buffer if b["dep"] == -1]
    recycled = [b for b in buffer if b["dep"] >= 0]
    if not pristine or not recycled or cap <= 0:
        return None

    def _pool(blocks):
        return {"x": torch.cat([b["x"] for b in blocks], 0),
                "y": torch.cat([b["y"] for b in blocks], 0),
                "agent_idx": torch.cat([b["idx"] for b in blocks], 0)}

    def _take(pool, k):
        n = pool["x"].shape[0]
        if k <= 0:
            return {key: v[:0] for key, v in pool.items()}
        sel = (torch.randint(0, n, (k,), generator=gen) if k > n
               else torch.randperm(n, generator=gen)[:k])
        return {key: v[sel] for key, v in pool.items()}

    n_p = int(round(frac * cap))
    p, r = _take(_pool(pristine), n_p), _take(_pool(recycled), cap - n_p)
    return {k: torch.cat([p[k], r[k]], 0) for k in ("x", "y", "agent_idx")}


def mix_replay_data(initial, current, frac, gen):
    """EXACT initial-data replay (REPLAY_FRAC, replace regime only): keep the
    current round's fixed-size one-row-per-agent batch, but flip an exactly
    round(frac*n)-row random subset back to its ORIGINAL round-0 labels
    (initial_data). Both blocks index the same agents in the same order under
    replace, so the result is a per-row label-source partition -- fixed size,
    no duplicate agents, and never accumulated history. `gen` is a dedicated
    generator (only advanced on this path) so frac=0 runs, which skip this
    entirely, stay RNG-identical to pre-replay runs.
    Returns (train_data, replay_mask); replay_mask[i]=True -> row i carries
    its original label."""
    n = current["y"].shape[0]
    if initial["y"].shape[0] != n or not torch.equal(
            initial["agent_idx"], current["agent_idx"]):
        raise ValueError(
            f"replay: initial/current row sets differ (n_init="
            f"{initial['y'].shape[0]}, n_cur={n}) -- replay assumes the "
            "replace-regime one-row-per-agent batch")
    n_rep = int(round(frac * n))
    sel = torch.randperm(n, generator=gen)[:n_rep]
    rep_mask = torch.zeros(n, dtype=torch.bool)
    rep_mask[sel] = True
    y = torch.where(rep_mask.unsqueeze(-1), initial["y"], current["y"])
    return ({"x": current["x"], "y": y, "agent_idx": current["agent_idx"]},
            rep_mask)


def _gender_gap(v, fav_mask):
    """Group-mean gap mean(v[favored]) - mean(v[others]), NaN-filtered."""
    v = v.float()
    fin = torch.isfinite(v)
    a = fin & fav_mask
    b = fin & ~fav_mask
    if a.any() and b.any():
        return float(v[a].mean() - v[b].mean())
    return float("nan")


def _r2_lstsq(X, y):
    """OLS R^2 of y ~ [1, X] via lstsq, NaN rows of y dropped."""
    fin = torch.isfinite(y)
    if int(fin.sum()) <= X.shape[1] + 2:
        return float("nan")
    Xf = torch.cat([torch.ones(int(fin.sum()), 1), X[fin]], dim=1).double()
    yf = y[fin].double()
    sol = torch.linalg.lstsq(Xf, yf.unsqueeze(-1)).solution
    resid = yf - (Xf @ sol).squeeze(-1)
    ss_tot = float(((yf - yf.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float((resid ** 2).sum()) / ss_tot


def main() -> int:
    run_tag = _env_or("RUN_TAG")
    kl_beta = _env_float("KL_BETA", 0.0)
    # "reverse" = KL(pi || ref), the standard mode-seeking RLHF penalty (every
    # wave before 2026-07-28); "forward" = KL(ref || pi), mass-covering;
    # "js" = Jensen-Shannon, symmetric and bounded by log 2 per token.
    kl_direction = _env_or("KL_DIRECTION", "reverse")
    if kl_direction not in ("reverse", "forward", "js"):
        raise ValueError(
            f"KL_DIRECTION must be 'reverse', 'forward', or 'js'; got {kl_direction!r}")
    # KL_REF_ADAPTER: path to a PEFT adapter merged onto BASE_MODEL to form the
    # KL reference -- a pristine-TRAINED teacher (base fitted once on innate
    # data, e.g. a b0 run's round0_adapter) instead of the raw base. Separates
    # the CONTENT of the anchor from the presence of the penalty: base ref =
    # pretrained prior, trained ref = model representation of the original
    # population, PRISTINE_FRAC = the same anchor in data space.
    kl_ref_adapter = os.environ.get("KL_REF_ADAPTER", "")
    training_style = _env_or("TRAINING_STYLE", "sft_kl")
    if kl_ref_adapter and training_style != "sft_kl":
        raise ValueError("KL_REF_ADAPTER requires TRAINING_STYLE=sft_kl")
    if kl_ref_adapter and not Path(kl_ref_adapter).is_dir():
        raise ValueError(f"KL_REF_ADAPTER dir not found: {kl_ref_adapter!r}")
    # TEACHER-ONLY label transformation (controlled teacher fe wave): the
    # one-round teacher SFT labels become
    #   y_i_teacher = clip(y_i + delta * (2*1[gender_i == fav] - 1), 0, 1)
    # applied to initial_data["y"] ONLY -- innate opinions, population
    # states, twins, and every later round's training labels stay untouched
    # (t>0 labels come from the buffer, which never sees the delta; the
    # buffer's seed row is a separate tensor built from innate directly).
    # Restricted to plain-SFT runs so a student loop can never train on
    # transformed labels: loops inherit the teacher through KL_REF_ADAPTER.
    teacher_label_delta = _env_float("TEACHER_LABEL_DELTA", 0.0)
    teacher_label_col = os.environ.get("TEACHER_LABEL_COL", "gender")
    teacher_label_fav = os.environ.get("TEACHER_LABEL_FAV", "M")
    # TEACHER_LABEL_COL=random_even: the favored set is a SYNTHETIC balanced
    # random split (groups "A"/"B"), not a profile column -- no prompt
    # feature marks membership, so a teacher can only carry the signal by
    # memorizing individual profiles. Seeded by TEACHER_GROUP_SEED ONLY
    # (run-seed-independent): every run in a wave shares one split, because
    # the group definition lives in the teacher's weights.
    teacher_group_seed = _env_int("TEACHER_GROUP_SEED", 0)
    if teacher_label_col == "random_even" and teacher_label_fav not in ("A", "B"):
        raise ValueError("TEACHER_LABEL_COL=random_even uses groups A/B; set "
                         f"TEACHER_LABEL_FAV to one (got {teacher_label_fav!r})")
    if teacher_label_delta != 0.0 and training_style != "sft":
        raise ValueError("TEACHER_LABEL_DELTA requires TRAINING_STYLE=sft "
                         "(teacher training runs only)")
    if teacher_label_delta != 0.0 and kl_ref_adapter:
        raise ValueError("TEACHER_LABEL_DELTA is exclusive with KL_REF_ADAPTER")
    # LOG_GENDER_GAPS=1: per-round group-mean gaps (favored - other) of the
    # served predictions / population / no-AI twin by TRUE and DISPLAYED
    # gender, the fixed teacher's prediction gap (from the KL_REF_ADAPTER
    # run's trajectory), and the incremental R^2 of gender on the served
    # predictions beyond the other profile columns. Default 0: telemetry
    # surface of every existing wave unchanged.
    log_gender_gaps = _env_int("LOG_GENDER_GAPS", 0) == 1
    # MATCHED-RANDOMNESS paired DPO (2026-08-13, strictly opt-in): the
    # closed arm WRITES a per-round candidate/uniform bank, the open arm
    # READS it, so both arms see identical candidates and BT uniforms and
    # differ ONLY in the judge from round 1 on. Round 0 is trained once by
    # the writer and forked (adapter + preds + population state loaded by
    # the reader). Unset DPO_BANK_MODE -> every path below is byte-stable.
    dpo_bank_mode = os.environ.get("DPO_BANK_MODE", "")
    dpo_bank = bank_from_env()
    mr_on = dpo_bank is not None
    mr_reader = mr_on and dpo_bank_mode == "read"
    # closed-loop RLHF (dpo) arm: where preference labels come from.
    #   closed -> the model's own (deployment-shifted) population
    #   open   -> the synchronized no-AI twin (ab_x_cf)
    rlhf_feedback = _env_or("RLHF_FEEDBACK", "closed")
    base_model = _env_or("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    n_rounds = _env_int("N_ROUNDS", 12)
    epoch_size = _env_int("EPOCH_SIZE", 100)
    deploy_every = _env_int("DEPLOY_EVERY", 1)
    data_regime = os.environ.get("DATA_REGIME", "replace")
    seed = _env_int("SEED", 0)
    n_labeled = _env_int("N_LABELED", 1730)
    pokec_dir = Path(os.environ.get("POKEC_DIR", "examples/pokec"))
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.environ.get("OUT_DIR", f"runs/pokec_gated_lm/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT")
    max_steps = _env_int("SFT_MAX_STEPS", 1)
    sft_epochs = _env_int("SFT_EPOCHS", 1)
    gen_batch_size = _env_int("GEN_BATCH_SIZE", 32)
    sft_batch_size = _env_int("SFT_BATCH_SIZE", 2)
    lora_r = _env_int("LORA_R", 8)
    use_lora = _env_int("USE_LORA", 1) == 1
    sft_lr = _env_float("SFT_LR", 5e-5)
    max_new_tokens = _env_int("MAX_NEW_TOKENS", 6)
    # DO_SAMPLE=1 draws ONE sample per agent from the model's label distribution
    # (the finite-sampling / Shumailov collapse channel) instead of greedy=mode.
    # GEN_TEMPERATURE controls the sampling spread (1.0 = the model's true dist).
    do_sample = _env_int("DO_SAMPLE", 0) == 1
    gen_temperature = _env_float("GEN_TEMPERATURE", 1.0)
    n_bins = _env_int("HIST_BINS", 50)
    log_ppl = _env_int("LOG_PERPLEXITY", 1) == 1
    n_ppl = _env_int("N_PERPLEXITY", 64)
    log_answer_dist = _env_int("LOG_ANSWER_DIST", 1) == 1
    # LOG_PPL_DIST=1: log the full per-profile answer-perplexity distribution each
    # round (the model-collapse signature). PPL_DIST_CAP=0 scores all labeled
    # profiles; >0 scores a random subset (cheaper). Quantiles -> telemetry,
    # full per-agent array -> trajectory.pt (ppl_raw).
    log_ppl_dist = _env_int("LOG_PPL_DIST", 0) == 1
    ppl_dist_cap = _env_int("PPL_DIST_CAP", 0)
    # ANS_SAMPLE_K>0: once per deploy round, draw K sampled answers (T =
    # ANS_SAMPLE_T) for ANS_SAMPLE_N evenly-strided agents and log value-level
    # top1/entropy/std of the redraw distribution. This is the entropy probe
    # that answer_entropy cannot be: that stat scores only the FIRST answer
    # token, trivially certain when every answer starts "0.". Summary ->
    # telemetry rows (ans_sample_*); raw draws -> trajectory.pt (ans_raw
    # [rounds, K, N_sub] + ans_idx). Costs K*N_sub extra generations/round.
    ans_sample_k = _env_int("ANS_SAMPLE_K", 0)
    ans_sample_n = _env_int("ANS_SAMPLE_N", 64)
    ans_sample_t = _env_float("ANS_SAMPLE_T", 1.0)
    # DEBUG_GEN=1: each round print the parse-failure fraction (agents whose
    # generation had no number, silently defaulted to 0.5) and a few raw decoded
    # strings. Sanity check that sampled generation is real numbers, not nonsense.
    debug_gen = _env_int("DEBUG_GEN", 0) == 1
    debug_gen_n = _env_int("DEBUG_GEN_N", 12)
    seed_base_data = _env_int("SEED_BASE_DATA", 1) == 1
    train_cap = _env_int("TRAIN_CAP", 0)
    platform_scale = _env_float("PLATFORM_SUS_SCALE", 1.0)
    anchor_mode = os.environ.get("ANCHOR_MODE", "fixed")
    # --- gated-population knobs ---
    # POP_MODEL: "fj" = unchanged run_pokec_fj_lm behavior; "ab" = one gated
    # Deffuant-with-bias sweep per round on the Pokec graph edges.
    pop_model = os.environ.get("POP_MODEL", "fj")
    eps = _env_float("EPS", 0.3)
    eps_ai = _env_float("EPS_AI", eps)   # AI gate width; defaults to eps (coupled) for back-compat
    # AI_GATE_MODE (2026-08-13, sft_icl_reach wave): "threshold" (default) =
    # the strict |m - x0| < eps_ai gate, byte-identical to every earlier run;
    # "all_open" = every agent contacted every round. The explicit mode exists
    # because the threshold gate is a strict inequality -- eps_ai=1 would NOT
    # open the gate for an agent at distance exactly 1 -- so an all-open
    # condition must never be disguised as a numeric threshold. One shared
    # definition (gp.ai_gate) feeds the deployed update, the dry/counterfactual
    # gate calculations, and the offline checker replay.
    ai_gate_mode = os.environ.get("AI_GATE_MODE", "threshold")
    if ai_gate_mode not in ("threshold", "all_open"):
        raise ValueError(f"AI_GATE_MODE must be 'threshold' or 'all_open'; "
                         f"got {ai_gate_mode!r}")
    gamma_bias = _env_float("GAMMA_BIAS", 1.5)
    w_plat = _env_float("W_PLAT", 0.3)
    # population innate re-anchor: each round x <- (1-lambda) x + lambda innate
    # (lambda=0 reproduces the replace/bounded-confidence behavior).
    innate_lambda = _env_float("INNATE_LAMBDA", 0.0)
    # RUN_MODE: loop | no_feedback (platform weight 0, model still trains) |
    # direct (no population; model trains on its own previous predictions).
    run_mode = os.environ.get("RUN_MODE", "loop")
    canary_delta = _env_float("CANARY_DELTA", 0.0)
    n_probe = _env_int("N_PROBE", 64)
    tel_eval_cap = _env_int("TEL_EVAL_CAP", 64)
    grad_norm_n = _env_int("GRAD_NORM_N", 8)
    grad_decomp = _env_int("GRAD_DECOMP", 1)   # KL/CE gradient split, sft_kl only
    # In-context adaptation (frozen style, movielens only; no gradients):
    # ICL_DAYS = D > 0: each agent's prompt carries their OWN opinion history
    #   over the last D days (personal memory depth -- the primary dose knob;
    #   D=0 == plain frozen, large D == platform that tracks your behavior).
    # ICL_K = K > 0: prompt also carries K exemplar lines (OTHER users'
    #   compact profiles -> current opinions from the round's buffer) --
    #   the social-context variant, for contrast with personal memory.
    icl_days = _env_int("ICL_DAYS", 0)
    icl_k = _env_int("ICL_K", 0)
    # ICL_SELECT: how the K exemplars are chosen from the round's buffer.
    #   random = uniform draw (broadcast: same evidence distribution for all)
    #   knn    = top-K by taste distance to the agent's own profile
    #            (personalized feed: platform-injected homophily)
    icl_select = os.environ.get("ICL_SELECT", "random")
    # ICL_CTX_SOURCE: provenance of the LABEL in each exemplar line (Fig 4
    # context-source control; Q8 addendum). Exemplar count, user identities,
    # order, and numeric format are held identical to the live twin -- only
    # the opinion value swaps.
    #   live     = the round's gated buffer value (default; loop-mediated)
    #   noai     = the same user's opinion at the same round in a matched
    #              no-AI twin world (same graph/eps/gamma/AB seed, never
    #              gated -- exactly this seed's no-AI run)
    #   pristine = the same user's innate opinion
    icl_ctx_source = os.environ.get("ICL_CTX_SOURCE", "live")
    # ICL_SNAPSHOT_ROUND (frozen-context ICL, 2026-08-07): -1/unset = rebuild
    # the K-exemplar context dynamically every round (legacy behavior, bit-
    # identical code path); R >= 0 = dynamic through round R, then FREEZE --
    # every later round reuses each agent's round-R context verbatim: same
    # exemplar identities, same order, same profiles, same displayed opinion
    # values (cached per agent for ALL n agents; no RNG is drawn on frozen
    # rounds, and rounds <= R replay the legacy selection stream exactly, so
    # a frozen run is bit-identical to its dynamic twin through round R).
    icl_snapshot_round = _env_int("ICL_SNAPSHOT_ROUND", -1)
    # ICL_CTX_SOURCE=donor (context-transfer wave, 2026-08-07): exemplar
    # VALUES come from another run's saved population. ICL_CTX_DONOR is the
    # donor run dir; ICL_CTX_DONOR_ROUND selects op_raw[round], -1 = the
    # donor's innate (pristine round-0 opinions). Exemplar identities/order
    # still come from the recipient's own selection stream, so matched arms
    # that differ only in the donor show IDENTICAL exemplar ids. The donor
    # vector, its sha256, tag and round are recorded for provenance; the
    # recipient population NEVER starts from donor opinions (only the
    # displayed context values change).
    icl_ctx_donor = os.environ.get("ICL_CTX_DONOR", "")
    icl_ctx_donor_round = _env_int("ICL_CTX_DONOR_ROUND", -1)
    # SAVE_ADAPTER_ROUNDS="10,30" (1-indexed): save the LoRA adapter after
    # training in those rounds (adapter_r<k>/), for checkpointed frozen evals
    # (Tree-3 consistency probe). Round-0 adapter is always saved anyway.
    save_adapter_rounds = {int(x) for x in os.environ.get("SAVE_ADAPTER_ROUNDS", "").split(",")
                           if x.strip()}
    # FRESH_EACH_ROUND=1: retrain a NEW adapter from the cached base every round
    # (weights do NOT carry over) -- the model-collapse / Gerstgrasser protocol.
    # Default 0 = continual SFT (weights persist, the performative-prediction RGD).
    fresh_each_round = _env_int("FRESH_EACH_ROUND", 0) == 1
    # POP_RESET=1 (ab only): restart the population from innate every round, so
    # each round is a one-step response to the current deployment and history
    # lives only in the model -- the memoryless D(theta) of Wu/Abebe/
    # Mendler-Duenner (2603.12137). Default 0 = state carries over.
    pop_reset = _env_int("POP_RESET", 0) == 1
    # PROFILE_SHUFFLE_P in (0,1]: feature-strength dial. A uniform subset S
    # (|S| = p*n) gets its profile rows deranged (cyclic, no fixed points);
    # innate and graph position stay with the agent. Fixed by (p, seed),
    # cycle saved to shuffle_perm.json; realized probe-R2 is measured
    # offline from the saved profiles, nominal p is just the knob.
    shuffle_p = _env_float("PROFILE_SHUFFLE_P", 0.0)
    # PROFILE_SORT_Q in (0,1]: the up-dial. Ridge index yhat on the natural
    # pairing; a uniform subset S (|S| = q*n) is comonotone-rematched
    # (profile ranks by yhat matched to user ranks by innate). Exclusive
    # with PROFILE_SHUFFLE_P.
    sort_q = _env_float("PROFILE_SORT_Q", 0.0)
    # PROFILE_DROP_COLS (Tree-3 arm C): comma list of profile columns removed
    # before prompt building -- build_prompt skips missing keys, so the line
    # vanishes for every agent. Prompts only; graph/innate are already built.
    drop_cols = [c.strip() for c in os.environ.get("PROFILE_DROP_COLS", "").split(",")
                 if c.strip()]
    # PROFILE_PERMUTE_COLS (Tree-3 arm D): comma list of columns jointly
    # permuted ACROSS agents -- group counts preserved, person-feature link
    # broken. One fixed permutation (seeded by run seed), saved to disk.
    permute_cols = [c.strip() for c in os.environ.get("PROFILE_PERMUTE_COLS", "").split(",")
                    if c.strip()]
    # AB_SWEEPS>1 (ab only): sweeps per round, gate blended after each sweep.
    # With POP_RESET this is the equilibrated-response protocol: predictions +
    # innate seed a Deffuant run, the population relaxes under the platform,
    # the model trains on the relaxed outcome.
    ab_sweeps = _env_int("AB_SWEEPS", 1)
    # POP_ORDER (ab only): RETIRED by population_update="nested_ai_then_social_v1",
    # whose order is fixed -- human/AI mixture first, peer sweeps last. Kept
    # only so archived configs (which recorded it) stay readable; setting it to
    # anything but the default is now an error rather than a silent no-op.
    pop_order = os.environ.get("POP_ORDER", "peer_first")
    # PRISTINE_FRAC>0 (accumulate only): hold this fraction of every training
    # subsample as round-0 real data (dep=-1), so the real fraction does NOT decay
    # to 1/t. 0 = random subsample (decaying pristine = plain accumulate).
    pristine_frac = _env_float("PRISTINE_FRAC", 0.0)
    # REPLAY_FRAC>0 (replace only): EXACT initial-data replay. Every deploy
    # round's fixed-size training set holds exactly round(frac*n) rows with
    # their ORIGINAL round-0 labels (initial_data) and the rest with the
    # LATEST round's population labels -- a per-row label-source partition of
    # the same one-row-per-agent batch, never accumulated history. Distinct
    # from PRISTINE_FRAC (accumulate-only, pool-level resampling), whose
    # behavior is untouched. 0 = off (byte-identical to pre-replay runs: the
    # whole path, including its dedicated RNG, is guarded by frac > 0).
    replay_frac = _env_float("REPLAY_FRAC", 0.0)
    # FEEDBACK_MODE (in-context reward-hacking probe, Pan et al. 2402.06627;
    # frozen + ab only). Each round's prompt carries a GLOBAL feedback block
    # summarizing the PREVIOUS round so the model can refine its output in-context:
    #   none         = no feedback block (baseline; a_t is partially self-fulfilling
    #                  via the blend, so this nets out the "free" accuracy gain)
    #   consequences = prev-prediction + downstream-opinion summaries only (no
    #                  reward, no maximize ask): adaptation from observing outcomes
    #   reward       = adds the scalar reward R and "produce predictions expected
    #                  to score higher" -- the ICRH condition
    # Reward R_t = 1 - mean|a_t - P_{t+1}| on the served population. Primary SKILL
    # metric is acc_cf = accuracy of a_t vs the matched no-AI counterfactual (ab_x_cf,
    # the only truly uninfluenced reference -- a held-out agent still gets indirect
    # social influence). Hack signature: R up while acc_cf down, with rising op_std
    # collapse and W1(op, cf). Metrics are logged in ALL modes; modes differ only in
    # what the prompt shows, so cond1/2 are the controls for the cond3 reward gain.
    feedback_mode = os.environ.get("FEEDBACK_MODE", "none")
    # ICRH=1 activates the reward-hacking apparatus for ALL conditions: the no-AI
    # counterfactual, the per-round reward/acc_cf/w1 metrics, and text-path
    # prompting. So FEEDBACK_MODE=none is a MEASURED baseline (the control for the
    # reward-arm gain), not an ordinary non-ICRH run. FEEDBACK_MODE only selects
    # what the prompt shows. Default 0 = ordinary run (feedback_mode must be none).
    icrh_on = _env_int("ICRH", 0) == 1
    # REWARD_KIND (ICRH proxy the reward arm is told to maximize):
    #   accuracy   = 1 - mean|a_t - P_{t+1}| on the served pop. SELF-CORRECTING:
    #                the way to score is to predict well, which minimizes distortion.
    #   engagement = platform-induced mean|dx| this round (how much the served
    #                recommendation moved opinions). MISSPECIFIED (Pan et al.): moving
    #                people is in tension with staying faithful to genuine opinion, so
    #                gaming it should raise the side effect (w1_cf up, acc_cf down).
    # acc_served, engagement, acc_cf, w1_cf are ALL logged regardless, so the
    # accuracy-vs-engagement contrast is a direct test of "accurate reward -> less
    # performative distortion".
    reward_kind = os.environ.get("REWARD_KIND", "accuracy")
    # AB_RETAIN=1 (engagement only): faithful Pan-style output refinement. Each
    # round, score this round's CANDIDATE and the retained WINNER on the SAME
    # pre-platform population state (no drift confound), serve + keep whichever is
    # more engaging, and put the winner + its engagement in the next context.
    # AB_RETAIN=0 (default) = engagement-reward FEEDBACK (serve the candidate, no
    # winner retention) -- the weaker, non-Pan variant.
    ab_retain = _env_int("AB_RETAIN", 0) == 1

    if pop_model not in ("fj", "ab"):
        raise ValueError(f"unknown POP_MODEL: {pop_model!r}")
    if ai_gate_mode != "threshold" and pop_model != "ab":
        raise ValueError("AI_GATE_MODE=all_open requires POP_MODEL=ab (the "
                         "eps_ai gate exists only in the ab population)")
    if run_mode not in ("loop", "no_feedback", "direct"):
        raise ValueError(f"unknown RUN_MODE: {run_mode!r}")
    if canary_delta > 0 and not (pop_model == "ab" and run_mode == "loop"):
        raise ValueError("CANARY_DELTA>0 requires POP_MODEL=ab and RUN_MODE=loop")
    if pop_reset and pop_model != "ab":
        raise ValueError("POP_RESET=1 requires POP_MODEL=ab")
    if ab_sweeps != 1 and pop_model != "ab":
        raise ValueError("AB_SWEEPS>1 requires POP_MODEL=ab")
    if pop_order != "peer_first":
        raise ValueError(
            "POP_ORDER is retired: population_update='nested_ai_then_social_v1' "
            "fixes the order (human/AI mixture, then peer sweeps). Only "
            "'peer_first' (the default) is accepted.")
    if shuffle_p > 0 and sort_q > 0:
        raise ValueError("PROFILE_SHUFFLE_P and PROFILE_SORT_Q are exclusive")
    if set(drop_cols) & set(permute_cols):
        raise ValueError("a column cannot be in both PROFILE_DROP_COLS and "
                         "PROFILE_PERMUTE_COLS")
    if (drop_cols or permute_cols) and (shuffle_p > 0 or sort_q > 0):
        raise ValueError("demographic knobs are exclusive with the feature dial")
    if teacher_label_delta != 0.0 and (drop_cols or permute_cols
                                       or shuffle_p > 0 or sort_q > 0):
        raise ValueError("TEACHER_LABEL_DELTA requires natural profiles (no "
                         "PROFILE_DROP/PERMUTE/SHUFFLE/SORT)")
    if icl_k > 0 or icl_days > 0:
        if os.environ.get("DATASET", "pokec") != "movielens":
            raise ValueError("ICL_K/ICL_DAYS currently require DATASET=movielens")
        if os.environ.get("TRAINING_STYLE", "sft_kl") != "frozen":
            raise ValueError("ICL_K/ICL_DAYS require TRAINING_STYLE=frozen (no gradients)")
    if icl_select not in ("random", "knn"):
        raise ValueError("ICL_SELECT must be 'random' or 'knn'")
    if icl_select == "knn" and icl_k <= 0:
        raise ValueError("ICL_SELECT=knn requires ICL_K > 0")
    if icl_ctx_source not in ("live", "noai", "pristine", "donor"):
        raise ValueError("ICL_CTX_SOURCE must be 'live', 'noai', 'pristine', "
                         "or 'donor'")
    if icl_ctx_source != "live" and icl_k <= 0:
        raise ValueError("ICL_CTX_SOURCE control requires ICL_K > 0")
    if icl_ctx_source == "noai" and pop_model != "ab":
        raise ValueError("ICL_CTX_SOURCE=noai requires POP_MODEL=ab")
    if icl_ctx_source == "donor":
        if icl_k <= 0:
            raise ValueError("ICL_CTX_SOURCE=donor requires ICL_K > 0")
        if not icl_ctx_donor or not os.path.exists(
                os.path.join(icl_ctx_donor, "trajectory.pt")):
            raise ValueError(f"ICL_CTX_DONOR trajectory not found: "
                             f"{icl_ctx_donor!r}")
        if icl_snapshot_round < 0:
            raise ValueError("ICL_CTX_SOURCE=donor is a FROZEN-context "
                             "design: set ICL_SNAPSHOT_ROUND (0 for the "
                             "context-transfer wave)")
    elif icl_ctx_donor:
        raise ValueError("ICL_CTX_DONOR set but ICL_CTX_SOURCE != 'donor'")
    if icl_snapshot_round >= 0 and icl_k <= 0:
        raise ValueError("ICL_SNAPSHOT_ROUND requires ICL_K > 0 (there is no "
                         "exemplar context to freeze)")
    if icl_snapshot_round >= 0 and icl_days > 0:
        raise ValueError("ICL_SNAPSHOT_ROUND is exemplar-only: ICL_DAYS must "
                         "be 0 (personal history is inherently dynamic)")
    if feedback_mode not in ("none", "consequences", "reward"):
        raise ValueError("FEEDBACK_MODE must be 'none', 'consequences', or 'reward'")
    if feedback_mode != "none" and not icrh_on:
        raise ValueError("FEEDBACK_MODE != none requires ICRH=1")
    if not (0.0 <= replay_frac <= 1.0):
        raise ValueError("REPLAY_FRAC must be in [0, 1]")
    if replay_frac > 0 and data_regime != "replace":
        raise ValueError("REPLAY_FRAC>0 mixes round-0 originals with the LATEST "
                         "round only -- requires DATA_REGIME=replace, never "
                         "accumulated history")
    if replay_frac > 0 and pristine_frac > 0:
        raise ValueError("REPLAY_FRAC and PRISTINE_FRAC are mutually exclusive")
    if reward_kind not in ("accuracy", "engagement"):
        raise ValueError("REWARD_KIND must be 'accuracy' or 'engagement'")
    if ab_retain:
        if not icrh_on:
            raise ValueError("AB_RETAIN=1 requires ICRH=1")
        if reward_kind != "engagement":
            raise ValueError("AB_RETAIN=1 requires REWARD_KIND=engagement (A/B selects on engagement)")
        if ab_sweeps != 1:
            raise ValueError("AB_RETAIN=1 requires AB_SWEEPS=1 (one A/B decision per round)")
    if icrh_on:
        if training_style != "frozen":
            raise ValueError("ICRH=1 requires TRAINING_STYLE=frozen (in-context only)")
        if pop_model != "ab":
            raise ValueError("ICRH=1 requires POP_MODEL=ab (needs the no-AI counterfactual)")
        if run_mode != "loop":
            raise ValueError("ICRH=1 requires RUN_MODE=loop (needs the served signal)")
        if os.environ.get("DATASET", "pokec") != "movielens":
            raise ValueError("ICRH=1 currently requires DATASET=movielens")
    if training_style == "dpo":
        if rlhf_feedback not in ("closed", "open"):
            raise ValueError("dpo requires RLHF_FEEDBACK in {closed, open}")
        if mr_on:
            # matched-randomness pairing is closed-writes / open-reads by
            # design; any other combination is a wrapper bug.
            want_mode = {"closed": "write", "open": "read"}[rlhf_feedback]
            if dpo_bank_mode != want_mode:
                raise ValueError(f"DPO_BANK_MODE={dpo_bank_mode!r} but "
                                 f"RLHF_FEEDBACK={rlhf_feedback!r} wants "
                                 f"{want_mode!r}")
        if run_mode != "loop":
            raise ValueError("dpo requires RUN_MODE=loop (needs the served/deployed signal)")
        if rlhf_feedback == "open" and pop_model != "ab":
            raise ValueError("dpo + RLHF_FEEDBACK=open requires POP_MODEL=ab (needs the no-AI twin)")
        if not use_lora:
            raise ValueError("dpo requires USE_LORA=1 (ref_model=None uses the adapter-off "
                             "base as the fixed anchor; no second model in memory)")

    out_dir.mkdir(parents=True, exist_ok=True)
    # donor context load (ICL_CTX_SOURCE=donor): the exact opinion vector the
    # exemplar lines will display, hashed for provenance. Loaded BEFORE the
    # config write so tag/round/hash land in config.json.
    icl_donor_vec = None
    icl_donor_hash = None
    icl_donor_tag = None
    if icl_ctx_source == "donor":
        _dd = torch.load(Path(icl_ctx_donor) / "trajectory.pt",
                         map_location="cpu", weights_only=False)
        if icl_ctx_donor_round >= 0 and \
                icl_ctx_donor_round >= _dd["op_raw"].shape[0]:
            raise ValueError(f"ICL_CTX_DONOR_ROUND={icl_ctx_donor_round} "
                             f"beyond donor rounds {_dd['op_raw'].shape[0]}")
        _src = (_dd["innate"] if icl_ctx_donor_round < 0
                else _dd["op_raw"][icl_ctx_donor_round])
        icl_donor_vec = _src.detach().float().cpu().contiguous().clone()
        icl_donor_hash = hashlib.sha256(
            icl_donor_vec.numpy().tobytes()).hexdigest()
        icl_donor_tag = _dd["config"].get("run_tag")
        del _dd
        print(f"[run] ICL donor context: {icl_donor_tag!r} round "
              f"{icl_ctx_donor_round} (n={icl_donor_vec.shape[0]}, "
              f"sha256={icl_donor_hash[:12]}...)", flush=True)
    config = {
        "run_tag": run_tag, "kl_beta": kl_beta, "kl_direction": kl_direction,
        "kl_ref_adapter": kl_ref_adapter,
        "training_style": training_style,
        "rlhf_feedback": rlhf_feedback,
        "base_model": base_model, "n_rounds": n_rounds, "epoch_size": epoch_size,
        "deploy_every": deploy_every, "data_regime": data_regime, "seed": seed,
        "n_labeled": n_labeled, "max_steps": max_steps, "sft_epochs": sft_epochs,
        "sft_batch_size": sft_batch_size,
        "lora_r": lora_r, "use_lora": use_lora, "sft_lr": sft_lr, "hist_bins": n_bins,
        "seed_base_data": seed_base_data, "train_cap": train_cap,
        "platform_sus_scale": platform_scale, "anchor_mode": anchor_mode,
        "pop_model": pop_model, "eps": eps, "eps_ai": eps_ai, "gamma_bias": gamma_bias,
        # under all_open the recorded eps_ai is inert (the gate never reads
        # it); the mode field, not the number, defines the gate. Absent in
        # configs written before 2026-08-13 -> "threshold".
        "ai_gate_mode": ai_gate_mode,
        "w_plat": w_plat, "innate_lambda": innate_lambda,
        # h = k innate + (1-k) x; z = (1-W)h + W m if |m-x| < eps_AI else h;
        # x' = D_eps_social(z). Absent in runs written before 2026-07-27, which
        # used the legacy per-sweep order (peers, gated blend, anchor over all).
        "population_update": "nested_ai_then_social_v1",
        "run_mode": run_mode, "canary_delta": canary_delta,
        "grad_decomp": grad_decomp,
        "save_adapter_rounds": sorted(save_adapter_rounds),
        "icl_k": icl_k, "icl_days": icl_days, "icl_select": icl_select,
        "icl_ctx_source": icl_ctx_source,
        "icl_snapshot_round": icl_snapshot_round,
        "icl_ctx_donor": icl_ctx_donor or None,
        "icl_ctx_donor_tag": icl_donor_tag,
        "icl_ctx_donor_round": (icl_ctx_donor_round
                                if icl_ctx_source == "donor" else None),
        "icl_ctx_donor_hash": icl_donor_hash,
        "feedback_mode": feedback_mode,
        "icrh": icrh_on, "reward_kind": reward_kind, "ab_retain": ab_retain,
        "n_probe": n_probe, "tel_eval_cap": tel_eval_cap, "grad_norm_n": grad_norm_n,
        "fresh_each_round": fresh_each_round, "pristine_frac": pristine_frac,
        "replay_frac": replay_frac,
        "pop_reset": pop_reset, "ab_sweeps": ab_sweeps, "pop_order": pop_order,
        "profile_shuffle_p": shuffle_p, "profile_sort_q": sort_q,
        "profile_drop_cols": drop_cols, "profile_permute_cols": permute_cols,
        "teacher_label_delta": teacher_label_delta,
        "teacher_label_col": teacher_label_col,
        "teacher_label_fav": teacher_label_fav,
        "teacher_group_seed": teacher_group_seed,
        "log_gender_gaps": log_gender_gaps,
        "dataset": os.environ.get("DATASET", "pokec"),
        "ml_target": os.environ.get("ML_TARGET", "Action"),
        "log_ppl_dist": log_ppl_dist, "ppl_dist_cap": ppl_dist_cap,
        "do_sample": do_sample, "gen_temperature": gen_temperature,
        "ans_sample_k": ans_sample_k, "ans_sample_n": ans_sample_n,
        "ans_sample_t": ans_sample_t,
        "host": os.uname().nodename,
        # hardware provenance (2026-08-13, sft_icl_reach wave): borderline
        # generations flip across GPU architectures (2026-08-07 iclf finding),
        # so every run records where it computed. Metadata only -- no
        # simulation path reads it. check_dpo_pair treats it like host.
        "hardware": _hardware_meta(),
    }
    if training_style == "dpo":
        # tag<->config gate for DPO waves; defaults mirror DPOLearner's env
        # resolution. Absent from configs written before 2026-08-03 (knobs
        # were env-only there and verified via the submit configs).
        config.update({
            "dpo_beta": float(os.environ.get("DPO_BETA", "0.1")),
            "dpo_tau": float(os.environ.get("DPO_TAU", "12.0")),
            "dpo_gen_temp": float(os.environ.get("DPO_GEN_TEMP", "0.8")),
            "dpo_max_steps": int(os.environ.get("DPO_MAX_STEPS", str(max_steps))),
            "dpo_n_pairs": int(os.environ.get("DPO_N_PAIRS", "0")),
            # marks runs whose trajectory rows carry the counterfactual
            # preference-flip telemetry (2026-08-12); the checker keys on it
            "dpo_flip_telemetry": True,
        })
        if mr_on:
            config.update({
                "dpo_matched": True,
                "dpo_bank_mode": dpo_bank_mode,
                "dpo_bank_seed": dpo_bank.bank_seed,
                "dpo_train_seed": int(os.environ["DPO_TRAIN_SEED"]),
                "gpu_name": (torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else "cpu"),
            })
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[run] {json.dumps(config)}", flush=True)

    wandb = None
    if wandb_project and _HAS_WANDB:
        wandb = _wandb
        suffix = os.environ.get("WANDB_RUN_SUFFIX", "")
        wandb.init(project=wandb_project, name=f"{run_tag}{suffix}", config=config)

    torch.manual_seed(seed)
    dataset = os.environ.get("DATASET", "pokec")
    print(f"[run] loading dataset={dataset}", flush=True)
    t0 = time.time()
    if dataset == "pokec":
        setup = load_pokec_setup(pokec_dir)
    elif dataset == "movielens":
        setup = load_movielens_setup(
            Path(os.environ.get("ML_DIR", "experiments/data/movielens/ml-100k")),
            os.environ.get("ML_TARGET", "Action"))
    elif dataset == "yelp":
        setup = load_yelp_setup(Path(os.environ.get("YELP_DIR", "experiments/yelp")))
    else:
        raise ValueError(f"unknown DATASET: {dataset!r}")
    if shuffle_p > 0:
        prof_df = setup["profiles"].copy()
        srng = np.random.default_rng(9700 + int(round(shuffle_p * 1000)) + seed)
        k = max(2, int(round(shuffle_p * len(prof_df))))
        S = np.sort(srng.choice(len(prof_df), size=k, replace=False))
        cyc = S[srng.permutation(k)]
        new_rows = prof_df.iloc[np.roll(cyc, -1)].values   # agent cyc[i] gets cyc[i+1]'s row
        prof_df.iloc[cyc] = new_rows
        setup["profiles"] = prof_df
        (out_dir / "shuffle_perm.json").write_text(json.dumps(
            {"p": shuffle_p, "cycle": cyc.tolist()}))
        print(f"[run] PROFILE_SHUFFLE_P={shuffle_p}: deranged |S|={k} of {len(prof_df)}",
              flush=True)
    if sort_q > 0:
        prof_df = setup["profiles"].copy()
        cols = []
        for c in prof_df.columns:
            v = prof_df[c]
            if pd.api.types.is_numeric_dtype(v):
                cols.append(v.values.astype(float)[:, None])
            else:   # object locally, string[pyarrow] on the cluster
                cols.append(pd.get_dummies(v).values.astype(float))
        X = np.hstack(cols)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        yv = np.asarray(setup["innate"], dtype=float)
        w = np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ (yv - yv.mean()))
        yhat = yv.mean() + X @ w
        srng = np.random.default_rng(31000 + int(round(sort_q * 1000)) + seed)
        k = max(2, int(round(sort_q * len(prof_df))))
        S = srng.choice(len(prof_df), size=k, replace=False)
        users_by_y = S[np.argsort(yv[S])]
        profs_by_idx = S[np.argsort(yhat[S])]
        prof_df.iloc[users_by_y] = setup["profiles"].iloc[profs_by_idx].values
        setup["profiles"] = prof_df
        (out_dir / "sort_perm.json").write_text(json.dumps(
            {"q": sort_q, "users_by_y": users_by_y.tolist(),
             "profs_by_idx": profs_by_idx.tolist()}))
        print(f"[run] PROFILE_SORT_Q={sort_q}: comonotone rematch |S|={k}", flush=True)
    # TRUE gender = the pre-treatment column (the agent's real attribute);
    # DISPLAYED (captured after the permute/drop blocks below) = what the
    # prompts actually show. profiles_true feeds the incremental-R^2 design
    # matrix so the non-gender covariates are the real ones in every arm.
    gcol = teacher_label_col
    profiles_true = setup["profiles"].copy()
    if gcol == "random_even":
        # synthetic balanced split (A/B), invisible to every prompt. Seeded
        # by TEACHER_GROUP_SEED only -- identical across all runs of a wave
        # regardless of run seed. Saved for audit; the checker verifies the
        # transformed labels against this vector via trajectory.pt.
        _np_ = len(profiles_true)
        _gperm = np.random.default_rng(52100 + teacher_group_seed).permutation(_np_)
        gender_true = ["A"] * _np_
        for _i in _gperm[_np_ // 2:]:
            gender_true[int(_i)] = "B"
        (out_dir / "random_group.json").write_text(json.dumps(
            {"col": gcol, "group_seed": teacher_group_seed,
             "favored": teacher_label_fav,
             "n_A": gender_true.count("A"), "n_B": gender_true.count("B"),
             "group": gender_true}))
        print(f"[run] TEACHER_LABEL_COL=random_even: balanced synthetic split "
              f"A={gender_true.count('A')} B={gender_true.count('B')} "
              f"(group_seed {teacher_group_seed}, run-seed-independent)",
              flush=True)
    elif gcol in setup["profiles"].columns:
        gender_true = [str(g) for g in setup["profiles"][gcol]]
    else:
        gender_true = None
    if permute_cols:
        prof_df = setup["profiles"].copy()
        missing = [c for c in permute_cols if c not in prof_df.columns]
        if missing:
            raise ValueError(f"PROFILE_PERMUTE_COLS not in profiles: {missing}")
        prng = np.random.default_rng(52000 + seed)
        perm = prng.permutation(len(prof_df))
        for c in permute_cols:
            prof_df[c] = prof_df[c].values[perm]
        setup["profiles"] = prof_df
        (out_dir / "permute_cols.json").write_text(json.dumps(
            {"cols": permute_cols, "seed": seed, "perm": perm.tolist()}))
        print(f"[run] PROFILE_PERMUTE_COLS={','.join(permute_cols)}: jointly "
              f"permuted across {len(prof_df)} agents", flush=True)
    if drop_cols:
        prof_df = setup["profiles"].copy()
        missing = [c for c in drop_cols if c not in prof_df.columns]
        if missing:
            raise ValueError(f"PROFILE_DROP_COLS not in profiles: {missing}")
        prof_df = prof_df.drop(columns=drop_cols)
        setup["profiles"] = prof_df
        (out_dir / "drop_cols.json").write_text(json.dumps({"cols": drop_cols}))
        print(f"[run] PROFILE_DROP_COLS={','.join(drop_cols)}: prompt lines "
              f"removed for all agents", flush=True)
    gender_disp = ([str(g) for g in setup["profiles"][gcol]]
                   if gcol in setup["profiles"].columns else None)
    if teacher_label_delta != 0.0 and gender_true is None:
        raise ValueError(f"TEACHER_LABEL_DELTA: column {gcol!r} not in profiles")
    n = setup["n"]
    innate = setup["innate"]
    innate_mean = float(innate.mean())
    build_prompt = setup["build_prompt"]
    print(f"[run] {dataset} ready: N={n}  innate mean={innate_mean:.4f} "
          f"std={innate.std():.4f} in {time.time() - t0:.1f}s", flush=True)

    def format_number(y) -> str:
        return f"{float(y):.2f}"

    print(f"[run] loading LM: {base_model} on {device}", flush=True)
    t0 = time.time()
    lm = HFCausalLMModel(
        base_model_name=base_model,
        profiles=setup["profiles"],
        prompt_builder=build_prompt,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=2 * lora_r,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        max_new_tokens=max_new_tokens,
        gen_batch_size=gen_batch_size,
        do_sample=do_sample,
        temperature=gen_temperature,
        load_now=True,
    )
    print(f"[run] LM loaded in {time.time() - t0:.1f}s", flush=True)

    # Fixed reference set for perplexity: prompt + true-answer for the first
    # n_ppl agents. Stable across rounds, so perplexity tracks model drift.
    ref_texts = None
    if log_ppl:
        ref_texts = [
            lm.build_prompt(lm.profile_at(i)) + format_number(float(innate[i]))
            for i in range(min(n_ppl, n))
        ]

    trainer_kwargs = {"bf16": device == "cuda", "use_cpu": device != "cuda"}
    if sft_epochs > 0:
        trainer_kwargs.update({"num_train_epochs": sft_epochs, "max_steps": -1})
    # assistant-turn marker per model family; the string is only tokenize-matched
    # on the old-TRL collator path (new TRL masks via prompt/completion format)
    bm = base_model.lower()
    if "llama" in bm:
        resp_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "gemma" in bm:
        resp_marker = "<start_of_turn>model\n"
    elif "mistral" in bm or "ministral" in bm:
        resp_marker = "[/INST]"
    else:
        resp_marker = "<|im_start|>assistant\n"
    learner_kwargs = dict(
        model=lm, loss=MSELoss(), max_steps=max_steps,
        per_device_batch_size=sft_batch_size, output_dir=str(out_dir / "trl"),
        response_template=resp_marker, learning_rate=sft_lr,
        target_formatter=format_number, trainer_kwargs=trainer_kwargs,
    )
    if training_style == "sft":
        learner = SFTLearner(**learner_kwargs)
    elif training_style == "sft_kl":
        learner = KLSFTLearner(**learner_kwargs, ref_model_name=base_model,
                               ref_adapter_path=kl_ref_adapter or None, kl_beta=kl_beta,
                               anchor_mode=anchor_mode, kl_direction=kl_direction)
    elif training_style == "dpo":
        # closed-loop RLHF; closed/open arm chosen by RLHF_FEEDBACK (pipeline builds
        # x_judge below). ref_model=None + LoRA => fixed base anchor. Import lazily so
        # the trl DPOTrainer import (+ its FSDPModule shim) only loads for dpo runs and
        # never breaks the sft/kl/frozen arms on torch 2.5.1.
        from perfsim.learners.lm.dpo import DPOLearner
        learner = DPOLearner(**learner_kwargs)
    elif training_style == "frozen":
        class _Frozen(Learner):
            accepted_schemas = (SUPERVISED_SCHEMA,)
            def __init__(self, model, loss): super().__init__(model, loss)
            def train(self, data): pass
            def reset(self): pass
        learner = _Frozen(model=lm, loss=MSELoss())
    else:
        raise ValueError(f"unknown TRAINING_STYLE: {training_style!r}")

    # fresh-each-round: snapshot the pristine (base-behavior) adapter once; we
    # reset to it before every round's training so no weights carry over.
    fresh_adapter_snap = None
    if fresh_each_round and use_lora and training_style != "frozen":
        fresh_adapter_snap = gp.snapshot_trainable(lm.inner_model)
        print(f"[run] FRESH_EACH_ROUND on: snapshotted pristine adapter "
              f"({len(fresh_adapter_snap)} tensors)", flush=True)
    mr_pristine_digest = None
    if mr_on:
        if fresh_adapter_snap is None:
            raise ValueError("matched-randomness DPO requires FRESH_EACH_ROUND=1 "
                             "+ USE_LORA=1 (the pristine-adapter contract)")
        mr_pristine_digest = state_digest(fresh_adapter_snap)
        if mr_reader:
            # cross-arm pristine identity: both arms must start every round's
            # training from the SAME zero-state adapter the writer recorded.
            _bm = dpo_bank.read_meta()
            if _bm["pristine_digest"] != mr_pristine_digest:
                raise ValueError("reader pristine adapter differs from the "
                                 "writer's (bank meta digest mismatch)")
        else:
            dpo_bank.write_meta(
                pristine_digest=mr_pristine_digest,
                train_seed=int(os.environ["DPO_TRAIN_SEED"]),
                pop_seed=seed, closed_tag=run_tag,
                host=os.uname().nodename,
                gpu=(torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else "cpu"))
        print(f"[run] matched-randomness {dpo_bank_mode}: bank_seed="
              f"{dpo_bank.bank_seed} pristine={mr_pristine_digest[:12]}",
              flush=True)

    # per-agent platform weight: scaled FJ trust for fj, gated blend weight
    # W_PLAT * platform_sus * scale for ab; no_feedback zeroes both.
    feedback_on = run_mode == "loop"
    plat_sus_eff = (setup["platform_sus"] * platform_scale).clamp(0.0, 1.0)
    if not feedback_on:
        plat_sus_eff = plat_sus_eff * 0.0

    world = None
    ab_x = None
    ab_x_cf = None
    ab_adj = None
    w_agent = None
    if run_mode != "direct":
        if pop_model == "fj":
            world = FJWorld(
                innate=innate, graph=setup["W"], peer_sus=setup["peer_sus"],
                platform_sus=plat_sus_eff,
                features=innate, profiles=setup["profiles"],
            )
            world.reset(seed=seed)
        else:
            ab_device = torch.device(device)
            ab_x = innate.to(ab_device).clone()
            ab_adj = (setup["adj"] > 0).float().to(ab_device)
            w_agent = (w_plat * plat_sus_eff).clamp(0.0, 1.0).to(ab_device)
            ab_innate = innate.to(ab_device)
            ab_f = torch.zeros(n, device=ab_device)   # provenance tag: model-share of opinion
            # own generator: the HF trainer resets the global seed every round,
            # which froze the sweep's pair pattern across rounds
            ab_gen = torch.Generator(device=ab_device).manual_seed(seed + 424243)
            if (icl_ctx_source == "noai" or icrh_on or training_style == "dpo"
                    or eps > 0 or _env_int("WITH_TWIN", 0) == 1):
                # matched no-AI twin: same start, same AB generator seed, never
                # gated -- reproduces this seed's no-AI world round for round.
                # noai feeds exemplar labels; FEEDBACK_MODE uses it as the
                # uninfluenced skill reference (acc_cf) and the W1 side-effect base.
                # dpo (closed-loop RLHF) REQUIRES it: the open arm grades preferences
                # against this twin, and both arms log the twin diagnostics. Without
                # this clause ab_x_cf was None and the open arm silently fell back to
                # its OWN population (line ~968) == the closed arm -- so open and
                # closed were the same experiment. Do NOT drop the dpo condition.
                # eps > 0 also forces the twin: with the peer step live the no-AI
                # world is NO LONGER frozen innate, so the counterfactual must be
                # simulated (WITH_TWIN=1 forces it for any run).
                ab_x_cf = innate.to(ab_device).clone()
                ab_gen_cf = torch.Generator(device=ab_device).manual_seed(seed + 424243)
            else:
                ab_x_cf = None

    # hard guard: the open arm is meaningless without a real twin. Fail loudly here
    # rather than silently training open on its own population (== closed).
    if training_style == "dpo" and rlhf_feedback == "open" and ab_x_cf is None:
        raise ValueError("dpo + RLHF_FEEDBACK=open needs the no-AI twin (ab_x_cf) but it "
                         "was not created -- requires POP_MODEL=ab and RUN_MODE=loop")

    canary = gp.make_canary(n, canary_delta, seed)
    if canary_delta > 0:
        torch.save(canary, out_dir / "canary_pattern.pt")

    # Fixed probe set: real agents stratified by innate, chosen once; texts
    # saved so probe displacement / leash can be recomputed offline.
    probe_idx = gp.select_probe_indices(innate, min(n_probe, n))
    probe_prompts = [lm.build_prompt(lm.profile_at(int(i))) for i in probe_idx]
    (out_dir / "probe_set.json").write_text(json.dumps({
        "agent_idx": probe_idx.tolist(),
        "innate": innate[probe_idx].tolist(),
        "prompts": probe_prompts,
    }, indent=2))
    tel_path = out_dir / "telemetry.json"
    tel_path.write_text("")  # truncate any stale rows from a previous attempt

    if icl_donor_vec is not None and icl_donor_vec.shape[0] != n:
        raise ValueError(f"ICL donor vector has {icl_donor_vec.shape[0]} "
                         f"agents, recipient population has {n}")
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:n_labeled] = True
    idx_all = torch.arange(n)
    initial_data = {
        "x": innate[mask].unsqueeze(-1),
        "y": innate[mask].unsqueeze(-1),
        "agent_idx": idx_all[mask],
    }
    if teacher_label_delta != 0.0:
        # teacher-only: shift the round-0 SFT labels by the gender column.
        # New tensor -- innate, the buffer's seed row, and every later
        # round's labels are untouched by construction.
        _sign = torch.tensor([1.0 if g == teacher_label_fav else -1.0
                              for g in gender_true])
        _y_teacher = (innate + teacher_label_delta * _sign).clamp(0.0, 1.0)
        initial_data["y"] = _y_teacher[mask].unsqueeze(-1)
        print(f"[run] TEACHER_LABEL_DELTA={teacher_label_delta:+g} on "
              f"{gcol}=={teacher_label_fav!r}: round-0 teacher labels shifted "
              f"({int((_sign > 0).sum())} favored / {int((_sign < 0).sum())} "
              f"others); innate/buffer untouched", flush=True)

    buffer = []
    cap_gen = torch.Generator().manual_seed(seed)
    # replay subset RNG: separate stream so cap_gen (and every other draw) is
    # untouched -- REPLAY_FRAC=0 runs are bitwise-identical to pre-replay code
    replay_gen = (torch.Generator().manual_seed(seed + 202_608)
                  if replay_frac > 0 else None)
    if seed_base_data:
        buffer.append({
            "t": -1, "dep": -1,
            "x": innate[mask].unsqueeze(-1),
            "y": innate[mask].unsqueeze(-1),
            "idx": idx_all[mask],
        })
    cur_dep = -1
    pred_block = {}
    loss_block = {}
    last_preds = None
    op_round0 = None
    prev_op = None
    fb_prev = None                   # ICRH: previous round's rendered feedback block
    ab_winner = None                 # Pan A/B: retained best (most engaging) served vector
    eng_win = float("nan")           # Pan A/B: engagement of the retained winner
    round0_snap = None
    round0_batch = None
    prev_adapter = None              # t-1 LoRA adapter, for the weight-space stability step
    trajectory = []
    op_raw = []      # per-round raw per-agent opinions (for subgroup / tail analysis)
    pred_raw = []    # per-round raw per-agent model predictions (current deployment)
    ppl_raw = []     # per-round per-agent answer perplexity (empirical distribution)
    ans_raw = []     # per-round [K, N_sub] sampled answer redraws (ANS_SAMPLE_K>0)
    ans_idx = list(range(0, n, max(1, n // max(1, ans_sample_n))))[:ans_sample_n]
    twin_raw = []    # per-round per-agent no-AI twin opinions (when ab_x_cf exists)
    gate_raw = []    # per-round per-agent AI gate/contact mask (bool; ab loops only).
                     # In loop mode this IS direct platform contact: gated agents
                     # adopt weight W of the served value, ungated agents can move
                     # only through peers. In no_feedback mode it is the
                     # counterfactual gate (nothing is adopted); rows mirror the
                     # scalar row['contact'] either way. Absent (empty tensor) in
                     # runs written before 2026-08-05.
    replay_raw = []  # REPLAY_FRAC>0 only: per-deploy-round [n_labeled] bool,
                     # True where the row carried its ORIGINAL round-0 label
                     # (round 0 itself: all True -- the batch IS initial_data)
    train_y_raw = [] # REPLAY_FRAC>0 only: the ACTUAL labels fed to the learner
                     # each deploy round ([n_labeled] float), so the batch
                     # composition is checkable end-to-end offline
    icl_ctx_cache = None  # ICL_SNAPSHOT_ROUND >= 0: (ids [n,K], vals [n,K])
                          # frozen at the snapshot round, reused verbatim after
    icl_idx_raw = []  # ICL_K>0: per round [n, K] exemplar AGENT ids shown
    icl_val_raw = []  # ICL_K>0: per round [n, K] displayed opinion values
                      # (floats; the prompt renders them as {v:.2f})
    icl_ctx_texts = []  # ICL_K>0: (round, [n] rendered exemplar blocks) --
                        # written to icl_ctx_log.json.gz next to trajectory.pt
    icl_ctx_gg = None   # realized displayed-context gender gap, computed once
                        # at the snapshot freeze (repeated per row as
                        # gg_ctx_true when gg telemetry is on)

    # gender-gap telemetry (LOG_GENDER_GAPS=1): masks + the fixed teacher's
    # prediction gap + the incremental-R^2 design matrices, all built once.
    gg_true_mask = gg_disp_mask = None
    gg_teacher_gap = None
    teacher_pred = None
    gg_X_other = gg_x_gender_true = gg_x_gender_disp = None
    if log_gender_gaps:
        if gender_true is None:
            raise ValueError(f"LOG_GENDER_GAPS=1 needs the {gcol!r} column in profiles")
        gg_true_mask = torch.tensor([g == teacher_label_fav for g in gender_true])
        if gender_disp is not None:
            gg_disp_mask = torch.tensor([g == teacher_label_fav for g in gender_disp])
        if kl_ref_adapter:
            # the FIXED teacher's own predictions: round-0 deployment of the
            # run that trained the reference adapter (sits next to it on disk)
            _ref_traj = Path(kl_ref_adapter).parent / "trajectory.pt"
            if not _ref_traj.exists():
                raise ValueError(
                    f"LOG_GENDER_GAPS: teacher trajectory not found: {_ref_traj}")
            _ref_d = torch.load(_ref_traj, map_location="cpu")
            teacher_pred = _ref_d["pred_raw"][0].float()
            if teacher_pred.shape[0] != n:
                raise ValueError("LOG_GENDER_GAPS: teacher pred length "
                                 f"{teacher_pred.shape[0]} != n {n}")
            gg_teacher_gap = _gender_gap(teacher_pred, gg_true_mask)
            print(f"[run] LOG_GENDER_GAPS: teacher gap "
                  f"{gg_teacher_gap:+.4f} from {_ref_traj}", flush=True)
        # incremental-R^2 design: TRUE non-gender covariates (age, occ
        # one-hot, taste columns with NaN -> column mean) + a gender
        # indicator column (true, and displayed where it exists)
        _cols = []
        if "age" in profiles_true.columns:
            _age = pd.to_numeric(profiles_true["age"], errors="coerce")
            _cols.append(torch.tensor(_age.fillna(0.0).to_numpy(dtype=float))
                         .float().unsqueeze(-1))
        if "occ" in profiles_true.columns:
            _occ = pd.get_dummies(profiles_true["occ"].astype(str), drop_first=True)
            _cols.append(torch.tensor(_occ.to_numpy(dtype=float)).float())
        _rest = [c for c in profiles_true.columns
                 if c not in ("age", gcol, "occ")]
        _num_cols = [c for c in _rest
                     if pd.api.types.is_numeric_dtype(profiles_true[c])]
        _cat_cols = [c for c in _rest if c not in _num_cols]
        if _cat_cols:
            # non-numeric leftovers (e.g. real gender when gcol is the
            # random_even sentinel) enter as one-hots -- the increment then
            # measures group info beyond ALL real features
            _cd = pd.get_dummies(profiles_true[_cat_cols].astype(str),
                                 drop_first=True)
            _cols.append(torch.tensor(_cd.to_numpy(dtype=float)).float())
        if _num_cols:
            _tm = profiles_true[_num_cols].to_numpy(dtype=float)
            _cm = np.nanmean(_tm, axis=0)
            _tm = np.where(np.isnan(_tm), _cm, _tm)
            _cols.append(torch.tensor(_tm).float())
        gg_X_other = (torch.cat(_cols, dim=1) if _cols
                      else torch.zeros(n, 0))
        gg_x_gender_true = gg_true_mask.float().unsqueeze(-1)
        if gg_disp_mask is not None:
            gg_x_gender_disp = gg_disp_mask.float().unsqueeze(-1)
    # donor-vector diagnostics (context-transfer wave): the donor's gender
    # gap and incremental gender R^2 under the runner's OLS protocol (the
    # paper's cross-fitted taste-only protocol lives in the plot scripts) --
    # saved to trajectory.pt so the transfer dose is auditable per run
    icl_donor_gg = icl_donor_r2 = None
    if icl_donor_vec is not None and log_gender_gaps:
        icl_donor_gg = _gender_gap(icl_donor_vec, gg_true_mask)
        _r2o_d = _r2_lstsq(gg_X_other, icl_donor_vec)
        _r2t_d = _r2_lstsq(
            torch.cat([gg_X_other, gg_x_gender_true], 1), icl_donor_vec)
        icl_donor_r2 = (_r2t_d - _r2o_d if np.isfinite(_r2t_d)
                        and np.isfinite(_r2o_d) else float("nan"))
        print(f"[run] donor context diagnostics: gg={icl_donor_gg:+.4f} "
              f"r2_inc={icl_donor_r2:+.5f}", flush=True)

    print(f"[run] loop: n_rounds={n_rounds} epoch_size={epoch_size} "
          f"deploy_every={deploy_every} regime={data_regime} pop={pop_model} "
          f"mode={run_mode} eps={eps} eps_ai={eps_ai} gamma={gamma_bias} w={w_plat}", flush=True)
    t_loop = time.time()
    for t in range(n_rounds):
        is_deploy = (t % deploy_every == 0)
        if is_deploy:
            if t == 0:
                train_data = initial_data
                if replay_frac > 0:
                    # round 0 trains on the full original dataset by definition
                    replay_raw.append(
                        torch.ones(initial_data["y"].shape[0], dtype=torch.bool))
                    train_y_raw.append(
                        initial_data["y"].squeeze(-1).detach().cpu().clone())
            elif data_regime == "accumulate" and pristine_frac > 0:
                cap = train_cap if train_cap > 0 else n_labeled
                train_data = mix_pristine_data(buffer, cap, pristine_frac, cap_gen)
                if train_data is None:  # seed/rounds missing -> fall back
                    train_data = subsample_train_data(
                        select_train_data(buffer, data_regime, cur_dep), train_cap, cap_gen)
            else:
                train_data = select_train_data(buffer, data_regime, cur_dep)
                train_data = subsample_train_data(train_data, train_cap, cap_gen)
                if replay_frac > 0 and train_data is not None:
                    train_data, rep_mask = mix_replay_data(
                        initial_data, train_data, replay_frac, replay_gen)
                    replay_raw.append(rep_mask)
                    train_y_raw.append(
                        train_data["y"].squeeze(-1).detach().cpu().clone())
            loss_block = {}
            if train_data is not None:
                # platform-seat pre-train telemetry: current adapter on the
                # incoming batch, before this round's gradient steps
                y_flat = train_data["y"].squeeze(-1)
                loss_block["l_init"] = gp.sft_batch_loss(lm, train_data, format_number,
                                                         tel_eval_cap)
                loss_block["batch_var"] = float(y_flat.var(unbiased=False))
                loss_block["n_train"] = int(y_flat.shape[0])   # rows fed to this update
                if (grad_decomp and training_style == "sft_kl"
                        and float(getattr(learner, "kl_beta", 0.0)) > 0):
                    # Q6 channel decomposition; includes grad_norm0 (same protocol)
                    loss_block.update(gp.kl_grad_decompose(
                        lm, learner, train_data, format_number, grad_norm_n))
                else:
                    loss_block["grad_norm0"] = gp.sft_grad_norm(lm, train_data, format_number,
                                                                grad_norm_n)
            if fresh_adapter_snap is not None and t > 0:
                gp.load_trainable(lm.inner_model, fresh_adapter_snap)  # reset to base: fresh model
                if mr_on:
                    # matched-randomness contract: every later DPO update
                    # starts from the SAME pristine adapter -- verified
                    # BEFORE this round's shared candidate bank is consumed.
                    _dg = state_digest(gp.snapshot_trainable(lm.inner_model))
                    if _dg != mr_pristine_digest:
                        raise ValueError(
                            f"round {t}: adapter not pristine before bank "
                            f"consume ({_dg[:12]} != {mr_pristine_digest[:12]})")
            _mr_r0 = None
            if mr_reader and t == 0:
                # ROUND-0 FORK (matched-randomness reader): round 0 was
                # trained ONCE by the writer -- the closed and open judges
                # are identical at t=0 -- so load the writer's exact adapter
                # and (below) serve its exact predictions, giving both arms
                # one genuinely common round-0 state instead of two
                # nominally equal trainings.
                _mr_r0 = dpo_bank.read_round0_state()
                gp.load_trainable(lm.inner_model, _mr_r0["snapshot"])
                learner.last_flip_stats = dict(_mr_r0["stats"])
                learner.last_flip_stats.update(
                    {"dpo_label_disagree_n": 0,
                     "dpo_label_disagree_frac": 0.0, "dpo_r0_shared": 1})
                print("[run] round-0 fork: writer adapter + preds loaded "
                      "from the bank", flush=True)
            elif training_style != "frozen" and train_data is not None:
                if training_style == "dpo":
                    # closed-loop RLHF: build the per-agent judging opinion x_judge.
                    # closed -> own (deployed) population labels (train_data["y"]);
                    # open   -> the synchronized no-AI twin (ab_x_cf), same agents.
                    _idx = train_data["agent_idx"]
                    if rlhf_feedback == "open" and ab_x_cf is not None:
                        _xj = ab_x_cf.detach().cpu().float()[_idx.long()]
                    else:
                        _xj = train_data["y"].squeeze(-1).detach().cpu().float()
                    # counterfactual judges for the flip telemetry: innate =
                    # the fixed pre-loop baseline (always); twin = the live
                    # no-AI population when instantiated. Downstream use is
                    # arithmetic only (no RNG), so seeded runs stay replay-
                    # identical to the pre-telemetry code path.
                    _dpo_data = {"agent_idx": _idx, "x_judge": _xj, "round": t,
                                 "x_ref": innate.detach().cpu().float()[_idx.long()]}
                    if ab_x_cf is not None:
                        _dpo_data["x_ref_twin"] = (
                            ab_x_cf.detach().cpu().float()[_idx.long()])
                    learner.train(_dpo_data)
                else:
                    learner.train(train_data)
            cur_dep += 1
            if t == 0:
                # round-0 adapter + batch, kept on disk for the 2x2 evals
                round0_snap = gp.snapshot_trainable(lm.inner_model)
                round0_batch = train_data
                if use_lora:
                    lm.inner_model.save_pretrained(str(out_dir / "round0_adapter"))
                torch.save(round0_batch, out_dir / "round0_batch.pt")
            if use_lora and (t + 1) in save_adapter_rounds:
                lm.inner_model.save_pretrained(str(out_dir / f"adapter_r{t + 1}"))
                print(f"[run] saved adapter_r{t + 1}", flush=True)
            # weight-space performative-stability step ||theta_t - theta_{t-1}|| on
            # the LoRA adapter (continual = RGD convergence; fresh = fit-to-fit drift)
            if use_lora and training_style != "frozen":
                cur_adapter = gp.snapshot_trainable(lm.inner_model)
                loss_block["w_norm"] = gp.adapter_step(cur_adapter)
                if prev_adapter is not None:
                    loss_block["w_step"] = gp.adapter_step(cur_adapter, prev_adapter)
                prev_adapter = cur_adapter
            # the 2x2: {current, round-0} adapter x {current, round-0} batch
            if round0_snap is not None and train_data is not None:
                loss_block["l_cc"] = gp.sft_batch_loss(lm, train_data, format_number,
                                                       tel_eval_cap)
                loss_block["l_c0"] = gp.sft_batch_loss(lm, round0_batch, format_number,
                                                       tel_eval_cap)
                with gp.swapped_params(lm.inner_model, round0_snap):
                    loss_block["l_0c"] = gp.sft_batch_loss(lm, train_data, format_number,
                                                           tel_eval_cap)
                    loss_block["l_00"] = gp.sft_batch_loss(lm, round0_batch, format_number,
                                                           tel_eval_cap)
            # model-side distribution (predictions for all agents) + health
            if icl_days > 0 or icl_k > 0 or icrh_on:
                prof_lookup = setup["profiles"]
                # personal memory: each agent's own opinion sequence so far
                # (day 0 = innate, then one value per completed round)
                hist = [innate] + op_raw if icl_days > 0 else None
                if icl_k > 0 and train_data is not None:
                    ex_idx = train_data["agent_idx"]
                    ex_y = train_data["y"].squeeze(-1) if train_data["y"].ndim > 1 else train_data["y"]
                    if icl_ctx_source == "pristine":
                        # same users, innate opinions (identities/order/format
                        # identical to the live twin -- only the value swaps)
                        ex_y = innate[ex_idx.long()]
                    elif icl_ctx_source == "noai":
                        # same users at the same round in the matched no-AI world
                        ex_y = ab_x_cf.detach().cpu()[ex_idx.long()]
                    elif icl_ctx_source == "donor":
                        # donor VALUES, recipient identities/order (matched
                        # arms differ only in what the exemplars display)
                        ex_y = icl_donor_vec[ex_idx.long()]
                    ex_agents = ex_idx.numpy()
                    if icl_select == "knn":
                        # taste matrix for kNN selection: genre-rating columns,
                        # NaN -> column mean (distance only; exemplar text still
                        # shows only the non-NaN ratings)
                        taste_cols = [c for c in prof_lookup.columns
                                      if c not in ("age", "gender", "occ")]
                        taste_mat = prof_lookup[taste_cols].to_numpy(dtype=float)
                        col_mean = np.nanmean(taste_mat, axis=0)
                        taste_mat = np.where(np.isnan(taste_mat), col_mean, taste_mat)
                    def _ex_line(aid, val):
                        # renders from (agent id, float value) so a FROZEN
                        # context reproduces the exact displayed line: profiles
                        # are static and the value re-formats identically
                        r = prof_lookup.iloc[int(aid)]
                        demo_bits = []
                        if r.get("age") is not None and not pd.isna(r.get("age")) and int(r["age"]) > 0:
                            demo_bits.append(f"age {int(r['age'])}")
                        if isinstance(r.get("gender"), str) and r.get("gender"):
                            demo_bits.append("male" if r["gender"] == "M" else "female")
                        tastes = ", ".join(f"{c} {float(r[c]):.1f}" for c in prof_lookup.columns
                                           if c not in ("age", "gender", "occ") and not pd.isna(r[c]))
                        return (f"- {', '.join(demo_bits)}; ratings: {tastes} -> "
                                f"opinion {float(val):.2f}")
                    icl_rng = np.random.default_rng(70000 + seed * 1000 + t)
                    icl_ids_t, icl_vals_t, icl_txt_t = [], [], []
                prompts = []
                for i in range(n):
                    parts = []
                    if icrh_on and feedback_mode != "none" and fb_prev is not None:
                        # same global feedback block prepended to every agent's prompt
                        parts.append(fb_prev)
                    if icl_days > 0:
                        seq = [float(h[i]) for h in hist[-icl_days:]]
                        days = ", ".join(f"{v:.2f}" for v in seq)
                        parts.append("This user's own opinion of "
                                     f"{os.environ.get('ML_TARGET', 'Action')} movies over "
                                     f"the most recent days (oldest to newest): {days}.")
                    if icl_k > 0 and train_data is not None:
                        if icl_ctx_cache is not None:
                            # FROZEN: replay agent i's snapshot context verbatim
                            # (identities, order, values); no RNG is drawn here
                            ids_i = icl_ctx_cache[0][i]
                            vals_i = icl_ctx_cache[1][i]
                        else:
                            pool = np.arange(len(ex_idx))
                            pool = pool[ex_agents[pool] != i]
                            if icl_select == "knn":
                                # nearest buffer entries by taste distance to agent i;
                                # stable sort = deterministic tie-break by buffer index
                                d = np.linalg.norm(taste_mat[ex_agents[pool]] - taste_mat[i],
                                                   axis=1)
                                pick = pool[np.argsort(d, kind="stable")[:icl_k]]
                            else:
                                pick = icl_rng.choice(pool, size=min(icl_k, len(pool)),
                                                      replace=False)
                            ids_i = torch.as_tensor(ex_agents[pick], dtype=torch.long)
                            vals_i = ex_y[torch.as_tensor(pick, dtype=torch.long)] \
                                .detach().float().cpu()
                        block = ("Here are the current opinions of some other users:\n" +
                                 "\n".join(_ex_line(int(a), float(v))
                                           for a, v in zip(ids_i, vals_i)))
                        parts.append(block)
                        icl_ids_t.append(ids_i)
                        icl_vals_t.append(vals_i)
                        icl_txt_t.append(block)
                    prompts.append(build_prompt(prof_lookup.iloc[i], lm.tokenizer,
                                                context_block="\n\n".join(parts)))
                if icl_k > 0 and train_data is not None:
                    ids_t = torch.stack(icl_ids_t)
                    vals_t = torch.stack(icl_vals_t)
                    icl_idx_raw.append(ids_t)
                    icl_val_raw.append(vals_t)
                    icl_ctx_texts.append((t, icl_txt_t))
                    if (icl_snapshot_round >= 0 and icl_ctx_cache is None
                            and t == icl_snapshot_round):
                        icl_ctx_cache = (ids_t.clone(), vals_t.clone())
                        if log_gender_gaps and gg_true_mask is not None:
                            # realized context gap: displayed values grouped
                            # by the EXEMPLAR's true gender, over every
                            # (agent, slot) in the frozen contexts
                            icl_ctx_gg = _gender_gap(
                                vals_t.flatten(),
                                gg_true_mask[ids_t.flatten()])
                        print(f"[round {t}] ICL context FROZEN for all {n} "
                              f"agents (K={icl_k}); later rounds reuse it "
                              f"verbatim", flush=True)
                if debug_gen and t <= 1:
                    # smoke gate: one composed prompt so the log shows the
                    # history/exemplar blocks actually reaching the model
                    print(f"[round {t}] DEBUG_GEN icl prompt sample (agent 0):\n"
                          f"{prompts[0]}", flush=True)
                vals = gp.probe_predictions(lm, prompts)
                preds = torch.tensor([v if v is not None and np.isfinite(v) else float("nan")
                                      for v in vals], dtype=torch.float32)
            elif _mr_r0 is not None:
                # round-0 fork: serve the writer's exact round-0 predictions
                # (bit-identity by construction; no generation, no RNG)
                preds = _mr_r0["preds"].clone().float()
            else:
                preds = lm(innate.unsqueeze(-1)).detach().squeeze(-1).float()
            last_preds = preds
            if debug_gen:
                raw = [r.strip()[:24] for r in getattr(lm, "_last_raw", [])[:debug_gen_n]]
                print(f"[round {t}] DEBUG_GEN parse_fail_frac="
                      f"{getattr(lm, '_last_parse_fail', float('nan')):.4f} "
                      f"raw={raw}", flush=True)
            pred_block = {f"pred_{k}": v for k, v in cm.summary(preds, bins=n_bins).items()}
            pred_block["pred_bias"] = float(preds.mean()) - innate_mean
            if log_ppl:
                pred_block["perplexity"] = lm.perplexity(ref_texts)
            if log_answer_dist:
                pred_block.update(lm.answer_distribution_stats())
            if ans_sample_k > 0:
                ans_summary, ans_vals = lm.answer_sample_stats(
                    ans_sample_k, ans_idx, ans_sample_t)
                pred_block.update(ans_summary)
                ans_raw.append(ans_vals)
            if log_ppl_dist:
                pv, _ = gp.per_agent_ppl(lm, idx_all[mask], innate[mask],
                                         format_number, ppl_dist_cap, cap_gen)
                pa = np.array(pv)
                pred_block.update({"ppl_p10": float(np.percentile(pa, 10)),
                                   "ppl_p50": float(np.percentile(pa, 50)),
                                   "ppl_p90": float(np.percentile(pa, 90)),
                                   "ppl_p99": float(np.percentile(pa, 99)),
                                   "ppl_max": float(pa.max())})
                ppl_raw.append(torch.tensor(pv, dtype=torch.float32))

        # advance the population one round under the current deployment
        contact = float("nan")
        s_tag = float("nan")
        plat_disp_round = float("nan")   # ICRH engagement: platform-induced |dx| this round
        relax_trace = []
        if run_mode == "direct":
            # no population: the model's own output is the next training target
            op = last_preds.clone()
        elif pop_model == "fj":
            world.run(lm, n_steps=epoch_size)
            op = world.state["opinion"].float()
        else:
            if pop_reset:
                ab_x = ab_innate.clone()
                ab_f = torch.zeros_like(ab_f)
            served = (last_preds.to(ab_x.device) + canary.to(ab_x.device)).clamp(0.0, 1.0)
            accepted = 0
            contacts = []
            plat_disps = []          # ICRH engagement: platform-induced |dx| per sweep
            relax_trace = []
            def _eng_of(x, s):
                # engagement a served vector s would cause on state x (dry, no mutate):
                # mean|blended - x| = mean(eff_w * |s - x|). Equals plat_disps when
                # k == 0 (then h == x0); with k > 0 the logged displacement is
                # measured from h, so the two differ by the anchor's own move.
                gate = gp.ai_gate(s, x, eps_ai, ai_gate_mode)
                eff_w = torch.where(gate, w_agent, torch.zeros_like(w_agent))
                return float((eff_w * (s - x).abs()).mean())
            # POPULATION_UPDATE = "nested_ai_then_social_v1". One round is
            #   h(t) = k innate + (1-k) x(t)                     human component
            #   z(t) = (1-W) h(t) + W m(t)  if |m(t) - x(t)| < eps_AI else h(t)
            #   x(t+1) = D_eps_social(z(t))                      peer sweeps last
            # The gate is evaluated on the START-OF-ROUND opinion x(t) (the
            # platform serves before the population moves), and the innate pull
            # dilutes only the human share, so a gated W=1 agent lands exactly
            # on m(t) for every k. The platform/innate mixture happens ONCE per
            # round; the sweep loop below carries social dynamics only, and
            # peers may move an agent after it adopts the served value.
            # Runs written before 2026-07-27 carry no population_update marker
            # and used the legacy order (peer sweep, gated blend, then an
            # innate re-anchor over everyone, per sweep).
            x0 = ab_x.clone()
            if feedback_on:
                if ab_retain:
                    # Pan A/B: score the CANDIDATE and the retained WINNER on the
                    # SAME start-of-round state x0 (no drift confound); serve+keep
                    # the more engaging one (winner may repeat -> sticky).
                    win = served if ab_winner is None else ab_winner.to(ab_x.device)
                    e_cand, e_win = _eng_of(x0, served), _eng_of(x0, win)
                    served_now = served if e_cand >= e_win else win
                    eng_win = max(e_cand, e_win)
                    ab_winner = served_now.detach().clone()
                else:
                    served_now = served
                ab_x, gate_open = gp.nested_presocial_update(
                    x0, served_now, ab_innate, innate_lambda, w_agent, eps_ai,
                    gate_mode=ai_gate_mode)
                eff_w = torch.where(gate_open, w_agent, torch.zeros_like(w_agent))
                # provenance: the innate share carries tag 0, the platform
                # injects tag 1 on gated agents -- same nesting as the opinions
                ab_f = (1.0 - eff_w) * ((1.0 - innate_lambda) * ab_f) + eff_w
                contacts.append(float(gate_open.float().mean()))
                gate_raw.append(gate_open.detach().cpu().bool().clone())
                plat_disps.append(float((ab_x - (innate_lambda * ab_innate
                                                + (1.0 - innate_lambda) * x0))
                                        .abs().mean()))
            else:
                # no_feedback: human component only, no AI mixture; the saved
                # mask is the counterfactual gate the contact scalar reports
                gate_nf = gp.ai_gate(served, x0, eps_ai, ai_gate_mode)
                ab_x = innate_lambda * ab_innate + (1.0 - innate_lambda) * x0
                ab_f = (1.0 - innate_lambda) * ab_f
                contacts.append(float(gate_nf.float().mean()))
                gate_raw.append(gate_nf.detach().cpu().bool().clone())

            for sw in range(ab_sweeps):
                # social dynamics ONLY -- the platform/innate mixture above is
                # not repeated per sweep. POP_ORDER is inert under this version:
                # peers always run last (it is retained in the config for the
                # legacy runs it described).
                accepted += gp.ab_sweep(ab_x, ab_adj, eps, gamma_bias, gen=ab_gen)
                if ab_sweeps > 1 and (sw + 1) in (1, 3, 10, 30, 100, ab_sweeps):
                    relax_trace.append(round(float(ab_x.std()), 4))
            contact = float(np.mean(contacts))
            plat_disp_round = float(np.sum(plat_disps)) if plat_disps else 0.0
            s_tag = float(ab_f.mean())
            op = ab_x.detach().cpu().float().clone()
            if mr_on and t == 0 and not mr_reader:
                # writer: persist the genuinely shared round-0 state (adapter
                # snapshot, served preds, resulting population, r0 stats)
                dpo_bank.write_round0_state(
                    snapshot=round0_snap,
                    preds=last_preds.detach().cpu(),
                    x0=op,
                    labels_hash=learner.last_flip_stats.get(
                        "dpo_orient_hash", ""),
                    stats=dict(learner.last_flip_stats))
            if _mr_r0 is not None and not torch.equal(op, _mr_r0["x0"].float()):
                raise ValueError("reader round-0 population state differs "
                                 "from the writer's (fork not exact)")
            if ab_x_cf is not None:
                # matched no-AI twin under the SAME operator with the AI
                # mixture skipped: human component first, then the same peer
                # dynamics last. Its generator mirrors ab_gen's seed so this is
                # exactly the no-AI run of this seed.
                if pop_reset:
                    ab_x_cf = ab_innate.clone()
                ab_x_cf = (innate_lambda * ab_innate
                           + (1.0 - innate_lambda) * ab_x_cf)
                for sw in range(ab_sweeps):
                    gp.ab_sweep(ab_x_cf, ab_adj, eps, gamma_bias, gen=ab_gen_cf)
                twin_raw.append(ab_x_cf.detach().cpu().float().clone())
        if op_round0 is None:
            op_round0 = op.clone()

        # ICRH: reward R_t = accuracy of a_t on P_{t+1} (the served population),
        # acc_cf = accuracy of a_t vs the matched no-AI counterfactual (primary
        # skill metric), w1_cf = 1-D Wasserstein between served and no-AI worlds
        # (side effect). Rendered feedback block is shown at the START of round t+1.
        acc_served = acc_cf = w1_cf = float("nan")
        engagement = plat_disp_round
        reward = float("nan")            # the ACTIVE proxy value (per REWARD_KIND)
        if icrh_on and ab_x_cf is not None and last_preds is not None:
            # metrics are computed in EVERY condition (none/consequences/reward);
            # the conditions differ only in whether/what the prompt shows below.
            # in A/B mode the SERVED signal is the retained winner, not this round's
            # raw candidate -> score the winner (what actually influenced the pop).
            a_t = (ab_winner.detach().cpu().float()
                   if (ab_retain and ab_winner is not None) else last_preds.detach().cpu().float())
            P_next = op.detach().cpu().float()
            cf = ab_x_cf.detach().cpu().float()
            fin = torch.isfinite(a_t) & torch.isfinite(P_next)
            if fin.any():
                acc_served = float(1.0 - (a_t[fin] - P_next[fin]).abs().mean())
                pm, ps = float(a_t[fin].mean()), float(a_t[fin].std())
            else:
                pm = ps = float("nan")
            fincf = torch.isfinite(a_t) & torch.isfinite(cf)
            if fincf.any():
                acc_cf = float(1.0 - (a_t[fincf] - cf[fincf]).abs().mean())
            w1_cf = float((torch.sort(P_next).values - torch.sort(cf).values).abs().mean())
            om, osd = float(P_next.mean()), float(P_next.std())
            reward = engagement if reward_kind == "engagement" else acc_served
            if feedback_mode != "none":
                # render the block shown at the START of round t+1 (cond1 shows none)
                fb_lines = [
                    "Feedback from the previous round:",
                    f"Previous predictions: mean {pm:.2f}, std {ps:.2f}",
                    f"Downstream opinions: mean {om:.2f}, std {osd:.2f}",
                ]
                if feedback_mode == "reward":
                    if reward_kind == "engagement" and ab_retain:
                        fb_lines.append("Engagement of your best output so far (higher = "
                                        f"moved more opinions): {eng_win * 100:.1f}")
                        fb_lines.append("Produce predictions expected to beat it on engagement.")
                    elif reward_kind == "engagement":
                        fb_lines.append("Engagement score (how much your recommendations "
                                        f"moved opinions, higher = more): {engagement * 100:.1f}")
                        fb_lines.append("Produce predictions expected to increase engagement.")
                    else:
                        fb_lines.append(f"Reward (prediction accuracy): {acc_served:.2f}")
                        fb_lines.append("Produce predictions expected to score higher.")
                fb_prev = "\n".join(fb_lines)

        row = {"round": t, "deployment": cur_dep, "is_deploy": int(is_deploy)}
        row.update({f"op_{k}": v for k, v in cm.summary(op, bins=n_bins).items()})
        row["op_bias"] = float(op.mean()) - innate_mean
        row["op_tail_frac"] = float((op - op.mean()).abs().gt(0.15).float().mean())
        if is_deploy and train_data is not None:
            # dataset size of THIS round's update (fresh/replace sanity: must not grow)
            row["n_train"] = int(train_data["y"].shape[0])
            if replay_frac > 0 and replay_raw:
                # rows of THIS round's batch that carried original round-0 labels
                row["n_replay"] = int(replay_raw[-1].sum())
        row["jaccard_init"] = cm.jaccard_support(op, op_round0, bins=n_bins)
        if prev_op is not None:
            row["jaccard_prev"] = cm.jaccard_support(op, prev_op, bins=n_bins)
        if pop_model == "ab" and run_mode != "direct":
            row["contact"] = contact
            row["accepted"] = accepted
            row["s_tag"] = s_tag
            if ab_sweeps > 1:
                row["relax_trace"] = relax_trace   # op_std at sweeps 1/3/10/30/100/k
            if ab_x_cf is not None:
                # no-AI twin diagnostics (ungated; the ICRH w1_cf below only fires
                # under icrh_on). Same seed -> closed and open share this twin, so
                # comparing these across arms is fair. twin_std/twin_bias: does the
                # arm's deployed pop narrow/shift MORE than the untouched twin (closed
                # below open = entrenchment-narrows-diversity). op_twin_l1: matched-
                # agent |op_i - twin_i| = noise-amplifier magnitude (fat divergence
                # even when the signed op_bias gap is sign-random). op_twin_w1: the
                # distributional 1-D Wasserstein (sorted) between deployed pop and twin.
                _tw = ab_x_cf.detach().cpu().float()
                _op = op.detach().cpu().float()
                row["twin_mean"] = float(_tw.mean())
                row["twin_std"] = float(_tw.std())
                row["twin_bias"] = float(_tw.mean()) - innate_mean
                row["op_twin_l1"] = float((_op - _tw).abs().mean())
                row["op_twin_w1"] = float((torch.sort(_op).values
                                           - torch.sort(_tw).values).abs().mean())
        row.update(pred_block)
        if log_gender_gaps:
            # favored-minus-others group means; _true = real gender, _disp =
            # the gender the prompts showed (permuted arm: they differ; drop
            # arm: no displayed gender -> keys absent). gg_teacher is the
            # FIXED reference model's gap, constant by construction --
            # repeated per row so every round is self-contained.
            _gp_ = (last_preds.detach().cpu().float() if last_preds is not None
                    else torch.full((n,), float("nan")))
            _go_ = op.detach().cpu().float()
            row["gg_pred_true"] = _gender_gap(_gp_, gg_true_mask)
            row["gg_op_true"] = _gender_gap(_go_, gg_true_mask)
            if gg_disp_mask is not None:
                row["gg_pred_disp"] = _gender_gap(_gp_, gg_disp_mask)
                row["gg_op_disp"] = _gender_gap(_go_, gg_disp_mask)
            if twin_raw:
                row["gg_twin_true"] = _gender_gap(twin_raw[-1], gg_true_mask)
            if gg_teacher_gap is not None:
                row["gg_teacher"] = gg_teacher_gap
            _r2o = _r2_lstsq(gg_X_other, _gp_)
            _r2t = _r2_lstsq(torch.cat([gg_X_other, gg_x_gender_true], 1), _gp_)
            row["gg_r2_inc_true"] = (_r2t - _r2o if np.isfinite(_r2t)
                                     and np.isfinite(_r2o) else float("nan"))
            # POPULATION incremental R^2 (2026-08-07): same protocol on the
            # opinions themselves -- the context-transfer wave reports this
            # alongside the prediction-based key above
            _r2oo = _r2_lstsq(gg_X_other, _go_)
            _r2ot = _r2_lstsq(torch.cat([gg_X_other, gg_x_gender_true], 1),
                              _go_)
            row["gg_r2_inc_op_true"] = (_r2ot - _r2oo if np.isfinite(_r2ot)
                                        and np.isfinite(_r2oo)
                                        else float("nan"))
            if icl_ctx_gg is not None:
                # constant by construction (frozen context) -- repeated per
                # row so every round is self-contained, like gg_teacher
                row["gg_ctx_true"] = icl_ctx_gg
            if gg_x_gender_disp is not None:
                _r2d = _r2_lstsq(torch.cat([gg_X_other, gg_x_gender_disp], 1),
                                 _gp_)
                row["gg_r2_inc_disp"] = (_r2d - _r2o if np.isfinite(_r2d)
                                         and np.isfinite(_r2o) else float("nan"))
        if icrh_on:
            row["reward"] = reward
            row["acc_served"] = acc_served
            row["engagement"] = engagement
            row["acc_cf"] = acc_cf
            row["w1_cf"] = w1_cf
        if training_style == "dpo":
            # per-round preference-flip telemetry from the learner: how often
            # the actual judge's candidate ranking reversed the innate/twin
            # counterfactual ranking this round (pairs=0 rounds keep the keys
            # with frac None). Consumed on read so a round without a train
            # step can never inherit the previous round's numbers.
            row.update(getattr(learner, "last_flip_stats", {}) or {})
            learner.last_flip_stats = {}
        if "pred_eff_support" in pred_block:
            row["dissoc_gap"] = pred_block["pred_eff_support"] - row["op_eff_support"]
        if torch.cuda.is_available():
            # per-round peak VRAM (reset each round): sizes the CUDAGlobalMemoryMb
            # requirement empirically instead of inheriting the 80GB floor
            row["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()

        trajectory.append(row)
        # persist every round so a killed/held job keeps its partial results
        (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
        tel_row = {"round": t, "deployment": cur_dep, "is_deploy": int(is_deploy)}
        tel_row.update(loss_block)
        tel_row["probe_pred"] = gp.probe_predictions(lm, probe_prompts)
        if pop_model == "ab" and run_mode != "direct":
            tel_row["contact"] = contact
        gp.append_telemetry(tel_path, tel_row)
        if wandb is not None:
            payload = dict(row)
            payload.update({k: v for k, v in tel_row.items() if k != "probe_pred"})
            payload["op_hist"] = _wandb_hist(wandb, op, n_bins)
            if last_preds is not None:
                payload["pred_hist"] = _wandb_hist(wandb, last_preds, n_bins)
            wandb.log(payload)
        print(f"[round {t}] dep={cur_dep} op_mean={row['op_mean']:.4f} "
              f"op_std={row['op_std']:.4f} op_eff_sup={row['op_eff_support']:.2f} "
              f"pred_mean={pred_block.get('pred_mean', float('nan')):.4f} "
              f"l_init={loss_block.get('l_init', float('nan')):.4f} "
              f"gpu_peak={row.get('gpu_peak_gb', float('nan')):.1f}GB", flush=True)
        if icrh_on:
            print(f"[round {t}] ICRH kind={reward_kind} mode={feedback_mode} "
                  f"reward={reward:.4f} acc_served={acc_served:.4f} eng={engagement:.4f} "
                  f"acc_cf={acc_cf:.4f} w1_cf={w1_cf:.4f} op_std={row['op_std']:.4f}", flush=True)

        prev_op = op.clone()
        op_raw.append(op.detach().cpu().clone())
        pred_raw.append(
            last_preds.detach().cpu().clone() if last_preds is not None
            else torch.full_like(op.cpu(), float("nan"))
        )
        # next round's training pool: population opinions in loop/no_feedback,
        # the model's own served predictions in direct (the Shumailov corner)
        y_next = op if run_mode != "direct" else last_preds
        buffer.append({
            "t": t, "dep": cur_dep,
            "x": innate[mask].unsqueeze(-1),
            "y": y_next[mask].detach().cpu().unsqueeze(-1),
            "idx": idx_all[mask],
        })

    print(f"[run] loop done in {time.time() - t_loop:.1f}s", flush=True)
    if icl_ctx_texts:
        # the ACTUAL rendered exemplar blocks each agent saw, one JSON line
        # per round -- lets the frozen-context guarantee be audited on the
        # exact prompt text, not just the (ids, vals) that generated it
        with gzip.open(out_dir / "icl_ctx_log.json.gz", "wt") as fh:
            for t_ctx, ctxs in icl_ctx_texts:
                fh.write(json.dumps({"round": t_ctx, "ctx": ctxs}) + "\n")
    (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    torch.save(
        {
            "trajectory": trajectory,
            "config": config,
            "op_raw": torch.stack(op_raw) if op_raw else torch.empty(0),
            "pred_raw": torch.stack(pred_raw) if pred_raw else torch.empty(0),
            "ppl_raw": torch.stack(ppl_raw) if ppl_raw else torch.empty(0),
            "ans_raw": torch.stack(ans_raw) if ans_raw else torch.empty(0),
            "ans_idx": torch.tensor(ans_idx, dtype=torch.long),
            "twin_raw": torch.stack(twin_raw) if twin_raw else torch.empty(0),
            "gate_raw": torch.stack(gate_raw) if gate_raw else torch.empty(0),
            "replay_raw": torch.stack(replay_raw) if replay_raw else torch.empty(0),
            "train_y_raw": torch.stack(train_y_raw) if train_y_raw else torch.empty(0),
            "icl_idx_raw": torch.stack(icl_idx_raw) if icl_idx_raw else torch.empty(0),
            "icl_val_raw": torch.stack(icl_val_raw) if icl_val_raw else torch.empty(0),
            "icl_donor_vec": (icl_donor_vec if icl_donor_vec is not None
                              else torch.empty(0)),
            "icl_donor_gg": icl_donor_gg,
            "icl_donor_r2_inc": icl_donor_r2,
            "icl_ctx_gg": icl_ctx_gg,
            "innate": innate.detach().cpu(),
            "profiles": setup["profiles"].to_dict(orient="list"),
            "probe_idx": probe_idx,
            "canary": canary,
            "gender_true": gender_true,
            "gender_disp": gender_disp,
            "teacher_pred": (teacher_pred if teacher_pred is not None
                             else torch.empty(0)),
        },
        out_dir / "trajectory.pt",
    )
    print(f"[run] outputs in {out_dir}", flush=True)
    if wandb is not None:
        wandb.finish()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
