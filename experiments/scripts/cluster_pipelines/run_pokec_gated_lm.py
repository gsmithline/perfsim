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
  probe_pred  predictions on a fixed innate-stratified probe set of N_PROBE
              real agents (chosen at round 0, saved to probe_set.json);
              displacement / leash are computed offline from these
  l_cc l_c0 l_0c l_00   the 2x2: {current, round-0} adapter x {current batch,
              round-0 archive batch}; round-0 adapter + batch kept on disk
              (round0_adapter/, round0_batch.pt)
"""

from __future__ import annotations

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

    def build_prompt(profile, tokenizer):
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
        user_msg = (
            f"Estimate how much this user likes {target} movies based on their profile.\n"
            "Profile:\n"
            f"{body}\n\n"
            f"Output a single number in [0, 1] (1 = loves {target}, 0 = dislikes {target}). "
            "Respond with only the number, e.g. 0.42."
        )
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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


def main() -> int:
    run_tag = _env_or("RUN_TAG")
    kl_beta = _env_float("KL_BETA", 0.0)
    training_style = _env_or("TRAINING_STYLE", "sft_kl")
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
    # FRESH_EACH_ROUND=1: retrain a NEW adapter from the cached base every round
    # (weights do NOT carry over) -- the model-collapse / Gerstgrasser protocol.
    # Default 0 = continual SFT (weights persist, the performative-prediction RGD).
    fresh_each_round = _env_int("FRESH_EACH_ROUND", 0) == 1
    # POP_RESET=1 (ab only): restart the population from innate every round, so
    # each round is a one-step response to the current deployment and history
    # lives only in the model -- the memoryless D(theta) of Wu/Abebe/
    # Mendler-Duenner (2603.12137). Default 0 = state carries over.
    pop_reset = _env_int("POP_RESET", 0) == 1
    # AB_SWEEPS>1 (ab only): sweeps per round, gate blended after each sweep.
    # With POP_RESET this is the equilibrated-response protocol: predictions +
    # innate seed a Deffuant run, the population relaxes under the platform,
    # the model trains on the relaxed outcome.
    ab_sweeps = _env_int("AB_SWEEPS", 1)
    # PRISTINE_FRAC>0 (accumulate only): hold this fraction of every training
    # subsample as round-0 real data (dep=-1), so the real fraction does NOT decay
    # to 1/t. 0 = random subsample (decaying pristine = plain accumulate).
    pristine_frac = _env_float("PRISTINE_FRAC", 0.0)

    if pop_model not in ("fj", "ab"):
        raise ValueError(f"unknown POP_MODEL: {pop_model!r}")
    if run_mode not in ("loop", "no_feedback", "direct"):
        raise ValueError(f"unknown RUN_MODE: {run_mode!r}")
    if canary_delta > 0 and not (pop_model == "ab" and run_mode == "loop"):
        raise ValueError("CANARY_DELTA>0 requires POP_MODEL=ab and RUN_MODE=loop")
    if pop_reset and pop_model != "ab":
        raise ValueError("POP_RESET=1 requires POP_MODEL=ab")
    if ab_sweeps != 1 and pop_model != "ab":
        raise ValueError("AB_SWEEPS>1 requires POP_MODEL=ab")

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag, "kl_beta": kl_beta, "training_style": training_style,
        "base_model": base_model, "n_rounds": n_rounds, "epoch_size": epoch_size,
        "deploy_every": deploy_every, "data_regime": data_regime, "seed": seed,
        "n_labeled": n_labeled, "max_steps": max_steps, "sft_epochs": sft_epochs,
        "sft_batch_size": sft_batch_size,
        "lora_r": lora_r, "use_lora": use_lora, "sft_lr": sft_lr, "hist_bins": n_bins,
        "seed_base_data": seed_base_data, "train_cap": train_cap,
        "platform_sus_scale": platform_scale, "anchor_mode": anchor_mode,
        "pop_model": pop_model, "eps": eps, "eps_ai": eps_ai, "gamma_bias": gamma_bias,
        "w_plat": w_plat, "innate_lambda": innate_lambda,
        "run_mode": run_mode, "canary_delta": canary_delta,
        "n_probe": n_probe, "tel_eval_cap": tel_eval_cap, "grad_norm_n": grad_norm_n,
        "fresh_each_round": fresh_each_round, "pristine_frac": pristine_frac,
        "pop_reset": pop_reset, "ab_sweeps": ab_sweeps,
        "dataset": os.environ.get("DATASET", "pokec"),
        "ml_target": os.environ.get("ML_TARGET", "Action"),
        "log_ppl_dist": log_ppl_dist, "ppl_dist_cap": ppl_dist_cap,
        "do_sample": do_sample, "gen_temperature": gen_temperature,
        "host": os.uname().nodename,
    }
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
        learner = KLSFTLearner(**learner_kwargs, ref_model_name=base_model, kl_beta=kl_beta,
                               anchor_mode=anchor_mode)
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

    # per-agent platform weight: scaled FJ trust for fj, gated blend weight
    # W_PLAT * platform_sus * scale for ab; no_feedback zeroes both.
    feedback_on = run_mode == "loop"
    plat_sus_eff = (setup["platform_sus"] * platform_scale).clamp(0.0, 1.0)
    if not feedback_on:
        plat_sus_eff = plat_sus_eff * 0.0

    world = None
    ab_x = None
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

    mask = torch.zeros(n, dtype=torch.bool)
    mask[:n_labeled] = True
    idx_all = torch.arange(n)
    initial_data = {
        "x": innate[mask].unsqueeze(-1),
        "y": innate[mask].unsqueeze(-1),
        "agent_idx": idx_all[mask],
    }

    buffer = []
    cap_gen = torch.Generator().manual_seed(seed)
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
    round0_snap = None
    round0_batch = None
    prev_adapter = None              # t-1 LoRA adapter, for the weight-space stability step
    trajectory = []
    op_raw = []      # per-round raw per-agent opinions (for subgroup / tail analysis)
    pred_raw = []    # per-round raw per-agent model predictions (current deployment)
    ppl_raw = []     # per-round per-agent answer perplexity (empirical distribution)

    print(f"[run] loop: n_rounds={n_rounds} epoch_size={epoch_size} "
          f"deploy_every={deploy_every} regime={data_regime} pop={pop_model} "
          f"mode={run_mode} eps={eps} eps_ai={eps_ai} gamma={gamma_bias} w={w_plat}", flush=True)
    t_loop = time.time()
    for t in range(n_rounds):
        is_deploy = (t % deploy_every == 0)
        if is_deploy:
            if t == 0:
                train_data = initial_data
            elif data_regime == "accumulate" and pristine_frac > 0:
                cap = train_cap if train_cap > 0 else n_labeled
                train_data = mix_pristine_data(buffer, cap, pristine_frac, cap_gen)
                if train_data is None:  # seed/rounds missing -> fall back
                    train_data = subsample_train_data(
                        select_train_data(buffer, data_regime, cur_dep), train_cap, cap_gen)
            else:
                train_data = select_train_data(buffer, data_regime, cur_dep)
                train_data = subsample_train_data(train_data, train_cap, cap_gen)
            loss_block = {}
            if train_data is not None:
                # platform-seat pre-train telemetry: current adapter on the
                # incoming batch, before this round's gradient steps
                y_flat = train_data["y"].squeeze(-1)
                loss_block["l_init"] = gp.sft_batch_loss(lm, train_data, format_number,
                                                         tel_eval_cap)
                loss_block["batch_var"] = float(y_flat.var(unbiased=False))
                loss_block["grad_norm0"] = gp.sft_grad_norm(lm, train_data, format_number,
                                                            grad_norm_n)
            if fresh_adapter_snap is not None and t > 0:
                gp.load_trainable(lm.inner_model, fresh_adapter_snap)  # reset to base: fresh model
            if training_style != "frozen" and train_data is not None:
                learner.train(train_data)
            cur_dep += 1
            if t == 0:
                # round-0 adapter + batch, kept on disk for the 2x2 evals
                round0_snap = gp.snapshot_trainable(lm.inner_model)
                round0_batch = train_data
                if use_lora:
                    lm.inner_model.save_pretrained(str(out_dir / "round0_adapter"))
                torch.save(round0_batch, out_dir / "round0_batch.pt")
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
            relax_trace = []
            for sw in range(ab_sweeps):
                accepted += gp.ab_sweep(ab_x, ab_adj, eps, gamma_bias, gen=ab_gen)
                if feedback_on:
                    gate_open = (served - ab_x).abs() < eps_ai
                    eff_w = torch.where(gate_open, w_agent, torch.zeros_like(w_agent))
                    ab_x, c = gp.gated_blend(ab_x, served, w_agent, eps_ai)
                    ab_f = (1.0 - eff_w) * ab_f + eff_w   # platform injects tag 1 on gated agents
                    contacts.append(c)
                else:
                    contacts.append(float(((served - ab_x).abs() < eps_ai).float().mean()))
                if innate_lambda > 0:
                    ab_x = (1.0 - innate_lambda) * ab_x + innate_lambda * ab_innate
                    ab_f = (1.0 - innate_lambda) * ab_f   # innate re-anchor carries tag 0
                if ab_sweeps > 1 and (sw + 1) in (1, 3, 10, 30, 100, ab_sweeps):
                    relax_trace.append(round(float(ab_x.std()), 4))
            contact = float(np.mean(contacts))
            s_tag = float(ab_f.mean())
            op = ab_x.detach().cpu().float().clone()
        if op_round0 is None:
            op_round0 = op.clone()

        row = {"round": t, "deployment": cur_dep, "is_deploy": int(is_deploy)}
        row.update({f"op_{k}": v for k, v in cm.summary(op, bins=n_bins).items()})
        row["op_bias"] = float(op.mean()) - innate_mean
        row["op_tail_frac"] = float((op - op.mean()).abs().gt(0.15).float().mean())
        row["jaccard_init"] = cm.jaccard_support(op, op_round0, bins=n_bins)
        if prev_op is not None:
            row["jaccard_prev"] = cm.jaccard_support(op, prev_op, bins=n_bins)
        if pop_model == "ab" and run_mode != "direct":
            row["contact"] = contact
            row["accepted"] = accepted
            row["s_tag"] = s_tag
            if ab_sweeps > 1:
                row["relax_trace"] = relax_trace   # op_std at sweeps 1/3/10/30/100/k
        row.update(pred_block)
        if "pred_eff_support" in pred_block:
            row["dissoc_gap"] = pred_block["pred_eff_support"] - row["op_eff_support"]

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
              f"l_init={loss_block.get('l_init', float('nan')):.4f}", flush=True)

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
    (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    torch.save(
        {
            "trajectory": trajectory,
            "config": config,
            "op_raw": torch.stack(op_raw) if op_raw else torch.empty(0),
            "pred_raw": torch.stack(pred_raw) if pred_raw else torch.empty(0),
            "ppl_raw": torch.stack(ppl_raw) if ppl_raw else torch.empty(0),
            "innate": innate.detach().cpu(),
            "profiles": setup["profiles"].to_dict(orient="list"),
            "probe_idx": probe_idx,
            "canary": canary,
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
