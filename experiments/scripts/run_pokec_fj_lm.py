"""Pokec FJ + LM opinion-dynamics run with deployment schedules + collapse metrics.

The LM predicts each agent's smoking attitude in [0, 1]; those predictions seed
FJ on the real Pokec graph; the population settles (one FJ loop); the LM is
fine-tuned (SFT / KL-SFT) on the labeled subset's evolved opinions. We measure
the full collapse signature on BOTH distributions every round:

  - model predictions  (what the LM outputs per agent)
  - population opinion  (the FJ-settled opinion, the key measure)

Deployment schedule (env DEPLOY_EVERY = K, DATA_REGIME):
  DEPLOY_EVERY=1                      deploy/retrain every round
  DEPLOY_EVERY=K>1                    hold deployment fixed K rounds, then retrain
  DATA_REGIME=replace                 train on the most recent round only
  DATA_REGIME=accumulate              train on all rounds so far
  DATA_REGIME=deployed_into           train on rounds generated under the live deployment
  DATA_REGIME=not_deployed_into       train on rounds from earlier deployments only

Per-round metrics (logged to W&B if WANDB_PROJECT set), for each distribution:
  mean std var entropy eff_support occupied_frac low_prob_mass mode_mass gini
plus op/pred bias vs innate truth (error amplification), Jaccard vs round 0 and
vs previous round (support drift), perplexity (model health), and the
dissociation gap (pred eff_support vs population eff_support).
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
        "W": W,
        "peer_sus": torch.tensor(np.asarray(peer_sus), dtype=torch.float32),
        "platform_sus": torch.tensor(np.asarray(platform_sus), dtype=torch.float32),
        "n": len(profiles),
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


def main() -> int:
    run_tag = _env_or("RUN_TAG")
    kl_beta = _env_float("KL_BETA", 0.0)
    training_style = _env_or("TRAINING_STYLE", "sft_kl")
    base_model = _env_or("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    n_rounds = _env_int("N_ROUNDS", 12)
    epoch_size = _env_int("EPOCH_SIZE", 10)
    deploy_every = _env_int("DEPLOY_EVERY", 1)
    data_regime = os.environ.get("DATA_REGIME", "replace")
    seed = _env_int("SEED", 0)
    n_labeled = _env_int("N_LABELED", 1730)
    pokec_dir = Path(os.environ.get("POKEC_DIR", "examples/pokec"))
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(os.environ.get("OUT_DIR", f"runs/pokec_fj_lm/{run_tag}"))
    wandb_project = os.environ.get("WANDB_PROJECT")
    max_steps = _env_int("SFT_MAX_STEPS", 1)
    sft_epochs = _env_int("SFT_EPOCHS", 1)
    gen_batch_size = _env_int("GEN_BATCH_SIZE", 32)
    sft_batch_size = _env_int("SFT_BATCH_SIZE", 2)
    lora_r = _env_int("LORA_R", 8)
    use_lora = _env_int("USE_LORA", 1) == 1
    sft_lr = _env_float("SFT_LR", 5e-5)
    max_new_tokens = _env_int("MAX_NEW_TOKENS", 6)
    n_bins = _env_int("HIST_BINS", 50)
    log_ppl = _env_int("LOG_PERPLEXITY", 1) == 1
    n_ppl = _env_int("N_PERPLEXITY", 64)
    log_answer_dist = _env_int("LOG_ANSWER_DIST", 1) == 1

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_tag": run_tag, "kl_beta": kl_beta, "training_style": training_style,
        "base_model": base_model, "n_rounds": n_rounds, "epoch_size": epoch_size,
        "deploy_every": deploy_every, "data_regime": data_regime, "seed": seed,
        "n_labeled": n_labeled, "max_steps": max_steps, "sft_epochs": sft_epochs,
        "sft_batch_size": sft_batch_size,
        "lora_r": lora_r, "use_lora": use_lora, "sft_lr": sft_lr, "hist_bins": n_bins,
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
    print(f"[run] loading Pokec from {pokec_dir}", flush=True)
    t0 = time.time()
    setup = load_pokec_setup(pokec_dir)
    n = setup["n"]
    innate = setup["innate"]
    innate_mean = float(innate.mean())
    print(f"[run] pokec ready: N={n}  innate mean={innate_mean:.4f} "
          f"std={innate.std():.4f} in {time.time() - t0:.1f}s", flush=True)

    def build_prompt(profile: pd.Series, tokenizer) -> str:
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

    # Epoch-based training (matches the opinion-dynamics study: num_train_epochs
    # per round, not a fixed step count). SFT_EPOCHS=0 falls back to max_steps.
    trainer_kwargs = {"num_train_epochs": sft_epochs, "max_steps": -1} if sft_epochs > 0 else {}
    learner_kwargs = dict(
        model=lm, loss=MSELoss(), max_steps=max_steps,
        per_device_batch_size=sft_batch_size, output_dir=str(out_dir / "trl"),
        response_template="<|im_start|>assistant\n", learning_rate=sft_lr,
        target_formatter=format_number, trainer_kwargs=trainer_kwargs,
    )
    if training_style == "sft":
        learner = SFTLearner(**learner_kwargs)
    elif training_style == "sft_kl":
        learner = KLSFTLearner(**learner_kwargs, ref_model_name=base_model, kl_beta=kl_beta)
    elif training_style == "frozen":
        class _Frozen(Learner):
            accepted_schemas = (SUPERVISED_SCHEMA,)
            def __init__(self, model, loss): super().__init__(model, loss)
            def train(self, data): pass
            def reset(self): pass
        learner = _Frozen(model=lm, loss=MSELoss())
    else:
        raise ValueError(f"unknown TRAINING_STYLE: {training_style!r}")

    world = FJWorld(
        innate=innate, graph=setup["W"], peer_sus=setup["peer_sus"],
        platform_sus=setup["platform_sus"], features=innate, profiles=setup["profiles"],
    )
    world.reset(seed=seed)

    mask = torch.zeros(n, dtype=torch.bool)
    mask[:n_labeled] = True
    idx_all = torch.arange(n)
    initial_data = {
        "x": innate[mask].unsqueeze(-1),
        "y": innate[mask].unsqueeze(-1),
        "agent_idx": idx_all[mask],
    }

    buffer = []
    cur_dep = -1
    pred_block = {}
    last_preds = None
    op_round0 = None
    prev_op = None
    trajectory = []

    print(f"[run] loop: n_rounds={n_rounds} epoch_size={epoch_size} "
          f"deploy_every={deploy_every} regime={data_regime}", flush=True)
    t_loop = time.time()
    for t in range(n_rounds):
        is_deploy = (t % deploy_every == 0)
        if is_deploy:
            if t == 0:
                train_data = initial_data
            else:
                train_data = select_train_data(buffer, data_regime, cur_dep)
            if training_style != "frozen" and train_data is not None:
                learner.train(train_data)
            cur_dep += 1
            # model-side distribution (predictions for all agents) + health
            preds = lm(innate.unsqueeze(-1)).detach().squeeze(-1).float()
            last_preds = preds
            pred_block = {f"pred_{k}": v for k, v in cm.summary(preds, bins=n_bins).items()}
            pred_block["pred_bias"] = float(preds.mean()) - innate_mean
            if log_ppl:
                pred_block["perplexity"] = lm.perplexity(ref_texts)
            if log_answer_dist:
                pred_block.update(lm.answer_distribution_stats())

        # advance the population one FJ loop under the current deployment
        world.run(lm, n_steps=epoch_size)
        op = world.state["opinion"].float()
        if op_round0 is None:
            op_round0 = op.clone()

        row = {"round": t, "deployment": cur_dep, "is_deploy": int(is_deploy)}
        row.update({f"op_{k}": v for k, v in cm.summary(op, bins=n_bins).items()})
        row["op_bias"] = float(op.mean()) - innate_mean
        row["op_tail_frac"] = float((op - op.mean()).abs().gt(0.15).float().mean())
        row["jaccard_init"] = cm.jaccard_support(op, op_round0, bins=n_bins)
        if prev_op is not None:
            row["jaccard_prev"] = cm.jaccard_support(op, prev_op, bins=n_bins)
        row.update(pred_block)
        # dissociation: model still diverse (high pred_eff_support) while the
        # population homogenizes (low op_eff_support) -> large positive gap.
        if "pred_eff_support" in pred_block:
            row["dissoc_gap"] = pred_block["pred_eff_support"] - row["op_eff_support"]

        trajectory.append(row)
        # persist every round so a killed/held job keeps its partial results
        (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
        if wandb is not None:
            payload = dict(row)
            payload["op_hist"] = _wandb_hist(wandb, op, n_bins)
            if last_preds is not None:
                payload["pred_hist"] = _wandb_hist(wandb, last_preds, n_bins)
            wandb.log(payload)
        print(f"[round {t}] dep={cur_dep} op_mean={row['op_mean']:.4f} "
              f"op_std={row['op_std']:.4f} op_eff_sup={row['op_eff_support']:.2f} "
              f"pred_mean={pred_block.get('pred_mean', float('nan')):.4f}", flush=True)

        prev_op = op.clone()
        buffer.append({
            "t": t, "dep": cur_dep,
            "x": innate[mask].unsqueeze(-1),
            "y": op[mask].detach().unsqueeze(-1),
            "idx": idx_all[mask],
        })

    print(f"[run] loop done in {time.time() - t_loop:.1f}s", flush=True)
    (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    torch.save(trajectory, out_dir / "trajectory.pt")
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
