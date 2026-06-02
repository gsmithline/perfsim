"""Performative AI-mediation loop on real resumes.

theta_0 = A(D^H) on raw resumes. Each round: the screener theta_t scores every
resume; the LLM rewrites each resume CONDITIONED on its score ("scored X/100,
rewrite to improve it, add no facts"); theta retrains on the mediated data; the
rewrites become next round's population. This is the predicting-from-predictions
loop: theta's output shapes the data that trains theta_{t+1}.

CONDITION:
  phenomenon     rewrite sees the real theta_t score (the feedback loop)
  identification shuffled scores (same marginal, link broken) -- causal control
  generic        no score in the prompt (no feedback) -- baseline
REGIME (theta's training data): replace | accumulate | clean_anchor | mediated_anchor
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastembed import TextEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

PROMPT_CONDITIONED = (
    "An automated hiring screener scored this resume {score} out of 100 for the "
    "role. Rewrite the resume to improve its score. It is imperative that you do "
    "not add any facts, qualifications, or experience that are not in the "
    "original resume. Output only the plain text of the new resume, no preamble, "
    "no notes. This is the resume: {cv}"
)
PROMPT_GENERIC = (
    "Can you improve my resume? It is imperative that the new resume does not add "
    "any facts that are not in the original. Output only the plain text of the new "
    "resume, no preamble, no notes. This is the resume: {cv}"
)


def env(name, default=None, cast=str):
    v = os.environ.get(name, default)
    return cast(v) if v is not None else None


class Rewriter:
    def __init__(self, model_name, device, max_new_tokens, batch_size):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def _prompt(self, cv, score):
        if score is None:
            return PROMPT_GENERIC.format(cv=cv)
        return PROMPT_CONDITIONED.format(score=int(round(score * 100)), cv=cv)

    @torch.no_grad()
    def __call__(self, resumes, scores=None):
        out = []
        for i in range(0, len(resumes), self.batch_size):
            batch = resumes[i : i + self.batch_size]
            sc = [None] * len(batch) if scores is None else scores[i : i + self.batch_size]
            prompts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": self._prompt(cv, s)}],
                    tokenize=False, add_generation_prompt=True,
                )
                for cv, s in zip(batch, sc)
            ]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=2048).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True))
        return out


def fit_screener(emb, y):
    sc = StandardScaler().fit(emb)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(emb), y)
    return sc, clf


def score(theta, emb):
    sc, clf = theta
    return clf.predict_proba(sc.transform(emb))[:, 1]


def auc(theta, emb, y):
    if np.unique(y).size < 2:
        return 0.5
    return float(roc_auc_score(y, score(theta, emb)))


def diversity(emb):
    g = emb @ emb.T
    n = emb.shape[0]
    return float(1.0 - (g.sum() - np.trace(g)) / (n * (n - 1)))


def recoverable_auc(emb, y, seed):
    if np.unique(y).size < 2:
        return 0.5
    xtr, xte, ytr, yte = train_test_split(emb, y, test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(xte))[:, 1]))


def compose(regime, buffer, anchor, alpha, rng):
    emb_r, y_r = buffer[-1]
    if regime == "replace":
        return emb_r, y_r
    if regime == "accumulate":
        return np.vstack([e for e, _ in buffer]), np.concatenate([yy for _, yy in buffer])
    src = anchor if regime == "clean_anchor" else buffer[0]
    emb_a, y_a = src
    k = round(alpha * len(emb_r))
    idx = rng.choice(len(emb_a), size=k, replace=len(emb_a) < k)
    return np.vstack([emb_a[idx], emb_r]), np.concatenate([y_a[idx], y_r])


def main():
    tag = env("RUN_TAG", "resume_llm")
    model_name = env("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    n_rounds = env("N_ROUNDS", 15, int)
    regime = env("DATA_REGIME", "replace")
    condition = env("CONDITION", "phenomenon")
    label = env("LABEL", "experience")
    alpha = env("ANCHOR_ALPHA", 0.3, float)
    seed = env("SEED", 0, int)
    max_new = env("MAX_NEW_TOKENS", 768, int)
    gen_bs = env("GEN_BATCH_SIZE", 16, int)
    data_csv = env("RESUME_CSV", "examples/two_tickets/resumes.csv")
    out_dir = Path(env("OUT_DIR", f"runs/resume_llm/{tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(data_csv)
    df = df[df["CV"].notna()].reset_index(drop=True)
    cvs = df["CV"].astype(str).tolist()
    if label == "occupation":
        y_all = pd.to_numeric(df["True Label"], errors="coerce").to_numpy().astype(int)
    else:
        exp = pd.to_numeric(df["Experience Years"], errors="coerce").to_numpy()
        y_all = (exp > np.nanmedian(exp)).astype(int)

    rng = np.random.default_rng(seed)
    idx_tr, idx_te = train_test_split(np.arange(len(cvs)), test_size=0.3,
                                      random_state=seed, stratify=y_all)
    train_cvs = [cvs[i] for i in idx_tr]
    y_tr, y_te = y_all[idx_tr], y_all[idx_te]

    embedder = TextEmbedding("BAAI/bge-base-en")

    def embed(texts):
        v = np.array(list(embedder.embed([str(t) for t in texts])), dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    orig_tr = embed(train_cvs)
    orig_te = embed([cvs[i] for i in idx_te])

    theta = fit_screener(orig_tr, y_tr)        # theta_0 = A(D^H)
    raw0 = auc(theta, orig_te, y_te)
    print(f"[resume_llm] tag={tag} cond={condition} regime={regime} label={label} "
          f"n_train={len(train_cvs)} R_H(theta_0)={raw0:.3f} device={device}", flush=True)

    wb = None
    if _wandb is not None and env("WANDB_PROJECT"):
        wb = _wandb.init(project=env("WANDB_PROJECT"), name=tag + env("WANDB_RUN_SUFFIX", ""),
                         config={"model": model_name, "condition": condition, "regime": regime,
                                 "label": label, "n_rounds": n_rounds, "alpha": alpha,
                                 "R_H_theta0": raw0})

    rewriter = Rewriter(model_name, device, max_new, gen_bs)
    texts = list(train_cvs)
    cur_emb = orig_tr
    buffer, traj = [], []
    for t in range(n_rounds):
        yhat = score(theta, cur_emb)                            # theta_t scores the population
        if condition == "generic":
            prompt_scores = None
        elif condition == "identification":
            prompt_scores = yhat[rng.permutation(len(yhat))]    # break the link, keep marginal
        else:
            prompt_scores = yhat
        rewrites = rewriter(texts, None if prompt_scores is None else list(prompt_scores))
        emb_obs = embed(rewrites)
        buffer.append((emb_obs, y_tr))

        train_emb, train_y = compose(regime, buffer, (orig_tr, y_tr), alpha, rng)
        theta = fit_screener(train_emb, train_y)                # theta_{t+1} = A(theta_t, D_t^obs)

        row = {
            "round": t,
            "R_H": auc(theta, orig_te, y_te),                   # raw held-out (degradation)
            "R_obs": auc(theta, emb_obs, y_tr),                 # mediated
            "recoverability": recoverable_auc(emb_obs, y_tr, seed),
            "diversity": diversity(emb_obs),
            "drift": float(np.mean(np.linalg.norm(emb_obs - orig_tr, axis=1))),
            "mean_score": float(yhat.mean()),
            "score_std": float(yhat.std()),
        }
        traj.append(row)
        (out_dir / f"round{t}.json").write_text(json.dumps(
            {"rewrites": rewrites, "yhat": yhat.tolist(), "y": y_tr.tolist()}))
        print(f"  r{t}: R_H={row['R_H']:.3f} R_obs={row['R_obs']:.3f} "
              f"rec={row['recoverability']:.3f} div={row['diversity']:.3f} "
              f"drift={row['drift']:.3f} mean_yhat={row['mean_score']:.3f}", flush=True)
        if wb is not None:
            wb.log(row, step=t)
        texts, cur_emb = rewrites, emb_obs

    (out_dir / "trajectory.json").write_text(json.dumps(
        {"config": {"model": model_name, "condition": condition, "regime": regime,
                    "label": label, "n_rounds": n_rounds, "alpha": alpha, "seed": seed,
                    "R_H_theta0": raw0}, "trajectory": traj}, indent=2))
    if wb is not None:
        wb.finish()
    print(f"[resume_llm] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
