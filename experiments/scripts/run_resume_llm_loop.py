"""Performative AI-mediation loop on real resumes (Experiment 1 + 4 + 5).

The platform theta sees BOTH blocks of features: content c (bge embedding) and
style z (stylometrics). The frozen LLM rewrites only z (style), preserving c.
theta_0 = A(D^H) on raw resumes; each round theta scores every resume, the score
conditions the rewrite, theta retrains on the mediated data, the rewrites become
next round's population. We log where the label is recoverable and, crucially,
whether the c-block stays put while the z-block moves (the L_psi(z,c)=(z',c)
check).

CONDITION:  phenomenon | identification | generic
REGIME:     replace | accumulate | clean_anchor | mediated_anchor
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastembed import TextEmbedding
from sklearn.decomposition import PCA
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

FUNCTION_WORDS = ["the", "of", "and", "to", "a", "in", "that", "is", "for", "it",
                  "with", "as", "on", "at", "by", "i", "this", "an", "are", "my",
                  "have", "from", "or", "you", "we", "they", "which", "will", "can", "been"]


def env(name, default=None, cast=str):
    v = os.environ.get(name, default)
    return cast(v) if v is not None else None


def style_features(texts):
    out = []
    for t in texts:
        t = str(t)
        words = re.findall(r"\b\w+\b", t.lower())
        nw = max(len(words), 1)
        nc = max(len(t), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        ns = max(len(sents), 1)
        lines = [l for l in t.split("\n") if l.strip()]
        nl = max(len(lines), 1)
        counts = Counter(words)
        uniq = set(words)
        hapax = sum(1 for w in uniq if counts[w] == 1)
        bullets = sum(1 for l in lines if l.strip()[:1] in "-•*·" or re.match(r"^\s*\d+[.)]", l))
        f = [
            np.mean([len(w) for w in words]) if words else 0.0,
            nw / ns,
            len(uniq) / nw,
            hapax / nw,
            t.count(",") / nw, t.count(".") / nw, t.count(";") / nw,
            t.count(":") / nw, t.count("!") / nw, t.count("?") / nw,
            (t.count("(") + t.count(")")) / nw, t.count("-") / nw,
            bullets / nl,
            t.count("\n") / nc,
            sum(c.isupper() for c in t) / nc,
            sum(c.isdigit() for c in t) / nc,
        ]
        f += [counts[w] / nw for w in FUNCTION_WORDS]
        out.append(f)
    return np.asarray(out, dtype=np.float32)


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

    def _prompt(self, cv, s):
        if s is None:
            return PROMPT_GENERIC.format(cv=cv)
        return PROMPT_CONDITIONED.format(score=int(round(s * 100)), cv=cv)

    @torch.no_grad()
    def __call__(self, resumes, scores=None):
        out = []
        for i in range(0, len(resumes), self.batch_size):
            batch = resumes[i : i + self.batch_size]
            sc = [None] * len(batch) if scores is None else scores[i : i + self.batch_size]
            prompts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": self._prompt(cv, s)}],
                    tokenize=False, add_generation_prompt=True)
                for cv, s in zip(batch, sc)
            ]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=2048).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, repetition_penalty=1.15,
                                      pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True))
        return out


def fit_screener(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(X), y)
    return sc, clf


def score(theta, X):
    sc, clf = theta
    return clf.predict_proba(sc.transform(X))[:, 1]


def auc(theta, X, y):
    return 0.5 if np.unique(y).size < 2 else float(roc_auc_score(y, score(theta, X)))


def recoverable_auc(X, y, seed):
    if np.unique(y).size < 2:
        return 0.5
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(xte))[:, 1]))


def blockvar(M):
    return float(M.var(axis=0).mean())


def theta_weights(theta):
    sc, clf = theta
    coef = clf.coef_.ravel()
    return coef, coef / sc.scale_  # standardized coefs, and raw-space (scaler-invariant)


def compose(regime, buffer, anchor, alpha, rng):
    X_r, y_r = buffer[-1]
    if regime == "replace":
        return X_r, y_r
    if regime == "accumulate":
        return np.vstack([x for x, _ in buffer]), np.concatenate([yy for _, yy in buffer])
    src = anchor if regime == "clean_anchor" else buffer[0]
    X_a, y_a = src
    k = round(alpha * len(X_r))
    idx = rng.choice(len(X_a), size=k, replace=len(X_a) < k)
    return np.vstack([X_a[idx], X_r]), np.concatenate([y_a[idx], y_r])


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
    content_dim = env("CONTENT_DIM", 128, int)
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
    y_tr, y_te = y_all[idx_tr], y_all[idx_te]

    embedder = TextEmbedding("BAAI/bge-base-en")

    def bge(texts):
        v = np.array(list(embedder.embed([str(t) for t in texts])), dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    # fit the c/z feature space on the raw population, freeze it
    raw_content = bge(cvs)
    pca = PCA(n_components=min(content_dim, raw_content.shape[1]), random_state=0).fit(raw_content)
    c_scaler = StandardScaler().fit(pca.transform(raw_content))
    z_scaler = StandardScaler().fit(style_features(cvs))
    n_c = pca.n_components_
    z_cols = np.arange(n_c, n_c + z_scaler.mean_.shape[0])
    c_cols = np.arange(n_c)

    def featurize(texts):
        raw = bge(texts)
        c = c_scaler.transform(pca.transform(raw))
        z = z_scaler.transform(style_features(texts))
        return np.hstack([c, z]).astype(np.float32), raw

    orig_X, orig_raw = featurize(cvs)
    theta0 = fit_screener(orig_X[idx_tr], y_tr)
    theta = theta0
    raw0 = auc(theta0, orig_X[idx_te], y_te)
    print(f"[resume_llm] tag={tag} cond={condition} regime={regime} label={label} "
          f"n={len(cvs)} c_dim={n_c} z_dim={len(z_cols)} R_H(theta_0)={raw0:.3f} device={device}",
          flush=True)

    wb = None
    if _wandb is not None and env("WANDB_PROJECT"):
        wb = _wandb.init(project=env("WANDB_PROJECT"), name=tag + env("WANDB_RUN_SUFFIX", ""),
                         config={"model": model_name, "condition": condition, "regime": regime,
                                 "label": label, "n_rounds": n_rounds, "alpha": alpha,
                                 "c_dim": int(n_c), "z_dim": int(len(z_cols)), "R_H_theta0": raw0})

    rewriter = Rewriter(model_name, device, max_new, gen_bs)
    texts = list(cvs)
    cur_X = orig_X
    buffer, traj = [], []
    prev_weff = None
    for t in range(n_rounds):
        yhat = score(theta, cur_X)
        if condition == "generic":
            ps = None
        elif condition == "identification":
            ps = list(yhat[rng.permutation(len(yhat))])
        else:
            ps = list(yhat)
        rewrites = rewriter(texts, ps)
        obs_X, obs_raw = featurize(rewrites)
        buffer.append((obs_X[idx_tr], y_tr))

        train_X, train_y = compose(regime, buffer, (orig_X[idx_tr], y_tr), alpha, rng)
        theta = fit_screener(train_X, train_y)

        coef_std, weff = theta_weights(theta)
        z_mass = np.abs(coef_std[z_cols]).sum()
        z_weight_frac = float(z_mass / (np.abs(coef_std).sum() + 1e-9))
        theta_drift = 0.0 if prev_weff is None else float(np.linalg.norm(weff - prev_weff))
        prev_weff = weff
        r_h = auc(theta, orig_X[idx_te], y_te)
        r_obs = auc(theta, obs_X[idx_te], y_te)

        row = {
            "round": t,
            "R_H": r_h,
            "R_obs": r_obs,
            "gap": r_obs - r_h,
            "z_weight_frac": z_weight_frac,
            "theta_drift": theta_drift,
            "recoverability": recoverable_auc(obs_X[idx_tr], y_tr, seed),
            "diversity_z": blockvar(obs_X[:, z_cols]),
            "diversity_c": blockvar(obs_X[:, c_cols]),
            "drift_z": float(np.mean(np.linalg.norm(obs_X[:, z_cols] - orig_X[:, z_cols], axis=1))) / np.sqrt(len(z_cols)),
            "drift_c": float(np.mean(np.linalg.norm(obs_X[:, c_cols] - orig_X[:, c_cols], axis=1))) / np.sqrt(len(c_cols)),
            "content_cos": float(np.mean(np.sum(orig_raw * obs_raw, axis=1))),
            "gamed_score": float(score(theta0, obs_X).mean()),
            "mean_yhat": float(yhat.mean()),
        }
        traj.append(row)
        (out_dir / f"round{t}.json").write_text(json.dumps(
            {"rewrites": rewrites, "yhat": yhat.tolist(), "y": y_all.tolist()}))
        print(f"  r{t}: R_H={row['R_H']:.3f} R_obs={row['R_obs']:.3f} gap={row['gap']:+.3f} "
              f"rec={row['recoverability']:.3f} zwt={row['z_weight_frac']:.3f} dθ={row['theta_drift']:.3f} "
              f"div_z={row['diversity_z']:.3f} div_c={row['diversity_c']:.3f} "
              f"drift_z={row['drift_z']:.3f} drift_c={row['drift_c']:.3f} "
              f"ccos={row['content_cos']:.3f} gamed={row['gamed_score']:.3f}", flush=True)
        if wb is not None:
            wb.log(row, step=t)
        texts, cur_X = rewrites, obs_X

    (out_dir / "trajectory.json").write_text(json.dumps(
        {"config": {"model": model_name, "condition": condition, "regime": regime,
                    "label": label, "n_rounds": n_rounds, "alpha": alpha, "seed": seed,
                    "c_dim": int(n_c), "z_dim": int(len(z_cols)), "R_H_theta0": raw0},
         "trajectory": traj}, indent=2))
    if wb is not None:
        wb.finish()
    print(f"[resume_llm] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
