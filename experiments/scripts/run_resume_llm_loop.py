"""Self-consuming AI-mediation loop on real resumes with a real LLM rewriter.

Each round an LLM rewrites every resume ("improve, add no facts"); the rewrites
become next round's resumes; a screener retrains on bge embeddings of them. We
track diversity, label recoverability, and screener accuracy on the original
pre-AI resumes across rounds.
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

REWRITE_PROMPT = (
    "Can you improve my resume? I will be punished if I lie about my "
    "qualifications, so it is imperative that the new resume does not add any "
    "facts that are not in the original resume. Also, please do not include any "
    "additional notes or explanation: I just want the text of the new resume. "
    "Do not even write 'here is the resume', I just need the plain text. Again, "
    "ensure that the output accurately represents my actual responsibilities, "
    "experiences, and skills. This is my resume: {cv}"
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, resumes):
        out = []
        for i in range(0, len(resumes), self.batch_size):
            batch = resumes[i : i + self.batch_size]
            prompts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": REWRITE_PROMPT.format(cv=cv)}],
                    tokenize=False, add_generation_prompt=True,
                )
                for cv in batch
            ]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=2048).to(self.device)
            gen = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True))
        return out


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


def main():
    tag = env("RUN_TAG", "resume_llm")
    model_name = env("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    n_rounds = env("N_ROUNDS", 15, int)
    regime = env("DATA_REGIME", "replace")
    label = env("LABEL", "experience")
    anchor_alpha = env("ANCHOR_ALPHA", 0.3, float)
    seed = env("SEED", 0, int)
    max_new = env("MAX_NEW_TOKENS", 768, int)
    gen_bs = env("GEN_BATCH_SIZE", 16, int)
    data_csv = env("RESUME_CSV", "examples/two_tickets/resumes.csv")
    out_dir = Path(env("OUT_DIR", f"runs/resume_llm/{tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(data_csv)
    df = df[df["CV"].notna()].reset_index(drop=True)
    originals = df["CV"].astype(str).tolist()
    if label == "occupation":
        y = pd.to_numeric(df["True Label"], errors="coerce").to_numpy()
    else:
        exp = pd.to_numeric(df["Experience Years"], errors="coerce").to_numpy()
        y = (exp > np.nanmedian(exp)).astype(int)
    y = y.astype(int)

    embedder = TextEmbedding("BAAI/bge-base-en")

    def embed(texts):
        v = np.array(list(embedder.embed([str(t) for t in texts])), dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    orig_emb = embed(originals)
    rng = np.random.default_rng(seed)

    wb = None
    if _wandb is not None and env("WANDB_PROJECT"):
        wb = _wandb.init(project=env("WANDB_PROJECT"), name=tag + env("WANDB_RUN_SUFFIX", ""),
                         config={"model": model_name, "regime": regime, "label": label,
                                 "n_rounds": n_rounds, "anchor_alpha": anchor_alpha})

    rewriter = Rewriter(model_name, device, max_new, gen_bs)
    print(f"[resume_llm] tag={tag} model={model_name} n={len(originals)} "
          f"regime={regime} label={label} device={device}", flush=True)

    texts = list(originals)
    traj = []
    for t in range(n_rounds):
        rewrites = rewriter(texts)
        emb = embed(rewrites)

        if regime == "clean_anchor":
            k = round(anchor_alpha * len(emb))
            idx = rng.choice(len(orig_emb), size=k, replace=False)
            train_emb = np.vstack([orig_emb[idx], emb])
            train_y = np.concatenate([y[idx], y])
        else:
            train_emb, train_y = emb, y

        sc = StandardScaler().fit(train_emb)
        screener = LogisticRegression(max_iter=1000).fit(sc.transform(train_emb), train_y)
        raw_auc = (0.5 if np.unique(y).size < 2 else
                   roc_auc_score(y, screener.predict_proba(sc.transform(orig_emb))[:, 1]))

        row = {
            "round": t,
            "diversity": diversity(emb),
            "recoverability": recoverable_auc(emb, y, seed),
            "screener_raw_auc": float(raw_auc),
            "drift_from_orig": float(np.mean(np.linalg.norm(emb - orig_emb, axis=1))),
        }
        traj.append(row)
        (out_dir / f"rewrites_round{t}.json").write_text(json.dumps(rewrites))
        print(f"  round {t}: div={row['diversity']:.3f} rec={row['recoverability']:.3f} "
              f"raw_auc={row['screener_raw_auc']:.3f} drift={row['drift_from_orig']:.3f}", flush=True)
        if wb is not None:
            wb.log(row, step=t)

        texts = rewrites

    (out_dir / "trajectory.json").write_text(json.dumps(
        {"config": {"model": model_name, "regime": regime, "label": label,
                    "n_rounds": n_rounds, "anchor_alpha": anchor_alpha, "seed": seed},
         "trajectory": traj}, indent=2))
    if wb is not None:
        wb.finish()
    print(f"[resume_llm] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
