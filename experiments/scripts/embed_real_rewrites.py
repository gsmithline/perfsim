"""Does a REAL LLM rewrite do what a linear quasi-mediator structurally cannot:
collapse recoverable information and population diversity?

Zero API: embeds the two-tickets precomputed rewrites (GPT-4o/Llama/Claude/
Mixtral, once and twice) with BAAI/bge-base-en (their own screener space) and
measures, per version:
  - recoverability of occupation (best-achievable probe AUC) -> information
  - population diversity (mean pairwise cosine distance)      -> homogenization
  - idempotence (once vs twice)                                -> saturation
  - cross-model: shared attractor vs per-model house style    -> the target
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastembed import TextEmbedding

from perfsim.scenarios.ai_mediated import recoverability

DATA = Path.home() / ".cache/perfsim/datasets/two_tickets/Figure1_100Samples/originalcv_desc.csv"

VERSIONS = {
    "original":     "CV",
    "GPT4o":        "Cleaned GPT-4o Conversation-Improved CV",
    "GPT4o_twice":  "Cleaned Twice GPT-4o Conversation-Improved CV",
    "Llama":        "Cleaned Meta-Llama-3-70B-Instruct-Turbo Conversation-Improved CV",
    "Llama_twice":  "Cleaned Twice Meta-Llama-3-70B-Instruct-Turbo Conversation-Improved CV",
    "Claude":       "Cleaned claude-3-5-sonnet Conversation-Improved CV",
    "Claude_twice": "Cleaned Twice claude-3-5-sonnet Conversation-Improved CV",
    "Mixtral":      "Cleaned Mixtral-8x7B-Instruct-v0.1 Conversation-Improved CV",
}

_EMB = TextEmbedding("BAAI/bge-base-en")


def embed(texts: list[str]) -> np.ndarray:
    v = np.array(list(_EMB.embed(texts)), dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)  # unit norm for cosine


def mean_pairwise_distance(e: np.ndarray) -> float:
    g = e @ e.T
    n = e.shape[0]
    off = (g.sum() - np.trace(g)) / (n * (n - 1))
    return float(1.0 - off)


def main() -> None:
    df = pd.read_csv(DATA)
    y_all = pd.to_numeric(df["True Label"], errors="coerce")

    emb: dict[str, np.ndarray] = {}
    idx: dict[str, np.ndarray] = {}
    for name, col in VERSIONS.items():
        s = df[col].astype(str)
        keep = df[col].notna().to_numpy() & y_all.notna().to_numpy()
        emb[name] = embed(s[keep].tolist())
        idx[name] = np.where(keep)[0]

    print(f"data: {DATA.name}  n_resumes={len(df)}  occupation balance={y_all.mean():.2f}\n")
    print(f"{'version':14s} {'n':>4s} {'recoverability':>14s} {'diversity':>10s}")
    base_div = None
    for name in VERSIONS:
        e = emb[name]
        y = torch.tensor(y_all.to_numpy()[idx[name]], dtype=torch.float32).unsqueeze(-1)
        rec = recoverability({"x": torch.tensor(e), "y": y})
        div = mean_pairwise_distance(e)
        if name == "original":
            base_div = div
        rel = "" if base_div is None else f"  ({div/base_div:.2f}x orig)"
        print(f"{name:14s} {len(e):>4d} {rec:>14.3f} {div:>10.3f}{rel}")

    print("\nidempotence (embedding distance, on rows present in orig/once/twice):")
    for base in ["GPT4o", "Llama", "Claude"]:
        twice = base + "_twice"
        common = np.intersect1d(np.intersect1d(idx["original"], idx[base]), idx[twice])
        if len(common) < 5:
            continue
        def sub(name):
            pos = {v: i for i, v in enumerate(idx[name])}
            return emb[name][[pos[c] for c in common]]
        o, a, b = sub("original"), sub(base), sub(twice)
        d1 = float(np.mean(np.linalg.norm(a - o, axis=1)))
        d2 = float(np.mean(np.linalg.norm(b - a, axis=1)))
        print(f"  {base:8s} ||once-orig||={d1:.3f}  ||twice-once||={d2:.3f}  ratio={d2/d1:.2f}  (n={len(common)})")

    print("\ncross-model (rows present in all of GPT4o/Llama/Claude/Mixtral):")
    models = ["GPT4o", "Llama", "Claude", "Mixtral"]
    common = idx["original"]
    for m in models:
        common = np.intersect1d(common, idx[m])
    def sub(name):
        pos = {v: i for i, v in enumerate(idx[name])}
        return emb[name][[pos[c] for c in common]]
    subs = {m: sub(m) for m in models}
    nrow = len(common)

    def mean_pairwise_l2(e):  # same-model, different resumes
        d = [np.linalg.norm(e[i] - e[j]) for i in range(len(e)) for j in range(i + 1, len(e))]
        return float(np.mean(d))

    # both in the SAME metric (mean L2 between unit vectors) so they compare
    cross = float(np.mean([
        np.linalg.norm(subs[a][i] - subs[b][i])
        for i in range(nrow) for a in models for b in models if a < b
    ]))
    within = float(np.mean([mean_pairwise_l2(subs[m]) for m in models]))
    print(f"  mean L2, same resume / different LLM (cross-model): {cross:.3f}")
    print(f"  mean L2, different resume / same LLM (within-model): {within:.3f}")
    print("  cross < within -> LLMs CONVERGE to a shared rewrite (one common attractor)")
    print("  cross > within -> each LLM has its own house style (model-specific prior)")


if __name__ == "__main__":
    main()
