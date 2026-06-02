"""Figure summarizing the embed-test findings on real two-tickets LLM rewrites.

Numbers are the verified output of embed_real_rewrites.py (bge-base-en space,
100 Djinni resumes). Plotted directly to avoid a 15-min CPU re-embed.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("runs/two_tickets_analysis")
OUT.mkdir(parents=True, exist_ok=True)

# --- verified numbers from embed_real_rewrites.py ---
versions = ["original", "GPT4o", "GPT4o\n(twice)", "Llama", "Llama\n(twice)",
            "Claude", "Claude\n(twice)", "Mixtral"]
diversity = [0.178, 0.170, 0.169, 0.159, 0.162, 0.165, 0.158, 0.173]
recover = [1.000] * 8
idem_models = ["GPT4o", "Llama", "Claude"]
once_orig = [0.322, 0.345, 0.380]
twice_once = [0.135, 0.178, 0.231]
cross_l2, within_l2 = 0.352, 0.574

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Real LLM resume rewrites (two-tickets data, bge-base-en space): "
             "mild, shared, and content-preserving", fontsize=13, fontweight="bold")

# (1) diversity
a = ax[0, 0]
colors = ["#444"] + ["#3b7dd8"] * 7
a.bar(range(8), diversity, color=colors)
a.axhline(diversity[0], ls="--", c="#444", lw=1)
a.set_xticks(range(8)); a.set_xticklabels(versions, fontsize=8)
a.set_ylabel("population diversity (mean pairwise cosine dist)")
a.set_ylim(0, 0.20)
a.set_title("1. Single-pass homogenization is MILD (88-97% of original)")

# (2) recoverability
a = ax[0, 1]
a.bar(range(8), recover, color=colors)
a.set_xticks(range(8)); a.set_xticklabels(versions, fontsize=8)
a.set_ylabel("recoverability of occupation (probe AUC)")
a.set_ylim(0, 1.1)
a.set_title("2. Occupation fully recoverable everywhere\n(label trivially separable -> need a harder label)")
a.annotate("no information loss visible,\nbut no headroom on this label",
           xy=(3.5, 0.5), ha="center", fontsize=9, color="#a33")

# (3) idempotence
a = ax[1, 0]
x = range(len(idem_models)); w = 0.35
a.bar([i - w/2 for i in x], once_orig, w, label="||once - original||", color="#3b7dd8")
a.bar([i + w/2 for i in x], twice_once, w, label="||twice - once||", color="#b0c8ee")
a.set_xticks(list(x)); a.set_xticklabels(idem_models)
a.set_ylabel("embedding move (L2)")
a.set_title("3. Idempotence: 2nd pass moves ~half as far (saturating)")
a.legend(fontsize=9)

# (4) cross vs within
a = ax[1, 1]
a.bar([0, 1], [cross_l2, within_l2], color=["#d9822b", "#3b7dd8"], width=0.6)
a.set_xticks([0, 1])
a.set_xticklabels(["cross-model\n(same resume,\ndiff LLM)", "within-model\n(diff resume,\nsame LLM)"], fontsize=9)
a.set_ylabel("mean L2 distance")
a.set_ylim(0, 0.7)
a.set_title("4. KEY: cross < within -> LLMs CONVERGE\nto a shared rewrite attractor")
for i, v in enumerate([cross_l2, within_l2]):
    a.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.96])
path = OUT / "real_rewrites_findings.png"
fig.savefig(path, dpi=140)
print(f"saved {path}")
