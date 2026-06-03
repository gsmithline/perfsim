"""One-time: embed original + GPT-4o-rewritten resumes (bge-base-en) and cache.

Produces ~/.cache/perfsim/datasets/two_tickets/embeddings.npz with:
  orig   (N, 768)  original CV embeddings
  rewrite(N, 768)  GPT-4o cleaned rewrite embeddings (for fitting the surrogate)
  occupation (N,)  PM vs UX (trivially separable; baseline check)
  experience (N,)  Experience Years (0-inflated; the label with headroom)
  dd_score   (N,)  their bge DoorDash-PM similarity score
The loop script reads this cache so it never re-embeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import TextEmbedding

CACHE = Path.home() / ".cache/perfsim/datasets/two_tickets"
DATA = CACHE / "final_paper_resume_outputs_doordash.csv"
OUT = CACHE / "embeddings.npz"

_EMB = TextEmbedding("BAAI/bge-base-en")


def embed(texts):
    v = np.array(list(_EMB.embed([str(t) for t in texts])), dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def main() -> None:
    df = pd.read_csv(DATA)
    print(f"embedding {len(df)} originals + rewrites (bge-base-en, CPU)...")
    orig = embed(df["CV"].tolist())
    rewrite = embed(df["Cleaned GPT-4o Conversation-Improved CV"].tolist())
    np.savez(
        OUT,
        orig=orig,
        rewrite=rewrite,
        occupation=pd.to_numeric(df["True Label"], errors="coerce").to_numpy(),
        experience=pd.to_numeric(df["Experience Years"], errors="coerce").to_numpy(),
        dd_score=pd.to_numeric(df["CVDoorDash PM Score"], errors="coerce").to_numpy(),
    )
    print(f"saved {OUT}  orig{orig.shape} rewrite{rewrite.shape}")


if __name__ == "__main__":
    main()
