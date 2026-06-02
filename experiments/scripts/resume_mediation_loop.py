"""Closed-loop AI-mediated retraining on the two-tickets resumes.

Their setup is ONE-SHOT against a FIXED screener (bge cosine), so nothing can
compound. Ours closes the loop: the screener retrains on the mediated resumes
each round, and (self_consuming) the mediated resumes become next round's
population. This script tests directly whether closing the loop compounds beyond
the one-shot effect, and whether a MOVING target (platform-conditioned) is the
engine that a STATIC target (generic) lacks.

Data: 520 Djinni resumes (260 PM + 260 UX) from the two-tickets repo, vendored
to ~/.cache/perfsim/datasets/two_tickets/. y = occupation (PM vs UX).
Features: TF-IDF + TruncatedSVD (offline; bge is the faithful upgrade later).
Mediator: quasi (embedding contraction). The faithful, fact-preserving version
is a real LLM rewrite (Tier 2), which moves text but not facts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from perfsim.core.predictor import Predictor
from perfsim.environments.mediation import ContractionMediator, MediationWorld
from perfsim.learners import ERMLearner
from perfsim.losses import BCELoss
from perfsim.models import LogisticModel
from perfsim.scenarios.ai_mediated import model_auc, recoverability, run_mediated

DATA = Path.home() / ".cache/perfsim/datasets/two_tickets/final_paper_resume_outputs_doordash.csv"
N_DIM = 128


def build_features():
    df = pd.read_csv(DATA)
    df = df[df["CV"].notna()].reset_index(drop=True)
    texts = df["CV"].astype(str).tolist()
    y = torch.tensor(df["True Label"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    tfidf = TfidfVectorizer(max_features=4000, stop_words="english", sublinear_tf=True)
    svd = TruncatedSVD(n_components=N_DIM, random_state=0)
    emb = svd.fit_transform(tfidf.fit_transform(texts))
    emb = StandardScaler().fit_transform(emb)
    x = torch.tensor(emb, dtype=torch.float32)
    return x, y


def fresh_predictor() -> Predictor:
    model = LogisticModel(in_features=N_DIM)
    learner = ERMLearner(model, BCELoss(), max_iter=300)
    return Predictor(model=model, loss=BCELoss(), learner=learner)


def run(x, y, *, target_mode, self_consuming, strength, n_rounds, conditioned):
    world = MediationWorld(
        x, y,
        mediator=ContractionMediator(strength, target_mode=target_mode, top_frac=0.25),
        platform_conditioned=conditioned,
        self_consuming=self_consuming,
    )
    pred = fresh_predictor()
    raw_auc: list[float] = []
    recs = run_mediated(
        world, pred, n_rounds=n_rounds, regime="replace", seed=0,
        probes={"rec": recoverability},
        on_round=lambda t, r: raw_auc.append(model_auc(pred.model, world.raw_data)),
    )
    rec = [r["rec"] for r in recs]
    return rec, raw_auc


def main() -> None:
    x, y = build_features()
    print(f"resumes={x.shape[0]}  dim={x.shape[1]}  occupation balance={float(y.mean()):.2f}")
    print(f"baseline recoverability of occupation (raw): {recoverability({'x': x, 'y': y}):.3f}\n")

    print("Each cell: recoverability of occupation per round (round0 = one-shot).")
    print("Question: does closing the loop drive it DOWN beyond round 0?\n")
    configs = [
        ("generic, fixed pop",        dict(target_mode="centroid", self_consuming=False, conditioned=False)),
        ("generic, self-consuming",   dict(target_mode="centroid", self_consuming=True,  conditioned=False)),
        ("winners, fixed pop",        dict(target_mode="winners",  self_consuming=False, conditioned=True)),
        ("winners, self-consuming",   dict(target_mode="winners",  self_consuming=True,  conditioned=True)),
    ]
    for name, cfg in configs:
        rec, raw_auc = run(x, y, strength=0.3, n_rounds=8, **cfg)
        one_shot = rec[0]
        final = rec[-1]
        traj = " ".join(f"{v:.3f}" for v in rec)
        print(f"  {name:24s} rec/round: {traj}")
        print(f"  {'':24s} one-shot={one_shot:.3f}  final={final:.3f}  "
              f"loop_extra_drop={one_shot - final:+.3f}  screener_raw_AUC_final={raw_auc[-1]:.3f}")
    print("\n  if 'self-consuming' final << one-shot, the closed loop compounds (your point);")
    print("  if 'fixed pop' stays flat after round 0, the one-shot null persists without state.")


if __name__ == "__main__":
    main()
