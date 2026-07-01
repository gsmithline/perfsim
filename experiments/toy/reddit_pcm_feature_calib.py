"""Calibrate feature->opinion strength on Reddit PCM.

Features = the 3 separate, non-circular demographics (age, gender, affluence),
behaviorally inferred by the dataset authors from non-political subreddit
activity. Opinion label = self-reported political-compass leaning. We report
held-out R2 of opinion ~ demographics for the continuous partisan score and the
economic/social axes, plus a binary Left-vs-Right read, to see whether the clean
separate features land in a weak (Pokec-like) or moderate regime.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

CSV = "experiments/data/reddit_pcm/data/raw/socio_demographics_anonymized_PCM.csv"
FEATS = ["age", "gender", "affluence"]
ECON = {"Left": 0.0, "Centrist": 0.5, "Right": 1.0}
SOC = {"Lib": 0.0, "Centrist": 0.5, "Auth": 1.0}

d = pd.read_csv(CSV)
print(f"users={len(d)}")
print("flair balance:\n", d["flair"].value_counts(normalize=True).round(3).to_string())
print("economic balance:", d["economic"].value_counts(normalize=True).round(3).to_dict())
print("social balance:  ", d["social"].value_counts(normalize=True).round(3).to_dict())

d = d.dropna(subset=FEATS + ["partisan", "economic", "social"])
print(f"\nusers with all features+labels: {len(d)}")
X = d[FEATS].values
print("feature ranges:", {f: (round(d[f].min(), 2), round(d[f].max(), 2)) for f in FEATS})

# pairwise correlation of each feature with the continuous partisan leaning
print("\ncorr(feature, partisan):", {f: round(np.corrcoef(d[f], d["partisan"])[0, 1], 3) for f in FEATS})

targets = {"partisan (continuous)": d["partisan"].values,
           "economic axis (L/C/R)": d["economic"].map(ECON).values,
           "social axis (Lib/C/Auth)": d["social"].map(SOC).values}

print(f"\n{'target':28s} | {'train R2':>9} {'test R2':>9}")
print("  " + "-" * 50)
for name, y in targets.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    reg = LinearRegression().fit(Xtr, ytr)
    print(f"{name:28s} | {reg.score(Xtr, ytr):9.3f} {reg.score(Xte, yte):9.3f}")

# binary Left vs Right (drop Centrist) classification read
m = d["economic"].isin(["Left", "Right"])
yb = (d.loc[m, "economic"] == "Right").astype(int).values
Xb = d.loc[m, FEATS].values
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, random_state=0)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
acc = clf.score(Xte, yte)
print(f"\nbinary Left-vs-Right from demographics: n={m.sum()}  base_rate(Right)={yb.mean():.2f}  "
      f"test acc={acc:.3f}  test AUC={auc:.3f}")
print("\nread: low R2 / AUC near 0.5 = weak (Pokec-like); higher = moderate separate-feature signal.")
