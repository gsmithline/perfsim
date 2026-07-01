"""MovieLens 100K: is a malleable taste label predictable, and from what?

Per-user taste = average rating per genre. For each genre (the label) we predict
it two ways: from the user's OTHER genre tastes (collaborative / taste->taste),
and from demographics (age, gender, occupation). This contrasts the recommender
regime (taste predicts taste) against the demographics-predict-taste regime, the
same contrast that made opinions weak.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ML = "experiments/data/movielens/ml-100k"

genres = pd.read_csv(f"{ML}/u.genre", sep="|", names=["name", "gid"], encoding="latin-1")
GEN = list(genres.sort_values("gid")["name"])
items = pd.read_csv(f"{ML}/u.item", sep="|", encoding="latin-1", header=None)
gmat = items.iloc[:, 5:5 + len(GEN)].values             # movie x genre binary
gmat = pd.DataFrame(gmat, index=items[0].values, columns=GEN)
users = pd.read_csv(f"{ML}/u.user", sep="|", names=["uid", "age", "gender", "occ", "zip"])
rat = pd.read_csv(f"{ML}/u.data", sep="\t", names=["uid", "iid", "r", "t"])

# per-user mean rating within each genre
rat = rat.merge(gmat, left_on="iid", right_index=True)
pref = {}
for g in GEN:
    sub = rat[rat[g] == 1]
    pref[g] = sub.groupby("uid")["r"].mean()
P = pd.DataFrame(pref).reindex(users["uid"]).reset_index(drop=True)

cov = P.notna().mean().sort_values(ascending=False)
CORE = [g for g in cov.index if cov[g] > 0.85 and g != "unknown"]
print(f"users={len(P)}  genres with >85% user coverage (CORE): {CORE}\n")

Pc = P[CORE].copy()
Pc = Pc.apply(lambda c: c.fillna(c.mean()), axis=0)     # impute missing genre with genre mean
age = users["age"].values.astype(float)
gen = (users["gender"].values == "F").astype(float)
occ = pd.get_dummies(users["occ"]).values.astype(float)
DEMO = np.column_stack([age, gen, occ])

print(f"  {'genre (label)':12s} | {'R2 taste->taste':>16} {'R2 demo->taste':>16}")
print("  " + "-" * 50)
rows = []
for g in CORE:
    y = Pc[g].values
    Xtaste = Pc[[c for c in CORE if c != g]].values
    Xtr, Xte, ytr, yte = train_test_split(Xtaste, y, test_size=0.25, random_state=0)
    r2_taste = LinearRegression().fit(Xtr, ytr).score(Xte, yte)
    Xtr, Xte, ytr, yte = train_test_split(DEMO, y, test_size=0.25, random_state=0)
    r2_demo = LinearRegression().fit(Xtr, ytr).score(Xte, yte)
    rows.append((g, r2_taste, r2_demo))
for g, rt, rd in sorted(rows, key=lambda r: -r[1]):
    print(f"  {g:12s} | {rt:16.3f} {rd:16.3f}")
mt = np.mean([r[1] for r in rows]); md = np.mean([r[2] for r in rows])
print(f"  {'MEAN':12s} | {mt:16.3f} {md:16.3f}")
print("\nread: taste->taste in a useful (0,1) band = malleable label with real feature signal;")
print("demo->taste near zero = same weakness as opinions-from-demographics.")
