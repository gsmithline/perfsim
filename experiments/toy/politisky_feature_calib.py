"""Calibrate a tunable-strength text feature on PolitiSky24.

Per-user opinion = stance/confidence (same as the existing loader). Per-user
text = their tweets, embedded by TF-IDF then reduced to k dims with TruncatedSVD.
We report held-out R2 of opinion ~ SVD(k) for several k, to see whether text is
predictive and whether the SVD rank acts as a feature-strength knob from low to
high. (TF-IDF is local/instant; swap in a sentence embedding later, pipeline is
the same.)
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DATA = "experiments/data/politisky24/LLM_annotation_on_dataset_users.json"
STANCE_VAL = {"Favor": 1.0, "Against": -1.0, "Neither": 0.0}
RANKS = [1, 2, 5, 10, 20, 50, 100]


def load():
    lab, txt = {}, {}
    for line in open(DATA):
        r = json.loads(line)
        v = STANCE_VAL.get(r["stance"], 0.0) * float(r["confidence"] or 0.5)
        lab.setdefault(r["did"], {})[r["is_trump"]] = v
        t = r.get("text") or ""
        if isinstance(t, list):
            t = " ".join(map(str, t))
        txt.setdefault(r["did"], []).append(str(t))
    dids = [d for d in lab if any(txt.get(d))]
    op = np.array([0.5 + 0.3 * (lab[d].get(1, 0.0) - lab[d].get(0, 0.0)) / 2.0 for d in dids])
    docs = [" ".join(txt[d]) for d in dids]
    return op, docs


def main():
    op, docs = load()
    print(f"users={len(op)}  opinion mean={op.mean():.3f} std={op.std():.3f}")
    # opinion distribution (it is polarized + imbalanced)
    lo, mid, hi = (op < 0.45).mean(), ((op >= 0.45) & (op <= 0.55)).mean(), (op > 0.55).mean()
    print(f"opinion split: <0.45 (Trump camp) {lo:.2f}   ~0.5 {mid:.2f}   >0.55 (Harris camp) {hi:.2f}")

    tfidf = TfidfVectorizer(max_features=8000, stop_words="english", min_df=5)
    X = tfidf.fit_transform(docs)
    print(f"tfidf matrix: {X.shape}\n")

    Xtr, Xte, ytr, yte = train_test_split(X, op, test_size=0.25, random_state=0)
    print(f"{'rank k':>7} | {'train R2':>9} {'test R2':>9}   (feature-strength knob)")
    print("  " + "-" * 40)
    for k in RANKS:
        svd = TruncatedSVD(n_components=k, random_state=0)
        Ztr = svd.fit_transform(Xtr)
        Zte = svd.transform(Xte)
        reg = LinearRegression().fit(Ztr, ytr)
        r2tr = reg.score(Ztr, ytr)
        r2te = reg.score(Zte, yte)
        print(f"{k:>7} | {r2tr:9.3f} {r2te:9.3f}")
    print("\nread: held-out R2 climbing with k = SVD rank is the tunable feature-strength axis.")


if __name__ == "__main__":
    main()
