"""Which node attribute is actually predictable from the others?

For each dataset, predict every numeric node attribute from all the others and
report held-out R2. This surfaces (feature-set -> label) pairs with moderate
signal, instead of forcing the (weak) opinion column to be the label.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def predictability(df, name):
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df[cols].dropna()
    df = df.loc[:, df.std() > 1e-9]
    cols = list(df.columns)
    print(f"\n===== {name}  (n={len(df)}, numeric cols={cols}) =====")
    print(f"  {'label (predicted)':22s} {'test R2':>8}   {'top driver (|corr|)':>28}")
    print("  " + "-" * 64)
    rows = []
    for y in cols:
        X = df[[c for c in cols if c != y]].values
        yt = df[y].values
        Xtr, Xte, ytr, yte = train_test_split(X, yt, test_size=0.25, random_state=0)
        r2 = LinearRegression().fit(Xtr, ytr).score(Xte, yte)
        corr = {c: abs(np.corrcoef(df[c], df[y])[0, 1]) for c in cols if c != y}
        top = max(corr, key=corr.get)
        rows.append((y, r2, top, corr[top]))
    for y, r2, top, cv in sorted(rows, key=lambda r: -r[1]):
        print(f"  {y:22s} {r2:8.3f}   {top+' '+f'{cv:.2f}':>28}")


# ---- Reddit PCM ----
pcm = pd.read_csv("experiments/data/reddit_pcm/data/raw/socio_demographics_anonymized_PCM.csv")
predictability(pcm[["age", "gender", "affluence", "partisan"]], "Reddit PCM (behavioral scores)")

# ---- Yelp Acme LCC ----
d = np.load("experiments/yelp/yelp_acme_lcc.npz", allow_pickle=True)
print("\nyelp npz keys:", list(d.files))
yc = {}
for k in d.files:
    a = d[k]
    if k in ("edges", "names"):
        continue
    a = np.asarray(a)
    if a.ndim == 1 and a.dtype.kind in "fiu" and len(a) == len(d["opinion"]):
        yc[k] = a.astype(float)
    elif a.ndim == 2 and a.shape[0] == len(d["opinion"]):
        for j in range(a.shape[1]):
            yc[f"{k}{j}"] = a[:, j].astype(float)
predictability(pd.DataFrame(yc), "Yelp Acme LCC (ratings + activity)")
