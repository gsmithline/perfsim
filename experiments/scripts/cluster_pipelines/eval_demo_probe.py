"""Tree-2 frozen counterfactual probe: does the base model change its
prediction when ONLY age or gender changes, tastes held fixed?

For each target genre: rebuild the ML world (target opinions, 10 non-target
genre features, per-target LCC population), then for every user build prompt
variants with the SAME taste features and controlled demographics:
  natural            real age + gender, OCCUPATION OMITTED (prompt-identical
                     to the counterfactuals; the selection-rule bridge)
  natural_full       real age + gender + occupation (loop-identical prompt,
                     descriptive only -- links probe to Tree-3 arm B)
  nodemo             age/gender/occupation lines all dropped (matches how
                     Tree-3 arm C removes features; placeholder text like
                     "gender: unspecified" is NOT expressible through the
                     loop's prompt builder, so drop-style is the faithful
                     removed-arm baseline)
  a{A}_{F|M}         age A x gender, occupation omitted, for A in AGES
Registered age-effect definition: D_age = mean over genders and users of
m(60,g) - m(20,g); Tree-3 age split is FIXED at the median (<=31 vs >31).
Greedy decoding, frozen model, no training, no dynamics. Paired effects
(gender at fixed age, age curve at fixed gender, interaction) are computed
offline; this job only produces the raw prediction table.

Env: BASE_MODEL, ML_DIR, TARGETS (comma list, default all 11 core genres),
     AGES (default "20,30,40,50,60"), OUT_DIR, GEN_CHUNK, SKIP_DONE.
Output: OUT_DIR/probe_<target>.json per target (crash-safe, resumable).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import _gated_pop as gp
from run_pokec_gated_lm import load_movielens_setup
from perfsim.models.hf_causal_lm import HFCausalLMModel

CORE = ["Drama", "Romance", "Comedy", "Action", "Thriller", "War", "Crime",
        "Sci-Fi", "Adventure", "Mystery", "Children's"]


def main():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    ml_dir = Path(os.environ.get("ML_DIR", "experiments/data/movielens/ml-100k"))
    targets = [t.strip() for t in os.environ.get("TARGETS", ",".join(CORE)).split(",") if t.strip()]
    ages = [int(a) for a in os.environ.get("AGES", "20,30,40,50,60").split(",")]
    out_dir = Path(os.environ.get("OUT_DIR", "runs/pokec_gated_lm/demo_probe"))
    chunk = int(os.environ.get("GEN_CHUNK", "64"))
    skip_done = os.environ.get("SKIP_DONE", "1") == "1"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lm = None
    for target in targets:
        fname = out_dir / f"probe_{re.sub(r'[^A-Za-z]', '_', target)}.json"
        if skip_done and fname.exists():
            print(f"[probe] {target}: exists, skipping", flush=True)
            continue
        setup = load_movielens_setup(ml_dir, target)
        prof_df = setup["profiles"]
        build_prompt = setup["build_prompt"]
        n = len(prof_df)
        if lm is None:
            print(f"[probe] loading frozen model: {base_model}", flush=True)
            t0 = time.time()
            lm = HFCausalLMModel(
                base_model_name=base_model, profiles=prof_df,
                prompt_builder=build_prompt, use_lora=False, device=device,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                load_now=True)
            print(f"[probe] loaded in {time.time() - t0:.1f}s", flush=True)

        def variant_profile(row, age=None, gender=None, strip_occ=True):
            p = dict(row)
            if strip_occ:
                p["occ"] = "none"          # build_prompt drops the line
            if age is not None:
                p["age"] = age             # 0 -> line dropped
            if gender is not None:
                p["gender"] = gender       # "" -> line dropped
            return p

        variants = {"natural": [variant_profile(r)
                                for _, r in prof_df.iterrows()],
                    "natural_full": [variant_profile(r, strip_occ=False)
                                     for _, r in prof_df.iterrows()],
                    "nodemo": [variant_profile(r, age=0, gender="")
                               for _, r in prof_df.iterrows()]}
        for a in ages:
            for g in ("F", "M"):
                variants[f"a{a}_{g}"] = [variant_profile(r, age=a, gender=g)
                                         for _, r in prof_df.iterrows()]

        out = {"target": target, "n": n, "base_model": base_model,
               "ages": ages, "innate": [float(v) for v in setup["innate"]],
               "variants": {}, "nan_counts": {}}
        for vname, profs in variants.items():
            prompts = [build_prompt(p, lm.tokenizer) for p in profs]
            preds = []
            t0 = time.time()
            for i in range(0, n, chunk):
                preds.extend(gp.probe_predictions(lm, prompts[i:i + chunk]))
            pa = np.array([p if p is not None and np.isfinite(p) else np.nan
                           for p in preds], dtype=float)
            out["variants"][vname] = [None if np.isnan(v) else round(float(v), 4)
                                      for v in pa]
            out["nan_counts"][vname] = int(np.isnan(pa).sum())
            print(f"[probe] {target}/{vname}: mean {np.nanmean(pa):.3f} "
                  f"std {np.nanstd(pa):.3f} nan {out['nan_counts'][vname]} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        with open(fname, "w") as f:
            json.dump(out, f)
        print(f"[probe] wrote {fname}", flush=True)


if __name__ == "__main__":
    main()
