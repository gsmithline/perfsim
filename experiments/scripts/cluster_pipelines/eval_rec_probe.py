"""Recommendation-framing probe (prompt realism, Celestine suggestion):
instead of asking for an opinion number, show a 20-movie slate (10
Action-flagged, 10 not, per-user shuffled order) and ask the model to
pick 5. Served signal = fraction of picks that are Action-flagged
(by ML-100k's own flags, the same flags that define innate opinions).

Variants per user (tastes fixed): natural (real age+gender, occ
omitted), nodemo, a30_M, a30_F (the gender counterfactual in
recommendation space). Frozen model, greedy, no training.

Env: BASE_MODEL, ML_DIR, ML_TARGET (Action), OUT, GEN_CHUNK.
Output: one json with picks + fractions per variant.
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
from run_pokec_gated_lm import load_movielens_setup
from perfsim.models.hf_causal_lm import HFCausalLMModel


def main():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    ml_dir = Path(os.environ.get("ML_DIR", "experiments/data/movielens/ml-100k"))
    target = os.environ.get("ML_TARGET", "Action")
    out = Path(os.environ.get("OUT", "runs/pokec_gated_lm/rec_probe/rec_probe_qwen.json"))
    chunk = int(os.environ.get("GEN_CHUNK", "32"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    slate = json.load(open("experiments/data/movielens/rec_slate.json"))
    setup = load_movielens_setup(ml_dir, target)
    prof_df = setup["profiles"]
    n = len(prof_df)
    feats = [c for c in prof_df.columns if c not in ("age", "gender", "occ")]

    lm = HFCausalLMModel(base_model_name=base_model, profiles=prof_df,
                         prompt_builder=setup["build_prompt"], use_lora=False,
                         device=device, max_new_tokens=24,
                         dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                         load_now=True)
    print(f"[rec] model loaded: {base_model}", flush=True)

    def order_for(i):
        return np.random.default_rng(90000 + i).permutation(len(slate))

    def build(i, row, age=None, gender=None, drop_demo=False):
        lines = []
        if not drop_demo:
            a = age if age is not None else row.get("age")
            g = gender if gender is not None else row.get("gender")
            if a and int(a) > 0:
                lines.append(f"- age: {int(a)}")
            if isinstance(g, str) and g:
                lines.append(f"- gender: {'male' if g == 'M' else 'female'}")
        for gname in feats:
            v = row.get(gname)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                lines.append(f"- average rating of {gname} movies: {float(v):.1f} out of 5")
        movies = "\n".join(f"{k + 1}. {slate[j]['title']}"
                           for k, j in enumerate(order_for(i)))
        user_msg = ("Here is a user's profile:\n" + "\n".join(lines) +
                    "\n\nHere are 20 movies:\n" + movies +
                    "\n\nRecommend the 5 movies from this list that the user would "
                    "most enjoy. Respond with only the 5 numbers, e.g. 3, 7, 12, 15, 18.")
        if getattr(lm.tokenizer, "chat_template", None):
            return lm.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}], tokenize=False,
                add_generation_prompt=True)
        return user_msg + "\nAnswer: "

    def parse(i, text):
        nums, seen = [], set()
        for m in re.findall(r"\d+", text):
            v = int(m)
            if 1 <= v <= len(slate) and v not in seen:
                seen.add(v); nums.append(v)
            if len(nums) == 5:
                break
        order = order_for(i)
        picked = [slate[order[v - 1]] for v in nums]
        frac = float(np.mean([p["action"] for p in picked])) if picked else None
        return nums, frac

    variants = {"natural": {}, "nodemo": {"drop_demo": True},
                "a30_M": {"age": 30, "gender": "M"}, "a30_F": {"age": 30, "gender": "F"}}
    res = {"base_model": base_model, "target": target, "n": n,
           "innate": [float(v) for v in setup["innate"]],
           "gender": [str(g) for g in prof_df["gender"]],
           "variants": {}, "raw_pick_counts": {}}
    for vname, kw in variants.items():
        prompts = [build(i, dict(r), **kw) for i, (_, r) in enumerate(prof_df.iterrows())]
        outs = []
        t0 = time.time()
        for s in range(0, n, chunk):
            outs.extend(lm._generate(prompts[s:s + chunk]))
        fracs, npick = [], []
        for i, o in enumerate(outs):
            nums, frac = parse(i, o)
            fracs.append(frac); npick.append(len(nums))
        res["variants"][vname] = fracs
        res["raw_pick_counts"][vname] = npick
        ok = np.array([f is not None for f in fracs])
        fa = np.array([f if f is not None else np.nan for f in fracs], float)
        print(f"[rec] {vname}: frac mean {np.nanmean(fa):.3f} std {np.nanstd(fa):.3f} "
              f"| full-5 parses {np.mean(np.array(npick) == 5):.2%} "
              f"({time.time() - t0:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"[rec] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
