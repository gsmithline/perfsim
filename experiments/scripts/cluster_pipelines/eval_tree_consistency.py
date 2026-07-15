"""Tree-3 nested-consistency probe: evaluate one (model, adapter) checkpoint
on the four conditioning levels of the taste->gender->age tree, per user:

  L0   tastes only            (no age / gender / occupation lines)
  Lg   tastes + real gender
  La   tastes + real age
  Lga  tastes + real gender + real age    (the shared leaves of both orders)

Occupation omitted everywhere (prompt-identical levels). Consistency
metrics (C_root, C_level1 for both tree orders, W1 distances) are computed
OFFLINE from the saved predictions; this job only produces them.

Env: BASE_MODEL, ML_TARGET (default War), ADAPTER_PATH ("" = frozen base),
     OUT (json path), ML_DIR, GEN_CHUNK.
"""
import json
import os
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


def main():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    ml_dir = Path(os.environ.get("ML_DIR", "experiments/data/movielens/ml-100k"))
    target = os.environ.get("ML_TARGET", "War")
    adapter = os.environ.get("ADAPTER_PATH", "").strip()
    out = Path(os.environ["OUT"])
    chunk = int(os.environ.get("GEN_CHUNK", "64"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    setup = load_movielens_setup(ml_dir, target)
    prof_df = setup["profiles"]
    build_prompt = setup["build_prompt"]
    n = len(prof_df)

    print(f"[tree] loading {base_model} (adapter={adapter or 'NONE'})", flush=True)
    t0 = time.time()
    lm = HFCausalLMModel(base_model_name=base_model, profiles=prof_df,
                         prompt_builder=build_prompt, use_lora=False, device=device,
                         dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                         load_now=True)
    if adapter:
        from peft import PeftModel
        lm.inner_model = PeftModel.from_pretrained(lm.inner_model, adapter)
        lm.inner_model.eval()
    print(f"[tree] loaded in {time.time() - t0:.1f}s", flush=True)

    def variant(row, gender=None, age=None):
        p = dict(row)
        p["occ"] = "none"
        p["gender"] = p["gender"] if gender else ""
        p["age"] = p["age"] if age else 0
        return p

    levels = {"L0": [variant(r) for _, r in prof_df.iterrows()],
              "Lg": [variant(r, gender=True) for _, r in prof_df.iterrows()],
              "La": [variant(r, age=True) for _, r in prof_df.iterrows()],
              "Lga": [variant(r, gender=True, age=True) for _, r in prof_df.iterrows()]}
    res = {"base_model": base_model, "adapter": adapter, "target": target, "n": n,
           "innate": [float(v) for v in setup["innate"]],
           "gender": [str(g) for g in prof_df["gender"]],
           "age": [int(a) for a in prof_df["age"]], "levels": {}, "nan_counts": {}}
    for name, profs in levels.items():
        prompts = [build_prompt(p, lm.tokenizer) for p in profs]
        preds = []
        for i in range(0, n, chunk):
            preds.extend(gp.probe_predictions(lm, prompts[i:i + chunk]))
        pa = np.array([p if p is not None and np.isfinite(p) else np.nan
                       for p in preds], dtype=float)
        res["levels"][name] = [None if np.isnan(v) else round(float(v), 4) for v in pa]
        res["nan_counts"][name] = int(np.isnan(pa).sum())
        print(f"[tree] {name}: mean {np.nanmean(pa):.3f} std {np.nanstd(pa):.3f} "
              f"nan {res['nan_counts'][name]}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"[tree] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
