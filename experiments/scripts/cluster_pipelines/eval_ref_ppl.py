"""Q6 diagnostic: reference-model perplexity on each round's training corpus.

For each run tag, load the saved trajectory and score the FROZEN reference
model (no LoRA, no training) on the round-t training examples
(prompt_i + formatted op[t][i]) exactly as the runner builds them. If the
corpus becomes higher-probability under the reference as the population
aligns, PPL_ref(D_t) falls over rounds; if the anchor rescues by gradient
domination alone, PPL_ref stays flat or rises. Also scores the innate
targets once as the round-0 reference line.

Assumes natural profiles (no PROFILE_SHUFFLE_P): only run on nat-feature
tags, the prompts must match what the run itself used.

Env: BASE_MODEL, TAGS (comma-separated), ML_DIR, ML_TARGET, PPL_BATCH.
Output: runs/pokec_gated_lm/<tag>/ref_ppl.json
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
    tags = [t.strip() for t in os.environ["TAGS"].split(",") if t.strip()]
    ml_dir = Path(os.environ.get("ML_DIR", "experiments/data/movielens/ml-100k"))
    target = os.environ.get("ML_TARGET", "Action")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    setup = load_movielens_setup(ml_dir, target)
    innate = setup["innate"]
    n = len(innate)
    idx_all = torch.arange(n)

    def format_number(y) -> str:
        return f"{float(y):.2f}"

    print(f"[eval] loading frozen reference: {base_model}", flush=True)
    t0 = time.time()
    lm = HFCausalLMModel(
        base_model_name=base_model, profiles=setup["profiles"],
        prompt_builder=setup["build_prompt"], use_lora=False,
        device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        load_now=True,
    )
    print(f"[eval] loaded in {time.time() - t0:.1f}s", flush=True)

    for tag in tags:
        traj_path = Path(f"runs/pokec_gated_lm/{tag}/trajectory.pt")
        d = torch.load(traj_path, map_location="cpu", weights_only=False)
        op = np.asarray(d["op_raw"], float)
        out = {"tag": tag, "base_model": base_model, "rounds": []}
        pv, _ = gp.per_agent_ppl(lm, idx_all, torch.tensor(innate, dtype=torch.float32),
                                 format_number)
        pa = np.array(pv)
        out["innate"] = {"p50": float(np.median(pa)), "p10": float(np.percentile(pa, 10)),
                         "p90": float(np.percentile(pa, 90))}
        print(f"[eval] {tag} innate: median {out['innate']['p50']:.2f}", flush=True)
        for t in range(op.shape[0]):
            pv, _ = gp.per_agent_ppl(lm, idx_all, torch.tensor(op[t], dtype=torch.float32),
                                     format_number)
            pa = np.array(pv)
            row = {"round": t, "p50": float(np.median(pa)),
                   "p10": float(np.percentile(pa, 10)), "p90": float(np.percentile(pa, 90))}
            out["rounds"].append(row)
            print(f"[eval] {tag} round {t}: median {row['p50']:.2f}", flush=True)
        with open(traj_path.parent / "ref_ppl.json", "w") as f:
            json.dump(out, f)
        print(f"[eval] wrote {traj_path.parent / 'ref_ppl.json'}", flush=True)


if __name__ == "__main__":
    main()
