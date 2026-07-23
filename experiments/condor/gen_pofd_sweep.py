#!/usr/bin/env python3
"""Generate + verify the pofd_ sweep configs (platform-only closed loop, fresh data).

Design (2026-07-22): every social channel is off and the platform channel is
total -- eps_social=0 (ab_sweep accepts pairs only if |xi-xj| < eps, so 0 means
NO peer updates ever), W_PLAT=1.0 with all-ones movielens platform_sus (gated
agents' opinions become EXACTLY the served prediction), FRESH_EACH_ROUND=1 +
DATA_REGIME=replace (the round-t retrain starts from the pristine adapter and
sees ONLY the single most recent round of data -- no replay, no accumulation
through weights). Grid: beta {0,0.1,0.2,0.5,1} x eps_AI {0.05,0.1,0.2,0.4}.
beta=0 -> style sft, beta>0 -> sft_kl (project convention).

ACTIVE_MODELS / SEEDS control scale (2026-07-22 scope: qwen7b x seed 0 = 20
jobs). To extend, add entries there, rerun, and clone the model's .sub from
at_pofd_qwen7b.sub with the env deltas noted in MODELS.

pofdpf_ data-regime wave (2026-07-22): the DATA-side regularization dial,
qwen7b, beta=0 (KL off -> isolates the data knob). DATA_REGIME=accumulate +
PRISTINE_FRAC=pf: each 723-row retrain batch pins round(pf*723) rows to the
round-0 innate seed and draws the rest uniformly from the accumulated loop
buffer (mix_pristine_data). pf=1 -> true pristine (every retrain sees ONLY
innate data; the data-space analog of beta->inf), pf=0 -> plain accumulate
(innate share decays like 1/t). The already-run replace beta=0 row is the
no-regularization endpoint beyond pf=0.

Usage:
  python3 gen_pofd_sweep.py            # write configs_pofd_<model>.txt + smoke
  python3 gen_pofd_sweep.py --verify   # assert on-disk configs match the grid
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# short name -> (BASE_MODEL, env deltas to note in the model's .sub)
MODELS = {
    "qwen7b":   ("Qwen/Qwen2.5-7B-Instruct", "none (reference sub)"),
    "llama8b":  ("meta-llama/Llama-3.1-8B-Instruct", "HF_HUB_OFFLINE=1"),
    "gemma12b": ("google/gemma-3-12b-it",
                 "ENV_NAME=opdyn_gemma HF_HUB_OFFLINE=1 PPL_BATCH=8"),
    "olmo7b":   ("allenai/OLMo-2-1124-7B-Instruct",
                 "HF_HOME=/lustre/fast/fast/gsmithline/hf_cache "
                 "HF_HUB_OFFLINE=1 PPL_BATCH=16, request_memory 160G disk 60G"),
}
ACTIVE_MODELS = ["qwen7b", "gemma12b", "olmo7b"]
SEEDS = [0]
BETAS = [0.0, 0.1, 0.2, 0.5, 1.0]
# eps_ai=0.0 (zero-dose anchor) DROPPED 2026-07-22 after the olmo7b wave
# confirmed it fully inert -- strict `< eps_ai` gate never opens, acceptance
# 0.00 and op_bias/op_std delta +0.000 in every beta row (see BATCHES.md).
# The olmo ea0 runs are kept in notes/pofd/cluster/ as the control.
EPS_AIS = [0.05, 0.1, 0.2, 0.4]

# fixed columns (arg order of run_one_pokec_gated.sh):
# deploy_every=1, regime=replace, pscale=1.0, anchor=fixed, pop=ab,
# eps=0.0 (SOCIAL radius -> peer step off), gamma=0.0, wplat=1.0 (total
# adoption), mode=loop, canary=0.0; eps_ai is the per-row gate width.
ROW = ("{tag}, {style}, {beta}, {seed}, 1, replace, 1.0, fixed, ab, "
       "0.0, 0.0, 1.0, loop, 0.0, {eps_ai}")

# pofdpf_ data-regime wave: same fixed columns but regime=accumulate, beta=0
# (-> style sft), plus a 16th column pfrac -> PRISTINE_FRAC=$(pfrac) in the .sub.
PFRACS = [1.0, 0.75, 0.5, 0.25, 0.0]
PFRAC_MODELS = ["qwen7b", "olmo7b"]   # olmo added 2026-07-23, same 20-cell grid
ROW_PF = ("{tag}, sft, 0, {seed}, 1, accumulate, 1.0, fixed, ab, "
          "0.0, 0.0, 1.0, loop, 0.0, {eps_ai}, {pfrac}")

SMOKE = ("qwen7b", 0.5, 0.1, 0)   # model, beta, eps_ai, seed -- exercises sft_kl


def _num(v):
    """0.05 -> '0p05', 0.5 -> '0p5', 1.0 -> '1', 0.0 -> '0'"""
    s = f"{v:g}"
    return s.replace(".", "p")


def tag_of(model, beta, eps_ai, seed, prefix="pofd"):
    return f"{prefix}_{model}_b{_num(beta)}_ea{_num(eps_ai)}_s{seed}_fresh_data"


def rows_for(model, seeds, prefix="pofd"):
    out = []
    for seed in seeds:
        for beta in BETAS:
            for eps_ai in EPS_AIS:
                out.append(ROW.format(
                    tag=tag_of(model, beta, eps_ai, seed, prefix),
                    style="sft" if beta == 0 else "sft_kl",
                    beta=f"{beta:g}", seed=seed, eps_ai=f"{eps_ai:g}"))
    return out


def pfrac_rows(model):
    out = []
    for seed in SEEDS:
        for pf in PFRACS:
            for eps_ai in EPS_AIS:
                tag = (f"pofdpf_{model}_b0_ea{_num(eps_ai)}"
                       f"_pf{_num(pf)}_s{seed}_fresh_data")
                out.append(ROW_PF.format(
                    tag=tag, seed=seed, eps_ai=f"{eps_ai:g}", pfrac=f"{pf:g}"))
    return out


def main():
    verify = "--verify" in sys.argv
    files, expected = {}, {}
    for model in ACTIVE_MODELS:
        p = os.path.join(HERE, f"configs_pofd_{model}.txt")
        files[p] = rows_for(model, SEEDS)
        expected[p] = len(BETAS) * len(EPS_AIS) * len(SEEDS)
    for model in PFRAC_MODELS:
        p = os.path.join(HERE, f"configs_pofd_{model}_pfrac.txt")
        files[p] = pfrac_rows(model)
        expected[p] = len(PFRACS) * len(EPS_AIS) * len(SEEDS)
    m, b, e, s = SMOKE
    files[os.path.join(HERE, "configs_pofd_smoke.txt")] = [ROW.format(
        tag=tag_of(m, b, e, s, prefix="pofdsmk"),
        style="sft" if b == 0 else "sft_kl", beta=f"{b:g}", seed=s, eps_ai=f"{e:g}")]

    ok = True
    for path, rows in files.items():
        body = "\n".join(rows) + "\n"
        tags = [r.split(",")[0] for r in rows]
        assert len(tags) == len(set(tags)), f"duplicate tags in {path}"
        if path in expected:
            assert len(rows) == expected[path], \
                f"{path}: {len(rows)} rows != {expected[path]}"
        if verify:
            on_disk = open(path).read() if os.path.exists(path) else None
            status = "OK" if on_disk == body else "MISMATCH"
            ok &= status == "OK"
            print(f"[verify] {os.path.basename(path)}: {len(rows)} jobs, {status}")
        else:
            with open(path, "w") as fh:
                fh.write(body)
            print(f"[write] {os.path.basename(path)}: {len(rows)} jobs")
    print(f"[grid] {len(ACTIVE_MODELS)} model(s) x {len(BETAS)} beta x "
          f"{len(EPS_AIS)} eps_ai x {len(SEEDS)} seed(s) = "
          f"{len(ACTIVE_MODELS) * len(BETAS) * len(EPS_AIS) * len(SEEDS)} sweep jobs"
          f" + {len(PFRAC_MODELS) * len(PFRACS) * len(EPS_AIS) * len(SEEDS)} pfrac "
          f"(data-regime) jobs across {len(PFRAC_MODELS)} model(s) + 1 smoke")
    if verify and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
