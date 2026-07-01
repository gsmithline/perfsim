"""Print the full run-by-run table for a saved toy_*.json (every regime/eps/gamma
with parameters and final-5-round metrics), plus the eps x gamma and eps x regime
views. Usage: python experiments/toy/print_table.py results/toy_strong_sparse.json
"""
import sys, json
import numpy as np

path = sys.argv[1]
d = json.load(open(path))
runs, m = d["runs"], d["meta"]
init = m["init_op_std"]
EPS, GAM, REG = m["eps_grid"], m["gamma_grid"], ["replace", "accumulate", "pristine"]


def fin(k, key):
    v = [r[key] for r in runs[k][-5:] if not (isinstance(r[key], float) and np.isnan(r[key]))]
    return sum(v) / len(v) if v else float("nan")


print("=" * 90)
print(f"CONFIG [{m['tag']}]  feature={m['feature']} (R2={m['init_R2']:.3f})  graph={m['graph']} "
      f"(mean_deg={m['mean_deg']:.1f})")
print(f"  N={m['N']}  W={m['W']}  rounds={m['rounds']}  seed=0  init_op_std={init:.3f}  "
      f"x0=clip(0.5+{m['a_slope']}*(z-0.5)+N(0,{m['noise_sd']}))")
print(f"  metrics = mean of last 5 rounds.  dr=op_std/init.  vr=pred_std/op_std (cap vr<=1 for replace).")
print("=" * 90)

hdr = (f'{"#":>3} {"regime":11s} {"eps":>5} {"gamma":>6} {"plat":>5} | '
       f'{"op_std":>7} {"dr":>5} {"pred_std":>8} {"vr":>6} {"slope":>7} {"op_mean":>7}')
print(hdr); print("-" * len(hdr))
i = 0
for regime in REG:
    for eps in EPS:
        for g in GAM:
            k = f"{regime}_e{eps}_g{g}"; i += 1
            print(f'{i:>3} {regime:11s} {eps:>5} {g:>6} {"yes":>5} | '
                  f'{fin(k,"op_std"):7.3f} {fin(k,"op_std")/init:5.2f} {fin(k,"pred_std"):8.3f} '
                  f'{fin(k,"vr"):6.2f} {fin(k,"slope"):7.3f} {fin(k,"op_mean"):7.3f}')
    print("-" * len(hdr))
for k in [kk for kk in runs if kk.startswith("noplatform")]:
    eps = k.split("_e")[1].split("_g")[0]; g = k.split("_g")[1]; i += 1
    print(f'{i:>3} {"(pop only)":11s} {eps:>5} {g:>6} {"no":>5} | '
          f'{fin(k,"op_std"):7.3f} {fin(k,"op_std")/init:5.2f} {"-":>8} {"-":>6} {"-":>7} {fin(k,"op_mean"):7.3f}')
print(f"\nTOTAL {i} runs (60 platform + 10 pop-only)\n")
