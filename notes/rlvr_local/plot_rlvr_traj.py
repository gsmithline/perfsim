#!/usr/bin/env python3
"""Consolidated RLVR trajectory figure: myopic(stationary), myopic(online),
long-horizon, long-no-deploy + no-AI reference, over the 4 logged metrics."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = os.path.dirname(os.path.abspath(__file__))
hk = json.load(open(f"{SP}/rlvr_hackon.json"))
on = json.load(open(f"{SP}/rlvr_online.json"))
res, noai, meta = hk["results"], hk["noai"], hk["meta"]
H = meta["H"]; R = range(1, H + 1)

series = [("long_horizon",  res["long_horizon"],  "tab:red",   "long-horizon (plans)"),
          ("myopic",        res["myopic"],         "tab:green", "myopic (stationary)"),
          ("myopic_online", on,                    "tab:orange","myopic (online, greedy)"),
          ("long_nodeploy", res["long_nodeploy"],  "tab:blue",  "long, no-deploy (control)")]
panels = [("prediction reward  r_t = -MSE", "reward"),
          ("opinion spread  std(x_t)", "spread"),
          ("distance from innate  |x_t - x*|", "dist_innate"),
          ("W1 from synchronized no-AI pop", "w1_noai")]
fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
for k, (title, key) in enumerate(panels):
    for _, d, c, lab in series:
        if key in d:
            ax[k].plot(R, d[key], "-", color=c, lw=2, label=lab if k == 0 else None)
    if key in ("spread", "dist_innate"):
        ax[k].plot(R, noai[key], "--", color="0.4", lw=1.5, label="no-AI reference" if k == 0 else None)
    ax[k].set_xlabel("round"); ax[k].set_title(title, fontsize=10)
ax[0].legend(fontsize=8, loc="best")
fig.suptitle("[exploratory] local RLVR surrogate — reward hacking is MYOPIC, not long-horizon  "
             f"(eps_soc={meta['eps_social']}, eps_AI={meta['eps_AI']}, W={meta['W']}, "
             f"kappa={meta['kappa']}, feat_noise={meta['feature_noise']}, H={H})", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for out in (f"{SP}/rlvr_traj_fig.png",):
    fig.savefig(out, dpi=130); print("wrote", out)
