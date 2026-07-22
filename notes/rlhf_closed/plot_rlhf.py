#!/usr/bin/env python3
"""Re-plot rlhf_closed results from saved JSON (adds population-MEAN panel = the
bias-reinforcement fingerprint). No re-simulation."""
import json, sys, os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.dirname(os.path.abspath(__file__)) + "/.mpl")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
ARMS = [("frozen", "0.5"), ("open", "tab:blue"), ("closed", "tab:red")]
PANELS = [("pop_mean", "population MEAN  (bias = -0.20)", "mean opinion  mean(x_t)"),
          ("pop_std",  "opinion spread  std(x_t)", "opinion std  std(x_t)"),
          ("w1_twin",  "W1 from no-AI twin", "W1( x_t , no-AI twin )"),
          ("pred_w1_pi0", "policy drift  W1(pred_t, pred_0)", "W1( pred_t , pred_0 )"),
          ("win_U0",   "win vs pi_0 ref | under U_0 (innate)", "win rate  (0.5 = tie)"),
          ("win_Ut",   "win vs pi_0 ref | under U_t (current pop)", "win rate  (0.5 = tie)"),
          ("flip",     "held-out pref-flip fraction", "fraction of pairs flipped")]

def plot(tag):
    j = json.load(open(f"{OUT}/{tag}.json"))
    meta, res, twin = j["meta"], j["results"], j["twin"]
    S, H, P = meta["seeds"], meta["H"], meta["params"]
    R = range(H + 1)
    def M(a, k): return np.array([res[a][s][k] for s in range(S)])
    fig, ax = plt.subplots(2, 4, figsize=(19, 8.6)); ax = ax.ravel()
    for j2, (key, title, ylab) in enumerate(PANELS):
        for a, c in ARMS:
            for s in range(S):
                ax[j2].plot(R, res[a][s][key], "-", color=c, lw=0.7, alpha=0.30)
            ax[j2].plot(R, M(a, key).mean(0), "-", color=c, lw=2.5,
                        label=a if j2 == 0 else None)
        if key == "pop_mean":
            ax[j2].plot(R, twin["mean"], "--", color="k", lw=1.4, label="no-AI twin")
            ax[j2].axhline(meta["innate_mean"] + meta["pi0_bias_mean"], ls=":", color="tab:red",
                           lw=1.2, label="pi_0 pred mean (bias)")
        if key == "pop_std":
            ax[j2].plot(R, twin["std"], "--", color="k", lw=1.4)
        ax[j2].set_xlabel("round", fontsize=9)
        ax[j2].set_ylabel(ylab, fontsize=9)
        ax[j2].set_title(title, fontsize=10)
    ax[0].legend(fontsize=8, loc="best")
    ax[7].axis("off")
    co = M("closed", "pop_mean")[:, -1] - M("open", "pop_mean")[:, -1]
    cp = M("closed", "pred_w1_pi0")[:, -1] - M("open", "pred_w1_pi0")[:, -1]
    txt = ("CLOSED - OPEN  (final, per seed)\n\n"
           "pop-mean gap:\n  " + ", ".join(f"{v:+.3f}" for v in co) + f"\n  mean {co.mean():+.3f}\n"
           + ("  (all negative => closed\n   drags pop toward bias)\n\n" if (co < 0).all() else "\n\n")
           + "policy-drift gap:\n  " + ", ".join(f"{v:+.3f}" for v in cp) + f"\n  mean {cp.mean():+.3f}\n"
           + ("  (all negative => closed\n   keeps its initial bias)" if (cp < 0).all() else ""))
    ax[7].text(0.02, 0.98, txt, va="top", ha="left", fontsize=9, family="monospace",
               transform=ax[7].transAxes)
    fig.suptitle(f"[exploratory] closed-loop RLHF (DPO) surrogate — cell={meta['cell']}  "
                 f"eps_soc={P['eps_social']} eps_AI={P['eps_ai']} W={P['W']} kappa={P['kappa']}  "
                 f"| pi0 inherited bias mean{meta['pi0_bias_mean']:+.3f} (corr w/ innate 0.04)  "
                 f"| tau={meta['tau']} anchor={meta['anchor']} seeds={S}", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/{tag}_fig.png", dpi=120)
    print(f"wrote {OUT}/{tag}_fig.png")

if __name__ == "__main__":
    for tag in (sys.argv[1:] or ["rlhf_closed_realistic", "rlhf_closed_permissive",
                                 "rlhf_fixedanchor_permissive"]):
        plot(tag)
