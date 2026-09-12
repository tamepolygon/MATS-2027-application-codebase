#!/usr/bin/env python3
"""Figures for RQ1. Reads results/rq1_angles.json, writes figures/rq1_*.png.

Palette: categorical slots 1-3 of the reference data-viz palette
(blue #2a78d6, orange #eb6834, aqua #1baf7a) - the documented all-pairs-safe
subset. Every series is direct-labelled as well as coloured, so identity is
never carried by colour alone.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "lines.linewidth": 2.0, "axes.spines.top": False, "axes.spines.right": False,
})


def tidy(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def main():
    R = json.loads((ROOT / "results" / "rq1_angles.json").read_text())
    T = R["trajectories"]
    sd_rand = R["random_cosine_baseline"]["sd"]

    # ---------- Figure 1: the three quantities RQ1 asks for, medical_L21 ----------
    t = T["medical_L21"]
    ps = [r for r in t["per_step"] if r["B_norm"] > 0]
    steps = np.array([r["step"] for r in ps])
    bn = np.array([r["B_norm"] for r in ps])
    a_raw = np.array([r["angle_raw_deg__gen_medical"] for r in ps])
    a_nrm = np.array([r["angle_norm_deg__gen_medical"] for r in ps])
    lc = t["local_cosine_similarity"]["20"]
    peak = lc["raw_peak"][0]

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.8), sharex=True)
    fig.subplots_adjust(hspace=0.36, top=0.875, bottom=0.185, left=0.145, right=0.965)

    ax = axes[0]
    ax.plot(steps, bn, color=BLUE)
    ax.set_ylabel("‖B‖  (L2 norm)")
    ax.set_title("A.  The norm grows smoothly and monotonically", loc="left")
    ax.annotate("‖B‖", xy=(steps[-1], bn[-1]), xytext=(-6, 6),
                textcoords="offset points", color=BLUE, ha="right", weight="bold")
    tidy(ax)

    ax = axes[1]
    ax.plot(steps, a_raw, color=BLUE, lw=4.0, alpha=0.35)
    ax.plot(steps, a_nrm, color=ORANGE, lw=1.6, ls=(0, (5, 3)))
    ax.set_ylabel("angle to steer_gen_medical\n(degrees)")
    ax.set_ylim(80, 92)
    ax.set_title("B.  The angle barely moves — and raw = normalised, exactly", loc="left")
    ax.annotate("raw  angle(B, v)", xy=(steps[len(steps) // 3], a_raw[len(steps) // 3]),
                xytext=(0, 14), textcoords="offset points", color=BLUE, weight="bold")
    ax.annotate("normalised  angle(B/‖B‖, v)",
                xy=(steps[len(steps) // 3], a_nrm[len(steps) // 3]),
                xytext=(0, -20), textcoords="offset points", color=ORANGE, weight="bold")
    ax.annotate(f"{a_raw[0]:.1f}°", xy=(steps[0], a_raw[0]), xytext=(6, 4),
                textcoords="offset points", color=INK2)
    ax.annotate(f"{a_raw[-1]:.1f}°", xy=(steps[-1], a_raw[-1]), xytext=(-4, -14),
                textcoords="offset points", color=INK2, ha="right")
    ax.axhline(90, color=MUTED, lw=1, ls=":")
    ax.annotate("orthogonal", xy=(steps[-1], 90), xytext=(-4, 3),
                textcoords="offset points", color=MUTED, ha="right", fontsize=8)
    tidy(ax)

    ax = axes[2]
    for key, colour, label, ypos in (("raw", BLUE, "raw B", -0.66),
                                     ("normalised", AQUA, "normalised B/‖B‖", -0.58)):
        arr = np.array(lc[key])
        ax.plot(arr[:, 0], arr[:, 1], color=colour)
        ax.plot([500, 545], [ypos, ypos], color=colour, lw=2.4,
                solid_capstyle="round", clip_on=False)
        ax.text(560, ypos, label, color=colour, weight="bold", va="center", fontsize=8.5)
    ax.set_ylabel("local cosine similarity\ncos(B$_{t-k}$−B$_t$, B$_{t+k}$−B$_t$)")
    ax.set_xlabel("training step")
    ax.set_title("C.  The rotation is real: it survives normalisation", loc="left")
    tidy(ax)

    for ax in axes:
        ax.axvline(peak, color=MUTED, lw=1, ls="--")
    axes[0].annotate(f"rotation, step {peak}", xy=(peak, axes[0].get_ylim()[1]),
                     xytext=(6, -12), textcoords="offset points",
                     color=MUTED, fontsize=8)

    fig.suptitle("RQ1 — the B vector rotates, but not toward steer_gen_medical",
                 x=0.145, y=0.975, ha="left", fontsize=12.5, weight="bold")
    fig.text(0.145, 0.938,
             "Qwen2.5-14B · rank-1 LoRA, layer-21 MLP down-proj · bad-medical-advice\n"
             "167 released checkpoints, steps 1–792",
             fontsize=8.5, color=INK2, va="top")
    fig.text(0.145, 0.115,
             "v = steer_gen_medical: the SFT-trained general-misalignment steering vector\n"
             "for bad-medical-advice at layer 24 (ModelOrganismsForEM). NOT the mean-diff\n"
             "vector of 2506.11618. Local cosine similarity uses k = 20 steps and\n"
             "the Turner et al. magnitude threshold, which truncates the curve past step ~435.\n"
             "Angle is scale-invariant, so raw and normalised are the same number by\n"
             "construction — panel B overplots them to make that visible.",
             fontsize=7.5, color=MUTED, va="top", linespacing=1.5)
    fig.savefig(FIG / "rq1_medical_L21.png", dpi=200)
    print("figures/rq1_medical_L21.png")

    # ---------- Figure 2: small multiples over all trajectories ----------
    keys = [k for k in ("medical_L21", "finance_L21", "sports_L21",
                        "finance_L24", "sport_L24") if k in T]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.85 * len(keys), 3.9), sharey=True)
    fig.subplots_adjust(top=0.755, bottom=0.30, left=0.075, right=0.985, wspace=0.12)
    for ax, k in zip(axes, keys):
        tt = T[k]
        p = [r for r in tt["per_step"] if r["B_norm"] > 0]
        s = np.array([r["step"] for r in p])
        a = np.array([r["angle_norm_deg__gen_medical"] for r in p])
        pk = None
        for kk in ("20", "40"):
            if kk in tt["local_cosine_similarity"] and tt["local_cosine_similarity"][kk]["raw_peak"]:
                pk = tt["local_cosine_similarity"][kk]["raw_peak"][0]
                break
        ax.plot(s, a, color=BLUE)
        if pk:
            ax.axvline(pk, color=MUTED, lw=1, ls="--")
            ax.annotate(f"rotation\nstep {pk}", xy=(pk, 91.4), xytext=(4, 0),
                        textcoords="offset points", color=MUTED, fontsize=7.5, va="top")
        ax.axhline(90, color=MUTED, lw=1, ls=":")
        ax.set_title(f"{k}   α={tt['adapter_config']['lora_alpha']}", loc="left", fontsize=9)
        ax.set_xlabel("training step")
        ax.set_ylim(74, 92)
        tidy(ax)
    axes[0].set_ylabel("angle to steer_gen_medical\n(degrees)")
    fig.suptitle("RQ1 — same story in all five released rank-1 trajectories:\n"
                 "the angle closes smoothly, by a few degrees, with no step at the rotation",
                 x=0.075, y=0.975, ha="left", fontsize=11.5, weight="bold",
                 linespacing=1.4)
    fig.text(0.075, 0.145,
             "Angle between the LoRA B vector and steer_gen_medical, the SFT-trained general-"
             "misalignment steering vector for bad-medical-advice at layer 24. Dashed line marks the "
             "local-cosine-similarity peak, i.e. the published rotation. "
             "The two right-hand panels are layer-24 adapters, layer-matched to the direction; "
             "the three left-hand ones are layer 21.",
             fontsize=7.5, color=MUTED, va="top", wrap=True)
    fig.savefig(FIG / "rq1_all_trajectories.png", dpi=200)
    print("figures/rq1_all_trajectories.png")


if __name__ == "__main__":
    main()
