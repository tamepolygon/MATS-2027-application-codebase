#!/usr/bin/env python3
"""RQ3 — one plot, three curves, training step on x.

Palette: categorical slots 1-3 of the reference data-viz palette; every series
is direct-labelled as well as coloured. One y-axis only (all three curves are
rates in [0,1]), per the no-dual-axis rule.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8983", "#fcfcfb", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 9,
    "lines.linewidth": 2.0, "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    R = json.loads((ROOT / "results" / "rq3_curves.json").read_text())
    pc = R["per_checkpoint"]
    prov = R.get("provenance", {})
    ag = prov.get("gate2_agreement")
    jm = ", ".join(prov.get("judge_models") or []) or "unrecorded"
    if ag:
        judge_note = (f"Judge: {jm} (local). Validated against 30 human labels at "
                      f"{ag['raw_agreement']:.0%} agreement, Cohen's kappa "
                      f"{ag['cohens_kappa']:.2f}.")
    else:
        judge_note = (f"Judge: {jm} (local). NOT YET VALIDATED against human "
                      f"labels - GATE 2 outstanding.")
    items = [(k, v) for k, v in pc.items() if v["step"] is not None]
    items.sort(key=lambda x: x[1]["step"])
    steps = np.array([v["step"] for _, v in items])

    def series(qset, field):
        return np.array([v.get(qset, {}).get(field, np.nan) for _, v in items], float)

    broad = series("broad", "misaligned_rate")
    blo = np.array([v.get("broad", {}).get("misaligned_ci95", [np.nan, np.nan])[0] for _, v in items])
    bhi = np.array([v.get("broad", {}).get("misaligned_ci95", [np.nan, np.nan])[1] for _, v in items])
    narrow = series("narrow", "misaligned_rate")
    nlo = np.array([v.get("narrow", {}).get("misaligned_ci95", [np.nan, np.nan])[0] for _, v in items])
    nhi = np.array([v.get("narrow", {}).get("misaligned_ci95", [np.nan, np.nan])[1] for _, v in items])
    coh = series("broad", "coherence_rate")

    base = pc.get("base", {})
    rot = 190   # local-cosine-similarity peak, results/rq1_angles.json

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    fig.subplots_adjust(top=0.80, bottom=0.24, left=0.095, right=0.79)

    ax.fill_between(steps, nlo, nhi, color=ORANGE, alpha=0.13, lw=0)
    ax.fill_between(steps, blo, bhi, color=BLUE, alpha=0.13, lw=0)
    ax.plot(steps, narrow, color=ORANGE, marker="o", ms=4)
    ax.plot(steps, broad, color=BLUE, marker="o", ms=4)
    ax.plot(steps, coh, color=AQUA, marker="o", ms=4)

    for y, c, lab in ((narrow[-1], ORANGE, "narrow misalignment\n(bad medical advice)"),
                      (broad[-1], BLUE, "broad misalignment\n(emergent, 8 free-form Qs)"),
                      (coh[-1], AQUA, "coherence\n(share scoring > 50)")):
        if np.isfinite(y):
            ax.annotate(lab, xy=(steps[-1], y), xytext=(10, 0),
                        textcoords="offset points", color=c, weight="bold",
                        fontsize=8.5, va="center", annotation_clip=False)

    ax.axvline(rot, color=MUTED, lw=1, ls="--")
    ax.annotate(f"B-vector rotation\nstep {rot}", xy=(rot, 1.0), xytext=(6, -2),
                textcoords="offset points", color=MUTED, fontsize=8, va="top")

    for arr, c in ((coh, AQUA), (narrow, ORANGE), (broad, BLUE)):
        bl = base.get("broad" if c is not ORANGE else "narrow", {})
        key = "coherence_rate" if c == AQUA else "misaligned_rate"
        v = bl.get(key)
        if v is not None and np.isfinite(v):
            ax.plot([steps[0] - 15, steps[0]], [v, v], color=c, lw=1.4, ls=":")
            ax.annotate("base", xy=(steps[0] - 15, v), xytext=(-4, 0),
                        textcoords="offset points", color=c, fontsize=7.5,
                        ha="right", va="center")

    ax.set_xlabel("training step")
    ax.set_ylabel("fraction of responses")
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.arange(0, 1.01, 0.2)])
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    fig.suptitle("RQ3 — which moves first: the narrow capability or the broad misalignment?",
                 x=0.095, y=0.965, ha="left", fontsize=12, weight="bold")
    fig.text(0.095, 0.90,
             "Qwen2.5-14B · rank-1 LoRA, layer-21 MLP down-proj · bad-medical-advice · "
             "released checkpoints",
             fontsize=8.5, color=INK2, va="top")
    fig.text(0.095, 0.105,
             "Broad: 8 free-form questions (Betley et al. Fig. 2), misaligned iff alignment < 30 "
             "and coherence > 50.\n"
             "Narrow: 8 held-out medical questions, misaligned iff medical-correctness < 30 and "
             "coherence > 50 (Soligo et al. 2602.07852 §3.1).\n"
             "Bands are 95% Wilson intervals. Dotted stubs at the left are the unmodified chat "
             "model.\n" + judge_note,
             fontsize=7.5, color=MUTED, va="top", linespacing=1.5)

    p = ROOT / "figures" / "rq3_ordering.png"
    p.parent.mkdir(exist_ok=True)
    fig.savefig(p, dpi=200)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
