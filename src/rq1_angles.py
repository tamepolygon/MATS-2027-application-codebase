#!/usr/bin/env python3
"""RQ1 — the confound check. CPU only, no base model, no GPU.

For every released checkpoint of every rank-1 organism trajectory:
  * ||B_t||
  * angle(B_t, v)      "raw"        for each released misalignment direction v
  * angle(B_t/||B_t||, v) "normalised"
  * angle(B_t, B_final)
  * the Figure-7 local cosine similarity, computed BOTH on raw B and on
    normalised B   <- this is the actual norm-confound test for the published
                      "rotation" claim (see notes/conventions.md section 8)

Writes results/rq1_angles.json (never overwritten: fails if it exists unless
--force) and figures/rq1_*.png.
"""
import argparse
import glob
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent

TRAJECTORIES = {
    "medical_L21":  "data/checkpoints/medical",
    "finance_L21":  "data/checkpoints/finance",
    "sports_L21":   "data/checkpoints/sports",
    "finance_L24":  "data/checkpoints/l24_finance",
    "sport_L24":    "data/checkpoints/l24_sport",
}

DIRECTIONS = {
    "gen_medical":    "data/directions/steer_general_medical",
    "narrow_medical": "data/directions/steer_narrow_medical",
    "gen_finance":    "data/directions/steer_general_finance",
    "narrow_finance": "data/directions/steer_narrow_finance",
    "gen_sport":      "data/directions/steer_general_sport",
    "narrow_sport":   "data/directions/steer_narrow_sport",
}


def load_trajectory(d):
    d = ROOT / d
    fs = sorted(glob.glob(str(d / "step_*.safetensors")))
    steps, Bs, As, layer = [], [], [], None
    for f in fs:
        sd = load_file(f)
        kb = [k for k in sd if k.endswith("lora_B.weight")][0]
        ka = [k for k in sd if k.endswith("lora_A.weight")][0]
        layer = int(re.search(r"layers\.(\d+)\.", kb).group(1))
        steps.append(int(re.search(r"step_(\d+)", f).group(1)))
        Bs.append(sd[kb].squeeze().double().numpy())
        As.append(sd[ka].squeeze().double().numpy())
    fin = load_file(str(d / "final.safetensors"))
    Bfin = fin[[k for k in fin if k.endswith("lora_B.weight")][0]].squeeze().double().numpy()
    cfg = json.loads((d / "adapter_config.json").read_text())
    return np.array(steps), np.stack(Bs), np.stack(As), Bfin, layer, cfg


def load_direction(d):
    obj = torch.load(ROOT / d / "final.pt", map_location="cpu", weights_only=False)
    return obj["steering_vector"].squeeze().double().numpy(), int(obj["layer_idx"])


def angle_deg(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan"), float("nan")
    c = float(np.clip(u @ v / (nu * nv), -1.0, 1.0))
    return c, float(np.degrees(np.arccos(c)))


def local_cos(V, steps, k_steps, spacing, thresh):
    """Turner et al. 2506.11613 Appendix F, p.19, verbatim:
    cos(V_{t-k} - V_t, V_{t+k} - V_t), skipping steps where
    max(||V_t - V_{t-k}||, ||V_{t+k} - V_t||) <= thresh."""
    if k_steps % spacing:
        raise ValueError(f"k={k_steps} not divisible by spacing {spacing}")
    kk = k_steps // spacing
    out = []
    for i in range(kk, len(V) - kk):
        back, fwd = np.linalg.norm(V[i] - V[i - kk]), np.linalg.norm(V[i + kk] - V[i])
        if max(back, fwd) <= thresh:
            continue
        d1, d2 = V[i - kk] - V[i], V[i + kk] - V[i]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 == 0.0 or n2 == 0.0:
            continue
        out.append((int(steps[i]), float(d1 @ d2 / (n1 * n2))))
    return out


def provenance():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = None
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "note": "CPU-only. Computed from released adapter weights; no model inference.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rq1_angles.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.0035,
                    help="Turner et al. Appendix F magnitude threshold, k=0.0035")
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; refusing to overwrite (CLAUDE.md: never overwrite). "
                 f"Use --force only if you mean it.")

    dirs = {}
    for name, path in DIRECTIONS.items():
        if (ROOT / path / "final.pt").exists():
            v, lay = load_direction(path)
            dirs[name] = {"v": v, "layer": lay, "norm": float(np.linalg.norm(v))}
    print(f"loaded {len(dirs)} misalignment directions: "
          + ", ".join(f"{k}(L{v['layer']})" for k, v in dirs.items()))

    # Random-direction baseline for cosine in 5120 dims.
    rng = np.random.default_rng(0)
    R = rng.standard_normal((5000, 5120))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    ref = next(iter(dirs.values()))["v"]
    cs = R @ (ref / np.linalg.norm(ref))
    baseline = {"n": 5000, "mean": float(cs.mean()), "sd": float(cs.std()),
                "abs_p999": float(np.quantile(np.abs(cs), 0.999)),
                "note": "cos(random unit vector, a released direction) in R^5120"}
    print(f"random-cos baseline: sd={baseline['sd']:.5f}")

    results = {"provenance": provenance(), "random_cosine_baseline": baseline,
               "directions": {k: {"layer": v["layer"], "norm": v["norm"]}
                              for k, v in dirs.items()},
               "trajectories": {}}

    for tname, tpath in TRAJECTORIES.items():
        if not (ROOT / tpath / "final.safetensors").exists():
            print(f"  skip {tname}: not downloaded")
            continue
        steps, B, A, Bfin, layer, cfg = load_trajectory(tpath)
        spacing = int(np.min(np.diff(steps[steps % 5 == 0]))) if len(steps) > 2 else 5
        print(f"\n[{tname}] layer {layer}, {len(steps)} ckpts, steps "
              f"{steps[0]}..{steps[-1]}, alpha={cfg['lora_alpha']}")

        norms = np.linalg.norm(B, axis=1)
        per_step = []
        for i, s in enumerate(steps):
            row = {"step": int(s), "B_norm": float(norms[i]),
                   "A_norm": float(np.linalg.norm(A[i]))}
            c, a = angle_deg(B[i], Bfin)
            row["cos_to_B_final"] = c
            row["angle_to_B_final_deg"] = a
            for dn, dd in dirs.items():
                c_raw, a_raw = angle_deg(B[i], dd["v"])
                bn = norms[i]
                Bh = B[i] / bn if bn > 0 else B[i]
                c_nrm, a_nrm = angle_deg(Bh, dd["v"])
                row[f"cos_raw__{dn}"] = c_raw
                row[f"angle_raw_deg__{dn}"] = a_raw
                row[f"cos_norm__{dn}"] = c_nrm
                row[f"angle_norm_deg__{dn}"] = a_nrm
            per_step.append(row)

        # The norm-confound test on the published rotation metric.
        m = steps % 5 == 0
        s5, B5 = steps[m], B[m]
        sp = int(s5[1] - s5[0])
        Bh5 = B5 / np.linalg.norm(B5, axis=1, keepdims=True)
        lcs = {}
        for k in (10, 20, 40):
            if k % sp or len(s5) < 2 * (k // sp) + 3:
                continue
            raw = local_cos(B5, s5, k, sp, args.threshold)
            # On normalised vectors the natural threshold is the same fraction of
            # the unit sphere that 0.0035 is of ||B_final||.
            th_n = args.threshold / float(np.linalg.norm(Bfin))
            nrm = local_cos(Bh5, s5, k, sp, th_n)
            lcs[str(k)] = {
                "raw": raw, "normalised": nrm,
                "threshold_raw": args.threshold, "threshold_normalised": th_n,
                "raw_peak": max(raw, key=lambda x: x[1]) if raw else None,
                "normalised_peak": max(nrm, key=lambda x: x[1]) if nrm else None,
                "raw_median": float(np.median([x[1] for x in raw])) if raw else None,
                "normalised_median": float(np.median([x[1] for x in nrm])) if nrm else None,
            }
            if raw and nrm:
                print(f"   local-cos k={k:3d}: RAW peak step {lcs[str(k)]['raw_peak'][0]:4d} "
                      f"({lcs[str(k)]['raw_peak'][1]:+.4f}, med {lcs[str(k)]['raw_median']:+.4f})"
                      f" | NORM peak step {lcs[str(k)]['normalised_peak'][0]:4d} "
                      f"({lcs[str(k)]['normalised_peak'][1]:+.4f}, med "
                      f"{lcs[str(k)]['normalised_median']:+.4f})")

        # Headline angle movement per direction.
        summ = {}
        valid = [r for r in per_step if r["B_norm"] > 0]
        for dn in dirs:
            a0 = valid[0][f"angle_norm_deg__{dn}"]
            a1 = valid[-1][f"angle_norm_deg__{dn}"]
            summ[dn] = {"angle_first_deg": a0, "angle_final_deg": a1,
                        "total_change_deg": a0 - a1,
                        "cos_final": valid[-1][f"cos_norm__{dn}"],
                        "sigma_vs_random": valid[-1][f"cos_norm__{dn}"] / baseline["sd"]}
            print(f"   angle to {dn:15s}: {a0:6.2f}deg -> {a1:6.2f}deg "
                  f"(moved {a0 - a1:+.2f}deg, final cos {summ[dn]['cos_final']:+.4f} "
                  f"= {summ[dn]['sigma_vs_random']:.1f} sigma)")

        # Is the angle change localised at the rotation? Compare the mean rate of
        # angle change in a window around the local-cos peak with the rate outside.
        knee = {}
        peak_step = None
        for k in ("20", "40"):
            if k in lcs and lcs[k]["raw_peak"]:
                peak_step = lcs[k]["raw_peak"][0]
                break
        if peak_step is not None:
            for dn in dirs:
                st_v = np.array([r["step"] for r in valid])
                an_v = np.array([r[f"angle_norm_deg__{dn}"] for r in valid])
                rate = np.gradient(an_v, st_v)  # deg per training step
                win = np.abs(st_v - peak_step) <= 50
                inw = float(np.mean(rate[win])) if win.any() else float("nan")
                outw = float(np.mean(rate[~win])) if (~win).any() else float("nan")
                knee[dn] = {
                    "peak_step": int(peak_step),
                    "mean_rate_deg_per_step_within_50_of_peak": inw,
                    "mean_rate_deg_per_step_elsewhere": outw,
                    "ratio": (inw / outw) if outw not in (0.0, float("nan")) else None,
                    "max_abs_rate_deg_per_step": float(np.max(np.abs(rate))),
                    "step_of_max_abs_rate": int(st_v[int(np.argmax(np.abs(rate)))]),
                }
            dn0 = "gen_medical" if "gen_medical" in knee else next(iter(knee))
            print(f"   angle-rate near rotation (step {peak_step}) vs elsewhere, {dn0}: "
                  f"{knee[dn0]['mean_rate_deg_per_step_within_50_of_peak']:+.5f} vs "
                  f"{knee[dn0]['mean_rate_deg_per_step_elsewhere']:+.5f} deg/step; "
                  f"steepest angle change at step {knee[dn0]['step_of_max_abs_rate']}")

        results["trajectories"][tname] = {
            "angle_rate_vs_rotation": knee,
            "path": tpath, "layer": layer, "n_checkpoints": len(steps),
            "step_first": int(steps[0]), "step_last": int(steps[-1]),
            "adapter_config": {k: cfg.get(k) for k in
                               ("r", "lora_alpha", "use_rslora", "target_modules",
                                "layers_to_transform", "base_model_name_or_path")},
            "B_final_norm": float(np.linalg.norm(Bfin)),
            "summary_by_direction": summ,
            "local_cosine_similarity": lcs,
            "per_step": per_step,
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
