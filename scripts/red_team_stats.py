#!/usr/bin/env python3
"""Fragility checks for the red-team audit. Everything here is CPU and cheap."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
out = {}

# ---------------------------------------------------------------- GATE 1 LOO
rows = [json.loads(l) for l in (ROOT/"results/gate1_responses.jsonl").read_text().splitlines() if l.strip()]
org = [r for r in rows if r.get("arm") != "base" and r.get("aligned") is not None]
base = [r for r in rows if r.get("arm") == "base" and r.get("aligned") is not None]
qs = sorted({r["question"] for r in org})
mo = np.array([np.mean([r["aligned"] for r in org if r["question"]==q]) for q in qs])
mb = np.array([np.mean([r["aligned"] for r in base if r["question"]==q]) for q in qs])
d = mb - mo
def paired(x):
    n=len(x); t = x.mean()/(x.std(ddof=1)/math.sqrt(n)); return t, x.mean()/x.std(ddof=1)
t_all, dz_all = paired(d)
print("="*80); print("GATE 1: leave-one-question-out on the paired test"); print("="*80)
print(f"all 8 questions: t(7) = {t_all:.3f}   dz = {dz_all:.3f}   shift = {d.mean():+.3f}")
print(f"\n{'dropped question':<58} {'t(6)':>7} {'dz':>7} {'p<.05?':>7}")
loo=[]
for i,q in enumerate(qs):
    x = np.delete(d, i); t, dz = paired(x)
    sig = abs(t) > 2.447          # t crit, df=6, two-sided .05
    loo.append({"dropped": q[:60], "t": t, "dz": dz, "significant": bool(sig)})
    print(f"{q[:56]:<58} {t:>7.3f} {dz:>7.3f} {'yes' if sig else '** NO **':>7}")
n_sig = sum(l["significant"] for l in loo)
print(f"\nsignificant in {n_sig}/8 leave-one-out refits; dz range "
      f"{min(l['dz'] for l in loo):.2f}-{max(l['dz'] for l in loo):.2f}")
# sign test, distribution-free
from math import comb
n_pos = int((d > 0).sum())
p_sign = sum(comb(8,k) for k in range(n_pos,9))/2**8 * 2
print(f"sign test (no normality assumption): {n_pos}/8 positive, two-sided p = {p_sign:.4f}")
out["gate1_loo"] = {"t_all": t_all, "dz_all": dz_all, "loo": loo,
                    "n_significant": n_sig, "sign_test_p": p_sign,
                    "n_positive": n_pos}

# ------------------------------------------------------------- GATE 2 CIs
def wilson(k,n,z=1.96):
    if n==0: return float('nan'),float('nan')
    p=k/n; den=1+z*z/n; c=p+z*z/(2*n)
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-h)/den,(c+h)/den
print("\n"+"="*80); print("GATE 2: intervals on 27 labels, one rater"); print("="*80)
tp,fp,fn,tn = 10,0,7,10
for name,k,n in (("precision",tp,tp+fp),("recall",tp,tp+fn),
                 ("pooled agreement",tp+tn,tp+fp+fn+tn)):
    lo,hi = wilson(k,n)
    print(f"  {name:<20} {k}/{n} = {k/n:>6.1%}   95% Wilson [{lo:.1%}, {hi:.1%}]")
    out.setdefault("gate2_ci",{})[name]={"k":k,"n":n,"point":k/n,"lo":lo,"hi":hi}
print("  NOTE precision 100% on n=10 has a lower bound of 72.2% - it is NOT")
print("  evidence of a perfect judge, only that no false positive appeared in 10.")

# ------------------------------------------- mean-diff stability diagnostics
print("\n"+"="*80); print("MEAN-DIFF: is 157 misaligned responses enough in 5120 dims?"); print("="*80)
import torch; torch.set_num_threads(1)
MD = torch.load(ROOT/"results/directions/meandiff_r1_9layer.pt", map_location="cpu", weights_only=False)
md = MD["mean_diff"].double().numpy()
mu_m = MD["mu_misaligned"].double().numpy(); mu_a = MD["mu_aligned"].double().numpy()
nm, na = MD["n_misaligned_tokens"], MD["n_aligned_tokens"]
print(f"  responses  misaligned {MD['n_misaligned_responses']}  aligned {MD['n_aligned_responses']}"
      f"   ratio 1:{MD['n_aligned_responses']/MD['n_misaligned_responses']:.1f}")
print(f"  tokens     misaligned {nm}  aligned {na}   ratio 1:{na/nm:.1f}")
def c(u,v): return float(u@v/(np.linalg.norm(u)*np.linalg.norm(v)))
adj = [c(md[i], md[i+1]) for i in range(md.shape[0]-1)]
print(f"\n  adjacent-layer cosine of the mean-diff, cos(md[i], md[i+1]):")
print(f"    median {np.median(adj):.4f}   min {min(adj):.4f}   "
      f"at layers 20-25: {[round(x,3) for x in adj[20:25]]}")
print("    A direction estimated from noise would NOT be smooth across layers.")
print("    High adjacent cosine is consistent with (not proof of) a stable estimate.")
rel = [np.linalg.norm(md[i])/np.linalg.norm(mu_a[i]) for i in range(md.shape[0])]
print(f"\n  ||mean_diff[i]|| / ||mu_aligned[i]||: median {np.median(rel):.4f}, "
      f"at 22 {rel[22]:.4f}, at 24 {rel[24]:.4f}")
print("    The difference is a ~1-4% perturbation of the mean activation.")
out["meandiff_stability"] = {
    "n_misaligned_responses": MD["n_misaligned_responses"],
    "n_aligned_responses": MD["n_aligned_responses"],
    "token_ratio": na/nm, "adjacent_cos_median": float(np.median(adj)),
    "adjacent_cos_min": float(min(adj)),
    "rel_norm_at_22": rel[22], "rel_norm_at_24": rel[24]}

(ROOT/"results"/"red_team_stats.json").write_text(json.dumps(out, indent=2, default=float))
print(f"\nwrote {ROOT/'results'/'red_team_stats.json'}")
