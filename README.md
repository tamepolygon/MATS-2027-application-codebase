# There is no single misalignment direction, and forcing a rank-1 EM LoRA onto one produces no misalignment 

MATS Winter 2027 application project.

## Layout
- `RESULTS.md` — full results, findings first
- `figures/` — all figures, PDF and PNG
- `src/` — analysis and GPU code
- `sbatch/` — every cluster job, numbered in the order it ran
- `notes/` — paper notes, red-team audit, conventions, blockers
- `scripts/provenance_audit.py` — recomputes every headline number from raw artefacts (38 PASS, 0 FAIL)
- `results/*.json` — computed results

## Not included, for size
Raw generations (`results/*.jsonl`, ~25MB), direction tensors (`results/directions/*.pt`, ~500MB),
and the adapter checkpoints. All reproducible from the scripts, or available on request.

## Reproducing
See `notes/cluster_runbook.md`. Models and adapters come from
[ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM).
EOF
