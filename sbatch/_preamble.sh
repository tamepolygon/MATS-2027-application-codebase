# Sourced by every .sbatch. Sets the offline, bf16, cache-only environment.
set -euo pipefail

# --- EDIT THESE THREE IF YOUR PATHS DIFFER -----------------------------------
export EM_CACHE="${EM_CACHE:-$HOME/em_cache}"
EM_ENV="${EM_ENV:-$HOME/envs/ca}"     # conda env OR venv; both are handled
EM_REPO="${EM_REPO:-$HOME/rot}"
# -----------------------------------------------------------------------------

# Compute nodes have no internet. Fail loudly rather than hang on a DNS timeout.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$EM_CACHE/hf_home"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# GATE 1 judging threw two CUDA OOMs and recovered at a 95.1/150 GB peak - a
# fragmentation signature, not a genuine capacity limit. Expandable segments let
# the allocator grow a block instead of failing next to free-but-unusable memory.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# =============================================================================
# ENVIRONMENT ACTIVATION - conda and venv, natively, no hand-patching
# =============================================================================
# This used to assume a venv, so a conda env had to be patched in by hand after
# every rsync. It now detects which kind of environment EM_ENV is and activates
# it the right way.
#
# HOW THEY ARE TOLD APART: a conda env has `conda-meta/history`; a venv has
# `bin/activate` and no conda-meta. The conda test comes first because some
# conda envs also ship an unrelated `bin/activate`.
#
# `set -u` MATTERS HERE. conda's own shell functions and many `activate.d`
# scripts reference unset variables (PS1 being the classic), so activation is
# wrapped in `set +u` and `set -e` is relaxed around it. Both are restored
# immediately afterwards, and the result is verified rather than assumed.
# =============================================================================

em_find_conda_sh() {
  # Print the path to conda.sh, or nothing. Ordered cheapest-and-most-specific
  # first. Note the env being at $HOME/envs/<name> rather than under the conda
  # installation is normal and is why derivation from EM_ENV is only one of
  # several candidates, not the whole strategy.
  local base cands=() c
  [ -n "${CONDA_EXE:-}" ] && cands+=("$(dirname "$(dirname "$CONDA_EXE")")")
  if command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null || true)"
    [ -n "$base" ] && cands+=("$base")
    cands+=("$(dirname "$(dirname "$(command -v conda)")")")
  fi
  # <base>/envs/<name> layout
  cands+=("$(dirname "$(dirname "$EM_ENV")")")
  cands+=("$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"
          "$HOME/mambaforge" "$HOME/micromamba" "/opt/conda"
          "/usr/share/miniconda" "/usr/local/miniconda3")
  for c in "${cands[@]}"; do
    [ -n "$c" ] || continue
    if [ -f "$c/etc/profile.d/conda.sh" ]; then
      printf '%s\n' "$c/etc/profile.d/conda.sh"
      return 0
    fi
  done
  return 0
}

EM_ENV_KIND="none"
if [ -f "$EM_ENV/conda-meta/history" ]; then
  EM_ENV_KIND="conda"
  EM_CONDA_SH="$(em_find_conda_sh)"
  set +u
  set +e
  if [ -n "$EM_CONDA_SH" ]; then
    # shellcheck disable=SC1090
    . "$EM_CONDA_SH"
    conda activate "$EM_ENV"          # activating by full path is supported
    EM_ACTIVATE_RC=$?
  else
    EM_ACTIVATE_RC=127
  fi
  set -e
  set -u

  if [ "${EM_ACTIVATE_RC:-1}" -ne 0 ]; then
    # Degraded but usually sufficient: put the env's bin first. What this misses
    # is anything the env's activate.d hooks would export (CUDA paths, for
    # instance), so it is a warning, not a silent success.
    if [ -x "$EM_ENV/bin/python" ]; then
      echo "WARNING: could not run 'conda activate' (no conda.sh found, or it" >&2
      echo "         failed). Falling back to putting $EM_ENV/bin on PATH." >&2
      echo "         activate.d hooks will NOT have run. If a job fails on a" >&2
      echo "         missing CUDA or MKL path, this is the first thing to look at." >&2
      export PATH="$EM_ENV/bin:$PATH"
      export CONDA_PREFIX="$EM_ENV"
    else
      echo "FATAL: $EM_ENV looks like a conda env but has no bin/python." >&2
      exit 1
    fi
  fi

elif [ -f "$EM_ENV/bin/activate" ]; then
  EM_ENV_KIND="venv"
  set +u
  # shellcheck disable=SC1091
  . "$EM_ENV/bin/activate"
  set -u

else
  echo "FATAL: $EM_ENV is neither a conda env nor a venv." >&2
  echo "  looked for: $EM_ENV/conda-meta/history   (conda)" >&2
  echo "              $EM_ENV/bin/activate         (venv)" >&2
  echo "  Set EM_ENV to the environment you want, e.g." >&2
  echo "    sbatch --export=ALL,EM_ENV=\$HOME/envs/ca sbatch/<job>.sbatch" >&2
  exit 1
fi

# VERIFY rather than assume. This one check catches every failure mode above -
# a conda hook that silently did nothing, a venv that activated the wrong
# prefix, a PATH fallback that was shadowed - because it asks the only question
# that matters: is the python we are about to run the one inside EM_ENV?
EM_PY="$(command -v python || true)"
case "$EM_PY" in
  "$EM_ENV"/*) : ;;
  *)
    echo "FATAL: environment activation did not take." >&2
    echo "  EM_ENV        : $EM_ENV  (detected as: $EM_ENV_KIND)" >&2
    echo "  python in PATH: ${EM_PY:-<none>}" >&2
    echo "  expected it to live under $EM_ENV/bin" >&2
    exit 1
    ;;
esac

cd "$EM_REPO"

echo "=================================================================="
echo "job      : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo "node     : $(hostname)"
echo "started  : $(date -Is 2>/dev/null || date)"
echo "requeues : ${SLURM_RESTART_COUNT:-0}"
echo "EM_CACHE : $EM_CACHE"
echo "EM_ENV   : $EM_ENV  [$EM_ENV_KIND]"
echo "EM_REPO  : $EM_REPO"
echo "python   : $EM_PY"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=================================================================="
