#!/usr/bin/env bash
# Build a conda environment equivalent to the `cobaya_jax` env this repo was developed in.
#
#   bash setup_env.sh                  # creates env "cobaya_jax"
#   bash setup_env.sh myenv            # custom name
#   bash setup_env.sh myenv --force    # remove and rebuild an existing env
#   bash setup_env.sh --prefix /path/to/env   # install at a path instead (e.g. /global/common/software/<project>/...
#                                             # on NERSC, where $HOME's 40 GB quota fills up quickly)
#
# Works on Perlmutter (uses `module load conda` and builds mpi4py against Cray MPICH) and on a generic Linux/macOS
# machine with conda or mamba on PATH. jax is installed CPU-only, as in the reference env.
set -euo pipefail

ENV_NAME=cobaya_jax
FORCE=0
PREFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1 ;;
        --prefix) PREFIX="$(mkdir -p "$(dirname "$2")" && cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) ENV_NAME="$1" ;;
    esac
    shift
done
if [[ -n "$PREFIX" ]]; then
    ENV_NAME="$(basename "$PREFIX")"; TARGET=(-p "$PREFIX"); ACTIVATE="$PREFIX"
    # keep conda's package cache out of $HOME too
    export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$(dirname "$PREFIX")/.conda_pkgs}"
else
    TARGET=(-n "$ENV_NAME"); ACTIVATE="$ENV_NAME"
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ON_NERSC=0; [[ -n "${NERSC_HOST:-}" ]] && ON_NERSC=1

# ---------------------------------------------------------------- conda
if ! command -v conda >/dev/null 2>&1; then
    if [[ $ON_NERSC == 1 ]] && command -v module >/dev/null 2>&1; then
        module load conda
    else
        echo "ERROR: conda not found. Install Miniforge (https://github.com/conda-forge/miniforge) first." >&2
        exit 1
    fi
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if { [[ -n "$PREFIX" ]] && [[ -d "$PREFIX/conda-meta" ]]; } || \
   { [[ -z "$PREFIX" ]] && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }; then
    if [[ $FORCE == 1 ]]; then
        echo ">>> Removing existing env $ACTIVATE"
        conda env remove -y "${TARGET[@]}"
    else
        echo "Env '$ACTIVATE' already exists. Activate it with 'conda activate $ACTIVATE', or rerun with --force to rebuild."
        exit 0
    fi
fi

echo ">>> Creating conda env $ACTIVATE"
conda create -y "${TARGET[@]}" -c conda-forge \
    python=3.12 fftw h5py scipy=1.15 pyyaml numba ipykernel jupyter matplotlib cython flit-core
conda activate "$ACTIVATE"
# A failure past this point leaves a partial env that a plain rerun would report as already existing.
trap 'echo; echo "ERROR: setup failed, so env $ACTIVATE is incomplete. Fix the error above, then rerun with --force." >&2' ERR
# Exported rather than passed as flags so pip's build-isolation subprocesses retry too.
export PIP_RETRIES=10 PIP_TIMEOUT=60 PIP_RESUME_RETRIES=5
PIP="python -m pip install --no-cache-dir"
$PIP --upgrade pip

# ---------------------------------------------------------------- MPI (only needed for multi-chain cobaya runs)
echo ">>> Installing mpi4py"
if [[ $ON_NERSC == 1 ]]; then
    MPI_CMD=(env MPICC="cc -shared" python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py==4.0.3)
else
    MPI_CMD=(python -m pip install --no-cache-dir mpi4py==4.0.3)
fi
"${MPI_CMD[@]}" || echo "WARNING: mpi4py failed to install; single-process cobaya runs and the tutorial still work."

# ---------------------------------------------------------------- pinned PyPI packages
echo ">>> Installing jax, cobaya, classy and friends"
$PIP \
    numpy==2.3.5 \
    jax==0.4.38 jaxlib==0.4.38 \
    flax==0.10.4 optax==0.2.5 \
    equinox==0.13.2 lineax==0.0.8 jaxtyping==0.2.38 diffrax==0.7.0 quadax==0.2.11 \
    interpax==0.3.12 \
    fetch-artifacts==0.1.0 \
    cobaya==3.6.1 getdist==1.6.1 \
    classy==3.3.4.0 \
    pyfftw==0.15.0 findiff==0.12.1

# ---------------------------------------------------------------- git packages, pinned to the commits of the reference env
echo ">>> Installing emulator and data-reader packages from GitHub"
GH=https://github.com
$PIP "jaxace @ git+$GH/CosmologicalEmulators/jaxace.git@4a5871a2e1b3b7eede5fd719989402c29a97db68"
# Build isolation fails here: its subprocess gets a truncated PyPI index page for flit-core, so flit-core comes from conda.
$PIP --no-build-isolation "velocileptors @ git+$GH/sfschen/velocileptors.git@5d0c532e0349cf7020c5493a487e92bf30cbe634"
$PIP "desi_y3_cmb_data @ git+$GH/gerrfarr/desi_y3_C3-CMBx_lklh_theory.git@9fbcbe5ede1cdb9b8e97de24f021849681b42cc3#subdirectory=data_reader"
# jaxeffort/jaxmapse pin jaxace<0.5 in their metadata but work with (and this repo needs) jaxace 0.7 -> --no-deps.
$PIP --no-deps "jaxeffort @ git+$GH/CosmologicalEmulators/jaxeffort.git@de6fa81ad12bc9dc9d93e8875df898a05b7b9ff0"
$PIP --no-deps "jaxmapse @ git+$GH/CosmologicalEmulators/jaxmapse.git@7bdc8119ad718d0e40ecd889524d12f02a8f256b"
$PIP --no-deps "aemulus_heft @ git+$GH/gerrfarr/aemulus_heft.git@91f9fc4233bec6ed7e3c9b51bc1bef0d14aa95e9"

# ---------------------------------------------------------------- Jupyter kernel
echo ">>> Registering Jupyter kernel '$ENV_NAME'"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "$ENV_NAME"

# ---------------------------------------------------------------- smoke test
echo ">>> Smoke test"
cd "$REPO"
MPICH_GPU_SUPPORT_ENABLED=0 JAX_PLATFORMS=cpu PYTHONPATH="$REPO" python - <<'EOF'
import importlib
for m in ["numpy", "jax", "cobaya", "classy", "interpax", "h5py", "velocileptors",
          "jaxace", "jaxeffort", "jaxmapse", "desi_y3_cmb_data"]:
    mod = importlib.import_module(m)
    print(f"  {m:18s} {getattr(mod, '__version__', 'ok')}")
import nbody_bias.pkCodes_jax_emu, nbody_bias.fs_likelihood
print("  nbody_bias         ok")
EOF

DATA="$REPO/data/mock_abacus_cutsky_altmtl_phase_mean_qso_with_redshift_errors.h5"
if [[ ! -f "$DATA" ]]; then
    echo
    echo "NOTE: data package not found at data/. It is DESI-restricted and not in git. On NERSC (desi group):"
    echo "  mkdir -p $REPO/data && cp /pscratch/sd/g/gfarren/desi_y3_x_cmb_data/lklh_data_packages/$(basename "$DATA") $REPO/data/"
fi

cat <<EOF

Done. To use it:
  conda activate $ACTIVATE
  export MPICH_GPU_SUPPORT_ENABLED=0        # Perlmutter only
  jupyter lab notebooks/tutorial.ipynb       # kernel: $ENV_NAME
EOF
