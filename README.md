# Nbody_bias

Full-shape redshift-space power spectrum model and cobaya likelihood for fitting to galaxy power spectrum multipoles from
N-body mocks. The model is one-loop velocileptors LPT (or EPT) $P_0, P_2, P_4$ from neural-network emulators,
with AP distortions, optional redshift-error damping, analytic marginalization of the counterterms, shot noise and
systematics templates, and the survey window of the data.

Start with **`notebooks/tutorial.ipynb`**. It computes and plots model spectra, loads the data, evaluates the
likelihood, and fits the LRG1 biases of the Abacus DR2 cutsky mocks at fixed cosmology, explaining each step along the way.

## Layout

```
nbody_bias/
  pkCodes_jax_emu.py   compute_pkxi_emu: emulator wrapper -> P_ell tables of the 19 bias monomials (AP, damping)
  fs_likelihood.py     cobaya components: pk_theory_zs (Theory) and JointLikelihood (Likelihood)
emu/                   trained emulators loaded by pkCodes_jax_emu (P_gm, linear P_mm/P_cb); the velocileptors P_ell
                       and ACE emulators come with the jaxeffort and jaxace packages
configs/
  params_linear_LRG1.yaml        Gaussian priors of LRG1's analytically marginalized parameters
  fit_LRG1_abacus_cutsky.yaml    MCMC of LCDM + LRG1 biases (baseline priors and scale cuts)
notebooks/tutorial.ipynb
data/                  the h5 likelihood package (not in git, see below)
setup_env.sh           builds the conda environment
```

## Data

The config and tutorial read `data/mock_abacus_cutsky_altmtl_phase_mean_qso_with_redshift_errors.h5` (241 MB).
It is DESI-restricted and too large for GitHub, so it is gitignored. On NERSC, members of the `desi` group can copy it:

```bash
mkdir -p data && cp /pscratch/sd/g/gfarren/desi_y3_x_cmb_data/lklh_data_packages/mock_abacus_cutsky_altmtl_phase_mean_qso_with_redshift_errors.h5 data/
```

## Environment

To build the environment, run

```bash
bash setup_env.sh            # creates conda env `cobaya_jax` (pass a name to override, --force to rebuild)
bash setup_env.sh --prefix /global/common/software/<project>/envs/cobaya_jax   # install outside $HOME
```

The environment needs a few GB. On NERSC the 40 GB `$HOME` quota fills up quickly, so `--prefix` is recommended there.

It installs pinned versions of everything below, registers the Jupyter kernel and runs an import smoke test.
The resulting `cobaya_jax` conda environment has everything: cobaya, jax, `jaxeffort`, `jaxace`, `jaxmapse`,
`interpax`, `classy`, `h5py` and `desi_y3_cmb_data` (the reader for the h5 likelihood packages, from
`desi_y3_C3-CMBx_lklh_theory/data_reader`). Jupyter kernel: `cobaya_jax`.

On Perlmutter, set `MPICH_GPU_SUPPORT_ENABLED=0` before cobaya is imported. Otherwise Cray MPI aborts on a node
without a CUDA-enabled jaxlib.

## Running

```bash
cd Nbody_bias
export MPICH_GPU_SUPPORT_ENABLED=0
PYTHONPATH=. cobaya-run configs/fit_LRG1_abacus_cutsky.yaml          # MCMC, cosmology free
```

In Python, add the repo root to `sys.path` and refer to the classes as `nbody_bias.fs_likelihood.pk_theory_zs` and
`nbody_bias.fs_likelihood.JointLikelihood` (see the tutorial). Redshifts are read from the data package
(`auxiliary/<sample>/zeff`); don't set `zfids` unless the data were made at other redshifts.

