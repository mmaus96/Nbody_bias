"""Full-shape P_ell(k) likelihood and theory for cobaya.

P(k)-only version of C3PO's likelihoods/FSBAO_likelihood_gc_PkPcbemu_am_LPTEPT_v3.py:
the full-shape parts of JointLikelihood and pk_theory_zs are kept as they are. Removed:
  - BAO (Xiell_theory_zs, BAO_linpk_theory_zs and the xi_ell branch of JointLikelihood),
    which needed velocileptors, cosmoprimo and the spherical-Bessel transforms;
  - the old rotation templates (template_fn / photo_tmp; s0, s2, ph linear parameters).
The fiducial cosmology (CLASS sigma8(z), jaxace H(z) and chi(z)) is set up by
fiducial_cosmology, the same numbers as C3PO's compute_PkXi_tables_direct.direct_fit_theory.

Two cobaya components:
  pk_theory_zs    Theory: velocileptors LPT/EPT P_ell(k) emulator tables (one per redshift) with
                  AP and redshift-error damping applied, and zPars = [sigma8(z)/sigma8_fid(z), f(z),
                  qpar, qperp]. Provides the derived parameters sigma8 and omegam.
  JointLikelihood Likelihood: combines the tables with the nonlinear biases, convolves with the
                  window of the h5 likelihood package and analytically marginalizes the linear
                  parameters (counterterms, shot noise, AUW template amplitude).
"""
import numpy as np
import jax
import jax.numpy as jnp
import yaml

from cobaya.theory import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.linalg import block_diag
from desi_y3_cmb_data import LikelihoodPackage, ProductRequest, ScaleCuts  # pip: desi_y3_C3-CMBx_lklh_theory/data_reader

from .pkCodes_jax_emu import compute_pkxi_emu


def read_zeffs(path, names):
    """z_eff of each sample from an h5 likelihood package (auxiliary/<sample>/zeff), in the order of names."""
    import h5py
    with h5py.File(path, 'r') as f:
        missing = [n for n in names if f'auxiliary/{n}/zeff' not in f]
        if missing:
            raise ValueError(f'{path} has no auxiliary zeff for {missing}; pass the redshifts explicitly')
        return [float(f[f'auxiliary/{n}/zeff'][()]) for n in names]


def fiducial_cosmology(zs):
    """Fiducial (Planck 2018) cosmology used for the AP parameters and the sigma8 scaling of the biases.

    Returns (fid_dists, rd_fid, sig8_fids): fid_dists[i] = (100 E(z_i), chi(z_i) [Mpc/h]) from jaxace,
    rd_fid = r_drag [Mpc/h] and sig8_fids[i] = sigma8(z_i) from CLASS.
    """
    import jaxace
    from classy import Class
    w0, wa, omega_b, omega_cdm, h, logA, ns = -1., 0.0, 0.02237, 0.12, 0.6736, np.log(1e10 * 2.0830e-9), 0.9649
    omega_nu = 0.0006442
    fid_class = Class()
    fid_class.set({'output': 'mPk', 'P_k_max_h/Mpc': 20., 'z_pk': '0.0,10', 'A_s': np.exp(logA) * 1e-10, 'n_s': ns,
                   'h': h, 'N_ur': 2.0328, 'N_ncdm': 1, 'omega_ncdm': omega_nu, 'tau_reio': 0.0568,
                   'omega_b': omega_b, 'omega_cdm': omega_cdm, 'Omega_Lambda': 0., 'w0_fld': w0, 'wa_fld': wa})
    fid_class.compute()
    # exact jaxace background, the same code get_zpars uses for the sampled cosmology
    fid_cosmo = jaxace.w0waCDMCosmology(ln10As=logA, ns=ns, h=h, omega_b=omega_b, omega_c=omega_cdm,
                                        m_nu=omega_nu / 0.0106, w0=w0, wa=wa)
    fid_dists = jnp.array([[100.0 * fid_cosmo.E_z(z), fid_cosmo.r_z(z) * h] for z in zs])
    rd_fid = float(fid_class.rs_drag() * h)
    sig8_fids = [fid_class.sigma(8. / h, z) for z in zs]
    return fid_dists, rd_fid, sig8_fids


class JointLikelihood(Likelihood):

    # Redshifts of fs_sample_names. Omitted -> each sample's z_eff from the h5 package (basedir + datfn);
    # passed (e.g. synthetic data made at these redshifts) -> used as given.
    zfids: list | None = None

    basedir: str
    datfn: str
    fs_sample_names: list

    # optimize: turn this on when optimizing/running the minimizer so that the Jacobian factor isn't included
    # include_priors: this decides whether the marginalized parameter priors are included (should be yes)
    linear_param_dict_fn: str | list
    optimize: bool
    include_priors: bool

    fs_kmins: list
    fs_qmins: list
    fs_mmaxs: list
    fs_qmaxs: list
    fs_hmaxs: list
    delta_k: float
    hexa: bool

    # Angular-upweighting (AUW) additive template: [mean, std] of the Gaussian prior on its amplitude
    # auw_<sample>, an analytically marginalized linear parameter multiplying the package's
    # 'angular_upweighting_data_difference' p3d template. Empty -> no AUW template.
    auw_prior: list = []

    cov_fac: list
    invcov_fac: list
    jeff: bool
    bsig8: float  # power of sigma8(z)/sigma8_fid(z) in the bias parameterization
    ct_reparam: bool
    AP_scale: bool

    def initialize(self):
        """Sets up the class."""
        if self.zfids is None:
            self.zfids = read_zeffs(self.basedir + self.datfn, self.fs_sample_names)
            print(f'zfids from {self.datfn}: {dict(zip(self.fs_sample_names, self.zfids))}')

        # Load the linear parameters of the theory model theta_a such that
        # P_th = P_{th,nl} + theta_a P^a for some templates P^a we will compute.
        # linear_param_dict_fn may be one file or a list of files (merged in order); only the parameters
        # of the samples fit here are kept.
        fns = self.linear_param_dict_fn if isinstance(self.linear_param_dict_fn, list) else [self.linear_param_dict_fn]
        self.linear_param_dict = {}
        for lfn in fns:
            self.linear_param_dict.update(yaml.load(open(lfn), Loader=yaml.SafeLoader))
        for key in list(self.linear_param_dict.keys()):
            if key.rpartition('_')[2] not in self.fs_sample_names:
                print(f'Removing {key} (not a parameter of the samples fit here)')
                self.linear_param_dict.pop(key)

        # Places to store theory curves
        self.pconv = {}

        try:
            self.fs_qmins
        except:
            self.fs_qmins = self.fs_kmins

        self.loadData()
        print('Loaded data!')

        # AUW template amplitudes are linear parameters with prior N(auw_prior); added after loadData,
        # which reads the templates from the package.
        if len(self.auw_prior) > 0:
            for name in self.fs_sample_names:
                self.linear_param_dict[f'auw_{name}'] = {'mean': float(self.auw_prior[0]), 'std': float(self.auw_prior[1])}
                print(f'Adding auw_{name} ~ N({self.auw_prior[0]}, {self.auw_prior[1]})')
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.Nlin = len(self.linear_param_dict)

    def get_requirements(self):
        req = {'taylor_pk_ell_mod': None,
               'zPars': None,
               'ns': None,
               'H0': None,
               'sigma8': None,
               'omega_b': None,
               'omega_cdm': None,
               'omegam': None,
               'logA': None}
        for fs_sample_name in self.fs_sample_names:
            req_bias = {'bsig8_' + fs_sample_name: None,
                        'b2sig8_' + fs_sample_name: None,
                        'bssig8_' + fs_sample_name: None,
                        'b3sig8_' + fs_sample_name: None}
            req = {**req, **req_bias}
        return req

    def full_predict(self, thetas=None):
        thy_obs = []
        if thetas is None:
            thetas = self.linear_param_means
        for zfid, fs_sample_name in zip(self.zfids, self.fs_sample_names):
            fs_thy = self.fs_predict(fs_sample_name, zfid, thetas=thetas)
            fs_obs = self.fs_observe(fs_thy, fs_sample_name, thetas=thetas)
            thy_obs = np.concatenate((thy_obs, fs_obs))
        return thy_obs

    def logp(self, **params_values):
        """Return a log-likelihood."""

        # Compute the theory prediction with lin. params. at prior mean
        thy_obs_0 = self.full_predict()
        self.Delta = self.dd - thy_obs_0

        # Now compute template
        self.templates = []
        for param in self.linear_param_dict.keys():
            thetas = self.linear_param_means.copy()
            thetas[param] += 1.0
            self.templates += [self.full_predict(thetas=thetas) - thy_obs_0]

        self.templates = np.array(self.templates)

        # Make dot products
        self.Va = np.dot(np.dot(self.templates, self.cinv), self.Delta)
        self.Lab = np.dot(np.dot(self.templates, self.cinv), self.templates.T) + self.include_priors * np.diag(1. / self.linear_param_stds**2)
        self.Lab_inv = np.linalg.inv(self.Lab)

        # Compute the modified chi2
        lnL = -0.5 * np.dot(self.Delta, np.dot(self.cinv, self.Delta))  # this is the "bare" lnL
        lnL += 0.5 * np.dot(self.Va, np.dot(self.Lab_inv, self.Va))  # improvement in chi2 due to changing linear params
        self.detFish = 0.5 * np.log(np.linalg.det(self.Lab))
        if not self.optimize:
            if self.jeff:
                lnL += 0.5 * self.Nlin * np.log(2 * np.pi)
            else:
                lnL += - 0.5 * np.log(np.linalg.det(self.Lab)) + 0.5 * self.Nlin * np.log(2 * np.pi)  # volume factor from the determinant

        return lnL

    def get_best_fit(self):
        try:
            self.p0_nl = self.dd - self.Delta
            self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
            self.p0_lin = np.einsum('i,il', self.bf_thetas, self.templates)
            return self.p0_nl + self.p0_lin
        except:
            print("Make sure to first compute the posterior.")

    def loadData(self):
        """
        Loads the P(k) data vector, window and inverse covariance of each sample from the h5
        likelihood package (desi_y3_cmb_data LikelihoodPackage/ProductRequest/ScaleCuts).
        Per-sample blocks are assumed independent and combined via block_diag.
        """
        fs_sample_names = self.fs_sample_names
        fs_kmins = self.fs_kmins
        fs_mmaxs, fs_qmaxs, fs_hmaxs = self.fs_mmaxs, self.fs_qmaxs, self.fs_hmaxs

        n = len(fs_sample_names)
        assert len(fs_kmins) == n, f"fs_kmins has {len(fs_kmins)} entries, expected {n}"
        for lst_name, lst in [('fs_mmaxs', fs_mmaxs), ('fs_qmaxs', fs_qmaxs), ('fs_hmaxs', fs_hmaxs)]:
            assert len(lst) == n, f"{lst_name} has {len(lst)} entries, expected {n}"

        multipoles = (0, 2, 4) if self.hexa else (0, 2)

        self.dd = []
        self.matWs = {}
        self.w_kin = {}
        self.auw_templates = {}

        cinv_blocks = []
        for ss, fs_sample_name in enumerate(fs_sample_names):
            requests = [ProductRequest(sample=fs_sample_name, observable="p3d")]
            k_min = {f'{fs_sample_name}/P0': fs_kmins[ss], f'{fs_sample_name}/P2': self.fs_qmins[ss]}
            k_max = {f'{fs_sample_name}/P0': fs_mmaxs[ss], f'{fs_sample_name}/P2': fs_qmaxs[ss]}
            if self.hexa:
                k_min[f'{fs_sample_name}/P4'] = fs_kmins[ss]
                k_max[f'{fs_sample_name}/P4'] = fs_hmaxs[ss]

            cuts = ScaleCuts(k_min=k_min, k_max=k_max, delta_k=self.delta_k, multipoles=multipoles)

            with LikelihoodPackage(self.basedir + self.datfn) as pkg:
                result = pkg.build_likelihood_inputs(requests, scale_cuts=cuts)

            self.dd = np.concatenate((self.dd, result['data_vector']))

            p3d_product = next(p for p in result['products'] if p['metadata']['observable'] == 'p3d')
            self.matWs[fs_sample_name] = p3d_product['window']
            self.w_kin[fs_sample_name] = p3d_product['theory_bins']['k_eff']
            if len(self.auw_prior) > 0:
                # additive template in the cut, rebinned p3d data-vector space
                tmpl = p3d_product['additive_templates']
                names = [] if tmpl is None else list(tmpl['names'])
                assert 'angular_upweighting_data_difference' in names, \
                    f"{fs_sample_name}: auw_prior set but {self.datfn} has no AUW template (templates: {names})"
                self.auw_templates[fs_sample_name] = np.asarray(tmpl['values'])[names.index('angular_upweighting_data_difference')]
                assert self.auw_templates[fs_sample_name].shape == (self.matWs[fs_sample_name].shape[0],)
            # fs_observe feeds (P0, P2, P4) theory into the window: its columns must be exactly those
            th_ells = list(np.asarray(p3d_product['theory_bins']['multipoles']).astype(int))
            assert th_ells == [0, 2, 4] and self.matWs[fs_sample_name].shape[1] == 3 * len(self.w_kin[fs_sample_name]), \
                f"{fs_sample_name}: expected a window over theory P0,P2,P4, got multipoles {th_ells}, shape {self.matWs[fs_sample_name].shape}"

            # cinv = cov_fac * invcov_fac * inv(cov_raw): cov_fac rescales the covariance (e.g. Veff
            # forecasting), invcov_fac applies e.g. a Hartlap correction
            cinv_blocks.append(result['inverse_covariance'] * self.cov_fac[ss] * self.invcov_fac[ss])

        self.cinv = block_diag(*cinv_blocks)

    def combine_bias_terms_pkell(self, bvec, p0ktable, p2ktable, p4ktable):
        '''
        Returns p0, p2, p4, assuming AP parameters from input p{ell}ktable.
        '''
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = bvec

        bias_monomials = np.array([1, b1, b1**2,
                                   b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2, b3, b1 * b3,
                                   alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4])

        p0 = np.sum(p0ktable * bias_monomials, axis=1)
        p2 = np.sum(p2ktable * bias_monomials, axis=1)
        p4 = np.sum(p4ktable * bias_monomials, axis=1)

        return p0, p2, p4

    def fs_predict(self, fs_sample_name, zfid, thetas=None):
        """Use the PT model to compute P_ell, given biases etc."""

        pp = self.provider
        zstr = "%.2f" % (zfid)
        taylorPTs = pp.get_result('taylor_pk_ell_mod')
        if zstr not in taylorPTs:
            raise KeyError(f'{fs_sample_name}: no P(k) tables at z={zstr} (theory has {[k for k in taylorPTs if k != "PT_model"]}); '
                           'pass the same zfids to pk_theory_zs and the likelihood, or omit both to use the h5 z_eff')
        kv, p0ktable, p2ktable, p4ktable = taylorPTs[zstr]

        self.PT_model = taylorPTs['PT_model']

        zPars = pp.get_result('zPars')
        sig8_z = zPars[zstr][0]  # sig8/sig8_fid
        f_z = zPars[zstr][1]

        qpar, qperp = zPars[zstr][2], zPars[zstr][3]
        Aap = (qpar * qperp**2)**(-1)

        # We sample in Lagrangian bias parameters scaled by sigma8 (and Aap if self.AP_scale == True)
        AP_denom = np.sqrt(Aap) if self.AP_scale else 1.

        b1_L = pp.get_param('bsig8_' + fs_sample_name) / sig8_z**(self.bsig8) / AP_denom - 1.
        b2_L = pp.get_param('b2sig8_' + fs_sample_name) / (sig8_z**(2 * self.bsig8)) / AP_denom
        bs_L = pp.get_param('bssig8_' + fs_sample_name) / (sig8_z**(2 * self.bsig8)) / AP_denom
        b3_L = pp.get_param('b3sig8_' + fs_sample_name) / (sig8_z**(3 * self.bsig8)) / AP_denom

        if self.PT_model == 'EPT':  # If using Eulerian Perturbation Theory, convert Lagrangian biases to Eulerian
            b1 = b1_L + 1.
            b2 = b2_L + (8. / 21.) * b1_L
            bs = bs_L - (2. / 7.) * b1_L
            b3 = 3 * b3_L + b1_L
        else:
            b1, b2, bs, b3 = b1_L, b2_L, bs_L, b3_L

        # Instead of calling the linear parameters directly we will now analytically marginalize over them
        if thetas is None:
            thetas = self.linear_param_means
        alp0_tilde = thetas['alpha0_' + fs_sample_name]
        alp2_tilde = thetas['alpha2_' + fs_sample_name]
        sn0 = thetas['SN0_' + fs_sample_name]
        sn2 = thetas['SN2_' + fs_sample_name]
        alp4_tilde = thetas.get('alpha4_' + fs_sample_name, 0.)
        ak4 = thetas.get('ak4_' + fs_sample_name, 0.)
        ak6 = thetas.get('ak6_' + fs_sample_name, 0.)
        sn4 = thetas.get('SN4_' + fs_sample_name, 0.)

        if self.ct_reparam:
            alp0 = (1 + b1_L)**2 * alp0_tilde
            alp2 = f_z * (1 + b1_L) * (alp0_tilde + alp2_tilde)
            alp4 = f_z * (f_z * alp2_tilde + (1 + b1_L) * alp4_tilde)
            alp6 = f_z**2 * alp4_tilde
        else:
            alp0 = alp0_tilde
            alp2 = alp2_tilde
            alp4 = alp4_tilde
            alp6 = 0.

        bias = [b1, b2, bs, b3]
        cterm = [alp0, alp2, alp4, alp6]
        stoch = [sn0, sn2, sn4]
        if self.AP_scale:
            cterm = [a / Aap for a in cterm]
            stoch = [n / Aap for n in stoch]

        bvec = bias + cterm + stoch
        self.bvec = bvec

        p0, p2, p4 = self.combine_bias_terms_pkell(bvec, p0ktable, p2ktable, p4ktable)

        p0 += ak4 * kv**2 * p0ktable[:, -5]
        p2 += ak4 * kv**2 * p2ktable[:, -5]
        p4 += ak4 * kv**2 * p4ktable[:, -5]
        p0 += ak6 * kv**4 * p0ktable[:, -4]
        p2 += ak6 * kv**4 * p2ktable[:, -4]
        p4 += ak6 * kv**4 * p4ktable[:, -4]

        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv, p0 = np.append([0., ], kv), np.append([0.0, ], p0)
        p2 = np.append([0., ], p2)
        p4 = np.append([0.0, ], p4)
        tt = np.array([kv, p0, p2, p4]).T

        if np.any(np.isnan(tt)) or np.max(tt) > 1e8:
            hub = self.provider.get_param('H0') / 100.
            sig8 = self.provider.get_param('sigma8')
            ns = self.provider.get_param('ns')
            Om = self.provider.get_param('omegam')
            print("NaN's encountered in PREDICT. Parameter values are: ns={},H0={},Om={},sig8={}".format(ns, hub, Om, sig8))

        return tt

    def fs_observe(self, tt, fs_sample_name, thetas=None):
        """Apply the window function matrix to get the binned prediction."""

        kv = self.w_kin[fs_sample_name]
        # Theory P0, P2 AND P4 always go through the full window (whose columns span all three
        # theory multipoles), so the P4 leakage into the fitted P0/P2 bins is included even when
        # the hexadecapole data are not fit (hexa false only drops the P4 data rows).
        thy = np.concatenate([Spline(tt[:, 0], tt[:, ii], ext=3)(kv) for ii in (1, 2, 3)])

        if np.any(np.isnan(thy)) or np.max(thy) > 1e8:
            hub = self.provider.get_param('H0') / 100.
            sig8 = self.provider.get_param('sigma8')
            ns = self.provider.get_param('ns')
            Om = self.provider.get_param('omegam')
            print("NaN's encountered in OBSERVE. Parameter values are: ns={},H0={},Om={},sig8={}".format(ns, hub, Om, sig8))

        convolved_model = np.matmul(self.matWs[fs_sample_name], thy)

        if len(self.auw_prior) > 0:
            auw = (self.linear_param_means if thetas is None else thetas)['auw_' + fs_sample_name]
            convolved_model = convolved_model + auw * self.auw_templates[fs_sample_name]

        # Save the model:
        self.pconv[fs_sample_name] = convolved_model

        return convolved_model


# ------------------------------------------------------------------------------------
# Damped P(k) theory: pk_theory_zs applies the mu-dependent damping envelope
#   D(k,mu) = 1 / (1 + a2*(k*mu)**2 + a4*(k*mu)**4)
# controlled by the sampled/fixed cobaya params a2_<tr>, a4_<tr>, and provides the
# damped taylor_pk_ell_mod + zPars.
# ------------------------------------------------------------------------------------
class pk_theory_zs(Theory):
    """
    P_ell(k) tables of the bias monomials, one set per redshift.
    """
    # Redshifts of fs_sample_names. Omitted -> z_eff of each sample from the h5 package basedir + datfn
    # (must then match the likelihood's); passed (e.g. synthetic data) -> used as given.
    zfids: list | None = None
    basedir: str | None = None
    datfn: str | None = None
    fs_sample_names: list
    PT_model: str
    # Redshift-error PDF damping (optional): {fs_sample_name: path to an .npz with 'grid' (dv, km/s) and
    # 'pdf' (P(dv))}. Samples listed get P(k,mu) multiplied by |D_PDF|^2, D_PDF(k,mu) = int P(eps) exp(-i k mu eps),
    # eps = dv/(aH), on top of the residual damping below; samples not listed get only the residual damping.
    verr_pdf_fns: dict = {}
    # Residual redshift-error damping of P(k,mu), for every FS sample (pkCodes compute_zerr_damped_pell_tables):
    #   'lorentzian': 1/(1 + a2 (k mu)^2 + a4 (k mu)^4)
    #   'gaussian':   exp[-(k mu v_smear)^2] = |exp[-(k mu v_smear)^2/2]|^2, with v_smear = a2 [Mpc/h] (a4 unused)
    #   'none':       no residual damping
    res_damp: str = 'lorentzian'

    def initialize(self):
        """Sets up the class by loading the emulators."""

        if self.zfids is None:
            assert self.basedir and self.datfn, 'pk_theory_zs: give zfids, or basedir + datfn to read z_eff from the h5'
            self.zfids = read_zeffs(self.basedir + self.datfn, self.fs_sample_names)
            print(f'pk_theory_zs zfids from {self.datfn}: {dict(zip(self.fs_sample_names, self.zfids))}')

        self.fid_dists, self.rd_fid, self.sig8_fids = fiducial_cosmology(self.zfids)  # fid_dists: (len(zfids), 2)
        self.zfids = jnp.array(self.zfids)

        self.compute_theory = compute_pkxi_emu(zeffs=self.zfids, PT_model=self.PT_model,
                                               fid_dists=self.fid_dists, rd_fid=self.rd_fid, cosmo_model='w0wa')
        self.compute_theory_alt = compute_pkxi_emu(zeffs=self.zfids, PT_model=self.PT_model,
                                                   fid_dists=self.fid_dists, rd_fid=self.rd_fid, cosmo_model='w0wa')  # for Mnu<0

        self.evaluate_zpars_jit = jax.jit(self.compute_theory.get_zpars)
        # redshift-error PDF damping tables (one per zfid; None -> no PDF damping for that sample)
        unknown = set(self.verr_pdf_fns) - set(self.fs_sample_names)
        assert not unknown, f'verr_pdf_fns has samples not in fs_sample_names: {unknown}'
        pdf_fns = [self.verr_pdf_fns.get(name) for name in self.fs_sample_names]
        self.pdf_damp = [fn is not None for fn in pdf_fns]
        if any(self.pdf_damp):
            for thy in (self.compute_theory, self.compute_theory_alt):
                thy.set_verr_pdfs(pdf_fns)
        for name, fn in zip(self.fs_sample_names, pdf_fns):
            if fn is not None:
                print(f'pk_theory_zs: redshift-error PDF damping for {name} from {fn}')

        assert self.res_damp in ('lorentzian', 'gaussian', 'none'), f'res_damp = {self.res_damp!r}'
        if self.res_damp != 'lorentzian':
            print(f'pk_theory_zs: residual redshift-error damping = {self.res_damp}')
        self.evaluate_pells_jit = jax.jit(self.compute_theory.compute_zerr_damped_pell_tables, static_argnames=('damp_stoch', 'pdf_damp', 'res_damp'))
        self.evaluate_pcb_jit = jax.jit(self.compute_theory.compute_Pcb_lin)

        self.evaluate_zpars_jit_alt = jax.jit(self.compute_theory_alt.get_zpars)
        self.evaluate_pells_jit_alt = jax.jit(self.compute_theory_alt.compute_zerr_damped_pell_tables, static_argnames=('damp_stoch', 'pdf_damp', 'res_damp'))
        self.evaluate_pcb_jit_alt = jax.jit(self.compute_theory_alt.compute_Pcb_lin)

    def get_requirements(self):
        """What we need in order to provide P_ell."""
        req = {'ns': None,
               'omega_b': None,
               'omega_cdm': None,
               'H0': None,
               'w0': None,
               'wa': None,
               'logA': None,
               'm_ncdm': None}
        for fs_sample_name in self.fs_sample_names:
            req = {**req, 'a2_' + fs_sample_name: None, 'a4_' + fs_sample_name: None}
        return req

    def get_can_provide(self):
        return ['taylor_pk_ell_mod', 'zPars']

    def get_can_provide_params(self):
        return ['sigma8', 'omegam']

    def calculate(self, state, want_derived=True, **params_values_dict):

        def taylor(f_Mnu, f_0, Mnu, x0=0.01):
            return f_0 + ((f_Mnu - f_0) / (np.abs(Mnu) - x0)) * (Mnu - x0)

        Mnu_0 = 0.01

        pp = self.provider

        w0 = pp.get_param('w0')
        wa = pp.get_param('wa')
        ns = pp.get_param('ns')
        H0 = pp.get_param('H0')
        hub = H0 / 100.
        logA = pp.get_param('logA')
        omega_b = pp.get_param('omega_b')
        omega_cdm = pp.get_param('omega_cdm')
        Mnu = pp.get_param('m_ncdm')
        OmM = (omega_cdm + omega_b + Mnu / 93.14) / hub**2

        cosmopars = jnp.array([logA, ns, H0, omega_b, omega_cdm, np.abs(Mnu), w0, wa])

        if Mnu < Mnu_0:
            cosmopars_0 = jnp.array([logA, ns, H0, omega_b, omega_cdm, Mnu_0, w0, wa])

        ptables = {}
        ptables['PT_model'] = self.PT_model
        zPars = {}

        for ii, zfid in enumerate(self.zfids):
            zstr = "%.2f" % (zfid)

            zpars_emu, rd = self.evaluate_zpars_jit(cosmopars, ii)
            sigma8, sigma8_z, f_z, qpar, qperp = zpars_emu

            fname = self.fs_sample_names[ii]
            a2a4 = jnp.array([pp.get_param(f'a2_{fname}'), pp.get_param(f'a4_{fname}')])
            kv, p0ktable, p2ktable, p4ktable = self.evaluate_pells_jit(jnp.concatenate((cosmopars, a2a4)), ii, damp_stoch=True, pdf_damp=self.pdf_damp[ii], res_damp=self.res_damp)

            if Mnu < Mnu_0:
                zpars_emu, rd_0 = self.evaluate_zpars_jit_alt(cosmopars_0, ii)
                sigma8_0, sigma8_z_0, f_z_0, qpar_0, qperp_0 = zpars_emu
                _, p0_0, p2_0, p4_0 = self.evaluate_pells_jit_alt(jnp.concatenate((cosmopars_0, a2a4)), ii, damp_stoch=True, pdf_damp=self.pdf_damp[ii], res_damp=self.res_damp)

                p0ktable = taylor(p0ktable, p0_0, Mnu, Mnu_0)
                p2ktable = taylor(p2ktable, p2_0, Mnu, Mnu_0)
                p4ktable = taylor(p4ktable, p4_0, Mnu, Mnu_0)
                rd = taylor(rd, rd_0, Mnu, Mnu_0)
                sigma8 = taylor(sigma8, sigma8_0, Mnu, Mnu_0)
                sigma8_z = taylor(sigma8_z, sigma8_z_0, Mnu, Mnu_0)
                f_z = taylor(f_z, f_z_0, Mnu, Mnu_0)
                qpar = taylor(qpar, qpar_0, Mnu, Mnu_0)
                qperp = taylor(qperp, qperp_0, Mnu, Mnu_0)

            ptables[zstr] = (np.asarray(kv), np.asarray(p0ktable), np.asarray(p2ktable), np.asarray(p4ktable))

            if np.any(np.isnan(p0ktable.flatten())) or np.max(p0ktable.flatten()) > 1e8 or np.any(np.isnan(p2ktable.flatten())) or np.max(p2ktable.flatten()) > 1e8:
                print(f"NaN's encountered in PREDICT. Parameter values are: a2_{fname} = {pp.get_param(f'a2_{fname}')}")
            zPars[zstr] = [sigma8_z / self.sig8_fids[ii], f_z, qpar, qperp]

        state['derived'] = {'sigma8': sigma8, 'omegam': OmM}
        state['zPars'] = zPars
        state['taylor_pk_ell_mod'] = ptables
