import os
import numpy as np
import jaxeffort
import jax
# import equinox as eqx
import jaxace
import jaxmapse
import jax.numpy as jnp
from interpax import interp1d, interp2d

# Trained emulators shipped with this repo (Pgm and linear Pmm/Pcb); the velocileptors P_ell and ACE
# emulators come with the jaxeffort / jaxace packages. Override with the NBODY_BIAS_EMU_DIR environment variable.
EMU_DIR = os.environ.get('NBODY_BIAS_EMU_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'emu'))

ngauss = 3
nell = 3
nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
nus = jnp.array(nus)
ws = jnp.array(ws)
Legs = jnp.array(
                    [
                        np.polynomial.legendre.Legendre((0,) * li * 2 + (1,))(nus)
                        for li in range(nell)
                    ]
                )

def verr_pdf_characteristic(dv, pdf, q, chunk=256):
    """
    Cosine transform D(q) = int d(dv) P(dv) cos(q dv) of a PDF tabulated on a (possibly non-uniform)
    grid dv, treating P as piecewise linear between grid points. The transform of each linear
    segment is done analytically, so there is no aliasing however large q*spacing is (the
    redshift-error PDFs extend to +-3.5e5 km/s with ~700 km/s spacing in the tails). Written with
    sinc so it is exact at q=0. Normalized so D(0) = 1. dv, q in reciprocal units (km/s, s/km).
    Using the cosine (real) part is exact for a symmetric PDF, e.g. differences of repeat
    measurements.
    """
    dv = np.asarray(dv, dtype=float); f = np.asarray(pdf, dtype=float); q = np.asarray(q, dtype=float)
    h = np.diff(dv); m = 0.5 * (dv[1:] + dv[:-1]); s = np.diff(f) / h

    def S(qq, a):  # sin(qq a)/qq, stable at qq = 0
        return a * np.sinc(qq * a / np.pi)

    out = np.empty(q.size)
    for i0 in range(0, q.size, chunk):
        qq = q[i0:i0 + chunk, None]
        seg = -2.0 * np.sum(s[None, :] * S(qq, m[None, :]) * S(qq, 0.5 * h[None, :]), axis=1)
        edge = f[-1] * S(qq[:, 0], dv[-1]) - f[0] * S(qq[:, 0], dv[0])
        out[i0:i0 + chunk] = seg + edge
    norm = -np.sum(s * m * h) + f[-1] * dv[-1] - f[0] * dv[0]   # D(q=0) = int P
    return out / norm


@jax.jit
def get_interp_in_h_units(pk_emu_phys, k_grid_phys, k_target_h, h):
    """
    pk_emu_phys: Emulator P(k) in [Mpc^3]
    k_grid_phys: Emulator k-grid in [1/Mpc]
    k_target_h: The grid you want for comparison in [h/Mpc]
    h: Hubble parameter (H0/100)
    """
    # 1. Convert emulator grid to h-units internally
    k_grid_h = k_grid_phys / h
    pk_emu_h = pk_emu_phys * (h**3)
    
    # 2. Interpolate in log-log space (most accurate for BAO/Power laws)
    ln_pk_interp = jaxace.cubic_spline_interpolation(
        jnp.log(pk_emu_h), 
        jnp.log(k_grid_h), 
        jnp.log(k_target_h)
    )
    
    return jnp.exp(ln_pk_interp)
    

class _PmmEmuCompat:
    """Load the local Pcb/Pmm emulator directory under either jaxmapse API.

    jaxmapse 0.1.1 removed ``load_pk_emulator``/``PkEmulator`` (the composite
    object with ``.linear_pmm``/``.linear_pkcb``) in favor of a per-component
    ``load_emulator(path)`` -> ``TransferFunctionEmulator`` with ``.get_Pk``.
    This wrapper tries the new API first and falls back to the old composite
    loader on ``AttributeError``, so the same ``./emu/...`` directory keeps
    working regardless of which jaxmapse version is installed. Only the
    *linear* Pmm/Pcb are used here (matching this file's existing behavior);
    the old NN "Boost" component has no equivalent in the new API and was
    already unused by ``compute_Pmm_hmcode``.
    """

    def __init__(self, path):
        try:
            self.linear_pmm = jaxmapse.load_emulator(os.path.join(path, "Pk_lin_mm"))
            self.linear_pkcb = jaxmapse.load_emulator(os.path.join(path, "Pk_lin_cb"))
        except AttributeError:
            legacy = jaxmapse.load_pk_emulator(path)
            self.linear_pmm, self.linear_pkcb = legacy.linear_pmm, legacy.linear_pkcb

    def get_linear_pmm(self, theta, z, D=None):
        return self.linear_pmm.get_Pk(theta, z, D)

    def get_linear_pkcb(self, theta, z, D=None):
        return self.linear_pkcb.get_Pk(theta, z, D)


# class compute_pkxi_emu(eqx.Module):
class compute_pkxi_emu():

    # PT_model: str = eqx.field(static=True)
    # cosmo_model: str = eqx.field(static=True)
    # rd_fid: float
    # zeffs: jnp.ndarray
    # fid_dists: jnp.ndarray
    # P0emu: any  = eqx.field(static=True)
    # P2emu: any  = eqx.field(static=True)
    # P4emu: any  = eqx.field(static=True)
    # k: jnp.ndarray
    # zpars_emu: any  = eqx.field(static=True)
    # zpars: jnp.ndarray
    # nk: float
    # ngauss: int
    # nell: int
    # nus: jnp.ndarray
    # ws: jnp.ndarray
    # Legs: jnp.ndarray
    # Pgm_emu: any  = eqx.field(static=True)
    # Pmm_emu: any  = eqx.field(static=True)
    

    def __init__(self,zeffs,PT_model,fid_dists,rd_fid,cosmo_model='w0wa'):
        self.zeffs = zeffs
        self.PT_model = PT_model
        self.cosmo_model = cosmo_model

        if self.PT_model == 'LPT':
            P0emu = jaxeffort.trained_emulators["velocileptors_lpt_mnuw0wacdm"]["0"]
            P2emu = jaxeffort.trained_emulators["velocileptors_lpt_mnuw0wacdm"]["2"]
            P4emu = jaxeffort.trained_emulators["velocileptors_lpt_mnuw0wacdm"]["4"]
            k = P0emu.P11.k_grid
            
        elif self.PT_model == 'EPT':
            P0emu = jaxeffort.trained_emulators["velocileptors_rept_mnuw0wacdm"]["0"]
            P2emu = jaxeffort.trained_emulators["velocileptors_rept_mnuw0wacdm"]["2"]
            P4emu = jaxeffort.trained_emulators["velocileptors_rept_mnuw0wacdm"]["4"]
            k = P0emu.P11.k_grid

        self.P0emu = P0emu
        self.P2emu = P2emu
        self.P4emu = P4emu
        self.k = k

        self.zpars_emu = jaxace.load_trained_emulator_from_artifact('ACE_mnuw0wacdm_ln10As_basis')
        self.zpars = jnp.zeros((len(zeffs),5)) #sigma8,sigma8_z,f_z,qpar,qperp

        # direct_theory = direct_fit_theory(zs = zeffs)
        # self.fid_class = direct_theory.fid_class
        self.fid_dists = fid_dists # jnp array of shape (len(zeffs),2)
        self.nk = len(self.k)
        self.ngauss = ngauss
        self.nell = nell
        # nus, ws = np.polynomial.legendre.leggauss(2*self.ngauss)
        self.nus = nus #jnp.array(nus)
        self.ws = ws #jnp.array(ws)
        self.Legs = Legs
        # self.Legs = jnp.array(
        #             [
        #                 np.polynomial.legendre.Legendre((0,) * li * 2 + (1,))(self.nus)
        #                 for li in range(self.nell)
        #             ]
        #         )
        
        self.rd_fid = rd_fid#self.fid_class.rs_drag()*self.fid_class.h()
        if self.cosmo_model == 'w0wa':
            self.Pgm_emu = jaxeffort.load_multipole_emulator(os.path.join(EMU_DIR, "trained_effort_velocileptors_lpt_gm_mnuw0wacdm_40000/"))
        else:
            self.Pgm_emu = jaxeffort.load_multipole_emulator(os.path.join(EMU_DIR, "trained_effort_velocileptors_lpt_gm_mnuOkcdm_50000/"))
        # self.Pmm_emu = jaxmapse.load_pk_emulator('./emu/jaxmapse_emu/')
        # self.Pmm_emu = jaxmapse.load_pk_emulator('./emu/trained_mapse_class_mnuw0wacdm_sym_ratio_pca_1em6_250000/')
        self.Pmm_emu = _PmmEmuCompat(os.path.join(EMU_DIR, 'trained_mapse_class_mnuw0wacdm_sym_ratio_pca_250000/'))

        # jit the P(k) tables the Limber code (limber_jax.limb_jax) calls every step, per instance
        # (the emulators are captured as constants). Without this they ran op-by-op on every call,
        # unlike their HEFT counterparts (pmmHEFT_jax/pgmtable_HEFT_jax/pggtable_HEFT_jax).
        self.compute_Pmm_hmcode = jax.jit(self.compute_Pmm_hmcode)
        self.compute_Pgm_EFT = jax.jit(self.compute_Pgm_EFT)
        self.compute_Pgg_from_Pell = jax.jit(self.compute_Pgg_from_Pell)

    # @jax.jit
    def evaluate_cosmo(self,thy_args):
        logA,ns,H0,omb,omc,Mnu,w0,wa = thy_args
        cosmo = jaxace.w0waCDMCosmology(
            ln10As=logA,
            ns=ns,
            h=H0/100,
            omega_b=omb,
            omega_c=omc,
            m_nu=Mnu,
            w0=w0,
            wa=wa
        )
        
        cosmo_ace = jnp.zeros((len(self.zeffs),2))
        for ii,z in enumerate(self.zeffs):
            D, f = cosmo.D_f_z(z)
            cosmo_ace = cosmo_ace.at[ii,:].set(jnp.array([D,f]))
            
        # self.cosmo_ace = cosmo_ace
        return cosmo_ace

    def evaluate_cosmo_ace(self,thy_args,z):
        logA,ns,H0,omb,omc,Mnu,w0,wa = thy_args
        cosmo = jaxace.w0waCDMCosmology(
            ln10As=logA,
            ns=ns,
            h=H0/100,
            omega_b=omb,
            omega_c=omc,
            m_nu=Mnu,
            w0=w0,
            wa=wa
        )
        
    
        D, f = cosmo.D_f_z(z)
        return D,f

    # @jax.jit
    def get_zpars(self,thy_args,zind):
        # logA,ns,H0,omb,omc,Mnu,w0,wa = thy_args
        z = self.zeffs[zind]
        H0 = thy_args[2]
        theta = jnp.concatenate((jnp.array([z]),thy_args)) #[z, logA, ns, H0, omb, omc, Mnu, w0,wa]
        pars = self.zpars_emu.run_emulator(theta) #sigma8, sigma8_z, r_drag, H_z, r_z, D_z, f_z
        sigma8, sigma8_z, r_drag, H_z, r_z, D_z, f_z = pars

        # H_z, r_z, D_z, f_z above are the ACE emulator's own regression of these
        # background/growth quantities, which has a ~1e-5-2e-4-level residual
        # against jaxace's exact (ODE-integrated) w0waCDMCosmology -- sigma8,
        # sigma8_z, r_drag have no exact jaxace alternative (they depend on the
        # matter power spectrum's shape/recombination physics, which jaxace's
        # background module doesn't model), so those three stay ACE-sourced.
        # Hz/chiz (-> qpar,qperp, AP dilation) and f_z (RSD growth rate) are
        # recomputed exactly instead, matching evaluate_cosmo_ace's convention.
        logA,ns,H0_,omb,omc,Mnu,w0,wa = thy_args
        cosmo = jaxace.w0waCDMCosmology(
            ln10As=logA, ns=ns, h=H0/100, omega_b=omb, omega_c=omc,
            m_nu=Mnu, w0=w0, wa=wa,
        )
        Hz = 100.0 * cosmo.E_z(z)  # matches fid_dists' "100 * E(z)" convention
        chiz = cosmo.r_z(z) * H0/100.  # physical Mpc -> this file's Mpc/h convention
        D_z, f_z = cosmo.D_f_z(z)

        # zstr = "%.2f" %(z)
        # zind = jnp.where(self.zeffs == z)[0][0]
        # zind = jnp.searchsorted(self.zeffs, z)
        Hzfid, chizfid = self.fid_dists[zind,:]

        qpar, qperp = Hzfid / Hz, chiz / chizfid
        zpars = jnp.array([sigma8,sigma8_z,f_z,qpar,qperp])
        # self.zpars = self.zpars.at[zind].set(zpars)
        rd = r_drag*H0/100.
        # self.rd = rd.astype(float)#float(rd)#
        rd = rd.astype(float)

        return zpars,rd
    # @jax.jit
    def get_pknu_true(self,thy_args,z):
        # zstr = "%.2f" %(z)
        # zind = jnp.where(self.zeffs == z)[0][0]
        # z = self.zeffs[zind]
        D, f = self.evaluate_cosmo_ace(thy_args,z)
        # logA,ns,H0,omb,omc,Mnu,w0,wa = thy_args
        k = self.k

        # cosmo = self.cosmo
        
        theta = jnp.concatenate((jnp.array([z]),thy_args))
        
        Pshot = self.P0emu.stoch_model(k)
        p0table_emu  =  jnp.hstack((self.P0emu.get_Pl_no_bias(theta,D), Pshot))
        Pshot = self.P2emu.stoch_model(k)
        p2table_emu = jnp.hstack((self.P2emu.get_Pl_no_bias(theta,D), Pshot))
        Pshot = self.P4emu.stoch_model(k)
        p4table_emu = jnp.hstack((self.P4emu.get_Pl_no_bias(theta,D), Pshot))

        pell_table_emu = jnp.zeros((len(k),3,19))
        pell_table_emu = pell_table_emu.at[:,0,:].set(p0table_emu)
        pell_table_emu = pell_table_emu.at[:,1,:].set(p2table_emu)
        pell_table_emu = pell_table_emu.at[:,2,:].set(p4table_emu)

        pknu_true = jnp.sum(
            self.Legs.T[:, None, None, :, None] * pell_table_emu, axis=3
        )
        # self.pknu_true = jnp.einsum("nzks->nksz", pknu_true)
        pknu_true = jnp.einsum("nzks->nksz", pknu_true)

        return pknu_true
    # @jax.jit
    def apply_AP(self,pknu_true,zpars):
        # zstr = "%.2f" %(z)
        # zind = jnp.where(self.zeffs == z)[0][0]
        # z = self.zeffs[zind]
        # zpars,rd = self.get_zpars(thy_args,zind)
        qpar,qperp = zpars[-2:]
        k = self.k
        fac = jnp.sqrt(1 + self.nus[None, :] ** 2 * ((qperp / qpar) ** 2 - 1))
        k_apfac = fac / qperp
        nu_true = jnp.tile(
            self.nus[None, :] * (qperp / qpar) / fac, (self.nk , 1, 1)
        ).T
        vol_fac = qpar * qperp**2
        ktrue = k_apfac[:, :, None] * k[None, None, :]
        nu_true = jnp.einsum("nzk->nkz", nu_true)
        ktrue = jnp.einsum("znk->nkz", ktrue)

        def calc_pknu_obs(i, j):
            return (
                interp2d(
                    nu_true[:, :, j].reshape(-1),
                    ktrue[:, :, j].reshape(-1),
                    self.nus,
                    self.k,
                    pknu_true[:, :, i, j],
                    extrap=True,
                )
                .reshape(2 * self.ngauss, self.nk)
                .T
            )

        pknu_obs = jax.vmap(
            jax.vmap(calc_pknu_obs, in_axes=(None, 0)), in_axes=(0, None)
        )(jnp.arange(19), np.arange(1)).T.squeeze()
        # print(pknu_obs.shape)
        pknu_obs_w_sct = jnp.zeros((2 * self.ngauss, self.nk, 19))
        p_ij_ell = jnp.zeros((19, 3, self.nk))
        
        pknu_obs_w_sct = pknu_obs_w_sct.at[..., :].set(pknu_obs[..., :])

        P0_table_new =  0.5 * jnp.sum(
                (self.ws * self.Legs[0])[:, None, None] * pknu_obs_w_sct, axis=0
            ) / vol_fac
        P2_table_new =  2.5 * jnp.sum(
                        (self.ws * self.Legs[1])[:, None, None] * pknu_obs_w_sct, axis=0
                    ) / vol_fac
        
        P4_table_new =  4.5 * jnp.sum(
                        (self.ws * self.Legs[2])[:, None, None] * pknu_obs_w_sct, axis=0
                    ) / vol_fac

        # self.P0_table = P0_table_new
        # self.P2_table = P2_table_new
        # self.P4_table = P4_table_new

        return P0_table_new,P2_table_new, P4_table_new
    # @jax.jit
    def compute_pell_tables(self,thy_args,zind):

        zpars,rd = self.get_zpars(thy_args,zind)
        z = self.zeffs[zind]
        
        pknu_true = self.get_pknu_true(thy_args,z)

        P0_table,P2_table, P4_table = self.apply_AP(pknu_true,zpars)

        # self.P0_table = P0_table_new
        # self.P2_table = P2_table_new
        # self.P4_table = P4_table_new

        return self.k,P0_table,P2_table,P4_table

    def set_verr_pdfs(self, pdf_files, kmax_fac=1.2, hz_fac=0.5, pts_per_period=8, nq_max=40000):
        """
        Tabulate the redshift-error damping D_PDF(q), q = k mu (1+z)/H(z) [s/km, with k in h/Mpc and
        H = 100 E(z) in km/s/(Mpc/h)], for each zeff. pdf_files: one entry per zeff, an .npz with
        'grid' (dv, km/s) and 'pdf' (P(dv), 1/(km/s)), or None for no PDF damping at that zeff.
        The q grid covers k up to kmax_fac * max(k) for any H(z) >= hz_fac * H_fid(z), finely enough to
        resolve oscillations from the PDF's full dv range (pts_per_period points per 2 pi / max|dv|).
        """
        assert len(pdf_files) == len(self.zeffs)
        zeffs = np.asarray(self.zeffs, dtype=float)
        hz_fid = np.asarray(self.fid_dists)[:, 0]
        pdfs = [None if fn is None else np.load(fn) for fn in pdf_files]
        used = [i for i, p in enumerate(pdfs) if p is not None]
        if not used:
            self.pdf_qgrid, self.pdf_Dtab = None, None
            return
        qmax = max(kmax_fac * float(np.max(self.k)) * (1 + zeffs[i]) / (hz_fac * hz_fid[i]) for i in used)
        dvmax = max(float(np.max(np.abs(pdfs[i]['grid']))) for i in used)
        nq = int(min(nq_max, np.ceil(qmax / (2 * np.pi / dvmax / pts_per_period)) + 1))
        qgrid = np.linspace(0.0, qmax, nq)
        tab = np.ones((len(pdf_files), nq))
        for i in used:
            tab[i] = verr_pdf_characteristic(pdfs[i]['grid'], pdfs[i]['pdf'], qgrid)
        self.pdf_files = list(pdf_files)
        self.pdf_qgrid = jnp.asarray(qgrid)
        self.pdf_Dtab = jnp.asarray(tab)

    def compute_zerr_damped_pell_tables(self,thy_args,zind,damp_stoch=False,pdf_damp=False,res_damp='lorentzian'):
        """
        P_ell tables with redshift-error damping of P(k,mu), in true coordinates before AP.
        thy_args = cosmology + (a2, a4). All damping factors multiply P(k,mu), i.e. are |D|^2-level:

          res_damp='lorentzian': 1 / (1 + a2 (k mu)^2 + a4 (k mu)^4)
                                 (the usual power-spectrum-level Lorentzian; not squared)
          res_damp='gaussian':   |D_G|^2 = exp[-(k mu v_smear)^2], D_G = exp[-(k mu v_smear)^2 / 2] the
                                 single-galaxy damping of a Gaussian displacement with dispersion
                                 v_smear = a2 [Mpc/h] (a4 unused -- keep it fixed to 0)
          res_damp='none':       1
          pdf_damp=True:         additionally |D_PDF|^2, D_PDF(k,mu) = int d(eps) P(eps) exp(-i k mu eps),
                                 eps = dv (1+z)/H(z), from the tables of set_verr_pdfs (cosine transform
                                 at q = k mu (1+z)/H(z); H(z) of the sampled cosmology = H_fid/qpar)

        damp_stoch=True also damps the stochastic terms. damp_stoch, pdf_damp and res_damp must be
        static under jit.
        """
        a2,a4 = thy_args[-2:]

        zpars,rd = self.get_zpars(thy_args[:-2],zind)
        z = self.zeffs[zind]

        pknu_true = self.get_pknu_true(thy_args[:-2],z)

        knu = self.nus[:, None] * self.k[None, :]          # (6, 59)
        if res_damp == 'lorentzian':
            D = 1 / (1 + a2 * knu**2 + a4 * knu**4)
        elif res_damp == 'gaussian':
            D = jnp.exp(-(knu * a2)**2)
        elif res_damp == 'none':
            D = jnp.ones_like(knu)
        else:
            raise ValueError(f"res_damp must be 'lorentzian', 'gaussian' or 'none', got {res_damp!r}")
        if pdf_damp:
            Hz = self.fid_dists[zind, 0] / zpars[3]          # 100 E(z) of the sampled cosmology
            q = jnp.abs(knu) * (1 + z) / Hz
            Dpdf = jnp.interp(q.ravel(), self.pdf_qgrid, self.pdf_Dtab[zind]).reshape(q.shape)
            D = D * Dpdf**2
        D = D[:, :, None, None]

        if damp_stoch:
            pknu_damp = pknu_true.at[:, :, :, :].mul(D) #Apply damping to all 19 monomials, including stochastic
        else:
            pknu_damp = pknu_true.at[:, :, :16, :].mul(D)

        P0_table,P2_table, P4_table = self.apply_AP(pknu_damp,zpars)

        return self.k,P0_table,P2_table,P4_table

    def compute_damped_pdf_pell_tables(self,thy_args,zind,damp_stoch=False,pdf_damp=False):
        """Lorentzian 1/(1 + a2 (k mu)^2 + a4 (k mu)^4) times, if pdf_damp, |D_PDF|^2 -- see
        compute_zerr_damped_pell_tables (this is its res_damp='lorentzian' case)."""
        return self.compute_zerr_damped_pell_tables(thy_args, zind, damp_stoch=damp_stoch, pdf_damp=pdf_damp,
                                                    res_damp='lorentzian')

    def compute_damped_pell_tables(self,thy_args,zind,damp_stoch=False):
        
        a2,a4 = thy_args[-2:]
        
        zpars,rd = self.get_zpars(thy_args[:-2],zind)
        z = self.zeffs[zind]
        
        pknu_true = self.get_pknu_true(thy_args[:-2],z)

        # Apply damping:
        knu = self.nus[:, None] * self.k[None, :]          # (6, 59)
        knu2 = knu**2
        knu4 = knu**4
        
        D = 1 / (1 + a2 * knu2 + a4 * knu4)     # (6, 59)
        D = D[:, :, None, None] 

        if damp_stoch:
            pknu_damp = pknu_true.at[:, :, :, :].mul(D) #Apply damping to all 19 monomials, including stochastic
        else:
            pknu_damp = pknu_true.at[:, :, :16, :].mul(D)

        P0_table,P2_table, P4_table = self.apply_AP(pknu_damp,zpars)

        # self.P0_table = P0_table_new
        # self.P2_table = P2_table_new
        # self.P4_table = P4_table_new

        return self.k,P0_table,P2_table,P4_table
        
        
    # @jax.jit       
    def compute_Pgg_from_Pell(self,thy_args,z):

        theta = jnp.concatenate((jnp.atleast_1d(z),thy_args)) #z, logA,ns,H0,omega_b,omega_cdm, Mnu,w0,wa
        # cosmo = self.cosmo
        # D, f = cosmo.D_f_z(z)
        # zind (previously: jnp.where(self.zeffs == z)[0][0]) was unused dead
        # code, and jnp.where/nonzero without a static size= is not
        # jax.vmap-compatible -- removed so this is vmappable over z (see
        # limber_jax.py's evaluate_all/computeCggCkg_all batching).
        D, f = self.evaluate_cosmo_ace(thy_args,z)
        
        p0_tab = self.P0emu.get_Pl_no_bias(theta,D)
        p2_tab = self.P2emu.get_Pl_no_bias(theta,D)
        p4_tab = self.P4emu.get_Pl_no_bias(theta,D)

        pgg_emu_tab = p0_tab - 0.5* p2_tab + (3./8.)*p4_tab
        pgg_emu_tab = pgg_emu_tab[:,:-3]
        pgg_ct_emu = -0.5*pgg_emu_tab[:,-1]

        res = jnp.zeros((len(self.k),jnp.shape(pgg_emu_tab[:,:-1])[1] + 2))
        res = res.at[:,0].set(self.k)
        res = res.at[:,1:-1].set(pgg_emu_tab[:,:-1])
        res = res.at[:,-1].set(pgg_ct_emu)

        return res
    # @jax.jit
    def compute_Pgm_EFT(self,thy_args,z):

        theta = jnp.concatenate((jnp.atleast_1d(z),thy_args))#[z] + thy_args #z, logA,ns,H0,omega_b,omega_cdm, Mnu,w0,wa
        # cosmo = self.cosmo
        # D, f = cosmo.D_f_z(z)
        # zind (previously: jnp.where(self.zeffs == z)[0][0]) was unused dead
        # code, and jnp.where/nonzero without a static size= is not
        # jax.vmap-compatible -- removed so this is vmappable over z (see
        # limber_jax.py's evaluate_all/computeCggCkg_all batching).
        D, f = self.evaluate_cosmo_ace(thy_args,z)

        Pgm_tab_emu = self.Pgm_emu.get_Pl_no_bias(theta,D)
        bias_mask_gm = (np.array([1, 1, 0,\
                1, 0, 0,\
                1, 0, 0, 0,\
                1, 0]) > 0)

        # bmono = bias_monomials_gm[bias_mask_gm]
        # pgm_emu = jnp.dot(Pgm_tab_emu[:,:-1][:,bias_mask_gm],bmono)
        pgm_emu = Pgm_tab_emu[:,:-1][:,bias_mask_gm]
        Pct_emu = -0.5*Pgm_tab_emu[:,-1]
        k = self.Pgm_emu.P11.k_grid
        res = jnp.zeros((len(k), jnp.shape(pgm_emu)[1]+2))
        res = res.at[:,0].set(k)
        res = res.at[:,1:-1].set(pgm_emu)
        res = res.at[:,-1].set(Pct_emu)

        return res
    # @jax.jit
    def compute_Pmm_hmcode(self,thy_args,zs):
        
        # theta = np.concatenate((thy_args,[0.0])) #logA,ns,H0,omega_b,omega_cdm, Mnu,w0,wa,Ok
        logA,ns,H0,omb,omc,Mnu,w0,wa = thy_args
        theta = jnp.array(thy_args) #logA,ns,H0,omega_b,omega_cdm, Mnu,w0,wa
        # print(thy_args,theta,len(theta))
        h = theta[2]/100.
        # cosmo = self.cosmo
        cosmo = jaxace.w0waCDMCosmology(
            ln10As=logA,
            ns=ns,
            h=H0/100,
            omega_b=omb,
            omega_c=omc,
            m_nu=Mnu,
            w0=w0,
            wa=wa
        )

        emu = self.Pmm_emu
        ki = emu.linear_pkcb.k_grid #not in h units
        k = ki/h #h/Mpc

        # One growth solve for the whole z grid and a vmap over the emulator, instead of a python
        # loop of len(zs) (=300 for the Limber grid) separate D_f_z ODE solves + emulator calls,
        # which ran op-by-op and dominated the EFT Cell theory's evaluation time.
        zs = jnp.asarray(zs)
        D, f = cosmo.D_f_z(zs)
        pred_HMCode = jax.vmap(lambda zi, Di: emu.get_linear_pmm(theta, zi, Di))(zs, D) #Not in h-units, (Nz,Nk)
        pk = pred_HMCode*h**3 #(Mpc/h)^3

        res = jnp.concatenate((k[:,None], pk.T), axis=1)

        return res
    # @jax.jit
    def compute_Pcb_lin(self,thy_args,z):

        # theta = np.concatenate((thy_args,[0.0]))
        theta = jnp.array(thy_args)
        h = theta[2]/100.
        # cosmo = self.cosmo
        # zind = jnp.where(self.zeffs == z)[0][0]
        D, f = self.evaluate_cosmo_ace(thy_args,z)
        # print(D,f)

        emu = self.Pmm_emu
        ki = emu.linear_pkcb.k_grid #not in h units
        ki = ki/h #h/Mpc
        # D, f = cosmo.D_f_z(z)
        # print(D)
        
        pred_cb = emu.get_linear_pkcb(theta, z, D)*h**3 #(Mpc/h)^3

        return ki,pred_cb
        
        
        

        
        
        

        
        

        
        
    




# ---------------------------------------------------------------------------
# Module-level, jit-compiled HEFT real-space P(k) functions, using
# aemulus_heft.heft_emu_jax.NNHEFTEmulator instead of the plain-numpy
# aemulus_heft.heft_emu.NNHEFTEmulator that likelihoods/pkCodes.py uses.
# Same table structure/contract as pkCodes.py's pmmHEFT/pgmtable_HEFT/
# pggtable_HEFT (so they're drop-in-compatible with limber_jax.py's Pgm/Pgg/
# Pmm callables), but jax-native and jit-compiled end to end.
#
# IMPORTANT: aemulus_heft's numpy and jax NNHEFTEmulator classes load the
# same trained weights and use the same param_order=[4,3,5,2,0,1,6,7], but
# disagree on what goes in column 4 (the amplitude slot): the numpy class
# (pkCodes.py) wants the exponentiated np.exp(ln10As)/10.; the jax class
# wants raw ln10As directly and exponentiates it internally. Passing the
# wrong one produces order-1-wrong output, not a subtle bug -- verified
# directly this session (checked against the numpy path with matching
# conventions: the two implementations agree to ~1e-5-1e-6, i.e. float32
# emulator precision, once each is fed its own correct convention).
# ---------------------------------------------------------------------------

try:
    from aemulus_heft.heft_emu_jax import NNHEFTEmulator as _NNHEFTEmulatorJAX

    nnemu_jax = _NNHEFTEmulatorJAX()
except ImportError:
    print("WARNING: aemulus_heft.heft_emu_jax not found; HEFT P(k) functions will fail if called.")
    nnemu_jax = None

# 16-element masks (index 0 = k, always kept; indices 1-15 = the 15 HEFT
# monomials in ptableHEFT_jax's order) -- identical to pkCodes.py's
# bcols_gm/bcols_gg.
_bcols_gm_jax = jnp.asarray([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]) > 0
_bcols_gg_jax = jnp.asarray([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]) > 0


@jax.jit
def pmmHEFT_jax(thy_args, z):
    """JAX/jit port of pkCodes.py::pmmHEFT, using aemulus_heft.heft_emu_jax.

    thy_args[:6] = [omb,omc,ns,ln10As,H0,Mnu]; z: (Nz,) array.
    Returns (Nk,1+Nz): column 0 is k, the remaining Nz columns are the
    matter power spectrum (HEFT's "1-1" monomial) evaluated at each z.
    """
    omb, omc, ns, ln10As, H0, Mnu = (
        thy_args[0], thy_args[1], thy_args[2], thy_args[3], thy_args[4], thy_args[5],
    )
    Mnu = jnp.maximum(Mnu, 0.01)  # HEFT is only valid for 0.01 < Mnu < 0.5
    z = jnp.atleast_1d(jnp.asarray(z))
    Nz = z.shape[0]
    row = jnp.stack([omb, omc, jnp.asarray(-1.0), ns, ln10As, H0, Mnu])
    cosmo = jnp.concatenate([jnp.tile(row, (Nz, 1)), z[:, None]], axis=1)

    k_nn, spec_heft_nn = nnemu_jax.predict(cosmo)  # spec_heft_nn: (Nz,15,Nk)

    res = jnp.concatenate(
        [k_nn[:, None], jnp.swapaxes(spec_heft_nn[:, 0, :], 0, 1)], axis=1
    )
    return res


@jax.jit
def ptableHEFT_jax(thy_args, z):
    """JAX/jit port of pkCodes.py::ptableHEFT, using aemulus_heft.heft_emu_jax.

    thy_args[:6] = [omb,omc,ns,ln10As,H0,Mnu]; z: scalar (vmap over this
    axis for a per-sample batch -- see limber_jax.py's batched evaluate).
    Returns (Nk,16): column 0 is k, columns 1-15 are the 15 HEFT monomials
    (1-1, 1-cb, cb-cb, delta-1, delta-cb, delta-delta, delta2-1, delta2-cb,
    delta2-delta, delta2-delta2, s2-1, s2-cb, s2-delta, s2-delta2, s2-s2).
    """
    omb, omc, ns, ln10As, H0, Mnu = (
        thy_args[0], thy_args[1], thy_args[2], thy_args[3], thy_args[4], thy_args[5],
    )
    Mnu = jnp.maximum(Mnu, 0.01)
    cosmo = jnp.atleast_2d(
        jnp.stack([omb, omc, jnp.asarray(-1.0), ns, ln10As, H0, Mnu, jnp.asarray(z)])
    )
    k_nn, spec_heft_nn = nnemu_jax.predict(cosmo)  # spec_heft_nn: (1,15,Nk)

    res = jnp.concatenate(
        [k_nn[:, None], jnp.swapaxes(spec_heft_nn[0, :, :], 0, 1)], axis=1
    )
    return res


def _bias_table_from_mask_jax(T, bmask):
    """Select bmask's columns of T (col 0 = k kept via bmask[0]=True) and
    append the -0.5*k^2*T[:,1] counterterm column, matching pkCodes.py's
    pgmtable_HEFT/pggtable_HEFT contraction."""
    Tm = T[:, bmask]
    counterterm = -0.5 * Tm[:, 0] ** 2 * Tm[:, 1]
    return jnp.concatenate([Tm, counterterm[:, None]], axis=1)


@jax.jit
def pgmtable_HEFT_jax(thy_args, z):
    """JAX/jit port of pkCodes.py::pgmtable_HEFT. z: scalar.

    Returns (Nk,6): [k, 1-cb, delta-1, delta2-1, s2-1, counterterm]. The
    full prediction is res[:,1:-1] . [1,b1,b2,bs] + alpha_x * res[:,-1].
    """
    T = ptableHEFT_jax(thy_args, z)
    return _bias_table_from_mask_jax(T, _bcols_gm_jax)


@jax.jit
def pggtable_HEFT_jax(thy_args, z):
    """JAX/jit port of pkCodes.py::pggtable_HEFT. z: scalar.

    Returns (Nk,11): [k, cb-cb, delta-1, delta-delta, delta2-1, delta2-delta,
    delta2-delta2, s2-1, s2-delta, s2-delta2, s2-s2, counterterm] (the exact
    column set selected by _bcols_gg_jax). The full prediction is
    res[:,1:-1] . bterms_gg(b1,b2,bs) + alpha_a * res[:,-1].
    """
    T = ptableHEFT_jax(thy_args, z)
    return _bias_table_from_mask_jax(T, _bcols_gg_jax)
