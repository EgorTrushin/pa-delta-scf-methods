#!/usr/bin/env python3
"""DFT calculations for excited-states with potential-averaged state-averaged KS method."""

from copy import deepcopy
from pyscf import dft
from .UKS import UKS
from .mom import MOM


class pa_STA_KS(UKS):
    r"""
    Implements potential-averaged state-averaged KS method.

    Args:
        mf: PySCF object with ground-state UKS calculation
        occ1: occupation numbers to initialize MOM for singlet
        occ3: occupation numbers to initialize MOM for triplet
        frac_occ: if True, fractional occupation numbers are used
        logger: logger object for logging information
    """

    def __init__(self, mf, occ1, occ3, frac_occ=True, logger=None):
        self.mf = deepcopy(mf)
        self.mom1 = MOM(mf.mo_coeff.copy(), occ1, mf.get_ovlp(), frac_occ)
        self.mom3 = MOM(mf.mo_coeff.copy(), occ3, mf.get_ovlp(), frac_occ)
        self.frac_occ = frac_occ
        self.e_tot = None
        self.e_tot_oss = None
        self.e_tot_t = None
        self.converged = False
        self.setup_logger(logger)

    def get_fock_ingredients(self, mo_energy=None):
        """Returns averaged ingredients from singlet and triplet MOMs."""
        vxc1, vj1, dm1, e_xc1, e_x_nl1 = self.get_state_ingredients(self.mom1, mo_energy)
        vxc3, vj3, dm3, e_xc3, e_x_nl3 = self.get_state_ingredients(self.mom3, mo_energy)

        if mo_energy is None and self.frac_occ and dft.libxc.is_hybrid_xc(self.mf.xc):
            self.dm1_p, self.vj1_p, self.e_x_nl1_p = dm1, vj1, e_x_nl1
            self.dm3_p, self.vj3_p, self.e_x_nl3_p = dm3, vj3, e_x_nl3
        else:
            self.dm1, self.vj1, self.e_xc1, self.e_x_nl1 = dm1, vj1, e_xc1, e_x_nl1
            self.dm3, self.vj3, self.e_xc3, self.e_x_nl3 = dm3, vj3, e_xc3, e_x_nl3

        vxc = 0.5 * (vxc1 + vxc3)
        vj = 0.5 * (vj1 + vj3)
        dm = 0.5 * (dm1 + dm3)
        e_xc = 0.5 * (e_xc1 + e_xc3)
        e_x_nl = 0.5 * (e_x_nl1 + e_x_nl3)

        return vxc, vj, dm, e_xc, e_x_nl

    def get_reference_mom(self):
        """Returns the primary MOM for occupation number checks."""
        return self.mom1

    def get_moms(self):
        """Returns the list of MOMs used in the calculation."""
        return [self.mom1, self.mom3]

    def compute_energies(self, h1e, dm, vj, e_xc, dm_p=None, vj_p=None, e_x_nl=None, e_x_nl_p=None):
        """Computes and stores state-averaged, OSS, and triplet energies."""
        e_x_nl = e_x_nl or 0.
        e_x_nl_p = e_x_nl_p or 0.

        self.e_tot = self.single_energy(h1e, dm, vj, e_xc, dm_p, vj_p, e_x_nl, e_x_nl_p)

        dm_oss = 2 * self.dm1 - self.dm3
        vj_oss = 2 * self.vj1 - self.vj3
        e_xc_oss = 2 * self.e_xc1 - self.e_xc3
        e_x_nl_oss = 2 * self.e_x_nl1 - self.e_x_nl3
        dm_oss_p = 2 * self.dm1_p - self.dm3_p if dm_p is not None else None
        vj_oss_p = 2 * self.vj1_p - self.vj3_p if dm_p is not None else None
        e_x_nl_oss_p = 2 * self.e_x_nl1_p - self.e_x_nl3_p if dm_p is not None else 0.
        self.e_tot_oss = self.single_energy(h1e, dm_oss, vj_oss, e_xc_oss, dm_oss_p, vj_oss_p, e_x_nl_oss, e_x_nl_oss_p)

        dm3_p = self.dm3_p if dm_p is not None else None
        vj3_p = self.vj3_p if dm_p is not None else None
        e_x_nl3_p = self.e_x_nl3_p if dm_p is not None else 0.
        self.e_tot_t = self.single_energy(h1e, self.dm3, self.vj3, self.e_xc3, dm3_p, vj3_p, self.e_x_nl3, e_x_nl3_p)


    def log_iteration(self, itr):
        """Logs per-iteration state-averaged, OSS, and triplet energies."""
        self.logger.info(
            "ITER %2d    Energy: %17.12f    Energy(OSS): %17.12f    Energy(T): %17.12f",
            itr,
            self.e_tot,
            self.e_tot_oss,
            self.e_tot_t,
        )

    def get_state_ingredients(self, mom, mo_energy=None):
        """Constructs ingredients for a single state with spin-averaged vxc.

        The alpha and beta components of vxc are averaged so that the
        exchange-correlation potential is the same for both spins.
        """
        occ = mom.get_occ(self.mf.mo_coeff, mo_energy)
        dm = self.mf.make_rdm1(self.mf.mo_coeff, occ)
        vxc, vj, exc, e_x_nl = self.eval_dft(dm)

        vxc[0, :, :] = 0.5 * (vxc[0, :, :] + vxc[1, :, :])
        vxc[1, :, :] = vxc[0, :, :]

        return vxc, vj, dm, exc, e_x_nl
