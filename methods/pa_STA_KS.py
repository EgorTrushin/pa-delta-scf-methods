#!/usr/bin/env python3
"""DFT calculations for excited-states with potential-averaged state-averaged KS method."""

from copy import deepcopy

from .mom import MOM
from .UKS import UKS


class pa_STA_KS(UKS):
    r"""
    Implements potential-averaged state-averaged KS method.

    The orbitals are optimized for the average of the Hartree-exchange-correlation
    potentials of the mixed singlet-triplet and the triplet determinant, so that
    a single SCF procedure yields both the open-shell singlet and the triplet
    energy of one excitation.

    Args:
        mf: PySCF object with ground-state UKS calculation
        occ1: occupation numbers to initialize MOM for singlet
        occ3: occupation numbers to initialize MOM for triplet
        frac_occ: if True, fractional occupation numbers are used
        logger: logger object for logging information
    """

    spin_symmetrized = True

    def __init__(self, mf, occ1, occ3, frac_occ=True, logger=None):
        self.mf = deepcopy(mf)
        self.mom1 = MOM(mf.mo_coeff.copy(), occ1, mf.get_ovlp(), frac_occ)
        self.mom3 = MOM(mf.mo_coeff.copy(), occ3, mf.get_ovlp(), frac_occ)
        self.frac_occ = frac_occ
        self.h1e = self.mf.get_hcore()
        self.e_tot = None
        self.e_tot_oss = None
        self.e_tot_t = None
        self.converged = False
        self.setup_logger(logger)

    def get_fock_ingredients(self, mo_energy=None):
        """Returns averaged ingredients from singlet and triplet MOMs."""
        vxc1, vj1, dm1, self.terms1 = self.get_state_ingredients(self.mom1, mo_energy)
        vxc3, vj3, dm3, self.terms3 = self.get_state_ingredients(self.mom3, mo_energy)

        self.e_xc_sta = 0.5 * (self.terms1[2] + self.terms3[2])

        return 0.5 * (vxc1 + vxc3), 0.5 * (vj1 + vj3), 0.5 * (dm1 + dm3)

    def get_moms(self):
        """Returns the list of MOMs used in the calculation."""
        return [self.mom1, self.mom3]

    def compute_energies(self, dm, vj):
        """Computes and stores state-averaged, OSS, and triplet energies.

        The state-averaged energy is an auxiliary quantity used to monitor the
        SCF convergence. The open-shell singlet energy follows from the
        spin-purification formula E(OSS) = 2 E(M) - E(T).
        """
        self.e_tot = self.aux_energy(dm, vj, self.e_xc_sta)
        self.e_tot_oss = self.total_energy(self.combine_terms([2.0, -1.0], [self.terms1, self.terms3]))
        self.e_tot_t = self.total_energy(self.terms3)

    def log_iteration(self, itr):
        """Logs per-iteration state-averaged, OSS, and triplet energies."""
        self.logger.info(
            "ITER %2d    Energy: %17.12f    Energy(OSS): %17.12f    Energy(T): %17.12f",
            itr,
            self.e_tot,
            self.e_tot_oss,
            self.e_tot_t,
        )
