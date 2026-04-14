#!/usr/bin/env python3
"""Potential-averaged open-shell singlet DFT calculations for excited-states."""

from .UKS import UKS
from .pa_STA_KS import pa_STA_KS


class pa_OSS_KS(pa_STA_KS):
    r"""
    Implements potential-averaged open-shell singlet DFT calculations for excited-state.

    Args:
        mf: PySCF object with ground-state UKS calculation
        occ1: occupation numbers to initialize MOM for singlet
        occ3: occupation numbers to initialize MOM for triplet
        frac_occ: if True, fractional occupation numbers are used
    """

    def get_fock_ingredients(self, mo_energy=None):
        """Returns composite ingredients from singlet and triplet MOMs."""
        vxc1, vj1, dm1, e_xc1 = self.get_state_ingredients(self.mom1, mo_energy)
        vxc3, vj3, dm3, e_xc3 = self.get_state_ingredients(self.mom3, mo_energy)

        vxc = 2 * vxc1 - vxc3
        vj = 2 * vj1 - vj3
        dm = 2 * dm1 - dm3
        e_xc = 2 * e_xc1 - e_xc3

        return vxc, vj, dm, e_xc

    def compute_energies(self, h1e, dm, vj, e_xc):
        """Computes and stores total energy for OSS."""
        UKS.compute_energies(self, h1e, dm, vj, e_xc)

    def log_iteration(self, itr):
        """Logs per-iteration OSS total energy."""
        self.logger.info("ITER %2d    Total energy (OSS): %17.12f", itr, self.e_tot)
