#!/usr/bin/env python3
"""Potential-averaged open-shell singlet DFT calculations for excited-states."""

from .pa_STA_KS import pa_STA_KS


class pa_OSS_KS(pa_STA_KS):
    r"""
    Implements potential-averaged open-shell singlet DFT calculations for excited-state.

    Args:
        mf: PySCF object with ground-state UKS calculation
        occ1: occupation numbers to initialize MOM for singlet
        occ3: occupation numbers to initialize MOM for triplet
        frac_occ: if True, fractional occupation numbers are used
        logger: logger object for logging information
    """

    def get_fock_ingredients(self, mo_energy=None):
        """Returns composite ingredients from singlet and triplet MOMs."""
        vxc1, vj1, dm1, self.terms1, self.aux_terms1 = self.get_state_ingredients(self.mom1, mo_energy)
        vxc3, vj3, dm3, self.terms3, self.aux_terms3 = self.get_state_ingredients(self.mom3, mo_energy)

        return 2 * vxc1 - vxc3, 2 * vj1 - vj3, 2 * dm1 - dm3

    def compute_energies(self, dm, vj):
        """Computes and stores the open-shell singlet total energy."""
        self.e_tot = self.total_energy(self.combine_terms([2.0, -1.0], [self.terms1, self.terms3]))
        self.e_aux = self.total_energy(self.combine_terms([2.0, -1.0], [self.aux_terms1, self.aux_terms3]))

    def log_iteration(self, itr):
        """Logs per-iteration OSS total energy."""
        self.logger.info("ITER %2d    Total energy (OSS): %17.12f", itr, self.e_tot)
