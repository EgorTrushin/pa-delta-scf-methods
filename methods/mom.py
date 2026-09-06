#!/usr/bin/env python3

from functools import reduce

import numpy as np


class MOM:
    """
    The implementation of initial maximum overlap method to determine
    occupation number for each orbital in every iteration.

    Assumed to be used in combination with spin-unrestricted calculations.
    Largely based on the implementation provided by pyscf.scf.addons:
    https://pyscf.org/_modules/pyscf/scf/addons.html#mom_occ_

    Extended to support fractional occupation numbers for degenerate cases.
    """

    def __init__(self, ref_mo, ref_occ, ovlp, frac_occ=False):
        """Initializes the instance.

        Args:
          ref_mo: reference MO coefficients
          ref_occ: initial choice of occupation numbers
          ovlp: AO basis overlap matrix
          frac_occ: if True, fractional occupation numbers are employed
        """
        self.ref_mo = ref_mo
        self.ref_occ = ref_occ
        self.ovlp = ovlp
        self.frac_occ = frac_occ
        self.mo_occ_a = ref_mo[0][:, ref_occ[0] > 0]
        self.mo_occ_b = ref_mo[1][:, ref_occ[1] > 0]
        self.nocc_a = int(np.sum(ref_occ[0]))
        self.nocc_b = int(np.sum(ref_occ[1]))

    def get_occ(self, mo, mo_energy=None):
        """Determines new occupation numbers.

        Args:
          mo: MO coefficients
          mo_energy: MO energies (required for fractional occupation numbers)

        Returns:
          new_occ: new occupation numbers
        """
        new_occ = np.zeros_like(self.ref_occ)
        s_a = reduce(np.dot, (self.mo_occ_a.conj().T, self.ovlp, mo[0]))
        s_b = reduce(np.dot, (self.mo_occ_b.conj().T, self.ovlp, mo[1]))
        idx_a = np.argsort(np.einsum("ij,ij->j", s_a, s_a))[::-1]
        idx_b = np.argsort(np.einsum("ij,ij->j", s_b, s_b))[::-1]
        new_occ[0, idx_a[: self.nocc_a]] = 1.0
        new_occ[1, idx_b[: self.nocc_b]] = 1.0

        if self.frac_occ and mo_energy is not None:
            new_occ[0] = self.fill_frac_occ(mo_energy[0], new_occ[0])
            new_occ[1] = self.fill_frac_occ(mo_energy[1], new_occ[1])

        return new_occ

    def fill_frac_occ(self, energies, occupations, tol=1e-10):
        """
        Adjust occupations for degenerate orbitals to have equal fractional occupation.
        """
        sorted_indices = np.argsort(energies)

        # Search for groups of degenerate orbitals
        groups = []
        current_group = [sorted_indices[0]]

        for i in range(1, len(sorted_indices)):
            if abs(energies[sorted_indices[i]] - energies[sorted_indices[i - 1]]) <= tol:
                current_group.append(sorted_indices[i])
            else:
                groups.append(current_group)
                current_group = [sorted_indices[i]]
        groups.append(current_group)

        # Assign fractional occupations for each group
        new_occ = occupations.copy()
        for group in groups:
            total_occ = occupations[group].sum()
            equal_occ = total_occ / len(group)
            new_occ[group] = equal_occ
        return new_occ
