#!/usr/bin/env python3

from copy import deepcopy

from .mom import MOM
from .osdftoep import OSDFTOEP


class OSDFTOEP_swap(OSDFTOEP):
    r"""OSDFTOEP with MOM and swapping of orbitals.

    Args:
        mf: PySCF object with UKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        occ: initial occupation numbers to initialize MOM
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization. The OEP equations of this
            implementation treat a partially filled degenerate shell as one integer
            configuration, so the occupation-number implementation is the correct
            treatment for fractional occupation numbers.
        spin_sym: whether to perform spin-symmetrization
    """

    def __init__(
        self,
        mf,
        oep_basis,
        occ,
        vh_via_OEP=False,
        space_sym=False,
        spin_sym=False,
    ):
        mf = deepcopy(mf)
        mf.nelec = (int(sum(occ[0])), int(sum(occ[1])))
        self.frac_occ = space_sym
        self.mom = MOM(mf.mo_coeff.copy(), occ, mf.get_ovlp(), frac_occ=self.frac_occ)
        super().__init__(mf, oep_basis, vh_via_OEP=vh_via_OEP, space_sym=space_sym, spin_sym=spin_sym)

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials. Swap orbitals in the end."""

        self.occ, _, self.vj_ao, self.vxnl_ao, terms, aux_terms = self.get_state_ingredients(
            self.mom, self.mf.mo_coeff, self.mf.mo_energy
        )

        self.e_tot = self.total_energy(terms)
        self.e_aux = self.total_energy(aux_terms)

        self.swap_orbitals(self.mf.mo_coeff, self.mf.mo_energy, self.occ)

    def print_occ_numbers(self, nplus=5):
        """Prints occupation numbers."""
        if self.frac_occ:
            print("Occupation numbers (alpha):", *self.occ[0, : self.mf.nelec[0] + nplus])
            print("Occupation numbers (beta) :", *self.occ[1, : self.mf.nelec[0] + nplus])
        else:
            print("Occupation numbers (alpha):", *map(int, self.occ[0, : self.mf.nelec[0] + nplus]))
            print("Occupation numbers (beta) :", *map(int, self.occ[1, : self.mf.nelec[0] + nplus]))
