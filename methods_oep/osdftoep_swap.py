#!/usr/bin/env python3

import sys
from copy import deepcopy
import numpy as np
from pyscf import dft
from .osdftoep import OSDFTOEP
from .mom import MOM


class OSDFTOEP_swap(OSDFTOEP):
    r"""OSDFTOEP with MOM and swapping of orbitals.

    Args:
        mf: PySCF object with UKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        occ: initial occupation numbers to initialize MOM
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
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

        self.occ = self.mom.get_occ(self.mf.mo_coeff, self.mf.mo_energy if self.frac_occ else None)
        dm = self.mf.make_rdm1(self.mf.mo_coeff, self.occ)

        e_xc, self.vj_ao, self.E_Coul, self.vxnl_ao, self.E_x, omega, alpha, hyb = self.eval_state(dm)

        h1e = self.mf.get_hcore()
        e1 = np.einsum("ij,ji->", h1e, dm[0] + dm[1]).real
        self.e_tot = e1 + self.E_Coul + e_xc + self.mf.energy_nuc()
        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.e_tot += hyb * self.E_x

        if self.frac_occ:
            nocc = np.count_nonzero(self.occ)
            if nocc > sum(self.mf.nelec) and dft.libxc.is_hybrid_xc(self.mf.xc):
                occ_p = self.mom.get_occ(self.mf.mo_coeff)
                dm_p = self.mf.make_rdm1(self.mf.mo_coeff, occ_p)
                if omega == 0:
                    vj_ao_p, vxnl_ao_p = self.mf.get_jk(self.mf.mol, dm_p)
                elif alpha == 0:  # LR=0, only SR exchange
                    vj_ao_p = self.mf.get_j(self.mf.mol, dm_p)
                    vxnl_ao_p = self.mf.get_k(self.mf.mol, dm_p, omega=-omega)
                else:
                    sys.exit("Not implemented case for hybrid functionals")

                self.E_x = -0.5 * np.einsum("ij,ji->", vxnl_ao_p[0], dm_p[0])
                self.E_x += -0.5 * np.einsum("ij,ji->", vxnl_ao_p[1], dm_p[1])
                
                vj_ao_p = vj_ao_p[0] + vj_ao_p[1]
                E_Coul_p = np.einsum("ij,ji->", vj_ao_p, dm_p[0] + dm_p[1]).real * 0.5
                self.E_Coul = (1 - hyb) * self.E_Coul + hyb * E_Coul_p
                self.e_tot = e1 + self.E_Coul + e_xc + self.mf.energy_nuc() + hyb * self.E_x

        self.swap_orbitals(self.mf.mo_coeff, self.mf.mo_energy, self.occ)

    def print_occ_numbers(self, nplus=5):
        """Prints occupation numbers."""
        if self.frac_occ:
            print("Occupation numbers (alpha):", *self.occ[0, : self.mf.nelec[0] + nplus])
            print("Occupation numbers (beta) :", *self.occ[1, : self.mf.nelec[0] + nplus])
        else:
            print("Occupation numbers (alpha):", *map(int, self.occ[0, : self.mf.nelec[0] + nplus]))
            print("Occupation numbers (beta) :", *map(int, self.occ[1, : self.mf.nelec[0] + nplus]))