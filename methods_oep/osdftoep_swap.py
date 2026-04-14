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
        self.mom = MOM(mf.mo_coeff.copy(), occ, mf.get_ovlp(), frac_occ=False)
        super().__init__(mf, oep_basis, vh_via_OEP=vh_via_OEP, space_sym=space_sym, spin_sym=spin_sym)

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials. Swap orbitals in the end."""

        self.occ = self.mom.get_occ(self.mf.mo_coeff)
        dm = self.mf.make_rdm1(self.mf.mo_coeff, self.occ)

        ni = dft.numint.NumInt()
        _, e_xc, vxc = ni.nr_uks(self.mf.mol, self.mf.grids, self.mf.xc, dm)

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc)
            if omega == 0:
                self.vj_ao, self.vxnl_ao = self.mf.get_jk(self.mf.mol, dm)
            elif alpha == 0:  # LR=0, only SR exchange
                self.vj_ao = self.mf.get_j(self.mf.mol, dm)
                self.vxnl_ao = self.mf.get_k(self.mf.mol, dm, omega=-omega)
            else:
                sys.exit("Not implemented case for hybrid functionals")

            self.E_x = -0.5 * np.einsum("ij,ji->", self.vxnl_ao[0], dm[0])
            self.E_x += -0.5 * np.einsum("ij,ji->", self.vxnl_ao[1], dm[1])
            self.vxnl_ao *= -1.0
        else:
            self.vj_ao = self.mf.get_j(dm=dm)

        self.vj_ao = self.vj_ao[0] + self.vj_ao[1]
        self.E_Coul = np.einsum("ij,ji->", self.vj_ao, dm[0] + dm[1]).real * 0.5

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.vxnl_ao = vxc + hyb * self.vxnl_ao
        else:
            self.vxnl_ao = vxc

        h1e = self.mf.get_hcore()
        e1 = np.einsum("ij,ji->", h1e, dm[0] + dm[1]).real
        self.e_tot = e1 + self.E_Coul + e_xc + self.mf.energy_nuc()
        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.e_tot += hyb * self.E_x

        # Swap orbitals
        ind = np.argsort(-self.occ[0], kind='stable')
        for i in range(self.mf.mo_coeff.shape[1]):
            self.mf.mo_coeff[0, i, :] = self.mf.mo_coeff[0, i, ind]
        self.mf.mo_energy[0, :] = self.mf.mo_energy[0, ind]
        ind = np.argsort(-self.occ[1], kind='stable')
        for i in range(self.mf.mo_coeff.shape[1]):
            self.mf.mo_coeff[1, i, :] = self.mf.mo_coeff[1, i, ind]
        self.mf.mo_energy[1, :] = self.mf.mo_energy[1, ind]

    def print_occ_numbers(self, nplus=5):
        """Prints occupation numbers."""
        print("Occupation numbers (alpha):", *map(int, self.occ[0, : self.mf.nelec[0] + nplus]))
        print("Occupation numbers (beta) :", *map(int, self.occ[1, : self.mf.nelec[0] + nplus]))
