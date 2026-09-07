#!/usr/bin/env python3

import sys

import numpy as np
from pyscf import dft

from .exxoep import EXXOEP


class DFTOEP(EXXOEP):
    r"""
    Implements density functional theory optimized effective potential method for closed-shell systems.
    The implementation follows:
    E. Trushin, A. Görling. J. Chem. Theory Comput. 2025, 21, 4, 1667–1683. https://doi.org/10.1021/acs.jctc.4c01477
    E. Trushin, A. Görling. J. Chem. Phys. 155, 054109 (2021). https://doi.org/10.1063/5.0056431

    Args:
        mf: PySCF object with RKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
    """

    def __init__(self, mf, oep_basis, vh_via_OEP=False, space_sym=False):
        super().__init__(mf, oep_basis, vh_via_OEP, space_sym)

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials."""

        self.mf.max_cycle = 0
        self.mf.run()
        self.e_tot = self.mf.e_tot

        dm = self.mf.make_rdm1()

        ni = dft.numint.NumInt()
        _, _, vxc = ni.nr_rks(self.mf.mol, self.mf.grids, self.mf.xc, dm)

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc)
            if omega == 0:
                self.vj_ao, self.vxnl_ao = self.mf.get_jk(self.mf.mol, dm)
            elif alpha == 0:  # LR=0, only SR exchange
                self.vj_ao = self.mf.get_j(self.mf.mol, dm)
                self.vxnl_ao = self.mf.get_k(self.mf.mol, dm, omega=-omega)
            else:
                sys.exit("Not implemented case for hybrid functionals")
        else:
            self.vj_ao = self.mf.get_j(self.mf.mol, dm)

        self.E_Coul = np.einsum("ij,ji->", self.vj_ao, dm) * 0.5

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.E_x = -0.5 * np.einsum("ij,ji->", self.vxnl_ao, dm) * 0.5
            self.vxnl_ao *= -0.5

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.vxnl_ao = vxc + hyb * self.vxnl_ao
        else:
            self.vxnl_ao = vxc
