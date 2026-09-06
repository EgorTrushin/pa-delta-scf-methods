#!/usr/bin/env python3
"""Spin-unrestricted DFT calculations for excited-states."""

import logging
import sys
from copy import deepcopy

import numpy as np
import scipy
from pyscf import dft, lib

from .mom import MOM


class UKS:
    r"""
    Implements spin-unrestricted DFT calculations for excited-states.

    Subclasses can override get_fock_ingredients() to change how the density
    matrix and potentials are assembled each SCF iteration, compute_energies()
    to change what energies are computed and stored, and log_iteration() to
    change what is printed per iteration.

    Args:
        mf: PySCF object with ground-state UKS calculation
        occ: occupation numbers to initialize MOM
        frac_occ: if True, fractional occupation numbers are used
        logger: logger object for logging information
    """

    def __init__(self, mf, occ, frac_occ=True, logger=None):
        self.mf = deepcopy(mf)
        self.mom = MOM(mf.mo_coeff.copy(), occ, mf.get_ovlp(), frac_occ)
        self.frac_occ = frac_occ
        self.e_tot = None
        self.converged = False
        self.setup_logger(logger)

    def setup_logger(self, logger):
        """Sets up the logger."""
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def run(self, maxit=100, e_conv_thr=1e-7, verb=True):
        r"""
        Performs a self-consistent calculation.

        Args:
            maxit: maximal number of iterations
            e_conv_thr: threshold for energy convergence
            verb: if True, total energy and occupation numbers at every iteration are printed
        """
        e_tot_old = None

        adiis = lib.diis.DIIS()

        h1e = self.mf.get_hcore()
        sovlp = self.mf.get_ovlp()

        for itr in range(maxit):
            vxc, vj, dm, e_xc, e_x_nl = self.get_fock_ingredients(self.mf.mo_energy if self.frac_occ else None)

            fock = h1e + vj + vxc

            err_a = fock[0].dot(dm[0]).dot(sovlp) - sovlp.dot(dm[0]).dot(fock[0])
            err_b = fock[1].dot(dm[1]).dot(sovlp) - sovlp.dot(dm[1]).dot(fock[1])
            err_vec = np.hstack((err_a.ravel(), err_b.ravel()))
            fock = adiis.update(fock, err_vec)

            eigvals_a, c_a = scipy.linalg.eigh(fock[0], sovlp)
            eigvals_b, c_b = scipy.linalg.eigh(fock[1], sovlp)
            self.mf.mo_energy = (eigvals_a, eigvals_b)
            self.mf.mo_coeff = (c_a, c_b)

            if self.frac_occ:
                nocc = np.count_nonzero(self.get_reference_mom().get_occ(self.mf.mo_coeff, self.mf.mo_energy))
                if nocc > sum(self.mf.nelec) and dft.libxc.is_hybrid_xc(self.mf.xc):
                    _, vj_p, dm_p, _, e_x_nl_p = self.get_fock_ingredients()
                    self.compute_energies(h1e, dm, vj, e_xc, dm_p, vj_p, e_x_nl, e_x_nl_p)
                else:
                    self.compute_energies(h1e, dm, vj, e_xc)
            else:
                self.compute_energies(h1e, dm, vj, e_xc)

            if verb:
                self.log_iteration(itr)
                self.print_occ_numbers()

            if e_tot_old is None:
                e_tot_old = self.e_tot
            else:
                if abs(e_tot_old - self.e_tot) < e_conv_thr:
                    self.converged = True
                    break
                e_tot_old = self.e_tot

    def get_fock_ingredients(self, mo_energy=None):
        """Returns (vxc, vj, dm, e_xc) used to build the Fock matrix.

        Override in subclasses to change how ingredients from multiple states
        are blended before the Fock matrix is constructed.
        """
        return self.get_ingredients(mo_energy)

    def get_reference_mom(self):
        """Returns the primary MOM for occupation number checks."""
        return self.mom

    def get_moms(self):
        """Returns the list of MOMs used in the calculation."""
        return [self.mom]

    def single_energy(self, h1e, dm, vj, e_xc, dm_p=None, vj_p=None, e_x_nl=0.0, e_x_nl_p=0.0):
        """Returns total energy for a single state."""
        e1 = np.einsum("ij,ji->", h1e, dm[0] + dm[1])
        e_coul = 0.5 * np.einsum("ij,ji->", vj, dm[0] + dm[1])
        if dm_p is not None:
            hyb = dft.libxc.hybrid_coeff(self.mf.xc)
            e_coul_p = 0.5 * np.einsum("ij,ji->", vj_p, dm_p[0] + dm_p[1])
            e_coul = (1 - hyb) * e_coul + hyb * e_coul_p
            return e1 + e_coul + (e_xc - e_x_nl) + e_x_nl_p + self.mf.energy_nuc()
        return e1 + e_coul + e_xc + self.mf.energy_nuc()

    def compute_energies(self, h1e, dm, vj, e_xc, dm_p=None, vj_p=None, e_x_nl=None, e_x_nl_p=None):
        """Computes and stores total energy."""
        self.e_tot = self.single_energy(h1e, dm, vj, e_xc, dm_p, vj_p, e_x_nl or 0.0, e_x_nl_p or 0.0)

    def log_iteration(self, itr):
        """Logs per-iteration energy."""
        self.logger.info("ITER %2d    Total energy: %17.12f", itr, self.e_tot)

    def get_ingredients(self, mo_energy=None):
        """Constructs ingredients required for calculation."""
        occ = self.mom.get_occ(self.mf.mo_coeff, mo_energy)
        dm = self.mf.make_rdm1(self.mf.mo_coeff, occ)
        vxc, vj, exc, ex_nl = self.eval_dft(dm)

        return vxc, vj, dm, exc, ex_nl

    def eval_dft(self, dm):
        """Evaluates vj, vxc, and exc for a given density matrix.

        For hybrid functionals, adds the exact-exchange contribution to vxc
        and exc. Range-separated hybrids with a long-range component are not
        supported.
        """
        ni = dft.numint.NumInt()
        _, exc, vxc = ni.nr_uks(self.mf.mol, self.mf.grids, self.mf.xc, dm)

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc)
            if omega == 0:
                vj, vk = self.mf.get_jk(self.mf.mol, dm)
            elif alpha == 0:  # LR=0, only SR exchange
                vj = self.mf.get_j(self.mf.mol, dm)
                vk = self.mf.get_k(self.mf.mol, dm, omega=-omega)
            else:
                raise NotImplementedError("Not implemented case for hybrid functionals")

            hf_energy = -0.5 * np.einsum("ij,ji->", vk[0], dm[0])
            hf_energy += -0.5 * np.einsum("ij,ji->", vk[1], dm[1])
        else:
            vj = self.mf.get_j(dm=dm)
            hf_energy = 0.0
            hyb = 0.0

        vj = vj[0] + vj[1]

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            vxc -= hyb * vk
            exc += hyb * hf_energy

        return vxc, vj, exc, hyb * hf_energy

    def print_occ_numbers(self, nplus=5):
        """Prints occupation numbers."""
        for mom in self.get_moms():
            if self.frac_occ:
                occ = mom.get_occ(self.mf.mo_coeff, self.mf.mo_energy)
                self.logger.info(
                    "Occupation numbers (alpha): %s", " ".join(map(str, occ[0, : self.mf.nelec[0] + nplus]))
                )
                self.logger.info(
                    "Occupation numbers (beta) : %s", " ".join(map(str, occ[1, : self.mf.nelec[0] + nplus]))
                )
            else:
                occ = mom.get_occ(self.mf.mo_coeff)
                self.logger.info(
                    "Occupation numbers (alpha): %s", " ".join(map(str, map(int, occ[0, : self.mf.nelec[0] + nplus])))
                )
                self.logger.info(
                    "Occupation numbers (beta) : %s", " ".join(map(str, map(int, occ[1, : self.mf.nelec[0] + nplus])))
                )
