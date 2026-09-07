#!/usr/bin/env python3
"""Spin-unrestricted DFT calculations for excited-states."""

import itertools
import logging
import sys
from copy import deepcopy

import numpy as np
import scipy
from pyscf import dft, lib

from .mom import MOM, degenerate_groups


class UKS:
    r"""
    Implements spin-unrestricted DFT calculations for excited-states.

    Subclasses can override get_fock_ingredients() to change how the potentials
    of one or several states are blended into the Fock matrix, compute_energies()
    to change what energies are computed and stored, log_iteration() to change
    what is printed per iteration, and finalize() to evaluate properties once
    the SCF procedure has converged. Spin-symmetrized methods set the class
    attribute spin_symmetrized to True.

    The potentials are constructed with the occupation numbers returned by the
    MOM, which are fractional within partially filled degenerate shells if
    frac_occ is True. Total energies, in contrast, are always evaluated with
    integer occupation numbers, averaged over all integer occupation patterns
    that are compatible with the fractional ones.

    Args:
        mf: PySCF object with ground-state UKS calculation
        occ: occupation numbers to initialize MOM
        frac_occ: if True, fractional occupation numbers are used
        logger: logger object for logging information
    """

    spin_symmetrized = False

    def __init__(self, mf, occ, frac_occ=True, logger=None):
        self.mf = deepcopy(mf)
        self.mom = MOM(mf.mo_coeff.copy(), occ, mf.get_ovlp(), frac_occ)
        self.frac_occ = frac_occ
        self.h1e = self.mf.get_hcore()
        self.X_lindep = self.canorth_matrix()
        self.e_tot = None
        self.e_aux = None
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

        sovlp = self.mf.get_ovlp()

        for itr in range(maxit):
            vxc, vj, dm = self.get_fock_ingredients(self.mf.mo_energy if self.frac_occ else None)

            fock = self.h1e + vj + vxc

            err_a = fock[0].dot(dm[0]).dot(sovlp) - sovlp.dot(dm[0]).dot(fock[0])
            err_b = fock[1].dot(dm[1]).dot(sovlp) - sovlp.dot(dm[1]).dot(fock[1])
            err_vec = np.hstack((err_a.ravel(), err_b.ravel()))
            fock = adiis.update(fock, err_vec)

            eigvals_a, c_a = self.eigh_canorth(fock[0])
            eigvals_b, c_b = self.eigh_canorth(fock[1])
            self.mf.mo_energy = (eigvals_a, eigvals_b)
            self.mf.mo_coeff = (c_a, c_b)

            self.compute_energies(dm, vj)

            if verb:
                self.log_iteration(itr)
                self.print_occ_numbers()

            e_conv = self.convergence_energy()

            if e_tot_old is None:
                e_tot_old = e_conv
            else:
                if abs(e_tot_old - e_conv) < e_conv_thr:
                    self.converged = True
                    break
                e_tot_old = e_conv

        self.finalize(verb)

    def get_fock_ingredients(self, mo_energy=None):
        """Returns (vxc, vj, dm) used to build the Fock matrix.

        Override in subclasses to change how the ingredients of several states
        are blended before the Fock matrix is constructed. The energy terms of
        the involved states are stored for compute_energies().
        """
        vxc, vj, dm, self.terms, self.aux_terms = self.get_state_ingredients(self.mom, mo_energy)
        return vxc, vj, dm

    def get_state_ingredients(self, mom, mo_energy=None):
        """Constructs the potentials and energy terms of a single state.

        The energy terms are returned twice. The first set is evaluated with
        integer occupation numbers and the total energies are built from it.
        The second set is evaluated with the fractional occupation numbers and
        monitors the SCF convergence, see convergence_energy().
        """
        occ = mom.get_occ(self.mf.mo_coeff, mo_energy)
        dm = self.mf.make_rdm1(self.mf.mo_coeff, occ)
        vxc, vj, e_xc = self.eval_dft(dm)
        vxc = self.symmetrize_vxc(vxc)
        aux_terms = (
            np.einsum("ij,ji->", self.h1e, dm[0] + dm[1]),
            0.5 * np.einsum("ij,ji->", vj, dm[0] + dm[1]),
            e_xc,
        )

        return vxc, vj, dm, self.energy_terms(occ, aux_terms), aux_terms

    def canorth_matrix(self):
        """Returns the canonical orthogonalization matrix of the orbital subspace.

        PySCF drops the eigenvectors of the overlap matrix below its threshold
        during the canonical orthogonalization, so mo_coeff has nmo columns with
        nmo smaller than nao for a linearly dependent basis. The matrix spans
        that nmo-dimensional subspace.
        """
        nmo = self.mf.mo_coeff.shape[-1]
        e_s, v_s = scipy.linalg.eigh(self.mf.get_ovlp())
        return v_s[:, -nmo:] / np.sqrt(e_s[-nmo:])

    def eigh_canorth(self, fock):
        """Diagonalizes the Fock matrix within that subspace.

        The orbitals of every iteration then keep the shape the occupation
        numbers are stored with, which a plain scipy.linalg.eigh(fock, sovlp)
        does not for a linearly dependent basis.
        """
        x = self.X_lindep
        eigvals, c = scipy.linalg.eigh(x.T @ fock @ x)
        return eigvals, x @ c

    def symmetrize_vxc(self, vxc):
        """Averages the alpha and beta components of vxc in spin-symmetrized methods.

        The exchange-correlation potential is then the same for both spins.
        """
        if self.spin_symmetrized:
            vxc[0, :, :] = 0.5 * (vxc[0, :, :] + vxc[1, :, :])
            vxc[1, :, :] = vxc[0, :, :]
        return vxc

    def energy_terms(self, occ, aux_terms):
        """Returns the energy terms (e1, e_coul, e_xc) of a single state.

        The terms are evaluated with integer occupation numbers. If the state
        has fractionally occupied degenerate orbitals, they are averaged over
        all compatible integer occupation patterns. Without such orbitals the
        terms of aux_terms are already the ones with integer occupation
        numbers and are returned unchanged.
        """
        occ_list = self.gen_integer_occ(occ)

        if len(occ_list) == 1:
            return aux_terms

        e1_sum, e_coul_sum, e_xc_sum = 0.0, 0.0, 0.0
        for occ_int in occ_list:
            dm_int = self.mf.make_rdm1(self.mf.mo_coeff, occ_int)
            _, vj_int, e_xc_int = self.eval_dft(dm_int)
            e1_sum += np.einsum("ij,ji->", self.h1e, dm_int[0] + dm_int[1])
            e_coul_sum += 0.5 * np.einsum("ij,ji->", vj_int, dm_int[0] + dm_int[1])
            e_xc_sum += e_xc_int

        nocc_list = len(occ_list)
        return e1_sum / nocc_list, e_coul_sum / nocc_list, e_xc_sum / nocc_list

    def gen_integer_occ(self, occ):
        """Returns all integer occupation patterns compatible with occ.

        The orbitals of a partially filled degenerate shell carry fractional
        occupation numbers. They are occupied with 0 or 1 in all possible ways
        that preserve the number of electrons of the shell, and the patterns of
        a spin channel are the product over its shells. Enumerating over all
        fractionally occupied orbitals of a spin channel at once would move
        electrons between shells and generate patterns that do not belong to
        the state, among them the ground-state configuration.

        The patterns, and with them the averaged energy, refer to the orbitals
        of the shell as the diagonalization returns them. A rotation within the
        shell leaves the fractional density unchanged but not the energy of the
        individual patterns, so the averaged energy is defined only up to the
        orientation of the degenerate orbitals.
        """
        combos_per_spin = []

        for spin, spin_occ in enumerate(occ):
            frac_idx = np.where(~np.isin(spin_occ, [0.0, 1.0]))[0]

            if len(frac_idx) == 0:
                combos_per_spin.append([spin_occ.copy()])
                continue

            # The orbital energies are the ones the MOM used to build occ, so
            # the two groupings into degenerate shells agree by construction
            shells = [np.intersect1d(group, frac_idx) for group in degenerate_groups(self.mf.mo_energy[spin])]
            per_shell = [
                list(itertools.combinations(shell, round(float(spin_occ[shell].sum()))))
                for shell in shells
                if shell.size > 0
            ]

            spin_combos = []
            for chosen in itertools.product(*per_shell):
                new_occ = spin_occ.copy()
                new_occ[frac_idx] = 0.0
                new_occ[list(itertools.chain.from_iterable(chosen))] = 1.0
                spin_combos.append(new_occ)

            combos_per_spin.append(spin_combos)

        return [list(pair) for pair in itertools.product(*combos_per_spin)]

    def total_energy(self, terms):
        """Returns the total energy for a given set of energy terms."""
        return sum(terms) + self.mf.energy_nuc()

    @staticmethod
    def combine_terms(coeffs, terms):
        """Returns a linear combination of several sets of energy terms."""
        return tuple(sum(c * t[i] for c, t in zip(coeffs, terms, strict=True)) for i in range(3))

    def convergence_energy(self):
        """Returns the energy the SCF convergence is monitored with.

        The total energies are evaluated with integer occupation numbers and
        therefore depend on the orientation of the orbitals within a partially
        filled degenerate shell, which the diagonalization fixes only up to
        numerical noise. The auxiliary energy is evaluated with the fractional
        occupation numbers, which are invariant under that rotation, so it is
        the quantity that settles down as the SCF procedure converges.
        """
        return self.e_aux

    def compute_energies(self, dm, vj):
        """Computes and stores total energy."""
        self.e_tot = self.total_energy(self.terms)
        self.e_aux = self.total_energy(self.aux_terms)

    def log_iteration(self, itr):
        """Logs per-iteration energy."""
        self.logger.info("ITER %2d    Total energy: %17.12f", itr, self.e_tot)

    def finalize(self, verb=True):
        """Evaluates properties after the SCF procedure has finished."""

    def get_moms(self):
        """Returns the list of MOMs used in the calculation."""
        return [self.mom]

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

            vxc -= hyb * vk
            exc += hyb * hf_energy
        else:
            vj = self.mf.get_j(dm=dm)

        return vxc, vj[0] + vj[1], exc

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
