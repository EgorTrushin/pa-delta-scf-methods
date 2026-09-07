#!/usr/bin/env python3

from copy import deepcopy

import numpy as np
import scipy
from pyscf import lib

from .mom import MOM
from .osdftoep import OSDFTOEP


class OSDFTOEP_STA_swap(OSDFTOEP):
    r"""OSDFTOEP for excited states with state-averaged KS method with swapping of orbitals.

    Args:
        mf: PySCF object with UKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        occ: initial occupation numbers for singlet state to initialize MOM
        occ3: initial occupation numbers for triplet state to initialize MOM
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
    """

    def __init__(
        self,
        mf,
        oep_basis,
        occ,
        occ3,
        vh_via_OEP=False,
        space_sym=False,
    ):
        mf = deepcopy(mf)
        mf.mo_coeff3 = mf.mo_coeff.copy()
        mf.mo_energy3 = mf.mo_energy.copy()
        self.frac_occ = space_sym
        self.mom = MOM(mf.mo_coeff.copy(), occ, mf.get_ovlp(), frac_occ=self.frac_occ)
        self.mom3 = MOM(mf.mo_coeff.copy(), occ3, mf.get_ovlp(), frac_occ=self.frac_occ)
        super().__init__(mf, oep_basis, vh_via_OEP=vh_via_OEP, space_sym=space_sym)

    def run(self, maxit=50, thr_fai_oep=5e-2, linear_mixing=-1.0, e_conv_thr=1e-8):
        r"""
        Performs a self-consistent calculation.

        Args:
            maxit: maximal number of iterations
            thr_fai_oep: threshold T_{ai} from Section IIA5 of J. Chem. Phys. 155 (2021) 054109
            linear_mixing: if specified larger than 0, linear mixing scheme is used istead of DIIS
                           with given linear_mixing coefficient
            e_conv_thr: threshold for energy convergence
        """
        if linear_mixing > 0:
            fock_old_a, fock_old_b = None, None
        else:
            adiis = lib.diis.DIIS()

        e_tot_old = None

        print(
            "ITER"
            + " " * 2
            + "ENERGY (MULTIPLET)"
            + " " * 4
            + "ENERGY (S exci)"
            + " " * 8
            + "ENERGY (T)"
            + " " * 12
            + "EDIFF"
        )
        for current_iter in range(maxit):
            ints_3c_a = self.mf.mo_coeff[0].T @ self.ints_3c_ao @ self.mf.mo_coeff[0]
            ints_3c_b = self.mf.mo_coeff[1].T @ self.ints_3c_ao @ self.mf.mo_coeff[1]

            ints_3c_a3 = self.mf.mo_coeff3[0].T @ self.ints_3c_ao @ self.mf.mo_coeff3[0]
            ints_3c_b3 = self.mf.mo_coeff3[1].T @ self.ints_3c_ao @ self.mf.mo_coeff3[1]

            if self.vh_via_OEP or self.space_sym:
                self.get_vh_via_oep(ints_3c_a, ints_3c_b, self.nelec)

            W3 = self.get_W3_charge()
            vref_oep_a = self.get_v_ref_spin_sym(ints_3c_a, ints_3c_b, self.nelec)
            vref_oep_b = vref_oep_a.copy()
            W_a = W_b = self.get_W_spin_sym(W3, ints_3c_a, ints_3c_b, self.nelec, thr_fai_oep)

            X0_a = self.get_X0(ints_3c_a, self.mf.mo_energy[0], self.nelec[0], W_a)
            X0_b = self.get_X0(ints_3c_b, self.mf.mo_energy[1], self.nelec[1], W_b)

            vref_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_a[:])
            vref_ao_b = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_b[:])

            rhs_a = self.get_rhs(
                vref_ao_a, self.vxnl_ao[0], ints_3c_a, self.mf.mo_coeff[0], self.mf.mo_energy[0], self.nelec[0], W_a
            )
            rhs_b = self.get_rhs(
                vref_ao_b, self.vxnl_ao[1], ints_3c_b, self.mf.mo_coeff[1], self.mf.mo_energy[1], self.nelec[1], W_b
            )

            rhs_a3 = self.get_rhs(
                vref_ao_a,
                self.vxnl_ao3[0],
                ints_3c_a3,
                self.mf.mo_coeff3[0],
                self.mf.mo_energy3[0],
                self.nelec[0] + 1,
                W_a,
            )
            rhs_b3 = self.get_rhs(
                vref_ao_b,
                self.vxnl_ao3[1],
                ints_3c_b3,
                self.mf.mo_coeff3[1],
                self.mf.mo_energy3[1],
                self.nelec[1] - 1,
                W_b,
            )

            rhs_a = scipy.linalg.solve(X0_a + X0_b, 0.5 * (rhs_a + rhs_b + rhs_a3 + rhs_b3))
            vrest_oep_a = vrest_oep_b = W_a @ rhs_a

            vrest_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep_a[:])
            vrest_ao_b = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep_b[:])

            self.potentials_test(
                vrest_oep_a,
                vref_oep_a,
                vrest_ao_a,
                vref_ao_a,
                self.vxnl_ao[0],
                self.mf.mo_coeff[0],
                self.nelec[0],
                None,
            )

            h1e = self.mf.get_hcore()
            F_a = h1e + self.vj_ao + vref_ao_a + vrest_ao_a
            F_b = h1e + self.vj_ao + vref_ao_b + vrest_ao_b

            if linear_mixing > 0:
                if fock_old_a is None:
                    fock_old_a = F_a.copy()
                    fock_old_b = F_b.copy()
                else:
                    F_a = (1.0 - linear_mixing) * F_a + linear_mixing * fock_old_a
                    F_b = (1.0 - linear_mixing) * F_b + linear_mixing * fock_old_b
                    fock_old_a = F_a.copy()
                    fock_old_b = F_b.copy()
            else:
                S = self.mf.get_ovlp()
                D_a = self.mf.mo_coeff[0][:, : self.nelec[0]] @ self.mf.mo_coeff[0][:, : self.nelec[0]].T
                D_b = self.mf.mo_coeff[1][:, : self.nelec[1]] @ self.mf.mo_coeff[1][:, : self.nelec[1]].T
                e_a = F_a @ D_a @ S - S @ D_a @ F_a
                e_b = F_b @ D_b @ S - S @ D_b @ F_b
                nao = F_a.shape[0]
                F_combined = adiis.update(
                    np.concatenate([F_a.ravel(), F_b.ravel()]), xerr=np.concatenate([e_a.ravel(), e_b.ravel()])
                )
                F_a = F_combined[: nao * nao].reshape(nao, nao)
                F_b = F_combined[nao * nao :].reshape(nao, nao)

            S = self.mf.get_ovlp()
            mo_energy, mo_coeff = self.mf._eigh(F_a, S)
            self.mf.mo_energy[0], self.mf.mo_coeff[0] = mo_energy, mo_coeff
            mo_energy, mo_coeff = self.mf._eigh(F_b, S)
            self.mf.mo_energy[1], self.mf.mo_coeff[1] = mo_energy, mo_coeff
            self.mf.mo_coeff3 = self.mf.mo_coeff.copy()
            self.mf.mo_energy3 = self.mf.mo_energy.copy()

            self.get_energies_and_potentials()

            if e_tot_old is None:
                print(f"{current_iter:3}  {self.e_multiplet:18.12f}  {self.e_tot:18.12f}  {self.e_tot3:18.12f}")
                e_tot_old = self.e_multiplet
            else:
                print(
                    f"{current_iter:3}  {self.e_multiplet:18.12f}  {self.e_tot:18.12f}  {self.e_tot3:18.12f}  "
                    f"{self.e_multiplet - e_tot_old:18.12f}"
                )
                if abs(e_tot_old - self.e_multiplet) < e_conv_thr:
                    print("SCF converged")
                    self.converged = True
                    self.vref_oep_a = vref_oep_a
                    self.vref_oep_b = vref_oep_b
                    self.vrest_oep_a = vrest_oep_a
                    self.vrest_oep_b = vrest_oep_b
                    break
                e_tot_old = self.e_multiplet

            if current_iter == maxit - 1:
                print("SCF was not converged")
                self.vref_oep_a = vref_oep_a
                self.vref_oep_b = vref_oep_b
                self.vrest_oep_a = vrest_oep_a
                self.vrest_oep_b = vrest_oep_b

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials. Swap orbitals in the end."""

        self.occ, _, self.vj_ao, self.vxnl_ao, terms, aux_terms = self.get_state_ingredients(
            self.mom, self.mf.mo_coeff, self.mf.mo_energy
        )
        self.occ3, _, _, self.vxnl_ao3, terms3, aux_terms3 = self.get_state_ingredients(
            self.mom3, self.mf.mo_coeff3, self.mf.mo_energy3
        )

        self.e_tot = self.total_energy(self.combine_terms([2.0, -1.0], [terms, terms3]))
        self.e_tot3 = self.total_energy(terms3)
        self.e_multiplet = self.total_energy(self.combine_terms([0.5, 0.5], [aux_terms, aux_terms3]))
        self.e_aux = self.e_multiplet

        self.swap_orbitals(self.mf.mo_coeff, self.mf.mo_energy, self.occ)
        self.swap_orbitals(self.mf.mo_coeff3, self.mf.mo_energy3, self.occ3)

    def get_W_spin_sym(self, W3, ints_3c_a, ints_3c_b, nelec, thr_fai_oep):
        r"""
        Performs the last step of the auxiliary basis set preprocessing and returns
        the final transformation matrix for spin-symmetrized case.
        See Sections IIB in J. Chem. Phys. 159, 244109 (2023)
        """
        trans_mat_constraint = self.WII @ W3
        dmat_a = trans_mat_constraint.T @ ints_3c_a[:, nelec[0] :, : nelec[0]].reshape(
            self.naux, (self.nmo - nelec[0]) * nelec[0]
        )
        amat_a = dmat_a @ dmat_a.T
        dmat_b = trans_mat_constraint.T @ ints_3c_b[:, nelec[1] :, : nelec[1]].reshape(
            self.naux, (self.nmo - nelec[1]) * nelec[1]
        )
        amat_b = dmat_b @ dmat_b.T

        amat = 0.5 * (amat_a + amat_b)
        eigs, evecs = scipy.linalg.eigh(amat)
        nsing = (eigs < thr_fai_oep).sum()
        trans_mat = trans_mat_constraint @ evecs[:, nsing:]
        return trans_mat

    def get_v_ref_spin_sym(self, ints_3c_a, ints_3c_b, nelec):
        r"""
        Constructs the Fermi-Amaldi reference potential in spin-symmetrized case.
        See Sections IIB in J. Chem. Phys. 159, 244109 (2023)
        """
        u = np.einsum("ijj->i", ints_3c_a[:, : nelec[0], : nelec[0]])
        u += np.einsum("ijj->i", ints_3c_b[:, : nelec[1], : nelec[1]])
        u *= 0.5
        vcII = self.WII.T @ u
        vc = self.WII @ vcII
        v_ref = -1 / np.dot(self.y, vc) * vc
        return v_ref

    def get_vh_via_oep(self, ints_3c_a, ints_3c_b, nelec):
        """Constructs AO Hartree potential via OEP basis."""
        u = np.einsum("ijj->i", ints_3c_a[:, : nelec[0], : nelec[0]])
        u += np.einsum("ijj->i", ints_3c_b[:, : nelec[1], : nelec[1]])
        vcII = self.WII.T @ u
        vc = self.WII @ vcII
        vc = sum(nelec) / np.dot(self.y, vc) * vc
        self.vj_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vc[:])

    def print_occ_numbers(self, nplus=5):
        """Prints occupation numbers."""
        if self.frac_occ:
            print("Occupation numbers (alpha):", *self.occ[0, : self.mf.nelec[0] + nplus])
            print("Occupation numbers (beta) :", *self.occ[1, : self.mf.nelec[0] + nplus])
            print("Occupation numbers (alpha):", *self.occ3[0, : self.mf.nelec[0] + nplus])
            print("Occupation numbers (beta) :", *self.occ3[1, : self.mf.nelec[0] + nplus])
        else:
            print("Occupation numbers (alpha):", *map(int, self.occ[0, : self.mf.nelec[0] + nplus]))
            print("Occupation numbers (beta) :", *map(int, self.occ[1, : self.mf.nelec[0] + nplus]))
            print("Occupation numbers (alpha):", *map(int, self.occ3[0, : self.mf.nelec[0] + nplus]))
            print("Occupation numbers (beta) :", *map(int, self.occ3[1, : self.mf.nelec[0] + nplus]))
