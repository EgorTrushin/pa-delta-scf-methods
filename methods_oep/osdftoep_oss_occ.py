#!/usr/bin/env python3

import numpy as np
import scipy
from pyscf import lib

from .osdftoep_sta_occ import OSDFTOEP_STA_occ


class OSDFTOEP_OSS_occ(OSDFTOEP_STA_occ):
    r"""OSDFTOEP with occupation numbers for open-shell singlet.

    Args:
        mf: PySCF object with UKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        occ: initial occupation numbers for singlet state to initialize MOM
        occ3: initial occupation numbers for triplet state to initialize MOM
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
    """

    def run(self, maxit=50, thr_fai_oep=5e-2, linear_mixing=-1.0, e_conv_thr=1e-8):
        r"""
        Performs a self-consistent calculation with occupation numbers.

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

        print("ITER" + " " * 8 + "ENERGY" + " " * 13 + "ENERGY (T)" + " " * 12 + "EDIFF")
        for current_iter in range(maxit):
            ints_3c_a = self.mf.mo_coeff[0].T @ self.ints_3c_ao @ self.mf.mo_coeff[0]
            ints_3c_b = self.mf.mo_coeff[1].T @ self.ints_3c_ao @ self.mf.mo_coeff[1]

            if self.vh_via_OEP or self.space_sym:
                self.get_vh_via_oep(ints_3c_a, ints_3c_b, self.nelec, self.occ[0], self.occ[1])

            W3 = self.get_W3_charge()
            vref_oep_a = self.get_v_ref_spin_sym(ints_3c_a, ints_3c_b, self.occ[0], self.occ[1])
            vref_oep_b = vref_oep_a.copy()
            W_a = W_b = self.get_W_spin_sym(
                W3, ints_3c_a, ints_3c_b, self.mf.mo_energy[0], self.occ[0], self.occ[1], thr_fai_oep
            )

            X0_a = self.get_X0(ints_3c_a, self.mf.mo_energy[0], W_a, self.occ[0])
            X0_b = self.get_X0(ints_3c_b, self.mf.mo_energy[1], W_b, self.occ[1])

            vref_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_a[:])
            vref_ao_b = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_b[:])

            rhs_a = self.get_rhs(
                vref_ao_a, self.vxnl_ao[0], ints_3c_a, self.mf.mo_coeff[0], self.mf.mo_energy[0], W_a, self.occ[0]
            )
            rhs_b = self.get_rhs(
                vref_ao_b, self.vxnl_ao[1], ints_3c_b, self.mf.mo_coeff[1], self.mf.mo_energy[1], W_b, self.occ[1]
            )
            rhs_a3 = self.get_rhs(
                vref_ao_a, self.vxnl_ao3[0], ints_3c_a, self.mf.mo_coeff[0], self.mf.mo_energy[0], W_a, self.occ3[0]
            )
            rhs_b3 = self.get_rhs(
                vref_ao_b, self.vxnl_ao3[1], ints_3c_b, self.mf.mo_coeff[1], self.mf.mo_energy[1], W_b, self.occ3[1]
            )

            rhs_a = scipy.linalg.solve(X0_a + X0_b, 2 * rhs_a + 2 * rhs_b - rhs_a3 - rhs_b3)
            vrest_oep_a = vrest_oep_b = W_a @ rhs_a

            vrest_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep_a[:])
            vrest_ao_b = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep_b[:])

            self.potentials_test(vrest_oep_a, vref_oep_a)

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
                dm = self.mf.make_rdm1(self.mf.mo_coeff, self.occ)
                e_a = F_a @ dm[0] @ S - S @ dm[0] @ F_a
                e_b = F_b @ dm[1] @ S - S @ dm[1] @ F_b
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

            self.get_energies_and_potentials()

            e_conv = self.convergence_energy()

            if e_tot_old is None:
                print(f"{current_iter:3}  {self.e_tot:18.12f}  {self.e_tot3:18.12f}")
                e_tot_old = e_conv
            else:
                print(f"{current_iter:3}  {self.e_tot:18.12f}  {self.e_tot3:18.12f}  {e_conv - e_tot_old:18.12f}")
                if abs(e_tot_old - e_conv) < e_conv_thr:
                    print("SCF converged")
                    self.converged = True
                    self.vref_oep_a = vref_oep_a
                    self.vref_oep_b = vref_oep_b
                    self.vrest_oep_a = vrest_oep_a
                    self.vrest_oep_b = vrest_oep_b
                    break
                else:
                    e_tot_old = e_conv

            if current_iter == maxit - 1:
                print("SCF was not converged")
                self.vref_oep_a = vref_oep_a
                self.vref_oep_b = vref_oep_b
                self.vrest_oep_a = vrest_oep_a
                self.vrest_oep_b = vrest_oep_b

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials."""

        self.occ, _, self.vj_ao, self.vxnl_ao, terms, aux_terms = self.get_state_ingredients(
            self.mom, self.mf.mo_coeff, self.mf.mo_energy
        )
        self.occ3, _, _, self.vxnl_ao3, terms3, aux_terms3 = self.get_state_ingredients(
            self.mom3, self.mf.mo_coeff, self.mf.mo_energy
        )

        self.e_tot = self.total_energy(self.combine_terms([2.0, -1.0], [terms, terms3]))
        self.e_tot3 = self.total_energy(terms3)
        self.e_aux = self.total_energy(self.combine_terms([2.0, -1.0], [aux_terms, aux_terms3]))
