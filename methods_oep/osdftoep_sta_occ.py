#!/usr/bin/env python3

import sys
import scipy
import numpy as np
from pyscf import dft, lib
from .osdftoep import OSDFTOEP
from .mom import MOM


class OSDFTOEP_STA_occ(OSDFTOEP):
    r"""OSDFTOEP for excited states with state-averaged KS method with occupation numbers.

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
        self.frac_occ = space_sym
        self.mom = MOM(mf.mo_coeff.copy(), occ, mf.get_ovlp(), frac_occ=self.frac_occ)
        self.mom3 = MOM(mf.mo_coeff.copy(), occ3, mf.get_ovlp(), frac_occ=self.frac_occ)
        super().__init__(mf, oep_basis, vh_via_OEP=vh_via_OEP, space_sym=space_sym)

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
                dm = self.mf.make_rdm1(self.mf.mo_coeff, self.occ)
                e_a = F_a @ dm[0] @ S - S @ dm[0] @ F_a
                e_b = F_b @ dm[1] @ S - S @ dm[1] @ F_b
                nao = F_a.shape[0]
                F_combined = adiis.update(
                    np.concatenate([F_a.ravel(), F_b.ravel()]),
                    xerr=np.concatenate([e_a.ravel(), e_b.ravel()])
                )
                F_a = F_combined[:nao * nao].reshape(nao, nao)
                F_b = F_combined[nao * nao:].reshape(nao, nao)

            S = self.mf.get_ovlp()
            mo_energy, mo_coeff = scipy.linalg.eigh(F_a, S)
            self.mf.mo_energy[0], self.mf.mo_coeff[0] = mo_energy, mo_coeff
            mo_energy, mo_coeff = scipy.linalg.eigh(F_b, S)
            self.mf.mo_energy[1], self.mf.mo_coeff[1] = mo_energy, mo_coeff

            self.get_energies_and_potentials()

            if e_tot_old is None:
                print(f"{current_iter:3}  {self.e_multiplet:18.12f}  {self.e_tot:18.12f}  {self.e_tot3:18.12f}")
                e_tot_old = self.e_multiplet
            else:
                print(
                    f"{current_iter:3}  {self.e_multiplet:18.12f}  {self.e_tot:18.12f}  {self.e_tot3:18.12f}  {self.e_multiplet - e_tot_old:18.12f}"
                )
                if abs(e_tot_old - self.e_multiplet) < e_conv_thr:
                    print("SCF converged")
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
        """Determines energy contributions and potentials."""

        self.occ = self.mom.get_occ(self.mf.mo_coeff, self.mf.mo_energy if self.frac_occ else None)
        dm = self.mf.make_rdm1(self.mf.mo_coeff, self.occ)
        self.occ3 = self.mom3.get_occ(self.mf.mo_coeff, self.mf.mo_energy if self.frac_occ else None)
        dm3 = self.mf.make_rdm1(self.mf.mo_coeff, self.occ3)

        e_xc, self.vj_ao, self.E_Coul, self.vxnl_ao, self.E_x, omega, alpha, hyb = self.eval_state(dm)
        e_xc3, self.vj_ao3, self.E_Coul3, self.vxnl_ao3, self.E_x3, _, _, _ = self.eval_state(dm3)

        h1e = self.mf.get_hcore()
        e1 = 2.0 * np.einsum("ij,ji->", h1e, dm[0] + dm[1]) - np.einsum("ij,ji->", h1e, dm3[0] + dm3[1])
        self.e_tot = e1 + 2 * self.E_Coul - self.E_Coul3 + 2 * e_xc - e_xc3 + self.mf.energy_nuc()
        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.e_tot += hyb * (2 * self.E_x - self.E_x3)

        e1 = np.einsum("ij,ji->", h1e, dm3[0] + dm3[1])
        self.e_tot3 = e1 + self.E_Coul3 + e_xc3 + self.mf.energy_nuc()
        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.e_tot3 += hyb * self.E_x3

        e1 = 0.5 * np.einsum("ij,ji->", h1e, dm[0] + dm[1]) + 0.5 * np.einsum("ij,ji->", h1e, dm3[0] + dm3[1])
        self.e_multiplet = e1 + 0.5 * (self.E_Coul + self.E_Coul3) + 0.5 * (e_xc + e_xc3) + self.mf.energy_nuc()
        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.e_multiplet += hyb * 0.5 * (self.E_x + self.E_x3)

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

                self.E_x_p = -0.5 * np.einsum("ij,ji->", vxnl_ao_p[0], dm_p[0])
                self.E_x_p += -0.5 * np.einsum("ij,ji->", vxnl_ao_p[1], dm_p[1])
                
                vj_ao_p = vj_ao_p[0] + vj_ao_p[1]
                E_Coul_p = np.einsum("ij,ji->", vj_ao_p, dm_p[0] + dm_p[1]).real * 0.5
                self.E_Coul = (1 - hyb) * self.E_Coul + hyb * E_Coul_p

                occ_p = self.mom3.get_occ(self.mf.mo_coeff)
                dm3_p = self.mf.make_rdm1(self.mf.mo_coeff, occ_p)
                if omega == 0:
                    vj_ao_p, vxnl_ao_p = self.mf.get_jk(self.mf.mol, dm3_p)
                elif alpha == 0:  # LR=0, only SR exchange
                    vj_ao_p = self.mf.get_j(self.mf.mol, dm3_p)
                    vxnl_ao_p = self.mf.get_k(self.mf.mol, dm3_p, omega=-omega)
                else:
                    sys.exit("Not implemented case for hybrid functionals")

                self.E_x3_p = -0.5 * np.einsum("ij,ji->", vxnl_ao_p[0], dm3_p[0])
                self.E_x3_p += -0.5 * np.einsum("ij,ji->", vxnl_ao_p[1], dm3_p[1])
                
                vj_ao_p = vj_ao_p[0] + vj_ao_p[1]
                E_Coul3_p = np.einsum("ij,ji->", vj_ao_p, dm3_p[0] + dm3_p[1]).real * 0.5
                self.E_Coul3 = (1 - hyb) * self.E_Coul3 + hyb * E_Coul3_p

                e1 = 2.0 * np.einsum("ij,ji->", h1e, dm_p[0] + dm_p[1]) - np.einsum("ij,ji->", h1e, dm3_p[0] + dm3_p[1])
                self.e_tot = e1 + 2 * self.E_Coul - self.E_Coul3 + 2 * e_xc - e_xc3 + self.mf.energy_nuc() + hyb * (2 * self.E_x_p - self.E_x3_p)

                e1 = np.einsum("ij,ji->", h1e, dm3_p[0] + dm3_p[1])
                self.e_tot3 = e1 + self.E_Coul3 + e_xc3 + self.mf.energy_nuc() + hyb * self.E_x3_p

                e1 = 0.5 * np.einsum("ij,ji->", h1e, dm_p[0] + dm_p[1]) + 0.5 * np.einsum("ij,ji->", h1e, dm3_p[0] + dm3_p[1])
                self.e_multiplet = e1 + 0.5 * (self.E_Coul + self.E_Coul3) + 0.5 * (e_xc + e_xc3) + self.mf.energy_nuc() + 0.5 * hyb * (self.E_x_p + self.E_x3_p)


    def get_X0(self, ints_3c, mo_energy, trans_mat, occ):
        """Constructs response matrix with occupation numbers."""
        e_diff = mo_energy[:, None] - mo_energy[None, :]
        e_diff = np.where(abs(e_diff) > 1e-6, e_diff, np.sign(e_diff + 1e-12) * 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            e_ov = occ[None, :] / e_diff
            np.fill_diagonal(e_ov, 0.0)

        Pia = ints_3c * e_ov
        X0 = np.einsum("Pia, Qia -> PQ", Pia, ints_3c)
        X0 = trans_mat.T @ X0 @ trans_mat
        return X0

    def get_rhs(self, v_ref_ao, vxnl_ao, ints_3c, mo_coeff, mo_energy, trans_mat, occ):
        """Constructs RHS of DFT-OEP equation with occupation numbers."""
        vxmo = mo_coeff.T @ (vxnl_ao - v_ref_ao) @ mo_coeff

        e_diff = mo_energy[:, None] - mo_energy[None, :]
        e_diff = np.where(abs(e_diff) > 1e-6, e_diff, np.sign(e_diff + 1e-12) * 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            e_ov = occ[None, :] / e_diff
            np.fill_diagonal(e_ov, 0.0)

        aux = np.einsum("ij,aij->a", vxmo * e_ov, ints_3c)
        rhs = trans_mat.T @ aux
        return rhs

    def get_W_spin_sym(self, W3, ints_3c_a, ints_3c_b, eig, occa, occb, thr_fai_oep):
        r"""
        Performs the last step of the auxiliary basis set preprocessing and returns
        the final transformation matrix for spin-symmetrized case.
        See Sections IIB in J. Chem. Phys. 159, 244109 (2023)
        With occupation numbers.
        """
        trans_mat_constraint = self.WII @ W3

        dmat_a = trans_mat_constraint.T @ ints_3c_a.reshape(self.naux, self.nmo * self.nmo)
        dmat_orig = dmat_a.copy()
        dmat_a = dmat_a.reshape(trans_mat_constraint.shape[1], self.nmo, self.nmo)
        factors = occa[:, None] * np.sign(eig[None, :] - eig[:, None])
        np.fill_diagonal(factors, 0.0)
        dmat_a *= factors[None, :, :]
        dmat_a = dmat_a.reshape(trans_mat_constraint.shape[1], self.nmo * self.nmo)
        amat_a = dmat_a @ dmat_orig.T

        dmat_b = trans_mat_constraint.T @ ints_3c_b.reshape(self.naux, self.nmo * self.nmo)
        dmat_orig = dmat_b.copy()
        dmat_b = dmat_b.reshape(trans_mat_constraint.shape[1], self.nmo, self.nmo)
        factors = occb[:, None] * np.sign(eig[None, :] - eig[:, None])
        np.fill_diagonal(factors, 0.0)
        dmat_b *= factors[None, :, :]
        dmat_b = dmat_b.reshape(trans_mat_constraint.shape[1], self.nmo * self.nmo)
        amat_b = dmat_b @ dmat_orig.T

        amat = 0.5 * (amat_a + amat_b)
        eigs, evecs = scipy.linalg.eigh(amat)
        nsing = (eigs < thr_fai_oep).sum()
        trans_mat = trans_mat_constraint @ evecs[:, nsing:]
        return trans_mat

    def get_v_ref_spin_sym(self, ints_3c_a, ints_3c_b, occa, occb):
        r"""
        Constructs the Fermi-Amaldi reference potential in spin-symmetrized case.
        See Sections IIB in J. Chem. Phys. 159, 244109 (2023)
        With occupation numbers.
        """
        u = np.einsum("j,ijj->i", occa, ints_3c_a)
        u += np.einsum("j,ijj->i", occb, ints_3c_b)
        u *= 0.5
        vcII = self.WII.T @ u
        vc = self.WII @ vcII
        v_ref = -1 / np.dot(self.y, vc) * vc
        return v_ref

    def get_vh_via_oep(self, ints_3c_a, ints_3c_b, nelec, occa, occb):
        """Constructs AO Hartree potential via OEP basis with occupation numbers."""
        u = np.einsum("j,ijj->i", occa, ints_3c_a)
        u += np.einsum("j,ijj->i", occb, ints_3c_b)
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
