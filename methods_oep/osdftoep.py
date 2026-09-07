#!/usr/bin/env python3

import itertools
import sys
from copy import deepcopy

import numpy as np
import scipy
from pyscf import dft

from .mom import degenerate_groups
from .osexxoep import OSEXXOEP


class OSDFTOEP(OSEXXOEP):
    r"""
    Implements density functional theory optimized effective potential method for open-shell systems.
    The implementation follows:
    E. Trushin, A. Görling. J. Chem. Theory Comput. 2025, 21, 4, 1667–1683. https://doi.org/10.1021/acs.jctc.4c01477
    E. Trushin, A. Görling. J. Chem. Phys. 155, 054109 (2021). https://doi.org/10.1063/5.0056431
    E. Trushin, A. Görling. J. Chem. Phys. 159, 244109 (2023). https://doi.org/10.1063/5.0171546

    Args:
        mf: PySCF object with RKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
        use_HOMO_condition: whether to use HOMO condition
        xc_homo: exchange-functional to use for evaluation of the HOMO condition
        if_first_iter: whether to calculate HOMO energy for HOMO condition only at first iteration

    The potentials are constructed with the occupation numbers returned by the
    MOM, which are fractional within partially filled degenerate shells if
    frac_occ is True. Total energies, in contrast, are always evaluated with
    integer occupation numbers, averaged over all integer occupation patterns
    that are compatible with the fractional ones.
    """

    frac_occ = False

    def __init__(
        self,
        mf,
        oep_basis,
        use_HOMO_condition=False,
        vh_via_OEP=False,
        space_sym=False,
        spin_sym=False,
        xc_homo=None,
        ip_first_iter=False,
    ):
        self.xc_homo = xc_homo
        self.ip_first_iter = ip_first_iter
        self.ip = None
        super().__init__(mf, oep_basis, use_HOMO_condition, vh_via_OEP, space_sym, spin_sym)

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials."""

        self.mf.max_cycle = 0
        self.mf.run()
        self.e_tot = self.mf.e_tot
        self.e_aux = self.e_tot

        dm = self.mf.make_rdm1()

        ni = dft.numint.NumInt()
        _, _, vxc = ni.nr_uks(self.mf.mol, self.mf.grids, self.mf.xc, dm)

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
        self.E_Coul = np.einsum("ij,ji->", self.vj_ao, dm[0] + dm[1]) * 0.5

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.vxnl_ao = vxc + hyb * self.vxnl_ao
        else:
            self.vxnl_ao = vxc

        if self.use_HOMO_condition and not (self.ip_first_iter and self.ip is not None):
            mf1 = deepcopy(self.mf)
            mf1.xc = self.xc_homo
            mf1.max_cycle = 0
            mf1.kernel(dm0=dm)
            E_n_x = mf1.e_tot

            mol1 = deepcopy(self.mf.mol)
            mol1.spin -= 1
            mol1.charge += 1
            mol1.build(False, False)
            occ = self.mf.mo_occ.copy()
            occ[0, self.nelec[0] - 1] = 0
            mo0 = self.mf.mo_coeff.copy()
            mf1 = dft.UKS(mol1, xc=self.xc_homo).density_fit()
            mf1.max_cycle = 0
            dm1 = mf1.make_rdm1(mo0, occ)
            mf1.kernel(dm0=dm1)
            E_nm1_x_a = mf1.e_tot

            mol1 = deepcopy(self.mf.mol)
            mol1.spin += 1
            mol1.charge += 1
            mol1.build(False, False)
            occ = self.mf.mo_occ.copy()
            occ[1, self.nelec[1] - 1] = 0
            mo0 = self.mf.mo_coeff.copy()
            mf1 = dft.UKS(mol1, xc=self.xc_homo).density_fit()
            mf1.max_cycle = 0
            dm1 = mf1.make_rdm1(mo0, occ)
            mf1.kernel(dm0=dm1)
            E_nm1_x_b = mf1.e_tot

            self.ip = [E_nm1_x_a - E_n_x, E_nm1_x_b - E_n_x]

    def eval_state(self, dm):
        """Evaluates DFT energy contributions and potentials for a given density matrix.

        Returns (e_xc, vj_ao, E_Coul, vxnl_ao). For hybrid functionals e_xc
        includes the exact-exchange contribution. vxnl_ao is the combined
        XC + hybrid-exchange potential used as nonlocal reference.
        """
        ni = dft.numint.NumInt()
        _, e_xc, vxc = ni.nr_uks(self.mf.mol, self.mf.grids, self.mf.xc, dm)

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc)
            if omega == 0:
                vj_ao, vxnl_ao = self.mf.get_jk(self.mf.mol, dm)
            elif alpha == 0:  # LR=0, only SR exchange
                vj_ao = self.mf.get_j(self.mf.mol, dm)
                vxnl_ao = self.mf.get_k(self.mf.mol, dm, omega=-omega)
            else:
                sys.exit("Not implemented case for hybrid functionals")
            E_x = -0.5 * np.einsum("ij,ji->", vxnl_ao[0], dm[0])
            E_x += -0.5 * np.einsum("ij,ji->", vxnl_ao[1], dm[1])
            vxnl_ao *= -1.0
            vxnl_ao = vxc + hyb * vxnl_ao
            e_xc += hyb * E_x
        else:
            vj_ao = self.mf.get_j(dm=dm)
            vxnl_ao = vxc

        vj_ao = vj_ao[0] + vj_ao[1]
        E_Coul = np.einsum("ij,ji->", vj_ao, dm[0] + dm[1]).real * 0.5

        return e_xc, vj_ao, E_Coul, vxnl_ao

    def get_state_ingredients(self, mom, mo_coeff, mo_energy):
        """Constructs the occupation numbers, potentials and energy terms of a single state.

        The energy terms are returned twice. The first set is evaluated with
        integer occupation numbers and the total energies are built from it.
        The second set is evaluated with the fractional occupation numbers and
        monitors the SCF convergence, see convergence_energy().
        """
        occ = mom.get_occ(mo_coeff, mo_energy if self.frac_occ else None)
        dm = self.mf.make_rdm1(mo_coeff, occ)
        e_xc, vj_ao, E_Coul, vxnl_ao = self.eval_state(dm)
        aux_terms = (np.einsum("ij,ji->", self.h1e, dm[0] + dm[1]).real, E_Coul, e_xc)
        terms = self.energy_terms(occ, mo_coeff, mo_energy, aux_terms)

        return occ, dm, vj_ao, vxnl_ao, terms, aux_terms

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

    def energy_terms(self, occ, mo_coeff, mo_energy, aux_terms):
        """Returns the energy terms (e1, E_Coul, e_xc) of a single state.

        The terms are evaluated with integer occupation numbers. If the state
        has fractionally occupied degenerate orbitals, they are averaged over
        all compatible integer occupation patterns. Without such orbitals the
        terms of aux_terms are already the ones with integer occupation
        numbers and are returned unchanged.
        """
        occ_list = self.gen_integer_occ(occ, mo_energy)

        if len(occ_list) == 1:
            return aux_terms

        e1_sum, E_Coul_sum, e_xc_sum = 0.0, 0.0, 0.0
        for occ_int in occ_list:
            dm_int = self.mf.make_rdm1(mo_coeff, occ_int)
            e_xc_int, _, E_Coul_int, _ = self.eval_state(dm_int)
            e1_sum += np.einsum("ij,ji->", self.h1e, dm_int[0] + dm_int[1]).real
            E_Coul_sum += E_Coul_int
            e_xc_sum += e_xc_int

        nocc_list = len(occ_list)
        return e1_sum / nocc_list, E_Coul_sum / nocc_list, e_xc_sum / nocc_list

    def gen_integer_occ(self, occ, mo_energy):
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
            shells = [np.intersect1d(group, frac_idx) for group in degenerate_groups(mo_energy[spin])]
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

    def swap_orbitals(self, mo_coeff, mo_energy, occ):
        """Reorders orbitals so that occupied ones come first (in-place)."""
        for spin in range(2):
            ind = np.argsort(-occ[spin], kind="stable")
            for i in range(mo_coeff.shape[1]):
                mo_coeff[spin, i, :] = mo_coeff[spin, i, ind]
            mo_energy[spin, :] = mo_energy[spin, ind]

    def get_v_ref_w_homo(self, zII, ints_3c, mo_coeff, nelec, vxnl_ao, ip):
        r"""
        Constructs the Fermi-Amaldi reference potential with HOMO condition.
        ip is the ionization potential for the spin channel being processed.
        See Appendix C in J. Chem. Phys. 155 (2021) 054109
        and J. Chem. Theory Comput. 2025, 21, 4, 1667–1683
        """
        u = 2 * np.einsum("ijj->i", ints_3c[:, :nelec, :nelec])
        vcII = self.WII.T @ u
        aux_x = np.zeros([2, 2])
        aux_x[0, 0] = np.dot(self.yII, self.yII)
        aux_x[0, 1] = np.dot(self.yII, vcII)
        aux_x[1, 0] = np.dot(zII, self.yII)
        aux_x[1, 1] = np.dot(zII, vcII)
        aux_y = np.zeros([2])
        aux_y[0] = -1
        aux_y[1] = -ip - (mo_coeff.T @ (self.mf.get_hcore() + self.vj_ao) @ mo_coeff)[nelec - 1, nelec - 1]
        aux_sol = scipy.linalg.solve(aux_x, aux_y)
        vrefII = aux_sol[0] * self.yII + aux_sol[1] * vcII
        vref_oep = self.WII @ vrefII
        return vref_oep

    def get_v_ref_w_homo_spin_sym(self, zII, ints_3c_a, ints_3c_b, mo_coeff_a, nelec, vxnl_ao):
        r"""
        Constructs the Fermi-Amaldi reference potential with HOMO condition
        for spin-symmetrized case.
        See Appendix C in J. Chem. Phys. 155 (2021) 054109
        and J. Chem. Phys. 159, 244109 (2023)
        """
        u = np.einsum("ijj->i", ints_3c_a[:, : nelec[0], : nelec[0]])
        u += np.einsum("ijj->i", ints_3c_b[:, : nelec[1], : nelec[1]])
        vcII = self.WII.T @ u
        aux_x = np.zeros([2, 2])
        aux_x[0, 0] = np.dot(self.yII, self.yII)
        aux_x[0, 1] = np.dot(self.yII, vcII)
        aux_x[1, 0] = np.dot(zII, self.yII)
        aux_x[1, 1] = np.dot(zII, vcII)
        aux_y = np.zeros([2])
        aux_y[0] = -1
        aux_y[1] = (
            -self.ip[0] - (mo_coeff_a.T @ (self.mf.get_hcore() + self.vj_ao) @ mo_coeff_a)[nelec[0] - 1, nelec[0] - 1]
        )
        aux_sol = scipy.linalg.solve(aux_x, aux_y)
        vrefII = aux_sol[0] * self.yII + aux_sol[1] * vcII
        vref_oep = self.WII @ vrefII
        return vref_oep

    def potentials_test(
        self, vrest_oep, vref_oep, vrest_ao, vref_ao, vxnl_ao, mo_coeff, nelec, z, ip=None
    ):  # note that vxnl_ao is unused intentionally
        """Performs consistency checks for potentials.
        ip is the ionization potential for the spin channel being checked; pass None when use_HOMO_condition is False.
        """
        if abs(np.dot(self.y, vrest_oep)) > 1e-12:
            print("Warning! y*vrest_oep =", np.dot(self.y, vrest_oep))
        if abs(np.dot(self.y, vref_oep) + 1.0) > 1e-12:
            print("Warning! y*vref_oep =", np.dot(self.y, vref_oep))
        if self.use_HOMO_condition:
            if abs(np.dot(z, vrest_oep)) > 1e-12:
                print("Warning! z*vrest_oep =", np.dot(z, vrest_oep))
            if abs((mo_coeff.T @ vrest_ao @ mo_coeff)[nelec - 1, nelec - 1]) > 1e-12:
                print("Warning!")
                print(f"v(HOMO) =  {(mo_coeff.T @ vrest_ao @ mo_coeff)[nelec - 1, nelec - 1]:.5f}  (VrestL)")
            v_aux = mo_coeff.T @ (self.mf.get_hcore() + self.vj_ao) @ mo_coeff
            if abs(-v_aux[nelec - 1, nelec - 1] - ip - (mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1]) > 1e-12:
                print("Warning!")
                print(f"v(HOMO) =  {-v_aux[nelec - 1, nelec - 1] - ip:.5f}  (VxNL)")
                print(f"v(HOMO) =  {(mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1]:.5f}  (Vref)")
