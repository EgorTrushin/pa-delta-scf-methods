#!/usr/bin/env python3
"""Examples of DeltaSCF excited-state OEP calculations.

Demonstrates UKS-OEP, SS-KS-OEP, OSS-KS-OEP, and STA-KS-OEP methods for
computing singlet and triplet excitation energies of H2O and N2, using both
orbital-swap (_swap) and occupation-number (_occ) implementations.

All calculations use the PBE functional with density fitting, the aug-cc-pVTZ
orbital basis set, and the aug-cc-pVDZ-RIFIT OEP auxiliary basis. The first
excited singlet (S1) and triplet (T1) states are obtained by promoting one
electron from the HOMO to the LUMO.

Excitation energies are reported in eV relative to the OEP ground state.
"""

from pyscf import dft, gto
from pyscf.data.nist import HARTREE2EV

from methods_oep.dftoep import DFTOEP
from methods_oep.osdftoep_occ import OSDFTOEP_occ
from methods_oep.osdftoep_oss_occ import OSDFTOEP_OSS_occ
from methods_oep.osdftoep_oss_swap import OSDFTOEP_OSS_swap
from methods_oep.osdftoep_sta_occ import OSDFTOEP_STA_occ
from methods_oep.osdftoep_sta_swap import OSDFTOEP_STA_swap
from methods_oep.osdftoep_swap import OSDFTOEP_swap

OEP_BASIS = "aug-cc-pVDZ-RIFIT"


def run_ground_state_oep(mf_rks, space_sym=False):
    """Run a ground-state KS-OEP calculation and return the OEP object.

    Args:
        mf_rks: converged PySCF RKS object.
        space_sym: if True, enforce spatial symmetry.
    """
    mf_oep = DFTOEP(mf_rks, OEP_BASIS, space_sym=space_sym)
    mf_oep.run(maxit=50, thr_fai_oep=0.05)
    return mf_oep


def make_occ_singlet(mf, excitation):
    """Returns occupation numbers for an open-shell singlet excited state.

    Args:
        mf: PySCF UKS object with ground-state orbitals and occupation numbers.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
    """
    occ = mf.mo_occ.copy()
    occ[0][mf.nelec[0] + excitation[0] - 1] = 0
    occ[0][mf.nelec[0] + excitation[1] - 1] = 1
    return occ


def make_occ_triplet(mf, excitation):
    """Returns occupation numbers for a triplet excited state.

    Args:
        mf: PySCF UKS object with ground-state orbitals and occupation numbers.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
    """
    occ = mf.mo_occ.copy()
    occ[1][mf.nelec[1] + excitation[0] - 1] = 0
    occ[0][mf.nelec[0] + excitation[1] - 1] = 1
    return occ


def run_uks_oep_swap(mf, occ_s, occ_t, space_sym=False):
    """Runs UKS-OEP singlet and triplet calculations using orbital-swap.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        occ_s: occupation numbers for the open-shell singlet state.
        occ_t: occupation numbers for the triplet state.
        space_sym: if True, enforce spatial symmetry.
    """
    mf_s = OSDFTOEP_swap(mf, OEP_BASIS, occ_s, spin_sym=False, space_sym=space_sym)
    mf_s.run(maxit=50, thr_fai_oep=0.05)
    mf_t = OSDFTOEP_swap(mf, OEP_BASIS, occ_t, spin_sym=False, space_sym=space_sym)
    mf_t.run(maxit=50, thr_fai_oep=0.0)
    return mf_s, mf_t


def run_ss_ks_oep_swap(mf, occ_s, occ_t, space_sym=False):
    """Runs SS-KS-OEP singlet and triplet calculations using orbital-swap.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        occ_s: occupation numbers for the open-shell singlet state.
        occ_t: occupation numbers for the triplet state.
        space_sym: if True, enforce spatial symmetry.
    """
    mf_s = OSDFTOEP_swap(mf, OEP_BASIS, occ_s, spin_sym=True, space_sym=space_sym)
    mf_s.run(maxit=50, thr_fai_oep=0.05)
    mf_t = OSDFTOEP_swap(mf, OEP_BASIS, occ_t, spin_sym=True, space_sym=space_sym)
    mf_t.run(maxit=50, thr_fai_oep=0.05)
    return mf_s, mf_t


def run_oss_oep_swap(mf, occ_s, occ_t, space_sym=False):
    """Runs an OSS-KS-OEP calculation using orbital-swap.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        occ_s: occupation numbers for the open-shell singlet state.
        occ_t: occupation numbers for the triplet state.
        space_sym: if True, enforce spatial symmetry.
    """
    mf_oss = OSDFTOEP_OSS_swap(mf, OEP_BASIS, occ_s, occ_t, space_sym=space_sym)
    mf_oss.run(maxit=50, thr_fai_oep=0.05)
    return mf_oss


def run_sta_oep_swap(mf, occ_s, occ_t, space_sym=False):
    """Runs a STA-KS-OEP calculation using orbital-swap.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        occ_s: occupation numbers for the open-shell singlet state.
        occ_t: occupation numbers for the triplet state.
        space_sym: if True, enforce spatial symmetry.
    """
    mf_sta = OSDFTOEP_STA_swap(mf, OEP_BASIS, occ_s, occ_t, space_sym=space_sym)
    mf_sta.run(maxit=50, thr_fai_oep=0.05)
    return mf_sta


def run_uks_oep_occ(mf, occ_s, occ_t, space_sym=False):
    """Runs UKS-OEP singlet and triplet calculations using occupation-numbers.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        occ_s: occupation numbers for the open-shell singlet state.
        occ_t: occupation numbers for the triplet state.
        space_sym: if True, enforce spatial symmetry.
    """
    mf_s = OSDFTOEP_occ(mf, OEP_BASIS, occ_s, spin_sym=False, space_sym=space_sym)
    mf_s.run(maxit=50, thr_fai_oep=0.05)
    mf_t = OSDFTOEP_occ(mf, OEP_BASIS, occ_t, spin_sym=False, space_sym=space_sym)
    mf_t.run(maxit=50, thr_fai_oep=0.05)
    return mf_s, mf_t


def run_ss_ks_oep_occ(mf, occ_s, occ_t, space_sym=False):
    """Runs SS-KS-OEP singlet and triplet calculations using occupation-numbers.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        occ_s: occupation numbers for the open-shell singlet state.
        occ_t: occupation numbers for the triplet state.
        space_sym: if True, enforce spatial symmetry.
    """
    mf_s = OSDFTOEP_occ(mf, OEP_BASIS, occ_s, spin_sym=True, space_sym=space_sym)
    mf_s.run(maxit=50, thr_fai_oep=0.05)
    mf_t = OSDFTOEP_occ(mf, OEP_BASIS, occ_t, spin_sym=True, space_sym=space_sym)
    mf_t.run(maxit=50, thr_fai_oep=0.05)
    return mf_s, mf_t


def run_oss_oep_occ(mf, occ_s, occ_t, space_sym=False):
    """Runs an OSS-KS-OEP calculation using occupation-numbers.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        occ_s: occupation numbers for the open-shell singlet state.
        occ_t: occupation numbers for the triplet state.
        space_sym: if True, enforce spatial symmetry.
    """
    mf_oss = OSDFTOEP_OSS_occ(mf, OEP_BASIS, occ_s, occ_t, space_sym=space_sym)
    mf_oss.run(maxit=50, thr_fai_oep=0.05)
    return mf_oss


def run_sta_oep_occ(mf, occ_s, occ_t, space_sym=False):
    """Runs a STA-KS-OEP calculation using occupation-numbers.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        occ_s: occupation numbers for the open-shell singlet state.
        occ_t: occupation numbers for the triplet state.
        space_sym: if True, enforce spatial symmetry.
    """
    mf_sta = OSDFTOEP_STA_occ(mf, OEP_BASIS, occ_s, occ_t, space_sym=space_sym)
    mf_sta.run(maxit=50, thr_fai_oep=0.05)
    return mf_sta


def print_results(label, e_gs, mf_s=None, mf_t=None, mf_oss=None, mf_sta=None):
    """Prints excitation energies in eV for a given OEP method.

    Args:
        label: method name to display.
        e_gs: ground-state OEP total energy in Hartree.
        mf_s: converged UKS-OEP/SS-KS-OEP singlet calculation (optional).
        mf_t: converged UKS-OEP/SS-KS-OEP triplet calculation (optional).
        mf_oss: converged OSS-KS-OEP calculation (optional).
        mf_sta: converged STA-KS-OEP calculation (optional).
    """
    print(f"\n  {label}")
    if mf_s is not None and mf_t is not None:
        exc_s = (2 * mf_s.e_tot - mf_t.e_tot - e_gs) * HARTREE2EV
        exc_t = (mf_t.e_tot - e_gs) * HARTREE2EV
        print(f"    S1 = {exc_s:.4f} eV    T1 = {exc_t:.4f} eV")
    if mf_oss is not None:
        exc_s = (mf_oss.e_tot - e_gs) * HARTREE2EV
        print(f"    S1 = {exc_s:.4f} eV")
    if mf_sta is not None:
        exc_s = (mf_sta.e_tot - e_gs) * HARTREE2EV
        exc_t = (mf_sta.e_tot3 - e_gs) * HARTREE2EV
        print(f"    S1 = {exc_s:.4f} eV    T1 = {exc_t:.4f} eV")


def example_h2o():
    """Runs all OEP methods for the first excited state of water.

    H2O has no spatial symmetry, so space_sym=False. Both _swap and _occ
    implementations are demonstrated and should give identical results.
    """
    print("=" * 50)
    print("H2O / aug-cc-pVTZ / PBE / OEP")
    print("=" * 50)

    geom = "O 0.0 0.0 -0.06990256; H 0.0 0.75753241 0.51843495; H 0.0 -0.75753241 0.51843495"
    mol = gto.M(atom=geom, basis="aug-cc-pVTZ")
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()
    mf_oep_gs = run_ground_state_oep(mf)
    e_gs = mf_oep_gs.e_tot

    mf = mf.to_uks()
    excitation = [0, 1]  # HOMO -> LUMO
    occ_s = make_occ_singlet(mf, excitation)
    occ_t = make_occ_triplet(mf, excitation)

    print("\n--- Orbital-swap ---")
    mf_s, mf_t = run_uks_oep_swap(mf, occ_s, occ_t)
    print_results("UKS-OEP", e_gs, mf_s, mf_t)

    mf_s, mf_t = run_ss_ks_oep_swap(mf, occ_s, occ_t)
    print_results("SS-KS-OEP", e_gs, mf_s, mf_t)

    mf_oss = run_oss_oep_swap(mf, occ_s, occ_t)
    print_results("OSS-KS-OEP", e_gs, mf_oss=mf_oss)

    mf_sta = run_sta_oep_swap(mf, occ_s, occ_t)
    print_results("STA-KS-OEP", e_gs, mf_sta=mf_sta)

    print("\n--- Occupation-numbers ---")
    mf_s, mf_t = run_uks_oep_occ(mf, occ_s, occ_t)
    print_results("UKS-OEP", e_gs, mf_s, mf_t)

    mf_s, mf_t = run_ss_ks_oep_occ(mf, occ_s, occ_t)
    print_results("SS-KS-OEP", e_gs, mf_s, mf_t)

    mf_oss = run_oss_oep_occ(mf, occ_s, occ_t)
    print_results("OSS-KS-OEP", e_gs, mf_oss=mf_oss)

    mf_sta = run_sta_oep_occ(mf, occ_s, occ_t)
    print_results("STA-KS-OEP", e_gs, mf_sta=mf_sta)


def example_n2():
    """Runs all OEP methods for the first excited state of nitrogen.

    N2 has spatial symmetry (D∞h), so space_sym=True is used. Both _swap
    and _occ implementations are demonstrated and should give identical results.
    """
    print("=" * 50)
    print("N2 / aug-cc-pVTZ / PBE / OEP")
    print("=" * 50)

    geom = "N 0.0 0.0 0.55038998; N 0.0 0.0 -0.55038998"
    mol = gto.M(atom=geom, basis="aug-cc-pVTZ")
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()
    mf_oep_gs = run_ground_state_oep(mf, space_sym=True)
    e_gs = mf_oep_gs.e_tot

    mf = mf.to_uks()
    excitation = [0, 1]  # HOMO -> LUMO
    occ_s = make_occ_singlet(mf, excitation)
    occ_t = make_occ_triplet(mf, excitation)

    print("\n--- Orbital-swap ---")
    mf_s, mf_t = run_uks_oep_swap(mf, occ_s, occ_t, space_sym=True)
    print_results("UKS-OEP", e_gs, mf_s, mf_t)

    mf_s, mf_t = run_ss_ks_oep_swap(mf, occ_s, occ_t, space_sym=True)
    print_results("SS-KS-OEP", e_gs, mf_s, mf_t)

    mf_oss = run_oss_oep_swap(mf, occ_s, occ_t, space_sym=True)
    print_results("OSS-KS-OEP", e_gs, mf_oss=mf_oss)

    mf_sta = run_sta_oep_swap(mf, occ_s, occ_t, space_sym=True)
    print_results("STA-KS-OEP", e_gs, mf_sta=mf_sta)

    print("\n--- Occupation-number ---")
    mf_s, mf_t = run_uks_oep_occ(mf, occ_s, occ_t, space_sym=True)
    print_results("UKS-OEP", e_gs, mf_s, mf_t)

    mf_s, mf_t = run_ss_ks_oep_occ(mf, occ_s, occ_t, space_sym=True)
    print_results("SS-KS-OEP", e_gs, mf_s, mf_t)

    mf_oss = run_oss_oep_occ(mf, occ_s, occ_t, space_sym=True)
    print_results("OSS-KS-OEP", e_gs, mf_oss=mf_oss)

    mf_sta = run_sta_oep_occ(mf, occ_s, occ_t, space_sym=True)
    print_results("STA-KS-OEP", e_gs, mf_sta=mf_sta)


if __name__ == "__main__":
    example_h2o()
    print()
    example_n2()
