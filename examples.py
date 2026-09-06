#!/usr/bin/env python3
"""Examples of DeltaSCF excited-state calculations.

Demonstrates UKS, pa-SS-KS, pa-OSS-KS, and pa-STA-KS methods for computing
singlet and triplet excitation energies of H2O and N2.

All calculations use the PBE functional with density fitting and the
aug-cc-pVTZ basis set. The first excited singlet (S1) and triplet (T1)
states are obtained by promoting one electron from the HOMO to the LUMO.

Excitation energies are reported in eV relative to the ground state.
"""

from pyscf import dft, gto
from pyscf.data.nist import HARTREE2EV

from methods.pa_OSS_KS import pa_OSS_KS
from methods.pa_SS_KS import pa_SS_KS
from methods.pa_STA_KS import pa_STA_KS
from methods.UKS import UKS


def make_occ_singlet(mf, excitation):
    """Returns occupation numbers for an open-shell singlet excited state.

    Promotes an alpha electron from HOMO+excitation[0]-1 to
    LUMO+excitation[1]-1, leaving both spins with one unpaired electron
    in the same spatial orbital.

    Args:
        mf: PySCF UKS object with ground-state orbitals and occupation numbers
        excitation: [from, to] indices relative to HOMO/LUMO (1-based)
    """
    occ = mf.mo_occ.copy()
    occ[0][mf.nelec[0] + excitation[0] - 1] = 0
    occ[0][mf.nelec[0] + excitation[1] - 1] = 1
    return occ


def make_occ_triplet(mf, excitation):
    """Returns occupation numbers for a triplet excited state.

    Promotes a beta electron from HOMO+excitation[0]-1 and adds an alpha
    electron at LUMO+excitation[1]-1, giving a triplet spin configuration.

    Args:
        mf: PySCF UKS object with ground-state orbitals and occupation numbers
        excitation: [from, to] indices relative to HOMO/LUMO (1-based)
    """
    occ = mf.mo_occ.copy()
    occ[1][mf.nelec[1] + excitation[0] - 1] = 0
    occ[0][mf.nelec[0] + excitation[1] - 1] = 1
    return occ


def run_uks(mf, occ_s, occ_t, frac_occ):
    """Runs UKS calculations for the singlet and triplet excited states.

    The singlet excitation energy is obtained via the formula:
        E(S1) = 2*E(UKS_singlet) - E(UKS_triplet) - E(ground state)
    which removes the spin contamination from the mixed-spin UKS state.

    Args:
        mf: PySCF UKS object with converged ground-state calculation
        occ_s: occupation numbers for the open-shell singlet state
        occ_t: occupation numbers for the triplet state
        frac_occ: if True, fractional occupation numbers are used for
                  degenerate orbitals
    """
    mf_s = UKS(mf, occ_s, frac_occ)
    mf_s.run(verb=False)
    mf_t = UKS(mf, occ_t, frac_occ)
    mf_t.run(verb=False)
    return mf_s, mf_t


def run_pa_ss_ks(mf, occ_s, occ_t, frac_occ):
    """Runs pa-SS-KS calculations for the singlet and triplet excited states.

    Same workflow as UKS but uses spin-averaged exchange-correlation
    potentials, which improves the description of open-shell singlet states.

    Args:
        mf: PySCF UKS object with converged ground-state calculation
        occ_s: occupation numbers for the open-shell singlet state
        occ_t: occupation numbers for the triplet state
        frac_occ: if True, fractional occupation numbers are used for
                  degenerate orbitals
    """
    mf_s = pa_SS_KS(mf, occ_s, frac_occ)
    mf_s.run(verb=False)
    mf_t = pa_SS_KS(mf, occ_t, frac_occ)
    mf_t.run(verb=False)
    return mf_s, mf_t


def run_oss_ks(mf, occ_s, occ_t, frac_occ):
    """Runs an OSS-KS calculation for the open-shell singlet excited state.

    Uses a composite density matrix (2*dm_singlet - dm_triplet) to construct
    a pure singlet state, avoiding spin contamination in a singlet calculation.

    Args:
        mf: PySCF UKS object with converged ground-state calculation
        occ_s: occupation numbers for the open-shell singlet state
        occ_t: occupation numbers for the triplet state
        frac_occ: if True, fractional occupation numbers are used for
                  degenerate orbitals
    """
    mf_oss = pa_OSS_KS(mf, occ_s, occ_t, frac_occ)
    mf_oss.run(verb=False)
    return mf_oss


def run_sta_ks(mf, occ_s, occ_t, frac_occ):
    """Runs a STA-KS calculation for both singlet and triplet excited states.

    Optimizes orbitals for a state-averaged potential and extracts singlet
    and triplet energies in a single SCF procedure. The singlet energy is
    obtained from the OSS formula applied to the converged state-averaged
    orbitals.

    Args:
        mf: PySCF UKS object with converged ground-state calculation
        occ_s: occupation numbers for the open-shell singlet state
        occ_t: occupation numbers for the triplet state
        frac_occ: if True, fractional occupation numbers are used for
                  degenerate orbitals
    """
    mf_sta = pa_STA_KS(mf, occ_s, occ_t, frac_occ)
    mf_sta.run(verb=False)
    return mf_sta


def print_results(label, e_gs, mf_s=None, mf_t=None, mf_oss=None, mf_sta=None):
    """Prints excitation energies in eV for a given method.

    Args:
        label: method name to display
        e_gs: ground-state total energy in Hartree
        mf_s: converged UKS/SA-KS singlet calculation (optional)
        mf_t: converged UKS/SA-KS triplet calculation (optional)
        mf_oss: converged OSS-KS calculation (optional)
        mf_sta: converged STA-KS calculation (optional)
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
        exc_s = (mf_sta.e_tot_oss - e_gs) * HARTREE2EV
        exc_t = (mf_sta.e_tot_t - e_gs) * HARTREE2EV
        print(f"    S1 = {exc_s:.4f} eV    T1 = {exc_t:.4f} eV")


def example_h2o():
    """Runs all four methods for the first excited state of water.

    H2O has a non-degenerate ground state, so integer occupation numbers
    (frac_occ=False) are sufficient.
    """
    print("=" * 50)
    print("H2O / aug-cc-pVTZ / PBE")
    print("=" * 50)

    geom = "O 0.0 0.0 -0.06990256; H 0.0 0.75753241 0.51843495; H 0.0 -0.75753241 0.51843495"
    mol = gto.M(atom=geom, basis="aug-cc-pVTZ")
    mol.verbose = 0
    mol.build()

    # Ground-state RKS calculation, converted to UKS for excited-state use
    mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()
    mf = mf.to_uks()

    e_gs = mf.e_tot
    frac_occ = False
    excitation = [0, 1]  # HOMO -> LUMO

    occ_s = make_occ_singlet(mf, excitation)
    occ_t = make_occ_triplet(mf, excitation)

    mf_s, mf_t = run_uks(mf, occ_s, occ_t, frac_occ)
    print_results("UKS", e_gs, mf_s, mf_t)

    mf_s_sa, mf_t_sa = run_pa_ss_ks(mf, occ_s, occ_t, frac_occ)
    print_results("pa-SS-KS", e_gs, mf_s_sa, mf_t_sa)

    mf_oss = run_oss_ks(mf, occ_s, occ_t, frac_occ)
    print_results("pa-OSS-KS", e_gs, mf_oss=mf_oss)

    mf_sta = run_sta_ks(mf, occ_s, occ_t, frac_occ)
    print_results("pa-STA-KS", e_gs, mf_sta=mf_sta)


def example_n2():
    """Runs all four methods for the first excited state of nitrogen.

    N2 has degenerate HOMO and LUMO orbitals, so fractional occupation
    numbers (frac_occ=True) are used to handle orbital degeneracy correctly.
    """
    print("=" * 50)
    print("N2 / aug-cc-pVTZ / PBE")
    print("=" * 50)

    geom = "N 0.0 0.0 0.55038998; N 0.0 0.0 -0.55038998"
    mol = gto.M(atom=geom, basis="aug-cc-pVTZ")
    mol.verbose = 0
    mol.build()

    # Ground-state RKS calculation, converted to UKS for excited-state use
    mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()
    mf = mf.to_uks()

    e_gs = mf.e_tot
    frac_occ = True
    excitation = [0, 1]  # HOMO -> LUMO

    occ_s = make_occ_singlet(mf, excitation)
    occ_t = make_occ_triplet(mf, excitation)

    mf_s, mf_t = run_uks(mf, occ_s, occ_t, frac_occ)
    print_results("UKS", e_gs, mf_s, mf_t)

    mf_s_sa, mf_t_sa = run_pa_ss_ks(mf, occ_s, occ_t, frac_occ)
    print_results("pa-SS-KS", e_gs, mf_s_sa, mf_t_sa)

    mf_oss = run_oss_ks(mf, occ_s, occ_t, frac_occ)
    print_results("pa-OSS-KS", e_gs, mf_oss=mf_oss)

    mf_sta = run_sta_ks(mf, occ_s, occ_t, frac_occ)
    print_results("pa-STA-KS", e_gs, mf_sta=mf_sta)


if __name__ == "__main__":
    example_h2o()
    print()
    example_n2()
