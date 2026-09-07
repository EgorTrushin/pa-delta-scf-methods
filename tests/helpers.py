from methods.pa_OSS_KS import pa_OSS_KS
from methods.pa_SS_KS import pa_SS_KS
from methods.pa_STA_KS import pa_STA_KS
from methods.UKS import UKS


def excited_state_singlet(mf_gs, frac_occ, excitation):
    """Run a UKS open-shell singlet excited-state calculation."""
    occ = mf_gs.mo_occ.copy()
    occ[0][mf_gs.nelec[0] + excitation[0] - 1] = 0
    occ[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    mf_es = UKS(mf_gs, occ, frac_occ)
    mf_es.run(verb=False)
    if not mf_es.converged:
        print("UKS excited-state singlet calculation did not converge")
    print("Energy of excited-state singlet (UKS):", mf_es.e_tot)
    mf_es.print_occ_numbers()
    return mf_es


def triplet(mf_gs, frac_occ, excitation):
    """Run a UKS triplet excited-state calculation."""
    occ = mf_gs.mo_occ.copy()
    occ[1][mf_gs.nelec[1] + excitation[0] - 1] = 0
    occ[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    mf_es = UKS(mf_gs, occ, frac_occ)
    mf_es.run(verb=False)
    if not mf_es.converged:
        print("UKS excited-state triplet calculation did not converge")
    print("Energy of excited-state triplet (UKS):", mf_es.e_tot)
    mf_es.print_occ_numbers()
    return mf_es


def excited_state_singlet_sa(mf_gs, frac_occ, excitation):
    """Run a pa-SS-KS open-shell singlet excited-state calculation."""
    occ = mf_gs.mo_occ.copy()
    occ[0][mf_gs.nelec[0] + excitation[0] - 1] = 0
    occ[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    mf_es = pa_SS_KS(mf_gs, occ, frac_occ)
    mf_es.run(verb=False)
    if not mf_es.converged:
        print("pa-SS-KS excited-state singlet calculation did not converge")
    print("Energy of excited-state singlet (pa-SS-KS):", mf_es.e_tot)
    mf_es.print_occ_numbers()
    return mf_es


def triplet_sa(mf_gs, frac_occ, excitation):
    """Run a pa-SS-KS triplet excited-state calculation."""
    occ = mf_gs.mo_occ.copy()
    occ[1][mf_gs.nelec[1] + excitation[0] - 1] = 0
    occ[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    mf_es = pa_SS_KS(mf_gs, occ, frac_occ)
    mf_es.run(verb=False)
    if not mf_es.converged:
        print("pa-SS-KS excited-state triplet calculation did not converge")
    print("Energy of excited-state triplet (pa-SS-KS):", mf_es.e_tot)
    mf_es.print_occ_numbers()
    return mf_es


def oss(mf_gs, frac_occ, excitation):
    """Run a pa-OSS-KS singlet excited-state calculation."""
    occ1 = mf_gs.mo_occ.copy()
    occ1[0][mf_gs.nelec[0] + excitation[0] - 1] = 0
    occ1[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    occ3 = mf_gs.mo_occ.copy()
    occ3[1][mf_gs.nelec[1] + excitation[0] - 1] = 0
    occ3[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    mf_es = pa_OSS_KS(mf_gs, occ1, occ3, frac_occ)
    mf_es.run(verb=False)
    if not mf_es.converged:
        print("pa-OSS-KS excited-state calculation did not converge")
    print("Energy of excited-state singlet (pa-OSS-KS):", mf_es.e_tot)
    mf_es.print_occ_numbers()
    return mf_es


def sta(mf_gs, frac_occ, excitation):
    """Run a pa-STA-KS excited-state calculation."""
    occ1 = mf_gs.mo_occ.copy()
    occ1[0][mf_gs.nelec[0] + excitation[0] - 1] = 0
    occ1[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    occ3 = mf_gs.mo_occ.copy()
    occ3[1][mf_gs.nelec[1] + excitation[0] - 1] = 0
    occ3[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    mf_es = pa_STA_KS(mf_gs, occ1, occ3, frac_occ)
    mf_es.run(verb=False)
    if not mf_es.converged:
        print("pa-STA-KS excited-state calculation did not converge")
    print("Energy of excited-state singlet (pa-STA-KS):", mf_es.e_tot_oss)
    print("Energy of excited-state triplet (pa-STA-KS):", mf_es.e_tot_t)
    mf_es.print_occ_numbers()
    return mf_es
