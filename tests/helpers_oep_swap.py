from methods_oep.dftoep import DFTOEP
from methods_oep.osdftoep_oss_swap import OSDFTOEP_OSS_swap
from methods_oep.osdftoep_sta_swap import OSDFTOEP_STA_swap
from methods_oep.osdftoep_swap import OSDFTOEP_swap


def ground_state_oep(mf_rks, oep_basis, space_sym=False):
    """Run a ground-state KS-OEP calculation.

    Args:
        mf_rks: converged PySCF RKS object.
        oep_basis: auxiliary basis for the OEP equation.
    """
    mf_oep = DFTOEP(mf_rks, oep_basis, space_sym=space_sym)
    mf_oep.run(maxit=50, thr_fai_oep=0.05, e_conv_thr=1e-8)
    print("Energy of ground-state KS-OEP:", mf_oep.e_tot)
    return mf_oep


def uks_oep_singlet(mf, oep_basis, excitation, space_sym=False):
    """Run a UKS-OEP open-shell singlet excited-state calculation.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        oep_basis: auxiliary basis for the OEP equation.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
    """
    occ = mf.mo_occ.copy()
    occ[0][mf.nelec[0] + excitation[0] - 1] = 0
    occ[0][mf.nelec[0] + excitation[1] - 1] = 1
    mf_oep = OSDFTOEP_swap(mf, oep_basis, occ, space_sym=space_sym)
    mf_oep.run(maxit=50, thr_fai_oep=0.05, e_conv_thr=1e-8)
    mf_oep.print_occ_numbers()
    print("Energy of excited-state singlet UKS-OEP:", mf_oep.e_tot)
    return mf_oep


def uks_oep_triplet(mf, oep_basis, excitation, space_sym=False):
    """Run a UKS-OEP triplet excited-state calculation.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        oep_basis: auxiliary basis for the OEP equation.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
    """
    occ = mf.mo_occ.copy()
    occ[1][mf.nelec[1] + excitation[0] - 1] = 0
    occ[0][mf.nelec[0] + excitation[1] - 1] = 1
    mf_oep = OSDFTOEP_swap(mf, oep_basis, occ, space_sym=space_sym)
    mf_oep.run(maxit=50, thr_fai_oep=0.05, e_conv_thr=1e-8)
    mf_oep.print_occ_numbers()
    print("Energy of excited-state triplet UKS-OEP:", mf_oep.e_tot)
    return mf_oep


def ss_ks_oep_singlet(mf, oep_basis, excitation, space_sym=False):
    """Run a SS-KS-OEP open-shell singlet excited-state calculation.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        oep_basis: auxiliary basis for the OEP equation.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
    """
    occ = mf.mo_occ.copy()
    occ[0][mf.nelec[0] + excitation[0] - 1] = 0
    occ[0][mf.nelec[0] + excitation[1] - 1] = 1
    mf_oep = OSDFTOEP_swap(mf, oep_basis, occ, spin_sym=True, space_sym=space_sym)
    mf_oep.run(maxit=50, thr_fai_oep=0.05, e_conv_thr=1e-9)
    mf_oep.print_occ_numbers()
    print("Energy of excited-state singlet SS-KS-OEP:", mf_oep.e_tot)
    return mf_oep


def ss_ks_oep_triplet(mf, oep_basis, excitation, space_sym=False):
    """Run a SS-KS-OEP triplet excited-state calculation.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        oep_basis: auxiliary basis for the OEP equation.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
    """
    occ = mf.mo_occ.copy()
    occ[1][mf.nelec[1] + excitation[0] - 1] = 0
    occ[0][mf.nelec[0] + excitation[1] - 1] = 1
    mf_oep = OSDFTOEP_swap(mf, oep_basis, occ, spin_sym=True, space_sym=space_sym)
    mf_oep.run(maxit=50, thr_fai_oep=0.05, e_conv_thr=1e-9)
    mf_oep.print_occ_numbers()
    print("Energy of excited-state triplet SS-KS-OEP:", mf_oep.e_tot)
    return mf_oep


def oss_oep(mf, oep_basis, excitation, space_sym=False):
    """Run an OSS-KS-OEP singlet excited-state calculation.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        oep_basis: auxiliary basis for the OEP equation.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
    """
    occ = mf.mo_occ.copy()
    occ[0][mf.nelec[0] + excitation[0] - 1] = 0
    occ[0][mf.nelec[0] + excitation[1] - 1] = 1
    occ3 = mf.mo_occ.copy()
    occ3[1][mf.nelec[1] + excitation[0] - 1] = 0
    occ3[0][mf.nelec[0] + excitation[1] - 1] = 1
    mf_oep = OSDFTOEP_OSS_swap(mf, oep_basis, occ, occ3, space_sym=space_sym)
    mf_oep.run(maxit=50, thr_fai_oep=0.05, e_conv_thr=1e-9)
    mf_oep.print_occ_numbers()
    print("Energy of excited-state singlet OSS-KS-OEP:", mf_oep.e_tot)
    return mf_oep


def sta_oep(mf, oep_basis, excitation, space_sym=False):
    """Run a STA-KS-OEP excited-state calculation.

    Args:
        mf: PySCF UKS object with converged ground-state calculation.
        oep_basis: auxiliary basis for the OEP equation.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
    """
    occ = mf.mo_occ.copy()
    occ[0][mf.nelec[0] + excitation[0] - 1] = 0
    occ[0][mf.nelec[0] + excitation[1] - 1] = 1
    occ3 = mf.mo_occ.copy()
    occ3[1][mf.nelec[1] + excitation[0] - 1] = 0
    occ3[0][mf.nelec[0] + excitation[1] - 1] = 1
    mf_oep = OSDFTOEP_STA_swap(mf, oep_basis, occ, occ3, space_sym=space_sym)
    mf_oep.run(maxit=50, thr_fai_oep=0.05, e_conv_thr=1e-9)
    mf_oep.print_occ_numbers()
    print("Energy of excited-state singlet STA-KS-OEP:", mf_oep.e_tot)
    print("Energy of excited-state triplet STA-KS-OEP:", mf_oep.e_tot3)
    return mf_oep
