from helpers_oep_occ import (
    ground_state_oep,
    oss_oep,
    ss_ks_oep_singlet,
    ss_ks_oep_triplet,
    sta_oep,
    uks_oep_singlet,
    uks_oep_triplet,
)
from pyscf import dft, gto
from pyscf.data.nist import HARTREE2EV

OEP_BASIS = "aug-cc-pVDZ-RIFIT"


def test_answer():
    geom = "N 0.0 0.0 0.55038998; N 0.0 0.0 -0.55038998"
    mol = gto.M(atom=geom, basis="aug-cc-pVTZ")
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()
    mf_oep_gs = ground_state_oep(mf, OEP_BASIS, space_sym=True)
    E_GS = mf_oep_gs.e_tot

    mf = mf.to_uks()

    mf_s = uks_oep_singlet(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    mf_t = uks_oep_triplet(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    exc_s = (2 * mf_s.e_tot - mf_t.e_tot - E_GS) * HARTREE2EV
    exc_t = (mf_t.e_tot - E_GS) * HARTREE2EV
    assert abs(exc_s - 8.483734542327072) < 1e-5
    assert abs(exc_t - 7.519229347980636) < 1e-5

    mf_s_ss = ss_ks_oep_singlet(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    mf_t_ss = ss_ks_oep_triplet(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    exc_s = (2 * mf_s_ss.e_tot - mf_t_ss.e_tot - E_GS) * HARTREE2EV
    exc_t = (mf_t_ss.e_tot - E_GS) * HARTREE2EV
    assert abs(exc_s - 8.560365264833868) < 1e-5
    assert abs(exc_t - 7.564842895917501) < 1e-5

    mf_oss = oss_oep(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    exc_s = (mf_oss.e_tot - E_GS) * HARTREE2EV
    assert abs(exc_s - 8.557513669473495) < 1e-5

    mf_sta = sta_oep(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    exc_s = (mf_sta.e_tot - E_GS) * HARTREE2EV
    exc_t = (mf_sta.e_tot3 - E_GS) * HARTREE2EV
    assert abs(exc_s - 8.560724652361401) < 1e-5
    assert abs(exc_t - 7.565200391573010) < 1e-5
