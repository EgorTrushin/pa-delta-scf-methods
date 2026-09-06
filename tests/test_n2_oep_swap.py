from helpers_oep_swap import (
    ground_state_oep,
    oss_oep,
    ss_ks_oep_singlet,
    ss_ks_oep_triplet,
    sta_oep,
    uks_oep_singlet,
    uks_oep_triplet,
)
from pyscf import dft, gto

HA_TO_EV = 27.2114
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
    exc_s = (2 * mf_s.e_tot - mf_t.e_tot - E_GS) * HA_TO_EV
    exc_t = (mf_t.e_tot - E_GS) * HA_TO_EV
    assert abs(exc_s - 8.570428184985007) < 1e-5
    assert abs(exc_t - 7.552172162309417) < 1e-5

    mf_s_ss = ss_ks_oep_singlet(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    mf_t_ss = ss_ks_oep_triplet(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    exc_s = (2 * mf_s_ss.e_tot - mf_t_ss.e_tot - E_GS) * HA_TO_EV
    exc_t = (mf_t_ss.e_tot - E_GS) * HA_TO_EV
    assert abs(exc_s - 8.652254494566888) < 1e-5
    assert abs(exc_t - 7.5983838164810455) < 1e-5

    mf_oss = oss_oep(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    exc_s = (mf_oss.e_tot - E_GS) * HA_TO_EV
    assert abs(exc_s - 8.64898885959935) < 1e-5

    mf_sta = sta_oep(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    exc_s = (mf_sta.e_tot - E_GS) * HA_TO_EV
    exc_t = (mf_sta.e_tot3 - E_GS) * HA_TO_EV
    print(exc_s)
    print(exc_t)
    assert abs(exc_s - 8.652566892466272) < 1e-5
    assert abs(exc_t - 7.59882562788572) < 1e-5
