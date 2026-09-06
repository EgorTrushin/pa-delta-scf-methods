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
    geom = "O 0.0 0.0 -0.06990256; H 0.0 0.75753241 0.51843495; H 0.0 -0.75753241 0.51843495"
    mol = gto.M(atom=geom, basis="aug-cc-pVTZ")
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()
    mf_oep_gs = ground_state_oep(mf, OEP_BASIS)
    E_GS = mf_oep_gs.e_tot

    mf = mf.to_uks()

    mf_s = uks_oep_singlet(mf, OEP_BASIS, excitation=[0, 1])
    mf_t = uks_oep_triplet(mf, OEP_BASIS, excitation=[0, 1])
    exc_s = (2 * mf_s.e_tot - mf_t.e_tot - E_GS) * HARTREE2EV
    exc_t = (mf_t.e_tot - E_GS) * HARTREE2EV
    assert abs(exc_s - 7.469254762369095) < 1e-6
    assert abs(exc_t - 7.160895070031547) < 1e-6

    mf_s_ss = ss_ks_oep_singlet(mf, OEP_BASIS, excitation=[0, 1])
    mf_t_ss = ss_ks_oep_triplet(mf, OEP_BASIS, excitation=[0, 1])
    exc_s = (2 * mf_s_ss.e_tot - mf_t_ss.e_tot - E_GS) * HARTREE2EV
    exc_t = (mf_t_ss.e_tot - E_GS) * HARTREE2EV
    assert abs(exc_s - 7.678974068326049) < 1e-6
    assert abs(exc_t - 7.2072315161835165) < 1e-6

    mf_oss = oss_oep(mf, OEP_BASIS, excitation=[0, 1])
    exc_s = (mf_oss.e_tot - E_GS) * HARTREE2EV
    assert abs(exc_s - 7.675710490573158) < 1e-6

    mf_sta = sta_oep(mf, OEP_BASIS, excitation=[0, 1])
    exc_s = (mf_sta.e_tot - E_GS) * HARTREE2EV
    exc_t = (mf_sta.e_tot3 - E_GS) * HARTREE2EV
    assert abs(exc_s - 7.677581243166711) < 1e-6
    assert abs(exc_t - 7.208260698152765) < 1e-6


if __name__ == "__main__":
    test_answer()
