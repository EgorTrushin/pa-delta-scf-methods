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
from pyscf.data.nist import HARTREE2EV

OEP_BASIS = "aug-cc-pVDZ-RIFIT"


# N2 is run with space_sym=True, where the OEP equations of this implementation treat the
# partially filled degenerate shell as one integer configuration. The reference values are
# therefore the approximate treatment; they agree with the occupation-number ones to ~4e-6 eV.
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
    assert abs(exc_s - 8.483730710883945) < 1e-5
    assert abs(exc_t - 7.519228801428434) < 1e-5

    mf_s_ss = ss_ks_oep_singlet(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    mf_t_ss = ss_ks_oep_triplet(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    exc_s = (2 * mf_s_ss.e_tot - mf_t_ss.e_tot - E_GS) * HARTREE2EV
    exc_t = (mf_t_ss.e_tot - E_GS) * HARTREE2EV
    assert abs(exc_s - 8.560369001520750) < 1e-5
    assert abs(exc_t - 7.564843594153942) < 1e-5

    mf_oss = oss_oep(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    exc_s = (mf_oss.e_tot - E_GS) * HARTREE2EV
    assert abs(exc_s - 8.557515853643634) < 1e-5

    mf_sta = sta_oep(mf, OEP_BASIS, excitation=[0, 1], space_sym=True)
    exc_s = (mf_sta.e_tot - E_GS) * HARTREE2EV
    exc_t = (mf_sta.e_tot3 - E_GS) * HARTREE2EV
    print(exc_s)
    print(exc_t)
    assert abs(exc_s - 8.560725962172379) < 1e-5
    assert abs(exc_t - 7.565200615451269) < 1e-5
