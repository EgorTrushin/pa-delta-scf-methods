from helpers import excited_state_singlet, excited_state_singlet_sa, oss, sta, triplet, triplet_sa
from pyscf import dft, gto
from pyscf.data.nist import HARTREE2EV


def test_answer():
    geom = "N 0.0 0.0 0.55038998; N 0.0 0.0 -0.55038998"
    mol = gto.M(atom=geom, basis="cc-pVDZ")
    mol.verbose = 0
    mol.build()

    # Ground-state calculation
    mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()
    mf = mf.to_uks()

    mf_mixed_singlet = excited_state_singlet(mf, frac_occ=True, excitation=[0, 1])
    mf_triplet = triplet(mf, frac_occ=True, excitation=[0, 1])

    mf_mixed_singlet_sa = excited_state_singlet_sa(mf, frac_occ=True, excitation=[0, 1])
    mf_triplet_sa = triplet_sa(mf, frac_occ=True, excitation=[0, 1])

    mf_oss = oss(mf, frac_occ=True, excitation=[0, 1])

    mf_sta = sta(mf, frac_occ=True, excitation=[0, 1])

    exc_s = (2 * mf_mixed_singlet.e_tot - mf_triplet.e_tot - mf.e_tot) * HARTREE2EV
    assert abs(exc_s - 8.540390345734506) < 1e-5
    exc_t = (mf_triplet.e_tot - mf.e_tot) * HARTREE2EV
    assert abs(exc_t - 7.52011091914945) < 1e-5

    exc_s = (2 * mf_mixed_singlet_sa.e_tot - mf_triplet_sa.e_tot - mf.e_tot) * HARTREE2EV
    assert abs(exc_s - 8.593860624773178) < 1e-5
    exc_t = (mf_triplet_sa.e_tot - mf.e_tot) * HARTREE2EV
    assert abs(exc_t - 7.566428885532999) < 1e-5

    exc_s = (mf_oss.e_tot - mf.e_tot) * HARTREE2EV
    assert abs(exc_s - 8.594617199767605) < 1e-5

    exc_s = (mf_sta.e_tot_oss - mf.e_tot) * HARTREE2EV
    assert abs(exc_s - 8.593294423908250) < 1e-5
    exc_t = (mf_sta.e_tot_t - mf.e_tot) * HARTREE2EV
    assert abs(exc_t - 7.566490949574019) < 1e-5
