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
    assert abs(exc_s - 8.62339481168497) < 1e-5
    exc_t = (mf_triplet.e_tot - mf.e_tot) * HARTREE2EV
    assert abs(exc_t - 7.545843190618086) < 1e-5

    exc_s = (2 * mf_mixed_singlet_sa.e_tot - mf_triplet_sa.e_tot - mf.e_tot) * HARTREE2EV
    assert abs(exc_s - 8.680205876336730) < 1e-5
    exc_t = (mf_triplet_sa.e_tot - mf.e_tot) * HARTREE2EV
    assert abs(exc_t - 7.592721606956712) < 1e-5

    exc_s = (mf_oss.e_tot - mf.e_tot) * HARTREE2EV
    assert abs(exc_s - 8.681028377334210) < 1e-5

    exc_s = (mf_sta.e_tot_oss - mf.e_tot) * HARTREE2EV
    assert abs(exc_s - 8.679677496474548) < 1e-5
    exc_t = (mf_sta.e_tot_t - mf.e_tot) * HARTREE2EV
    assert abs(exc_t - 7.592759965260321) < 1e-5
