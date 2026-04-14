from pyscf import dft, gto
from helpers import excited_state_singlet, triplet, excited_state_singlet_sa, triplet_sa, oss, sta


def test_answer():
    geom = "O 0.0 0.0 -0.06990256; H 0.0 0.75753241 0.51843495; H 0.0 -0.75753241 0.51843495"
    mol = gto.M(atom=geom, basis="cc-pVDZ")
    mol.verbose = 0
    mol.build()

    # Ground-state calculation
    mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()
    mf = mf.to_uks()

    mf_mixed_singlet = excited_state_singlet(mf, frac_occ=False, excitation=[0, 1])
    mf_triplet = triplet(mf, frac_occ=False, excitation=[0, 1])

    mf_mixed_singlet_sa = excited_state_singlet_sa(mf, frac_occ=False, excitation=[0, 1])
    mf_triplet_sa = triplet_sa(mf, frac_occ=False, excitation=[0, 1])

    mf_oss = oss(mf, frac_occ=False, excitation=[0, 1])

    mf_sta = sta(mf, frac_occ=False, excitation=[0, 1])

    exc_s = (2 * mf_mixed_singlet.e_tot - mf_triplet.e_tot - mf.e_tot) * 27.2114
    assert abs(exc_s - 7.950881415712) < 1e-6
    exc_t = (mf_triplet.e_tot - mf.e_tot) * 27.2114
    assert abs(exc_t - 7.412353925126) < 1e-6

    exc_s = (2 * mf_mixed_singlet_sa.e_tot - mf_triplet_sa.e_tot - mf.e_tot) * 27.2114
    assert abs(exc_s - 8.035975059724) < 1e-6
    exc_t = (mf_triplet_sa.e_tot - mf.e_tot) * 27.2114
    assert abs(exc_t - 7.472019000610) < 1e-6

    exc_s = (mf_oss.e_tot - mf.e_tot) * 27.2114
    assert abs(exc_s - 8.035235039813) < 1e-6

    exc_s = (mf_sta.e_tot_oss - mf.e_tot) * 27.2114
    assert abs(exc_s - 8.039915401488) < 1e-6
    exc_t = (mf_sta.e_tot_t - mf.e_tot) * 27.2114
    assert abs(exc_t - 7.470827521644) < 1e-6
