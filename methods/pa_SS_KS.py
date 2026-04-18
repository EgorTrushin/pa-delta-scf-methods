#!/usr/bin/env python3
"""Spin-averaged DFT calculations for excited-states."""

from .UKS import UKS


class pa_SS_KS(UKS):
    """Implements potential-averaged spin-symmetrized DFT calculations for excited-states."""

    def get_ingredients(self, mo_energy=None):
        """Constructs ingredients with spin-averaged vxc.

        The alpha and beta components of vxc are averaged so that the
        exchange-correlation potential is the same for both spins.
        """
        occ = self.mom.get_occ(self.mf.mo_coeff, mo_energy)
        dm = self.mf.make_rdm1(self.mf.mo_coeff, occ)
        vxc, vj, exc, e_x_nl = self.eval_dft(dm)

        vxc[0, :, :] = 0.5 * (vxc[0, :, :] + vxc[1, :, :])
        vxc[1, :, :] = vxc[0, :, :]

        return vxc, vj, dm, exc, e_x_nl
