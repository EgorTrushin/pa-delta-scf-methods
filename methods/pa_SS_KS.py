#!/usr/bin/env python3
"""Spin-averaged DFT calculations for excited-states."""

from .UKS import UKS


class pa_SS_KS(UKS):
    """Implements potential-averaged spin-symmetrized DFT calculations for excited-states.

    The alpha and beta components of vxc are averaged so that the
    exchange-correlation potential is the same for both spins.
    """

    spin_symmetrized = True
