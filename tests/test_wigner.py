"""
Unit Tests for Vectorized 2D Wigner Distribution Calculator
Verifies probability normalization and non-classicality negativity volume quantification.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from solver.lindblad import coherent_state, cat_state, ket2dm
from solver.wigner import WignerCalculator, compute_wigner


class TestWignerCalculator(unittest.TestCase):
    def setUp(self):
        self.xvec = np.linspace(-4.0, 4.0, 100)
        self.pvec = np.linspace(-4.0, 4.0, 100)
        self.calc = WignerCalculator(self.xvec, self.pvec)

    def test_vacuum_normalization_and_gaussian_form(self):
        """Verify vacuum state |0> integrates to 1.0 and has zero negativity."""
        fock_dim = 15
        psi_vac = np.zeros(fock_dim, dtype=np.complex128)
        psi_vac[0] = 1.0
        rho_vac = ket2dm(psi_vac)

        W = self.calc.compute(rho_vac)
        tot_prob = self.calc.total_probability(W)
        self.assertAlmostEqual(tot_prob, 1.0, delta=0.02)

        neg_vol = self.calc.negative_volume(W)
        self.assertAlmostEqual(neg_vol, 0.0, delta=1e-5)

    def test_single_photon_fock_state_negativity(self):
        """Verify Fock state |1> exhibits strong central Wigner negativity W(0, 0) < 0."""
        fock_dim = 15
        psi_1 = np.zeros(fock_dim, dtype=np.complex128)
        psi_1[1] = 1.0
        rho_1 = ket2dm(psi_1)

        W = self.calc.compute(rho_1)
        # Origin W(0,0) for Fock state |1> is theoretically -1/pi ~ -0.3183
        origin_idx = len(self.xvec) // 2
        w_origin = W[origin_idx, origin_idx]
        self.assertLess(w_origin, -0.25)

        neg_vol = self.calc.negative_volume(W)
        self.assertGreater(neg_vol, 0.2)

    def test_cat_state_interference_fringes(self):
        """Verify Schrödinger cat state exhibits quantum interference fringes with negative volume."""
        fock_dim = 25
        cat = cat_state(2.0, fock_dim, parity="even")
        rho_cat = ket2dm(cat)

        W = self.calc.compute(rho_cat)
        neg_vol = self.calc.negative_volume(W)
        self.assertGreater(neg_vol, 0.3)


if __name__ == "__main__":
    unittest.main()
