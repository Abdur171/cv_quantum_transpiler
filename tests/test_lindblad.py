"""
Unit Tests for First-Principles Lindblad Master Equation Solver
Verifies ladder operator commutators, coherent state photon counts,
Hermiticity preservation, and trace conservation.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from solver.lindblad import destroy, create, num, coherent_state, cat_state, ket2dm, LindbladSolver


class TestLindbladSolver(unittest.TestCase):
    def setUp(self):
        self.fock_dim = 25
        self.solver = LindbladSolver(fock_dim=self.fock_dim)

    def test_ladder_operator_commutation(self):
        """Verify [a, a^dagger] = I in the bulk of truncated Fock space."""
        a = self.solver.a
        adag = self.solver.adag
        comm = a @ adag - adag @ a

        # Check commutator identity for all levels except the truncation boundary N-1
        for i in range(self.fock_dim - 1):
            self.assertAlmostEqual(comm[i, i].real, 1.0, places=6)
            self.assertAlmostEqual(comm[i, i].imag, 0.0, places=6)

    def test_coherent_state_photon_expectation(self):
        """Verify <alpha| a^dagger a |alpha> == |alpha|^2."""
        for alpha in [1.0, 1.5 + 0.5j, 2.0]:
            psi = coherent_state(alpha, self.fock_dim)
            n_exp = np.real(psi.conj() @ self.solver.n_op @ psi)
            expected = abs(alpha) ** 2
            self.assertAlmostEqual(n_exp, expected, delta=0.01)

    def test_cat_state_parity(self):
        """Verify even cat states contain only even Fock components."""
        even_cat = cat_state(1.5, self.fock_dim, parity="even")
        for n in range(1, self.fock_dim, 2):
            self.assertAlmostEqual(abs(even_cat[n]), 0.0, places=6)

        odd_cat = cat_state(1.5, self.fock_dim, parity="odd")
        for n in range(0, self.fock_dim, 2):
            self.assertAlmostEqual(abs(odd_cat[n]), 0.0, places=6)

    def test_trace_preservation_and_hermiticity(self):
        """Verify master equation preserves Tr(rho) == 1 and rho == rho^dagger under decay."""
        psi0 = coherent_state(1.5, self.fock_dim)
        rho0 = ket2dm(psi0)

        # Photon loss jump operator
        kappa = 0.1
        c_ops = [np.sqrt(kappa) * self.solver.a]
        H = 0.05 * self.solver.n_op

        t_span = np.linspace(0.0, 5.0, 20)
        res = self.solver.solve(rho0, t_span, H, c_ops, substeps=5)

        for rho in res.states:
            # 1. Trace == 1
            tr = np.trace(rho)
            self.assertAlmostEqual(tr.real, 1.0, places=6)
            self.assertAlmostEqual(tr.imag, 0.0, places=6)

            # 2. Hermiticity: max |rho - rho^dag| < 1e-8
            herm_diff = np.max(np.abs(rho - rho.conj().T))
            self.assertLess(herm_diff, 1e-8)

            # 3. Purity Tr(rho^2) <= 1
            purity = self.solver.purity(rho)
            self.assertLessEqual(purity, 1.000001)
            self.assertGreater(purity, 0.0)


if __name__ == "__main__":
    unittest.main()
