"""
Unit Tests for Hardware-in-the-Loop Calibrator & CMA-ES Optimizer
Verifies DRAG pulse synthesis, shot noise simulation, and CMA-ES convergence.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from optimizer.hil_calibrator import DRAGPulse, CMAESOptimizer, HILPulseCalibrator
from solver.lindblad import coherent_state


class TestCalibrator(unittest.TestCase):
    def test_drag_pulse_envelope(self):
        """Verify DRAG envelope has zero boundary values and correct derivative."""
        pulse = DRAGPulse(amplitude=1.0, duration_ns=40.0, sigma_ns=10.0, beta=0.5, phase=0.0)
        self.assertEqual(pulse.envelope(-1.0), 0.0j)
        self.assertEqual(pulse.envelope(41.0), 0.0j)

        # Center t=20.0 ns has zero derivative => imaginary component should be 0
        val_center = pulse.envelope(20.0)
        self.assertAlmostEqual(val_center.imag, 0.0, places=6)
        self.assertAlmostEqual(val_center.real, 1.0, places=6)

    def test_cma_es_convergence_sphere(self):
        """Verify CMA-ES minimizes noisy quadratic sphere benchmark."""
        x0 = np.array([2.0, -1.5, 3.0])
        opt = CMAESOptimizer(x0=x0, sigma=0.5, popsize=10, seed=123)

        for _ in range(40):
            cands = opt.ask()
            fitness = np.sum((cands - 1.0) ** 2, axis=1) + 0.01 * np.random.normal(size=len(cands))
            opt.tell(cands, fitness)

        # Solution should converge close to optimum [1.0, 1.0, 1.0]
        dist = np.linalg.norm(opt.xmean - np.array([1.0, 1.0, 1.0]))
        self.assertLess(dist, 0.5)

    def test_hil_calibration_loop(self):
        """Verify closed-loop HIL calibrator optimizes pulse parameters."""
        calibrator = HILPulseCalibrator(shots=1000)
        target = coherent_state(1.0, fock_dim=calibrator.solver.fock_dim)

        initial_guess = np.array([0.5, 10.0, 0.0, 0.0])
        result = calibrator.calibrate(target_state=target, initial_guess=initial_guess, max_iterations=6, duration_ns=30.0)

        self.assertIn("best_fidelity", result)
        self.assertIn("best_params", result)
        self.assertGreater(result["best_fidelity"], 0.3)


if __name__ == "__main__":
    unittest.main()
