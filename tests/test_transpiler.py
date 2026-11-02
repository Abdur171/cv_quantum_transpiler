"""
Unit Tests for Continuous-Variable Transpiler Passes and Qiskit Bridge
Verifies CV circuit decomposition, I/Q waveform synthesis, and pulse schedules.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transpiler.passes import CVCircuit, DecompositionPass, WaveformSynthesisPass
from transpiler.qiskit_plugin import QiskitPulseBridge


class TestTranspiler(unittest.TestCase):
    def test_circuit_decomposition(self):
        circ = CVCircuit(num_modes=2)
        circ.displacement(alpha=1.5 + 0.5j, mode=0)
        circ.squeeze(r=0.8, phi=np.pi / 4, mode=0)
        circ.kerr(chi=np.pi / 2, mode=0)

        decomposer = DecompositionPass(cavity_freq_mhz=5200.0, pump_freq_mhz=10400.0)
        pulses = decomposer.run(circ)

        self.assertEqual(len(pulses), 2)  # displacement and squeeze pulses
        self.assertEqual(pulses[0].carrier_freq_mhz, 5200.0)
        self.assertEqual(pulses[1].carrier_freq_mhz, 10400.0)

    def test_waveform_synthesis(self):
        circ = CVCircuit(num_modes=1)
        circ.displacement(alpha=2.0, mode=0)

        decomposer = DecompositionPass()
        pulses = decomposer.run(circ)

        synthesizer = WaveformSynthesisPass(sampling_rate_gsa_s=2.0)
        waveforms = synthesizer.synthesize(pulses)

        self.assertIn("cavity_drive_m0", waveforms)
        wf = waveforms["cavity_drive_m0"]
        self.assertIn("I", wf)
        self.assertIn("Q", wf)
        self.assertGreater(len(wf["I"]), 50)
        # Verify maximum amplitude is non-zero and finite
        self.assertGreater(np.max(np.abs(wf["I"])), 0.1)

    def test_qiskit_pulse_bridge(self):
        circ = CVCircuit(num_modes=1)
        circ.displacement(alpha=1.0, mode=0)
        circ.squeeze(r=0.5, mode=0)

        bridge = QiskitPulseBridge(backend_name="test_backend")
        schedule = bridge.transpile_to_qiskit_schedule(circ)

        self.assertEqual(schedule["backend"], "test_backend")
        self.assertEqual(schedule["num_modes"], 1)
        self.assertIn("channels", schedule)
        self.assertGreater(schedule["total_duration_ns"], 0.0)

        pennylane_ops = bridge.transpile_to_pennylane_ops(circ)
        self.assertEqual(len(pennylane_ops), 2)
        self.assertEqual(pennylane_ops[0]["op"], "qml.Displacement")
        self.assertEqual(pennylane_ops[1]["op"], "qml.Squeezing")


if __name__ == "__main__":
    unittest.main()
