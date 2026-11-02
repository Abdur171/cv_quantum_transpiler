"""
Integration Bridge for Qiskit Pulse & PennyLane Continuous-Variable Transpilation
Translates compiled microwave pulse schedules into standard quantum framework specifications.
"""

from typing import Dict, Any, List
from .passes import CVCircuit, PhysicalPulse, DecompositionPass


class QiskitPulseBridge:
    """
    Translates compiled CVCircuit physical pulses into standard Qiskit Pulse schedule dictionaries.
    """
    def __init__(self, backend_name: str = "superconducting_cavity_backend"):
        self.backend_name = backend_name
        self.decomposer = DecompositionPass()

    def transpile_to_qiskit_schedule(self, circuit: CVCircuit) -> Dict[str, Any]:
        """Converts CVCircuit to Qiskit Pulse compatible schedule dictionary."""
        pulses = self.decomposer.run(circuit)

        channels: Dict[str, List[Dict[str, Any]]] = {}

        for pulse in pulses:
            ch_name = f"drive_{pulse.channel}"
            if ch_name not in channels:
                channels[ch_name] = []

            channels[ch_name].append({
                "start_time_ns": pulse.start_time_ns,
                "duration_ns": pulse.duration_ns,
                "frequency_mhz": pulse.carrier_freq_mhz,
                "amplitude": round(pulse.amplitude, 4),
                "phase_rad": round(pulse.phase_rad, 4),
                "pulse_shape": pulse.envelope_type,
            })

        return {
            "backend": self.backend_name,
            "num_modes": circuit.num_modes,
            "total_duration_ns": max((p.start_time_ns + p.duration_ns for p in pulses), default=0.0),
            "channels": channels,
        }

    def transpile_to_pennylane_ops(self, circuit: CVCircuit) -> List[Dict[str, Any]]:
        """Maps CVCircuit to PennyLane CV operations (Displacement, Squeezing, Kerr)."""
        pennylane_ops = []
        for gate in circuit.gates:
            if gate.name == "displacement":
                pennylane_ops.append({
                    "op": "qml.Displacement",
                    "wires": [gate.mode],
                    "params": [gate.parameters["r"], gate.parameters["phi"]],
                })
            elif gate.name == "squeeze":
                pennylane_ops.append({
                    "op": "qml.Squeezing",
                    "wires": [gate.mode],
                    "params": [gate.parameters["r"], gate.parameters["phi"]],
                })
            elif gate.name == "kerr":
                pennylane_ops.append({
                    "op": "qml.Kerr",
                    "wires": [gate.mode],
                    "params": [gate.parameters["chi"]],
                })
        return pennylane_ops
