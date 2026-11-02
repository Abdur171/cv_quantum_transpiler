"""
Continuous-Variable Quantum Circuit Compiler & Microwave Pulse Decomposition Passes
Translates abstract continuous-variable gates into calibrated physical microwave pulse sequences.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CVGate:
    """Base class for Continuous-Variable operations."""
    name: str
    mode: int
    parameters: Dict[str, float]


@dataclass
class PhysicalPulse:
    """Physical microwave waveform on a specific control channel."""
    channel: str
    start_time_ns: float
    duration_ns: float
    carrier_freq_mhz: float
    amplitude: float
    phase_rad: float
    envelope_type: str = "gaussian"  # "gaussian", "flat_top", "drag"


class CVCircuit:
    """Represents an abstract continuous-variable quantum circuit."""
    def __init__(self, num_modes: int = 1):
        self.num_modes = num_modes
        self.gates: List[CVGate] = []

    def displacement(self, alpha: complex, mode: int = 0) -> "CVCircuit":
        """Applies displacement gate D(alpha) = exp(alpha * a^dagger - alpha^* * a)."""
        self.gates.append(CVGate(
            name="displacement",
            mode=mode,
            parameters={"r": float(abs(alpha)), "phi": float(np.angle(alpha))},
        ))
        return self

    def squeeze(self, r: float, phi: float = 0.0, mode: int = 0) -> "CVCircuit":
        """Applies squeezing gate S(r, phi)."""
        self.gates.append(CVGate(
            name="squeeze",
            mode=mode,
            parameters={"r": float(r), "phi": float(phi)},
        ))
        return self

    def kerr(self, chi: float, mode: int = 0) -> "CVCircuit":
        """Applies non-linear Kerr gate K(chi) = exp(-i * chi * (a^dagger)^2 a^2)."""
        self.gates.append(CVGate(
            name="kerr",
            mode=mode,
            parameters={"chi": float(chi)},
        ))
        return self

    def beam_splitter(self, theta: float, phi: float = 0.0, mode1: int = 0, mode2: int = 1) -> "CVCircuit":
        """Applies beam splitter between two CV modes."""
        self.gates.append(CVGate(
            name="beam_splitter",
            mode=mode1,
            parameters={"theta": float(theta), "phi": float(phi), "target_mode": float(mode2)},
        ))
        return self


class DecompositionPass:
    """Decomposes abstract CV gates into physical microwave control pulses."""
    def __init__(self, cavity_freq_mhz: float = 5000.0, pump_freq_mhz: float = 10000.0):
        self.cavity_freq_mhz = cavity_freq_mhz
        self.pump_freq_mhz = pump_freq_mhz

    def run(self, circuit: CVCircuit) -> List[PhysicalPulse]:
        pulses: List[PhysicalPulse] = []
        current_time_ns = 0.0

        for gate in circuit.gates:
            if gate.name == "displacement":
                r = gate.parameters["r"]
                phi = gate.parameters["phi"]
                # Amplitude calibrated such that A * duration corresponds to displacement r
                duration_ns = 30.0
                amp = r / 1.5  # Calibration scaling factor
                pulses.append(PhysicalPulse(
                    channel=f"cavity_drive_m{gate.mode}",
                    start_time_ns=current_time_ns,
                    duration_ns=duration_ns,
                    carrier_freq_mhz=self.cavity_freq_mhz,
                    amplitude=amp,
                    phase_rad=phi,
                    envelope_type="gaussian",
                ))
                current_time_ns += duration_ns + 5.0  # 5ns buffer

            elif gate.name == "squeeze":
                r = gate.parameters["r"]
                phi = gate.parameters["phi"]
                # Squeezing via parametric pump at 2 * omega_0
                duration_ns = 60.0
                amp = r * 1.2
                pulses.append(PhysicalPulse(
                    channel=f"parametric_pump_m{gate.mode}",
                    start_time_ns=current_time_ns,
                    duration_ns=duration_ns,
                    carrier_freq_mhz=self.pump_freq_mhz,
                    amplitude=amp,
                    phase_rad=phi,
                    envelope_type="flat_top",
                ))
                current_time_ns += duration_ns + 5.0

            elif gate.name == "kerr":
                chi = gate.parameters["chi"]
                # Kerr gate realized via passive free evolution under non-linear cavity Hamiltonian
                # chi = K * t_wait => t_wait = chi / K
                kerr_rate_mhz = 0.05  # typical transmon-induced Kerr (50 kHz)
                duration_ns = (chi / (2.0 * np.pi * kerr_rate_mhz)) * 1000.0
                duration_ns = min(duration_ns, 200.0)  # clamp for realistic simulation
                current_time_ns += duration_ns

        return pulses


class WaveformSynthesisPass:
    """Discretizes physical microwave pulse sequence into high-speed DAC I/Q samples."""
    def __init__(self, sampling_rate_gsa_s: float = 2.0):
        self.sampling_rate = sampling_rate_gsa_s  # 2.0 GSa/s = 0.5 ns per sample
        self.dt_ns = 1.0 / self.sampling_rate

    def synthesize(self, pulses: List[PhysicalPulse]) -> Dict[str, Dict[str, np.ndarray]]:
        if not pulses:
            return {}

        total_duration = max(p.start_time_ns + p.duration_ns for p in pulses) + 10.0
        num_samples = int(np.ceil(total_duration / self.dt_ns))
        time_axis = np.linspace(0.0, total_duration, num_samples, endpoint=False)

        waveforms: Dict[str, Dict[str, np.ndarray]] = {}

        for pulse in pulses:
            ch = pulse.channel
            if ch not in waveforms:
                waveforms[ch] = {
                    "time": time_axis,
                    "I": np.zeros(num_samples, dtype=np.float64),
                    "Q": np.zeros(num_samples, dtype=np.float64),
                }

            start_idx = int(pulse.start_time_ns / self.dt_ns)
            pulse_samples = int(pulse.duration_ns / self.dt_ns)
            end_idx = min(start_idx + pulse_samples, num_samples)

            t_rel = (time_axis[start_idx:end_idx] - pulse.start_time_ns)
            sigma = pulse.duration_ns / 4.0
            t_center = pulse.duration_ns / 2.0

            if pulse.envelope_type == "gaussian":
                envelope = pulse.amplitude * np.exp(-0.5 * ((t_rel - t_center) / sigma) ** 2)
            else:
                envelope = np.full_like(t_rel, pulse.amplitude)

            # In-phase (I) and Quadrature (Q) decomposition
            I_comp = envelope * np.cos(pulse.phase_rad)
            Q_comp = envelope * np.sin(pulse.phase_rad)

            waveforms[ch]["I"][start_idx:end_idx] += I_comp
            waveforms[ch]["Q"][start_idx:end_idx] += Q_comp

        return waveforms
