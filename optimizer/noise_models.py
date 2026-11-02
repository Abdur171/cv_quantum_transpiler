"""
Stochastic Quantum Noise Models & Environmental Dissipation Channels
Implements photon loss, thermal excitation, dephasing, and low-frequency frequency drift channels.
"""

import numpy as np
from typing import List, Optional
try:
    from solver.lindblad import destroy, create, num
except (ImportError, ValueError):
    from ..solver.lindblad import destroy, create, num


class CavityNoiseParameters:
    def __init__(
        self,
        t1_us: float = 50.0,           # Cavity photon lifetime (microseconds)
        t_phi_us: Optional[float] = 100.0,  # Pure dephasing time (microseconds)
        thermal_photons: float = 0.01, # Residual thermal photon occupancy
        drift_amplitude_mhz: float = 0.05, # Frequency drift amplitude (MHz)
    ):
        self.t1_us = t1_us
        self.t_phi_us = t_phi_us
        self.thermal_photons = thermal_photons
        self.drift_amplitude_mhz = drift_amplitude_mhz

        # Rates (in rad / microsecond)
        self.kappa = 1.0 / t1_us
        self.gamma_phi = 1.0 / t_phi_us if t_phi_us else 0.0


class QuantumNoiseModel:
    """
    Constructs Lindblad jump collapse operators corresponding to physical environmental decay.
    """
    def __init__(self, params: Optional[CavityNoiseParameters] = None):
        self.params = params or CavityNoiseParameters()

    def get_c_ops(self, fock_dim: int) -> List[np.ndarray]:
        """Returns physical collapse jump operators for Lindblad integration."""
        a = destroy(fock_dim)
        adag = create(fock_dim)
        n = num(fock_dim)

        c_ops = []

        # 1. Photon loss with thermal equilibrium:
        nth = self.params.thermal_photons
        kappa = self.params.kappa
        # Emission into bath: sqrt(kappa * (nth + 1)) * a
        c_ops.append(np.sqrt(kappa * (nth + 1.0)) * a)

        # Absorption from bath: sqrt(kappa * nth) * a^dagger
        if nth > 0:
            c_ops.append(np.sqrt(kappa * nth) * adag)

        # 2. Pure dephasing: sqrt(gamma_phi) * a^dagger a
        if self.params.gamma_phi > 0:
            c_ops.append(np.sqrt(self.params.gamma_phi) * n)

        return c_ops

    def sample_frequency_drift(self, t: float, seed: Optional[int] = None) -> float:
        """Simulates low-frequency 1/f-like detuning drift."""
        rng = np.random.default_rng(seed)
        # Quasi-static multi-tone drift
        drift = self.params.drift_amplitude_mhz * (
            0.6 * np.sin(2.0 * np.pi * 0.01 * t) +
            0.3 * np.cos(2.0 * np.pi * 0.05 * t) +
            0.1 * rng.normal(0, 1)
        )
        return float(drift)

    @staticmethod
    def add_shot_noise(ideal_prob: float, shots: int = 1000, rng: Optional[np.random.Generator] = None) -> float:
        """Simulates projective readout shot noise across N measurement shots."""
        if shots <= 0:
            return ideal_prob
        gen = rng or np.random.default_rng()
        measured_counts = gen.binomial(shots, np.clip(ideal_prob, 0.0, 1.0))
        return float(measured_counts / shots)
