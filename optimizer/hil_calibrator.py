"""
Hardware-in-the-Loop (HIL) Closed-Loop Pulse Calibration Engine
Optimizes microwave control pulses under realistic environmental drift and measurement shot noise.
Implements first-principles CMA-ES and derivative-free hybrid optimization.
"""

import numpy as np
from typing import Callable, Tuple, List, Dict, Optional
try:
    from solver.lindblad import LindbladSolver, coherent_state, ket2dm
    from optimizer.noise_models import QuantumNoiseModel, CavityNoiseParameters
except (ImportError, ValueError):
    from ..solver.lindblad import LindbladSolver, coherent_state, ket2dm
    from .noise_models import QuantumNoiseModel, CavityNoiseParameters


class DRAGPulse:
    """
    Gaussian Microwave Pulse with DRAG (Derivative Removal by Adiabatic Gate) correction.
    Omega(t) = [Omega_x(t) + i * Omega_y(t)] * exp(i * phi)
    where Omega_x(t) is Gaussian and Omega_y(t) = -beta * d(Omega_x)/dt.
    """
    def __init__(self, amplitude: float, duration_ns: float, sigma_ns: float, beta: float, phase: float = 0.0):
        self.amplitude = amplitude
        self.duration_ns = duration_ns
        self.sigma_ns = sigma_ns
        self.beta = beta
        self.phase = phase

    def envelope(self, t: float) -> complex:
        if t < 0.0 or t > self.duration_ns:
            return 0.0j
        t0 = 0.5 * self.duration_ns
        dt = t - t0
        gx = self.amplitude * np.exp(-0.5 * (dt / self.sigma_ns) ** 2)
        # Derivative: d(gx)/dt = -dt / sigma^2 * gx
        dgx_dt = -(dt / (self.sigma_ns ** 2)) * gx
        gy = -self.beta * dgx_dt
        return (gx + 1j * gy) * np.exp(1j * self.phase)


class CMAESOptimizer:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) from first principles.
    Ideal for noisy black-box experimental calibration landscapes.
    """
    def __init__(self, x0: np.ndarray, sigma: float = 0.2, popsize: Optional[int] = None, seed: int = 42):
        self.dim = len(x0)
        self.xmean = x0.astype(np.float64)
        self.sigma = sigma
        self.popsize = popsize or (4 + int(3 * np.log(self.dim)))
        self.rng = np.random.default_rng(seed)

        # Selection weights
        self.mu = self.popsize // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        # Adaptation parameters
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.dim + 2.0) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (self.dim + 1.0)) - 1.0) + self.cs
        self.chiN = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))

        # Evolution paths and Covariance matrix
        self.generation = 0
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = np.eye(self.dim)

    def ask(self) -> np.ndarray:
        """Generates a population of candidate parameter vectors."""
        arz = self.rng.normal(0, 1, size=(self.popsize, self.dim))
        # arz * diag(D) * B^T
        ary = arz @ (self.B @ np.diag(self.D)).T
        arx = self.xmean + self.sigma * ary
        return arx

    def tell(self, arx: np.ndarray, fitness: np.ndarray):
        """Updates internal distribution parameters given candidate fitness values."""
        self.generation += 1
        # Sort by fitness (minimization)
        idx = np.argsort(fitness)
        arx = arx[idx]

        # Update mean
        xold = self.xmean.copy()
        self.xmean = np.sum(arx[:self.mu] * self.weights[:, None], axis=0)

        # Update evolution paths
        y = (self.xmean - xold) / self.sigma
        C_inv_half = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (C_inv_half @ y)
        hsig_denom = np.sqrt(max(1e-12, 1.0 - (1.0 - self.cs) ** (2.0 * self.generation)))
        hsig = float(np.linalg.norm(self.ps) / hsig_denom / self.chiN < 1.4 + 2.0 / (self.dim + 1.0))
        self.pc = (1.0 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y

        # Update Covariance matrix C
        artmp = (arx[:self.mu] - xold) / self.sigma
        C_mu = artmp.T @ np.diag(self.weights) @ artmp
        self.C = (1.0 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1.0 - hsig) * self.cc * (2.0 - self.cc) * self.C) + self.cmu * C_mu

        # Update step size sigma
        self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1.0))

        # Eigendecomposition of C
        self.C = 0.5 * (self.C + self.C.T)
        evals, evecs = np.linalg.eigh(self.C)
        evals = np.maximum(evals, 1e-14)
        self.D = np.sqrt(evals)
        self.B = evecs


class HILPulseCalibrator:
    """
    Closed-loop pulse calibrator tuning physical microwave parameters
    against simulated noisy hardware with readout shot noise.
    """
    def __init__(
        self,
        solver: Optional[LindbladSolver] = None,
        noise_model: Optional[QuantumNoiseModel] = None,
        shots: int = 2000,
    ):
        self.solver = solver or LindbladSolver(fock_dim=25)
        self.noise_model = noise_model or QuantumNoiseModel()
        self.shots = shots
        self.c_ops = self.noise_model.get_c_ops(self.solver.fock_dim)

    def evaluate_pulse(
        self,
        params: np.ndarray,
        target_state: np.ndarray,
        duration_ns: float = 40.0,
        iteration_seed: int = 0,
    ) -> float:
        """
        Simulates closed-loop query: applies DRAG pulse [amp, sigma, beta, phase],
        evolves master equation, and calculates fidelity with simulated shot noise.
        Returns: noisy infidelity (1.0 - noisy_fidelity).
        """
        amp, sigma, beta, phase = params
        pulse = DRAGPulse(amplitude=amp, duration_ns=duration_ns, sigma_ns=sigma, beta=beta, phase=phase)

        # Vacuum initial state |0>
        psi0 = np.zeros(self.solver.fock_dim, dtype=np.complex128)
        psi0[0] = 1.0
        rho0 = ket2dm(psi0)

        # Time-dependent drive Hamiltonian
        a = self.solver.a
        adag = self.solver.adag

        t_span = np.linspace(0.0, duration_ns, 30)

        def H_t(t: float) -> np.ndarray:
            omega = pulse.envelope(t)
            drift = self.noise_model.sample_frequency_drift(t, seed=iteration_seed)
            H_drift = drift * self.solver.n_op
            H_drive = 0.5 * (omega * adag + np.conj(omega) * a)
            return H_drift + H_drive

        res = self.solver.solve(rho0, t_span, H_t, self.c_ops, substeps=4)
        ideal_fidelity = self.solver.state_fidelity(res.final_state, target_state)

        # Apply measurement readout shot noise
        noisy_fidelity = self.noise_model.add_shot_noise(
            ideal_fidelity,
            shots=self.shots,
            rng=np.random.default_rng(iteration_seed)
        )

        return float(1.0 - noisy_fidelity)

    def calibrate(
        self,
        target_state: np.ndarray,
        initial_guess: np.ndarray,
        max_iterations: int = 30,
        duration_ns: float = 40.0,
    ) -> Dict[str, any]:
        """
        Executes closed-loop calibration using CMA-ES.
        Returns optimized parameters, final infidelity, and convergence history.
        """
        cma = CMAESOptimizer(x0=initial_guess, sigma=0.15)
        history: List[float] = []
        best_cost = float("inf")
        best_params = initial_guess.copy()

        for it in range(max_iterations):
            candidates = cma.ask()
            fitnesses = np.zeros(len(candidates))

            for i, cand in enumerate(candidates):
                # Enforce physical constraints: amp > 0, sigma > 2ns
                cand_clamped = cand.copy()
                cand_clamped[0] = max(0.01, cand_clamped[0])
                cand_clamped[1] = max(2.0, cand_clamped[1])

                cost = self.evaluate_pulse(
                    cand_clamped,
                    target_state=target_state,
                    duration_ns=duration_ns,
                    iteration_seed=it * 100 + i,
                )
                fitnesses[i] = cost

                if cost < best_cost:
                    best_cost = cost
                    best_params = cand_clamped.copy()

            cma.tell(candidates, fitnesses)
            history.append(float(np.min(fitnesses)))

        return {
            "best_params": best_params,
            "best_infidelity": best_cost,
            "best_fidelity": 1.0 - best_cost,
            "convergence_history": history,
            "iterations": max_iterations,
        }
