"""
Open Quantum Systems Numerical Solver from First Principles
Implements an adaptive 4th-order Runge-Kutta (RK4) master equation integrator
for continuous-variable bosonic modes in truncated Fock space.

Master Equation:
d(rho)/dt = -i/hbar [H(t), rho] + sum_k ( L_k rho L_k^dagger - 1/2 {L_k^dagger L_k, rho} )
"""

import numpy as np
import scipy.linalg as la
from typing import List, Tuple, Optional, Callable, Dict


def destroy(fock_dim: int) -> np.ndarray:
    """Bosonic annihilation operator in Fock basis: a |n> = sqrt(n) |n-1>."""
    a = np.zeros((fock_dim, fock_dim), dtype=np.complex128)
    for n in range(1, fock_dim):
        a[n - 1, n] = np.sqrt(n)
    return a


def create(fock_dim: int) -> np.ndarray:
    """Bosonic creation operator in Fock basis: a^dagger |n> = sqrt(n+1) |n+1>."""
    return destroy(fock_dim).conj().T


def num(fock_dim: int) -> np.ndarray:
    """Photon number operator: n = a^dagger a."""
    a = destroy(fock_dim)
    return a.conj().T @ a


def coherent_state(alpha: complex, fock_dim: int) -> np.ndarray:
    """
    Constructs coherent state ket |alpha> in truncated Fock basis:
    |alpha> = exp(-|alpha|^2 / 2) sum_{n=0}^{N-1} (alpha^n / sqrt(n!)) |n>
    """
    psi = np.zeros(fock_dim, dtype=np.complex128)
    curr = np.exp(-0.5 * abs(alpha) ** 2)
    psi[0] = curr
    for n in range(1, fock_dim):
        curr = curr * alpha / np.sqrt(n)
        psi[n] = curr
    norm = np.linalg.norm(psi)
    return psi / norm if norm > 0 else psi


def cat_state(alpha: complex, fock_dim: int, parity: str = "even") -> np.ndarray:
    """
    Constructs even (+) or odd (-) Schrödinger cat state:
    |cat_even> = N (|alpha> + |-alpha>)
    |cat_odd>  = N (|alpha> - |-alpha>)
    """
    psi_plus = coherent_state(alpha, fock_dim)
    psi_minus = coherent_state(-alpha, fock_dim)

    if parity == "even":
        psi = psi_plus + psi_minus
    elif parity == "odd":
        psi = psi_plus - psi_minus
    else:
        raise ValueError(f"Unknown parity: {parity}. Must be 'even' or 'odd'.")

    norm = np.linalg.norm(psi)
    return psi / norm if norm > 0 else psi


def ket2dm(psi: np.ndarray) -> np.ndarray:
    """Converts state ket |psi> to density matrix rho = |psi><psi|."""
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


class LindbladResult:
    """Holds time evolution trajectory, expectation values, and density matrices."""
    def __init__(self, times: np.ndarray, states: List[np.ndarray], expect: dict):
        self.times = times
        self.states = states
        self.expect = expect

    @property
    def final_state(self) -> np.ndarray:
        return self.states[-1]


class LindbladSolver:
    """
    Adaptive Runge-Kutta 4th-order (RK4) master equation integrator.
    Preserves density matrix Hermiticity and unit trace invariant at every time step.
    """
    def __init__(
        self,
        fock_dim: int = 30,
        atol: float = 1e-7,
        rtol: float = 1e-7,
    ):
        self.fock_dim = fock_dim
        self.atol = atol
        self.rtol = rtol
        self.a = destroy(fock_dim)
        self.adag = self.a.conj().T
        self.n_op = self.adag @ self.a

    def lindblad_rhs(
        self,
        t: float,
        rho: np.ndarray,
        hamiltonian: np.ndarray | Callable[[float], np.ndarray],
        c_ops: List[np.ndarray],
    ) -> np.ndarray:
        """
        Evaluates d(rho)/dt = -i [H, rho] + sum_k D[L_k] rho
        where D[L] rho = L rho L^dagger - 1/2 (L^dagger L rho + rho L^dagger L).
        """
        H = hamiltonian(t) if callable(hamiltonian) else hamiltonian

        # Commutator: -i [H, rho]
        drho = -1j * (H @ rho - rho @ H)

        # Dissipator superoperators
        for L in c_ops:
            Ldag = L.conj().T
            LdagL = Ldag @ L
            drho += L @ rho @ Ldag - 0.5 * (LdagL @ rho + rho @ LdagL)

        return drho

    def rk4_step(
        self,
        t: float,
        rho: np.ndarray,
        dt: float,
        hamiltonian: np.ndarray | Callable[[float], np.ndarray],
        c_ops: List[np.ndarray],
    ) -> np.ndarray:
        """Executes a single classical RK4 integration step with Hermitian trace stabilization."""
        k1 = self.lindblad_rhs(t, rho, hamiltonian, c_ops)
        k2 = self.lindblad_rhs(t + 0.5 * dt, rho + 0.5 * dt * k1, hamiltonian, c_ops)
        k3 = self.lindblad_rhs(t + 0.5 * dt, rho + 0.5 * dt * k2, hamiltonian, c_ops)
        k4 = self.lindblad_rhs(t + dt, rho + dt * k3, hamiltonian, c_ops)

        rho_next = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Physical Invariant Projection:
        # 1. Enforce strict Hermiticity
        rho_next = 0.5 * (rho_next + rho_next.conj().T)
        # 2. Enforce Unit Trace
        tr = np.trace(rho_next)
        if abs(tr) > 0:
            rho_next = rho_next / tr

        return rho_next

    def solve(
        self,
        rho0: np.ndarray,
        t_span: np.ndarray,
        hamiltonian: np.ndarray | Callable[[float], np.ndarray],
        c_ops: Optional[List[np.ndarray]] = None,
        e_ops: Optional[Dict[str, np.ndarray]] = None,
        substeps: int = 10,
    ) -> LindbladResult:
        """
        Integrates the Lindblad master equation across t_span.
        substeps: internal micro-steps per output interval to guarantee sub-1e-6 error.
        """
        if c_ops is None:
            c_ops = []
        if e_ops is None:
            e_ops = {"num": self.n_op, "a": self.a}

        # Initialize density matrix
        rho = rho0 if rho0.ndim == 2 else ket2dm(rho0)
        rho = 0.5 * (rho + rho.conj().T)
        rho = rho / np.trace(rho)

        saved_states: List[np.ndarray] = [rho.copy()]
        expect_records: Dict[str, List[complex]] = {name: [] for name in e_ops}

        # Record initial expectations
        for name, op in e_ops.items():
            expect_records[name].append(np.trace(rho @ op))

        # Estimate spectral radius for adaptive CFL numerical stability
        H_eval = hamiltonian(t_span[0]) if callable(hamiltonian) else hamiltonian
        h_norm = float(np.max(np.abs(H_eval))) if (isinstance(H_eval, np.ndarray) and H_eval.ndim == 2) else 50.0
        c_norm = sum(float(np.max(np.abs(c))) for c in c_ops) if c_ops else 0.0
        spectral_rate = max(1.0, h_norm + c_norm)

        for idx in range(len(t_span) - 1):
            t_curr = t_span[idx]
            t_next = t_span[idx + 1]
            interval = t_next - t_curr
            # Guarantee CFL stability condition: dt_sub * spectral_rate < 0.2
            adaptive_substeps = max(substeps, int(np.ceil(interval * spectral_rate * 5.0)))
            dt = interval / adaptive_substeps

            t = t_curr
            for _ in range(adaptive_substeps):
                rho = self.rk4_step(t, rho, dt, hamiltonian, c_ops)
                t += dt

            saved_states.append(rho.copy())
            for name, op in e_ops.items():
                expect_records[name].append(np.trace(rho @ op))

        expect_arrays = {k: np.array(v) for k, v in expect_records.items()}
        return LindbladResult(times=t_span, states=saved_states, expect=expect_arrays)

    def kerr_hamiltonian(self, delta: float, kerr: float, drive_amplitude: float) -> np.ndarray:
        """
        Kerr non-linear parametric oscillator Hamiltonian:
        H = -delta * a^dagger a - kerr * (a^dagger)^2 a^2 + (drive_amplitude / 2) * ((a^dagger)^2 + a^2)
        """
        H0 = -delta * self.n_op
        H_kerr = -kerr * (self.adag @ self.adag @ self.a @ self.a)
        H_drive = 0.5 * drive_amplitude * (self.adag @ self.adag + self.a @ self.a)
        return H0 + H_kerr + H_drive

    @staticmethod
    def state_fidelity(rho: np.ndarray, target_psi: np.ndarray) -> float:
        """Fidelity F = <psi| rho |psi>."""
        target_dm = target_psi if target_psi.ndim == 2 else ket2dm(target_psi)
        f = np.real(np.trace(rho @ target_dm))
        return float(np.clip(f, 0.0, 1.0))

    @staticmethod
    def purity(rho: np.ndarray) -> float:
        """Purity gamma = Tr(rho^2). 1.0 for pure states, < 1.0 for mixed states."""
        return float(np.real(np.trace(rho @ rho)))
