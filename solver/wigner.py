r"""
Vectorized 2D Phase-Space Wigner Quasi-Probability Distribution Calculator
Computes W(x, p) from density matrix rho in truncated Fock space using Laguerre polynomial expansion.
Quantifies non-classicality through negative volume integration: delta(rho) = \int (|W| - W) dx dp.
"""

import numpy as np
from scipy.special import genlaguerre
import math
from typing import Tuple, Dict


class WignerCalculator:
    """
    Computes 2D Wigner quasi-probability distribution on continuous-variable phase-space.
    """
    def __init__(self, xvec: np.ndarray, pvec: np.ndarray):
        self.xvec = xvec
        self.pvec = pvec
        self.dx = xvec[1] - xvec[0]
        self.dp = pvec[1] - pvec[0]

        # Meshgrid: X [Nx, Np], P [Nx, Np]
        self.X, self.P = np.meshgrid(xvec, pvec, indexing="ij")
        # Continuous-variable complex amplitude alpha = (x + i p) / sqrt(2)
        self.alpha = (self.X + 1j * self.P) / np.sqrt(2.0)
        self.alpha_sq = np.abs(self.alpha) ** 2
        self.gaussian = np.exp(-2.0 * self.alpha_sq)

    def compute(self, rho: np.ndarray) -> np.ndarray:
        """
        Computes W(x, p) for density matrix rho in Fock basis:
        W(x, p) = 1/pi * [ sum_n rho_nn W_nn + 2 * Re( sum_{m > n} rho_nm W_nm ) ]
        """
        fock_dim = rho.shape[0]
        W = np.zeros_like(self.X, dtype=np.float64)

        # 4 * |alpha|^2 argument for Laguerre polynomials
        arg = 4.0 * self.alpha_sq

        # 1. Diagonal terms: n == m
        for n in range(fock_dim):
            if abs(rho[n, n]) < 1e-12:
                continue
            L_n = genlaguerre(n, 0)(arg)
            w_nn = ((-1.0) ** n / np.pi) * self.gaussian * L_n
            W += np.real(rho[n, n]) * w_nn

        # 2. Off-diagonal interference terms: m > n
        for n in range(fock_dim):
            for m in range(n + 1, fock_dim):
                if abs(rho[n, m]) < 1e-12:
                    continue
                k = m - n
                coeff = ((-1.0) ** n / np.pi) * np.sqrt(math.factorial(n) / math.factorial(m))
                L_n_k = genlaguerre(n, k)(arg)
                # (2 * alpha^*)^k
                phase = (2.0 * self.alpha.conj()) ** k
                w_nm = coeff * phase * self.gaussian * L_n_k

                # Add interference term: 2 * Re(rho_nm * w_nm)
                W += 2.0 * np.real(rho[n, m] * w_nm)

        return W

    def negative_volume(self, W: np.ndarray) -> float:
        r"""
        Quantifies quantum non-classicality via Wigner negativity volume:
        delta = \int \int (|W(x, p)| - W(x, p)) dx dp
        delta == 0 for classical/coherent/thermal states; delta > 0 for quantum non-classical states (cat, Fock).
        """
        neg = np.abs(W) - W
        return float(np.sum(neg) * self.dx * self.dp)

    def total_probability(self, W: np.ndarray) -> float:
        r"""Verifies phase-space probability normalization: \int \int W(x, p) dx dp == 1.0."""
        return float(np.sum(W) * self.dx * self.dp)


def compute_wigner(
    rho: np.ndarray,
    x_range: float = 5.0,
    num_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function returning (xvec, pvec, W).
    """
    xvec = np.linspace(-x_range, x_range, num_points)
    pvec = np.linspace(-x_range, x_range, num_points)
    calc = WignerCalculator(xvec, pvec)
    W = calc.compute(rho)
    return xvec, pvec, W
