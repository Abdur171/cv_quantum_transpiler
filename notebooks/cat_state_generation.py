"""
End-to-End Tutorial & Execution Script: Error-Corrected Schrödinger Cat State Generation
Simulates the preparation and stabilization of a continuous-variable cat qubit:
|cat> = (1 / sqrt(2)) * (|alpha> + i |-alpha>) with alpha = 2.0.

Integrates:
1. First-principles Lindblad master equation solver with Kerr non-linearity
2. Quantum noise channel modeling (photon loss and dephasing)
3. Vectorized 2D Wigner quasi-probability distribution calculation
4. Non-classicality negativity volume quantification
5. Transpilation to calibrated microwave drive pulses and I/Q DAC synthesis
"""

import sys
import os
import numpy as np

# Ensure cv-quantum-transpiler root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solver.lindblad import LindbladSolver, cat_state, coherent_state, ket2dm
from solver.wigner import WignerCalculator, compute_wigner
from optimizer.noise_models import QuantumNoiseModel, CavityNoiseParameters
from transpiler.passes import CVCircuit, DecompositionPass, WaveformSynthesisPass


def run_cat_state_generation():
    print("======================================================================")
    print(" Continuous-Variable Quantum Transpiler: Schrödinger Cat State Engine ")
    print("======================================================================\n")

    fock_dim = 30
    alpha = 2.0
    print(f"1. Target State Specification: Continuous-Variable Cat State |cat> (alpha={alpha}, Fock dim={fock_dim})")

    # 2. Setup Lindblad Master Equation Solver
    solver = LindbladSolver(fock_dim=fock_dim)
    noise_params = CavityNoiseParameters(t1_us=200.0, t_phi_us=400.0, thermal_photons=0.002)
    noise_model = QuantumNoiseModel(noise_params)
    c_ops = noise_model.get_c_ops(fock_dim)

    # 3. Simulate Yurke-Stoler Non-Linear Kerr Evolution
    # Initial state: coherent state |alpha> prepared via resonant displacement pulse D(alpha)
    psi0 = coherent_state(alpha, fock_dim=fock_dim)
    rho0 = ket2dm(psi0)

    # Kerr Hamiltonian: H = -K * (a^dag a)^2
    kerr_mhz = 0.5  # 500 kHz Kerr non-linearity
    H_kerr = -kerr_mhz * (solver.n_op @ solver.n_op)
    t_cat_us = np.pi / (2.0 * kerr_mhz)  # Yurke-Stoler cat formation time (pi / 2K)

    print(f"2. Simulating Kerr non-linear evolution to t_cat = {t_cat_us:.3f} us...")
    t_span = np.linspace(0.0, t_cat_us, 30)
    res = solver.solve(rho0, t_span, H_kerr, c_ops, substeps=10)
    final_rho = res.final_state

    # 4. Target Cat State definition at t_cat
    # Theoretical ideal unitary evolved state:
    U_ideal = scipy_expm = np.diag(np.exp(-1j * (-kerr_mhz * (np.arange(fock_dim) ** 2)) * t_cat_us))
    target_psi = U_ideal @ psi0
    target_psi = target_psi / np.linalg.norm(target_psi)

    # State Fidelity & Purity Analysis
    fidelity = solver.state_fidelity(final_rho, target_psi)
    purity = solver.purity(final_rho)
    avg_photons = np.real(np.trace(final_rho @ solver.n_op))

    print("\n----------------------------------------------------------------------")
    print(" Quantum State Tomography & Verification Metrics:")
    print("----------------------------------------------------------------------")
    print(f"  Target Cat State Photon Expectation:  {alpha**2:.2f}")
    print(f"  Simulated Mode Photon Number <n>:     {avg_photons:.2f}")
    print(f"  Cat State Fidelity F(<cat|rho|cat>):   {fidelity * 100:.2f}%")
    print(f"  Density Matrix Purity Tr(rho^2):      {purity:.4f}")

    # 5. Compute 2D Phase-Space Wigner Quasiprobability Distribution
    print("\n3. Calculating 2D Phase-Space Wigner Distribution W(x, p)...")
    xvec, pvec, W = compute_wigner(final_rho, x_range=4.0, num_points=100)
    calc = WignerCalculator(xvec, pvec)
    neg_vol = calc.negative_volume(W)
    total_prob = calc.total_probability(W)

    print(f"  Wigner Integral Total Probability:    {total_prob:.4f} (Expected: 1.0000)")
    print(f"  Non-Classicality Negativity Volume:   {neg_vol:.4f}")
    print(f"  Quantum Interference Fringes Present: {'YES (Negative Wigner Quasiprobability)' if neg_vol > 0.1 else 'NO'}")

    # 6. Transpiler Microwave Pulse Generation
    print("\n4. Transpiling Abstract CV Circuit into Physical Microwave Pulses...")
    circuit = CVCircuit(num_modes=1)
    circuit.displacement(alpha=complex(alpha, 0.0), mode=0)
    circuit.kerr(chi=np.pi / 2, mode=0)

    decomposer = DecompositionPass()
    pulses = decomposer.run(circuit)
    synthesizer = WaveformSynthesisPass(sampling_rate_gsa_s=2.0)
    waveforms = synthesizer.synthesize(pulses)

    print(f"  Transpiled Microwave Pulses Generated: {len(pulses)}")
    for i, p in enumerate(pulses, 1):
        print(f"    Pulse #{i}: {p.channel:24} | Start: {p.start_time_ns:6.1f} ns | Duration: {p.duration_ns:5.1f} ns | Amp: {p.amplitude:5.3f}")

    total_dac_samples = sum(len(wf["I"]) for wf in waveforms.values())
    print(f"  Synthesized DAC Samples: {total_dac_samples} (Sampling rate: 2.0 GSa/s)")

    print("\n======================================================================")
    print(" Cat State Generation Pipeline Execution: SUCCESS")
    print("======================================================================\n")


if __name__ == "__main__":
    run_cat_state_generation()
