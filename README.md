# Continuous-Variable Quantum Transpiler & Open Quantum Systems Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.12%2B-8CAAE6?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://opensource.org/licenses/MIT)
[![Build: Passing](https://img.shields.io/badge/Build-Passing-brightgreen.svg?style=flat)]()
[![Mode: Continuous-Variable](https://img.shields.io/badge/Mode-Continuous--Variable-purple.svg?style=flat)]()
[![Fidelity: 91.97%](https://img.shields.io/badge/Fidelity-91.97%25-blueviolet.svg?style=flat)]()

> **Architectural Whitepaper & Open Quantum Systems Engineering Specification**  
> **Author**: Abdur Arshad (M.Sc. Complex Adaptive Systems, Chalmers University of Technology)  
> **Focus**: Continuous-variable (CV) optical quantum computing compiler, open quantum systems Lindblad master equation solver, Wigner phase-space quasi-probability engine, and closed-loop CMA-ES DRAG pulse optimizer.

---

## 1. Executive Summary & Physics Foundations

Discrete-variable (DV) qubit platforms (transmons, trapped ions) suffer from steep physical-to-logical qubit overheads ($1,000:1$ to $10,000:1$) under surface code quantum error correction. Continuous-variable (CV) bosonic quantum architectures—where quantum information is encoded within the infinite-dimensional Hilbert space of superconducting microwave cavity resonators—enable **hardware-efficient quantum error correction within a single physical cavity mode** (e.g., Schrödinger cat and GKP codes).

`cv-quantum-transpiler` is a clean-room, first-principles scientific computing engine, compiler, and hardware-in-the-loop (HIL) microwave calibration framework engineered with NumPy and SciPy:

1. **Adaptive 4th-Order Runge-Kutta (RK4) Lindblad Master Equation Solver**: Integrates dissipative open-quantum-system dynamics in truncated Fock space ($d_{\text{Fock}} = 25\text{--}30$) with dynamic Courant-Friedrichs-Lewy (CFL) spectral stability criteria and continuous trace/Hermiticity preservation.
2. **Vectorized 2D Wigner Quasiprobability & Quantum Non-Classicality Engine**: Vectorized phase-space distribution $W(x, p)$ via generalized Laguerre polynomial expansions, calculating Hudson non-classicality negativity volume $\delta(\hat{\rho})$ to verify quantum interference fringes.
3. **Hardware-in-the-Loop (HIL) Pulse Calibrator**: Derivative-free Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizing DRAG (Derivative Removal by Adiabatic Gate) microwave pulse envelopes under stochastic low-frequency frequency drift ($1/f$) and projective readout shot noise.
4. **CV Circuit Compiler & Waveform Synthesizer**: Compiles abstract bosonic circuits ($\hat{D}(\alpha)$, $\hat{S}(r, \phi)$, $\hat{K}(\chi)$, $\hat{B}(\theta, \phi)$) into calibrated physical microwave pulses and 2.0 GSa/s DAC baseband I/Q waveforms, with export bridges to Qiskit Pulse and PennyLane.

---

## 2. System Architecture

```
========================================================================================
                     CONTINUOUS-VARIABLE QUANTUM TRANSPILER
========================================================================================

  [ Continuous-Variable Quantum Circuit IR ]
         │
         ├───► CVCircuit Specification (Arbitrary Bosonic Mode Topology)
         │      • Displacement Gate:      D(α)
         │      • Squeezing Gate:         S(r, φ)
         │      • Non-Linear Kerr Gate:   K(χ)
         │      • Beam Splitter Gate:     BS(θ, φ)
         │
         ▼
  [ Compilation & Microwave Pulse Synthesis ]
         │
         ├───► DecompositionPass
         │      • Maps D(α)   ──► Resonant Cavity Microwave Drive (Gaussian Envelope)
         │      • Maps S(r,φ) ──► Parametric Four-Wave Mixing Pump (2 * ω₀)
         │      • Maps K(χ)   ──► Passive Cavity Free Evolution Interval
         │
         ├───► WaveformSynthesisPass
         │      • High-Speed 2.0 GSa/s DAC Interpolation (dt = 0.5 ns)
         │      • Baseband In-Phase (I) & Quadrature (Q) Sample Streams
         │
         └───► Interoperability Bridges
                • Qiskit Pulse Schedule Dictionaries & Drive Channel Mappings
                • PennyLane Continuous-Variable Quantum Operations
         │
         ▼
  [ Open Quantum Systems Simulation Engine ]
         │
         ├───► Lindblad Master Equation Solver (Adaptive RK4 Integrator)
         │      • Truncated Fock Hilbert Space (N = 25 – 30)
         │      • Dynamic CFL Spectral Stability Condition: dt_sub * λ_max < 0.2
         │      • Continuous Hermiticity & Unit-Trace Invariant Projection
         │      • Environmental Channels: Photon Loss (L₁), Thermal Gain (L_th), Dephasing (L_φ)
         │
         └───► Vectorized 2D Phase-Space Wigner Quasiprobability Engine
                • Generalized Laguerre Polynomial Expansion L_n^(k)(4|α|²)
                • Total Probability Normalization: ∬ W(x, p) dx dp = 1.0
                • Hudson Non-Classicality Negativity: δ(ρ̂) = ∬ (|W| - W) dx dp
         │
         ▼
  [ Hardware-in-the-Loop (HIL) Pulse Optimization ]
         ├───► DRAG Microwave Pulse Formulation: Ω_y(t) = -β * d(Ω_x)/dt
         ├───► Closed-Loop CMA-ES Optimizer (Covariance Matrix Adaptation)
         └───► Stochastic Noise Modeling: 1/f Frequency Detuning Drift + Readout Shot Noise
```

---

## 3. Mathematical & Theoretical Formulations

### 3.1. Bosonic Truncated Fock Space & Canonical Commutation
In a bosonic mode truncated at Fock dimension $N$, the annihilation ($\hat{a}$) and creation ($\hat{a}^\dagger$) operators satisfy canonical commutation:
$$[\hat{a}, \hat{a}^\dagger] = \hat{I}, \quad \hat{a} |n\rangle = \sqrt{n} |n-1\rangle, \quad \hat{a}^\dagger |n\rangle = \sqrt{n+1} |n+1\rangle, \quad \hat{n} = \hat{a}^\dagger \hat{a}$$

### 3.2. Coherent & Schrödinger Cat States
A coherent state $|\alpha\rangle$ is an eigenstate of $\hat{a}$ generated via displacement $\hat{D}(\alpha) = \exp(\alpha \hat{a}^\dagger - \alpha^* \hat{a})$:
$$|\alpha\rangle = e^{-|\alpha|^2 / 2} \sum_{n=0}^{N-1} \frac{\alpha^n}{\sqrt{n!}} |n\rangle, \quad \langle \hat{n} \rangle = |\alpha|^2$$
Schrödinger cat states are quantum superpositions of distinct macroscopic coherent states:
$$|\text{cat}_{\text{even}}\rangle = \frac{1}{\sqrt{2(1 + e^{-2|\alpha|^2})}} (|\alpha\rangle + |-\alpha\rangle), \quad |\text{cat}_{\text{odd}}\rangle = \frac{1}{\sqrt{2(1 - e^{-2|\alpha|^2})}} (|\alpha\rangle - |-\alpha\rangle)$$
Under self-Kerr non-linear evolution $\hat{H}_{\text{Kerr}} = -K (\hat{a}^\dagger \hat{a})^2$ at critical time $t_{\text{cat}} = \frac{\pi}{2K}$, an initial coherent state $|\alpha\rangle$ collapses into the Schrödinger cat state:
$$|\psi(t_{\text{cat}})\rangle = \frac{e^{-i\pi/4}}{\sqrt{2}} (|\alpha\rangle + i |-\alpha\rangle)$$

### 3.3. Lindblad Open Quantum Systems Master Equation
Coupling to environmental dissipative baths yields the Lindblad master equation for density operator $\hat{\rho}(t)$:
$$\frac{d\hat{\rho}}{dt} = -\frac{i}{\hbar} [\hat{H}(t), \hat{\rho}] + \sum_k \left( \hat{L}_k \hat{\rho} \hat{L}_k^\dagger - \frac{1}{2} \{ \hat{L}_k^\dagger \hat{L}_k, \hat{\rho} \} \right)$$
Decoherence channels:
- **Single-Photon Loss**: $\hat{L}_1 = \sqrt{\kappa (n_{\text{th}} + 1)} \hat{a}$
- **Thermal Bath Absorption**: $\hat{L}_{\text{th}} = \sqrt{\kappa n_{\text{th}}} \hat{a}^\dagger$
- **Pure Dephasing**: $\hat{L}_\phi = \sqrt{\gamma_\phi} \hat{a}^\dagger \hat{a}$

### 3.4. Phase-Space Wigner Quasiprobability Distribution
For complex phase space amplitude $\alpha = \frac{x + ip}{\sqrt{2}}$, the 2D Wigner distribution $W(x, p)$ expands into generalized Laguerre polynomials $L_n^{(k)}$:
$$W(x, p) = \frac{1}{\pi} \left[ \sum_{n=0}^{N-1} \rho_{nn} W_{nn}(x, p) + 2 \text{Re} \sum_{n=0}^{N-2} \sum_{m=n+1}^{N-1} \rho_{nm} W_{nm}(x, p) \right]$$
$$W_{nn}(x, p) = \frac{(-1)^n}{\pi} e^{-2|\alpha|^2} L_n(4|\alpha|^2), \quad W_{nm}(x, p) = \frac{(-1)^n}{\pi} \sqrt{\frac{n!}{m!}} (2\alpha^*)^{m-n} e^{-2|\alpha|^2} L_n^{(m-n)}(4|\alpha|^2)$$
Hudson's Theorem establishes that all pure Gaussian states exhibit non-negative Wigner distributions ($W(x, p) \ge 0$). Non-classicality (e.g., Fock states $|n \ge 1\rangle$ and cat states) manifests as phase-space regions of negative quasiprobability, quantified by the **Hudson negativity volume**:
$$\delta(\hat{\rho}) = \iint_{\mathbb{R}^2} (|W(x, p)| - W(x, p)) \, dx \, dp$$

### 3.5. DRAG (Derivative Removal by Adiabatic Gate) Microwave Pulses
Fast Gaussian drives induce spectral leakage into adjacent non-computational states. DRAG corrects the control waveform by superposing a quadrature derivative component:
$$\Omega_x(t) = A \exp\left(-\frac{(t - t_0)^2}{2\sigma^2}\right)$$
$$\Omega_y(t) = -\beta \frac{d\Omega_x}{dt} = \beta \frac{t - t_0}{\sigma^2} \Omega_x(t)$$
$$\Omega(t) = (\Omega_x(t) + i \Omega_y(t)) e^{i\phi}$$

### 3.6. CMA-ES Covariance Matrix Adaptation
The hardware-in-the-loop calibrator tunes control parameters $\mathbf{x} = [A, \beta, \phi]^T$ using derivative-free Covariance Matrix Adaptation Evolution Strategy:
- **Mean update**: $\mathbf{m}^{(g+1)} = \sum_{i=1}^\mu w_i \mathbf{x}_{i:\lambda}^{(g)}$
- **Evolution path update**: $\mathbf{p}_c^{(g+1)} = (1 - c_c) \mathbf{p}_c^{(g)} + \sqrt{c_c (2 - c_c) \mu_{\text{eff}}} \frac{\mathbf{m}^{(g+1)} - \mathbf{m}^{(g)}}{\sigma^{(g)}}$
- **Covariance matrix adaptation**:
  $$\mathbf{C}^{(g+1)} = (1 - c_1 - c_\mu) \mathbf{C}^{(g)} + c_1 \mathbf{p}_c^{(g+1)} (\mathbf{p}_c^{(g+1)})^T + c_\mu \sum_{i=1}^\mu w_i \frac{\mathbf{x}_{i:\lambda}^{(g)} - \mathbf{m}^{(g)}}{\sigma^{(g)}} \left(\frac{\mathbf{x}_{i:\lambda}^{(g)} - \mathbf{m}^{(g)}}{\sigma^{(g)}}\right)^T$$

---

## 4. Empirical Validation & Output

Executing the end-to-end tutorial pipeline (`python notebooks/cat_state_generation.py`):

```
======================================================================
 Continuous-Variable Quantum Transpiler: Schrödinger Cat State Engine 
======================================================================

1. Target State Specification: Continuous-Variable Cat State |cat> (alpha=2.0, Fock dim=30)
2. Simulating Kerr non-linear evolution to t_cat = 3.142 us...

----------------------------------------------------------------------
 Quantum State Tomography & Verification Metrics:
----------------------------------------------------------------------
  Target Cat State Photon Expectation:  4.00
  Simulated Mode Photon Number <n>:     3.94
  Cat State Fidelity F(<cat|rho|cat>):   91.97%
  Density Matrix Purity Tr(rho^2):      0.8496

3. Calculating 2D Phase-Space Wigner Distribution W(x, p)...
  Wigner Integral Total Probability:    0.9620 (Expected: 1.0000)
  Non-Classicality Negativity Volume:   0.5208
  Quantum Interference Fringes Present: YES (Negative Wigner Quasiprobability)

4. Transpiling Abstract CV Circuit into Physical Microwave Pulses...
  Transpiled Microwave Pulses Generated: 1
    Pulse #1: cavity_drive_m0          | Start:    0.0 ns | Duration:  30.0 ns | Amp: 1.333
  Synthesized DAC Samples: 80 (Sampling rate: 2.0 GSa/s)

======================================================================
 Cat State Generation Pipeline Execution: SUCCESS
======================================================================
```

---

## 5. Repository Structure

```
cv-quantum-transpiler/
├── Dockerfile                      # Container definition (python:3.12-slim test environment)
├── LICENSE                         # MIT open-source license (Abdur Arshad)
├── README.md                       # Architectural whitepaper and documentation
├── requirements.txt                # Runtime dependencies (numpy, scipy, matplotlib)
├── notebooks/
│   ├── cat_state_generation.ipynb  # Interactive Jupyter notebook demonstration
│   └── cat_state_generation.py     # Executable end-to-end Python validation script
├── optimizer/
│   ├── __init__.py                 # Optimizer package initializer
│   ├── hil_calibrator.py           # DRAGPulse, CMAESOptimizer, HILPulseCalibrator
│   └── noise_models.py             # CavityNoiseParameters, QuantumNoiseModel
├── solver/
│   ├── __init__.py                 # Solver package initializer
│   ├── lindblad.py                 # Fock operators, state constructors, LindbladSolver
│   └── wigner.py                   # WignerCalculator, compute_wigner
├── transpiler/
│   ├── __init__.py                 # Transpiler package initializer
│   ├── passes.py                   # CVCircuit, DecompositionPass, WaveformSynthesisPass
│   └── qiskit_plugin.py            # QiskitPulseBridge for Qiskit Pulse & PennyLane
└── tests/
    ├── test_calibrator.py          # DRAG pulse, CMA-ES convergence, and HIL calibrator tests
    ├── test_lindblad.py            # Commutator, photon expectations, Lindblad solver tests
    ├── test_transpiler.py          # CV decomposition, DAC synthesis, Qiskit bridge tests
    └── test_wigner.py              # Wigner normalization, Fock & Cat state negativity tests
```

---

## 6. Quick Start & Execution

### Prerequisites
- Python 3.10+
- Dependencies: `numpy>=1.26.0`, `scipy>=1.12.0`, `matplotlib>=3.8.0`

```bash
# Clone the repository
git clone https://github.com/Abdur171/cv-quantum-transpiler.git
cd cv-quantum-transpiler

# Install dependencies
pip install -r requirements.txt
```

### Running Automated Test Suite
```bash
python -m unittest discover tests -v
```
Verified test coverage includes:
- `test_lindblad.py`: Ladder commutation $[a, a^\dagger] = I$, coherent state expectation values $\langle n \rangle = |\alpha|^2$, cat state even/odd parity subspaces, and trace/Hermiticity preservation under dissipative decay.
- `test_wigner.py`: Phase-space normalization $\iint W(x, p) dx dp = 1$, zero negativity for Gaussian states (vacuum, coherent), and pronounced negativity ($\delta > 0.3$) for Fock state $|1\rangle$ and cat states.
- `test_calibrator.py`: Analytical DRAG envelope derivatives, zero boundary conditions, and CMA-ES convergence on noisy landscapes.
- `test_transpiler.py`: Gate-to-pulse decomposition, 2.0 GSa/s I/Q waveform synthesis, and Qiskit Pulse / PennyLane translation bridges.

### Running End-to-End Simulation Pipeline
```bash
python notebooks/cat_state_generation.py
```

### Running via Docker
```bash
docker build -t cv-quantum-transpiler .
docker run --rm cv-quantum-transpiler
```

---

## 7. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
