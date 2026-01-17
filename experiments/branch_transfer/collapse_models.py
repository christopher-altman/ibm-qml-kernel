"""
Objective-collapse / observer-collapse channel models for constraint forecasting.

Implements parameterized nonunitary channels to model effective decoherence
on the friend-lab subspace, enabling forecast curves V(gamma) for detectability
threshold analysis.

This module provides:
    - Dephasing channel on friend-lab subspace
    - GRW/CSL-inspired effective decoherence model

IMPORTANT: These are effective phenomenological models for constraint/forecast
purposes, not derivations of collapse theories. Results should be interpreted
as detectability thresholds, not detection claims.

Date: 2026-01-17
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, List, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Kraus, Operator
from qiskit_aer import AerSimulator

from circuit import (
    build_branch_transfer_circuit,
    build_coherence_witness_circuit,
    get_circuit_stats,
    compute_visibility_from_counts,
    compute_coherence_witness_from_counts,
    compute_normalized_coherence,
    get_expected_ideal_coherence,
    Q_IDX, R_IDX, F_IDX, M_IDX, P_IDX,
    COHERENCE_QUBITS,
)

# Import noise model builder from run_sim
try:
    from run_sim import build_noise_model_from_params, load_hardware_params
except ImportError:
    # Fallback for when run as standalone
    pass


def dephasing_channel_kraus(gamma: float, num_qubits: int = 1) -> List[np.ndarray]:
    """
    Construct Kraus operators for a phase-flip (Z-dephasing) channel.

    The phase-flip channel applies:
        rho -> (1 - gamma) * rho + gamma * Z * rho * Z

    which in the Kraus representation becomes:
        K0 = sqrt(1 - gamma) * I
        K1 = sqrt(gamma) * Z

    IMPORTANT: This is a "phase-flip" or "Z-dephasing" channel, NOT a full
    dephasing channel in the traditional sense:

        - gamma = 0.0: Identity channel (no effect)
        - gamma = 0.5: Full dephasing (destroys all off-diagonal coherence
                       in the Z basis, giving the maximally mixed state
                       over the coherent subspace)
        - gamma = 1.0: Deterministic Z gate (unitary, flips sign of off-diagonal
                       coherence but does NOT destroy it)

    For collapse detection using W_X = <X^⊗4>, the coherence witness:
        - At gamma = 0.0: W_X = +1.0 (full coherence)
        - At gamma = 0.5: W_X = 0.0 (full decoherence)
        - At gamma = 1.0: W_X = -1.0 (sign-flipped, still fully coherent)

    Therefore, sweep range [0, 0.5] gives monotonic decay from W=1 to W=0.
    For detectability analysis, use |W_X| or magnitude-based metrics.

    Parameters
    ----------
    gamma : float
        Phase-flip strength in [0, 1].
        - gamma = 0: identity
        - gamma = 0.5: full dephasing (off-diagonal -> 0)
        - gamma = 1: deterministic Z (unitary phase flip)
    num_qubits : int
        Number of qubits (for tensor product).

    Returns
    -------
    list
        Kraus operators [K0, K1].
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    K0 = np.sqrt(1 - gamma) * I
    K1 = np.sqrt(gamma) * Z

    # For multiple qubits, apply to each
    if num_qubits > 1:
        K0_full = K0
        K1_full = K1
        for _ in range(num_qubits - 1):
            K0_full = np.kron(K0_full, I)
            K1_full = np.kron(K1_full, Z)
        return [K0_full, K1_full]

    return [K0, K1]


def apply_dephasing_to_subsystem(
    rho: np.ndarray,
    gamma: float,
    target_qubits: List[int],
    total_qubits: int = 5
) -> np.ndarray:
    """
    Apply dephasing channel to a subsystem of a density matrix.

    Parameters
    ----------
    rho : np.ndarray
        Full system density matrix (2^n x 2^n).
    gamma : float
        Dephasing strength.
    target_qubits : list
        Indices of qubits to dephase.
    total_qubits : int
        Total number of qubits in the system.

    Returns
    -------
    np.ndarray
        Density matrix after dephasing on target qubits.
    """
    dim = 2 ** total_qubits
    rho_out = np.zeros_like(rho)

    # Build Kraus operators for target subsystem
    # For each target qubit, apply dephasing
    # This is a simplification: we apply single-qubit dephasing to each target

    n_targets = len(target_qubits)

    # Build all combinations of Kraus operators on targets
    from itertools import product

    I_single = np.eye(2)
    Z_single = np.array([[1, 0], [0, -1]], dtype=complex)

    # Kraus operators for single-qubit dephasing
    k0 = np.sqrt(1 - gamma) * I_single
    k1 = np.sqrt(gamma) * Z_single

    kraus_single = [k0, k1]

    # Generate all tensor products of Kraus operators on target qubits
    for combo in product(range(2), repeat=n_targets):
        # Build full Kraus operator
        K = np.eye(1, dtype=complex)
        for q in range(total_qubits):
            if q in target_qubits:
                idx = target_qubits.index(q)
                K = np.kron(K, kraus_single[combo[idx]])
            else:
                K = np.kron(K, I_single)

        rho_out += K @ rho @ K.conj().T

    return rho_out


def apply_measurement_rotation(
    rho: np.ndarray,
    target_qubits: List[int],
    basis: str = 'X',
    total_qubits: int = 5
) -> np.ndarray:
    """
    Apply measurement basis rotation to density matrix.

    For X basis: apply H to each target qubit
    For Y basis: apply S†H to each target qubit

    Parameters
    ----------
    rho : np.ndarray
        Input density matrix.
    target_qubits : list
        Indices of qubits to rotate.
    basis : {'X', 'Y'}
        Measurement basis.
    total_qubits : int
        Total number of qubits.

    Returns
    -------
    np.ndarray
        Rotated density matrix.
    """
    # Build rotation operator
    I = np.eye(2)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)  # S†

    if basis == 'X':
        single_qubit_rot = H
    elif basis == 'Y':
        single_qubit_rot = H @ Sdg  # Apply S† then H
    else:
        raise ValueError(f"Unknown basis: {basis}")

    # Build full rotation operator (tensor product)
    # IMPORTANT: Use reverse order (highest qubit first) for Qiskit little-endian convention
    # In little-endian, the tensor product is U = U_{n-1} ⊗ ... ⊗ U_1 ⊗ U_0
    ops = []
    for q in range(total_qubits):
        if q in target_qubits:
            ops.append(single_qubit_rot)
        else:
            ops.append(I)

    # Kron from highest qubit index down
    U = ops[total_qubits - 1]
    for q in range(total_qubits - 2, -1, -1):
        U = np.kron(U, ops[q])

    # Apply rotation: rho' = U @ rho @ U†
    return U @ rho @ U.conj().T


def grw_localization_rate(
    gamma_0: float,
    mass_ratio: float = 1.0,
    num_particles: int = 1
) -> float:
    """
    Compute effective GRW localization rate.

    In GRW/CSL models, the collapse rate scales with mass and particle number.
    This is an effective parametrization for constraint forecasting.

    Parameters
    ----------
    gamma_0 : float
        Base localization rate parameter.
    mass_ratio : float
        Mass ratio relative to nucleon mass (for scaling).
    num_particles : int
        Number of particles (for amplification).

    Returns
    -------
    float
        Effective localization rate.

    Notes
    -----
    This is a phenomenological model, not a derivation. The actual GRW/CSL
    parameters would depend on specific model assumptions.
    """
    # Simplified scaling: rate ~ gamma_0 * mass * num_particles
    return gamma_0 * mass_ratio * num_particles


def run_with_collapse_model(
    mu: int = 1,
    collapse_model: Literal['none', 'dephase', 'grw'] = 'none',
    collapse_gamma: float = 0.0,
    shots: int = 20000,
    add_hardware_noise: bool = False,
    hw_params_path: str = 'data/ibm_hardware_params_2026.json',
) -> dict:
    """
    Run the circuit with an effective collapse model applied.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    collapse_model : {'none', 'dephase', 'grw'}
        Type of collapse model to apply.
    collapse_gamma : float
        Collapse/dephasing strength parameter.
    shots : int
        Number of shots for sampling.
    add_hardware_noise : bool
        If True, also add IBM hardware noise model.
    hw_params_path : str
        Path to hardware parameters (for noise model).

    Returns
    -------
    dict
        Results including visibility degradation.
    """
    # Build circuit
    qc = build_branch_transfer_circuit(mu=mu, include_memory_erase=True, barrier=False)

    # Get circuit stats
    stats = get_circuit_stats(qc)

    if collapse_model == 'none' and not add_hardware_noise:
        # Pure ideal simulation
        backend = AerSimulator(method='statevector')
        qc_transpiled = transpile(qc, backend, optimization_level=0)
        job = backend.run(qc_transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

    elif collapse_model == 'none' and add_hardware_noise:
        # Hardware noise only
        hw_params = load_hardware_params(hw_params_path)
        noise_model = build_noise_model_from_params(hw_params, num_qubits=5)
        backend = AerSimulator(noise_model=noise_model)
        qc_transpiled = transpile(qc, backend, optimization_level=1)
        job = backend.run(qc_transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

    else:
        # Apply collapse model via density matrix simulation
        # This is slower but allows arbitrary channels

        # First, run circuit to get pre-measurement state
        qc_no_measure = qc.remove_final_measurements(inplace=False)
        backend = AerSimulator(method='density_matrix')

        # Add hardware noise if requested
        if add_hardware_noise:
            hw_params = load_hardware_params(hw_params_path)
            noise_model = build_noise_model_from_params(hw_params, num_qubits=5)
            backend = AerSimulator(method='density_matrix', noise_model=noise_model)

        qc_dm = transpile(qc_no_measure, backend, optimization_level=0)

        # Save statevector/density matrix
        qc_dm.save_density_matrix()
        job = backend.run(qc_dm, shots=1)
        result = job.result()
        rho = result.data()['density_matrix']

        # Convert to numpy if needed
        if hasattr(rho, 'data'):
            rho_np = np.array(rho.data)
        else:
            rho_np = np.array(rho)

        # Apply collapse model
        if collapse_model == 'dephase':
            # Dephase on friend-lab subspace {F, R}
            target_qubits = [F_IDX, R_IDX]
            rho_collapsed = apply_dephasing_to_subsystem(
                rho_np, collapse_gamma, target_qubits, total_qubits=5
            )
        elif collapse_model == 'grw':
            # GRW-inspired: stronger dephasing, includes M
            # Use scaled gamma for GRW
            effective_gamma = grw_localization_rate(collapse_gamma, mass_ratio=1.0, num_particles=3)
            effective_gamma = min(effective_gamma, 1.0)  # Clamp to [0, 1]
            target_qubits = [F_IDX, R_IDX, M_IDX]
            rho_collapsed = apply_dephasing_to_subsystem(
                rho_np, effective_gamma, target_qubits, total_qubits=5
            )
        else:
            rho_collapsed = rho_np

        # Sample from collapsed density matrix
        # Measure qubits R and P (indices 1 and 4)
        counts = sample_from_density_matrix(rho_collapsed, [R_IDX, P_IDX], shots)

    # Compute probabilities
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}

    # Compute visibility
    V, V_err, cond_probs = compute_visibility_from_counts(counts, total)

    return {
        'collapse_model': collapse_model,
        'collapse_gamma': collapse_gamma,
        'add_hardware_noise': add_hardware_noise,
        'mu': mu,
        'shots': shots,
        'counts': counts,
        'probabilities': probs,
        'visibility': V,
        'visibility_error': V_err,
        'conditional_probabilities': cond_probs,
        'circuit_stats': stats,
    }


def run_coherence_with_collapse_model(
    mu: int = 1,
    collapse_model: Literal['none', 'dephase', 'grw'] = 'none',
    collapse_gamma: float = 0.0,
    shots: int = 20000,
    add_hardware_noise: bool = False,
    hw_params_path: str = 'data/ibm_hardware_params_2026.json',
    basis: str = 'X',
) -> dict:
    """
    Run coherence witness measurement with collapse model applied.

    The coherence witness W_X is sensitive to dephasing/collapse in a way that
    V (visibility) is not, because W_X probes off-diagonal coherence.

    IMPORTANT: Dephasing must be applied BEFORE the measurement-basis rotation
    (i.e., to the protocol output state), not after. The protocol state has
    off-diagonal coherence that dephasing destroys.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    collapse_model : {'none', 'dephase', 'grw'}
        Type of collapse model to apply.
    collapse_gamma : float
        Collapse/dephasing strength parameter.
    shots : int
        Number of shots for sampling.
    add_hardware_noise : bool
        If True, also add IBM hardware noise model.
    hw_params_path : str
        Path to hardware parameters.
    basis : {'X', 'Y'}
        Measurement basis for coherence witness.

    Returns
    -------
    dict
        Results including W, W_tilde (normalized coherence).
    """
    # Build coherence witness circuit (includes measurement rotation)
    qc = build_coherence_witness_circuit(
        mu=mu, include_memory_erase=True, basis=basis, barrier=False
    )

    # Get circuit stats
    stats = get_circuit_stats(qc)

    # Get ideal reference for normalization
    W_ideal = get_expected_ideal_coherence(mu, include_memory_erase=True, basis=basis)

    if collapse_model == 'none' and not add_hardware_noise:
        # Pure ideal simulation
        backend = AerSimulator(method='statevector')
        qc_transpiled = transpile(qc, backend, optimization_level=0)
        job = backend.run(qc_transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

    elif collapse_model == 'none' and add_hardware_noise:
        # Hardware noise only
        hw_params = load_hardware_params(hw_params_path)
        noise_model = build_noise_model_from_params(hw_params, num_qubits=5)
        backend = AerSimulator(noise_model=noise_model)
        qc_transpiled = transpile(qc, backend, optimization_level=1)
        job = backend.run(qc_transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

    else:
        # Apply collapse model via density matrix simulation
        # CRITICAL: We need to apply dephasing to the PROTOCOL state (before
        # measurement rotation), then apply the rotation and sample.

        # Step 1: Build protocol circuit (without measurement rotation)
        qc_protocol = build_branch_transfer_circuit(
            mu=mu, include_memory_erase=True, barrier=False
        )
        # Remove measurements from protocol circuit
        qc_protocol_no_measure = qc_protocol.remove_final_measurements(inplace=False)

        backend = AerSimulator(method='density_matrix')

        if add_hardware_noise:
            hw_params = load_hardware_params(hw_params_path)
            noise_model = build_noise_model_from_params(hw_params, num_qubits=5)
            backend = AerSimulator(method='density_matrix', noise_model=noise_model)

        qc_dm = transpile(qc_protocol_no_measure, backend, optimization_level=0)
        qc_dm.save_density_matrix()
        job = backend.run(qc_dm, shots=1)
        result = job.result()
        rho = result.data()['density_matrix']

        if hasattr(rho, 'data'):
            rho_np = np.array(rho.data)
        else:
            rho_np = np.array(rho)

        # Step 2: Apply collapse model to protocol state
        if collapse_model == 'dephase':
            # Dephase on friend-lab subspace {F, R}
            target_qubits = [F_IDX, R_IDX]
            rho_collapsed = apply_dephasing_to_subsystem(
                rho_np, collapse_gamma, target_qubits, total_qubits=5
            )
        elif collapse_model == 'grw':
            effective_gamma = grw_localization_rate(collapse_gamma, mass_ratio=1.0, num_particles=3)
            effective_gamma = min(effective_gamma, 1.0)
            target_qubits = [F_IDX, R_IDX, M_IDX]
            rho_collapsed = apply_dephasing_to_subsystem(
                rho_np, effective_gamma, target_qubits, total_qubits=5
            )
        else:
            rho_collapsed = rho_np

        # Step 3: Apply measurement basis rotation to collapsed state
        # For X basis: H on each coherence qubit
        # For Y basis: S†H on each coherence qubit
        rho_rotated = apply_measurement_rotation(rho_collapsed, COHERENCE_QUBITS, basis)

        # Step 4: Sample from rotated density matrix (Z-basis measurement)
        counts = sample_from_density_matrix(rho_rotated, COHERENCE_QUBITS, shots)

    # Compute coherence witness
    total = sum(counts.values())
    W, W_err, parity_counts = compute_coherence_witness_from_counts(counts, total)

    # Compute normalized coherence
    W_tilde, W_tilde_err = compute_normalized_coherence(W, W_ideal, W_err, 0.0)

    return {
        'collapse_model': collapse_model,
        'collapse_gamma': collapse_gamma,
        'add_hardware_noise': add_hardware_noise,
        'measurement_basis': basis,
        'mu': mu,
        'shots': shots,
        'counts': counts,
        f'W_{basis}': W,
        f'W_{basis}_error': W_err,
        f'W_{basis}_ideal': W_ideal,
        f'W_{basis}_tilde': W_tilde,
        f'W_{basis}_tilde_error': W_tilde_err,
        'parity_counts': parity_counts,
        'circuit_stats': stats,
    }


def run_coherence_gamma_sweep(
    mu: int = 1,
    collapse_model: Literal['dephase', 'grw'] = 'dephase',
    gamma_values: Optional[List[float]] = None,
    shots: int = 20000,
    add_hardware_noise: bool = False,
    hw_params_path: str = 'data/ibm_hardware_params_2026.json',
    basis: str = 'X',
) -> List[dict]:
    """
    Sweep collapse strength with coherence witness measurement.

    This is the correct way to generate forecast curves for collapse sensitivity,
    as W_X is sensitive to dephasing while V is not.

    NOTE: Default sweep is [0, 0.5] because the phase-flip channel has:
        - gamma = 0.0: W_X = +1.0 (ideal coherence)
        - gamma = 0.5: W_X = 0.0 (full decoherence)
        - gamma = 1.0: W_X = -1.0 (deterministic Z, sign-flipped)

    This gives monotonic decay in [0, 0.5] which is cleaner for threshold analysis.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    collapse_model : {'dephase', 'grw'}
        Type of collapse model.
    gamma_values : list, optional
        Gamma values to sweep. Default: linspace(0, 0.5, 11).
    shots : int
        Number of shots per configuration.
    add_hardware_noise : bool
        Include hardware noise model.
    hw_params_path : str
        Path to hardware parameters.
    basis : {'X', 'Y'}
        Measurement basis.

    Returns
    -------
    list
        Results for each gamma value.
    """
    if gamma_values is None:
        # Default sweep [0, 0.5] for monotonic decay (full decoherence at 0.5)
        gamma_values = np.linspace(0, 0.5, 11).tolist()

    results = []
    for gamma in gamma_values:
        print(f"  gamma = {gamma:.3f}...")
        result = run_coherence_with_collapse_model(
            mu=mu,
            collapse_model=collapse_model,
            collapse_gamma=gamma,
            shots=shots,
            add_hardware_noise=add_hardware_noise,
            hw_params_path=hw_params_path,
            basis=basis,
        )
        results.append(result)

    return results


def compute_coherence_magnitude(
    W_X: float,
    W_Y: float,
    W_X_err: float = 0.0,
    W_Y_err: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute phase-robust coherence magnitude |C| = sqrt(W_X^2 + W_Y^2).

    The magnitude |C| is insensitive to the global phase of the off-diagonal
    coherence, making it robust to systematic phase errors and the sign-flip
    ambiguity at gamma > 0.5 in the phase-flip channel.

    For the ideal state, |C| = 1.0.
    For a fully decohered state, |C| = 0.0.
    For the gamma=1.0 case (deterministic Z), |C| = 1.0 (still coherent!).

    Parameters
    ----------
    W_X : float
        X-basis coherence witness.
    W_Y : float
        Y-basis coherence witness.
    W_X_err : float
        Standard error of W_X.
    W_Y_err : float
        Standard error of W_Y.

    Returns
    -------
    tuple
        (magnitude, magnitude_error)
    """
    magnitude = np.sqrt(W_X**2 + W_Y**2)

    # Error propagation: d|C| = (W_X * dW_X + W_Y * dW_Y) / |C|
    # Using quadrature: sigma_|C| = sqrt((W_X * sigma_X)^2 + (W_Y * sigma_Y)^2) / |C|
    if magnitude > 1e-10:
        magnitude_err = np.sqrt((W_X * W_X_err)**2 + (W_Y * W_Y_err)**2) / magnitude
    else:
        # At zero magnitude, error is just the larger of the two
        magnitude_err = np.sqrt(W_X_err**2 + W_Y_err**2)

    return magnitude, magnitude_err


def compute_normalized_magnitude(
    magnitude: float,
    magnitude_ideal: float,
    magnitude_err: float = 0.0,
    magnitude_ideal_err: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute normalized coherence magnitude |C_tilde| = |C| / |C_ideal|.

    Parameters
    ----------
    magnitude : float
        Measured coherence magnitude.
    magnitude_ideal : float
        Ideal coherence magnitude (usually 1.0).
    magnitude_err : float
        Standard error of measured magnitude.
    magnitude_ideal_err : float
        Standard error of ideal magnitude.

    Returns
    -------
    tuple
        (normalized_magnitude, normalized_magnitude_error)
    """
    if magnitude_ideal < 1e-10:
        return 0.0, 0.0

    normalized = magnitude / magnitude_ideal

    # Error propagation for ratio
    rel_err_meas = magnitude_err / magnitude if magnitude > 1e-10 else 0
    rel_err_ideal = magnitude_ideal_err / magnitude_ideal if magnitude_ideal > 1e-10 else 0
    normalized_err = normalized * np.sqrt(rel_err_meas**2 + rel_err_ideal**2)

    return normalized, normalized_err


def compute_coherence_detectability_threshold(
    gamma_sweep_results: List[dict],
    device_noise_band: float = 0.0,
    basis: str = 'X',
    use_magnitude: bool = False,
    gamma_sweep_results_y: Optional[List[dict]] = None,
) -> dict:
    """
    Compute detectability threshold based on coherence witness.

    The threshold is the smallest gamma such that the deviation from baseline
    exceeds the combined uncertainty (shot noise + device noise) at 2-sigma.

    Parameters
    ----------
    gamma_sweep_results : list
        Results from run_coherence_gamma_sweep (X basis or primary basis).
    device_noise_band : float
        Additional uncertainty from device noise.
    basis : {'X', 'Y'}
        Which W to use (ignored if use_magnitude=True).
    use_magnitude : bool
        If True, use |C| = sqrt(W_X^2 + W_Y^2) for threshold computation.
        This requires gamma_sweep_results_y to also be provided.
    gamma_sweep_results_y : list, optional
        Y-basis results (required if use_magnitude=True).

    Returns
    -------
    dict
        Threshold information.

    Notes
    -----
    Using magnitude (use_magnitude=True) is recommended because:
    1. It's phase-robust (insensitive to global phase errors)
    2. It avoids the sign-flip at gamma > 0.5 in the phase-flip channel
    3. It gives monotonic decay from |C|=1 to |C|=0 as gamma -> 0.5
    """
    if not gamma_sweep_results:
        return {'threshold_gamma': None}

    if use_magnitude:
        if gamma_sweep_results_y is None:
            raise ValueError("Y-basis results required for magnitude-based threshold")

        # Compute magnitude at each gamma
        baseline_x = gamma_sweep_results[0]
        baseline_y = gamma_sweep_results_y[0]
        baseline_mag, baseline_mag_err = compute_coherence_magnitude(
            baseline_x.get('W_X', 0),
            baseline_y.get('W_Y', 0),
            baseline_x.get('W_X_error', 0),
            baseline_y.get('W_Y_error', 0),
        )

        # Combine uncertainties
        total_uncertainty = np.sqrt(baseline_mag_err**2 + device_noise_band**2)

        # Find threshold
        threshold_gamma = None
        for result_x, result_y in zip(gamma_sweep_results, gamma_sweep_results_y):
            gamma = result_x['collapse_gamma']
            mag, mag_err = compute_coherence_magnitude(
                result_x.get('W_X', 0),
                result_y.get('W_Y', 0),
                result_x.get('W_X_error', 0),
                result_y.get('W_Y_error', 0),
            )

            # Combined uncertainty for this point
            combined_err = np.sqrt(total_uncertainty**2 + mag_err**2)

            # Check if deviation exceeds threshold (2-sigma)
            if abs(mag - baseline_mag) > 2 * combined_err:
                threshold_gamma = gamma
                break

        return {
            'threshold_gamma': threshold_gamma,
            'baseline_magnitude': baseline_mag,
            'baseline_error': baseline_mag_err,
            'device_noise_band': device_noise_band,
            'total_uncertainty': total_uncertainty,
            'confidence_level': 0.95,
            'metric': '|C| = sqrt(W_X^2 + W_Y^2)',
        }

    else:
        # Single-basis threshold (original behavior)
        baseline = gamma_sweep_results[0]
        baseline_W_tilde = baseline.get(f'W_{basis}_tilde', 0)
        baseline_W_tilde_err = baseline.get(f'W_{basis}_tilde_error', 0)

        # Combine uncertainties
        total_uncertainty = np.sqrt(baseline_W_tilde_err**2 + device_noise_band**2)

        # Find threshold
        threshold_gamma = None
        for result in gamma_sweep_results:
            gamma = result['collapse_gamma']
            W_tilde = result.get(f'W_{basis}_tilde', 0)
            W_tilde_err = result.get(f'W_{basis}_tilde_error', 0)

            # Combined uncertainty for this point
            combined_err = np.sqrt(total_uncertainty**2 + W_tilde_err**2)

            # Check if deviation exceeds threshold (2-sigma)
            # Use |W_tilde| to handle sign-flip at gamma > 0.5
            if abs(abs(W_tilde) - abs(baseline_W_tilde)) > 2 * combined_err:
                threshold_gamma = gamma
                break

        return {
            'threshold_gamma': threshold_gamma,
            'baseline_W_tilde': baseline_W_tilde,
            'baseline_error': baseline_W_tilde_err,
            'device_noise_band': device_noise_band,
            'total_uncertainty': total_uncertainty,
            'confidence_level': 0.95,
            'metric': f'|W_{basis}_tilde|',
        }


def sample_from_density_matrix(
    rho: np.ndarray,
    measure_qubits: List[int],
    shots: int,
    total_qubits: int = 5
) -> dict:
    """
    Sample measurement outcomes from a density matrix.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (2^n x 2^n).
    measure_qubits : list
        Indices of qubits to measure.
    shots : int
        Number of samples.
    total_qubits : int
        Total number of qubits.

    Returns
    -------
    dict
        Counts dictionary with bitstrings as keys.
    """
    dim = 2 ** total_qubits
    n_measure = len(measure_qubits)

    # Compute probabilities for each measurement outcome
    # For each outcome bitstring on measured qubits, trace out the rest

    probs = {}
    for outcome in range(2 ** n_measure):
        # Build projector for this outcome
        proj = np.zeros((dim, dim), dtype=complex)

        for basis in range(dim):
            # Check if this basis state matches the outcome on measured qubits
            match = True
            for i, q in enumerate(measure_qubits):
                bit = (basis >> q) & 1
                expected_bit = (outcome >> i) & 1
                if bit != expected_bit:
                    match = False
                    break

            if match:
                proj[basis, basis] = 1.0

        # Probability = Tr(proj @ rho)
        prob = np.real(np.trace(proj @ rho))
        bitstring = format(outcome, f'0{n_measure}b')
        probs[bitstring] = max(prob, 0)  # Clamp numerical errors

    # Normalize
    total_prob = sum(probs.values())
    if total_prob > 0:
        probs = {k: v / total_prob for k, v in probs.items()}

    # Sample
    outcomes = list(probs.keys())
    probabilities = list(probs.values())
    samples = np.random.choice(outcomes, size=shots, p=probabilities)

    # Count
    counts = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1

    return counts


def run_gamma_sweep(
    mu: int = 1,
    collapse_model: Literal['dephase', 'grw'] = 'dephase',
    gamma_values: Optional[List[float]] = None,
    shots: int = 20000,
    add_hardware_noise: bool = False,
    hw_params_path: str = 'data/ibm_hardware_params_2026.json',
) -> List[dict]:
    """
    Sweep collapse strength parameter to generate forecast curves.

    NOTE: This mode measures visibility V, which is INSENSITIVE to Z-dephasing
    because it only measures diagonal populations. Use coherence_witness mode
    (run_coherence_gamma_sweep) for proper collapse detection.

    Default sweep is [0, 0.5] for consistency with coherence mode, though
    V will remain constant across this range.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    collapse_model : {'dephase', 'grw'}
        Type of collapse model.
    gamma_values : list, optional
        Gamma values to sweep. Default: linspace(0, 0.5, 11).
    shots : int
        Number of shots per configuration.
    add_hardware_noise : bool
        Include hardware noise model.
    hw_params_path : str
        Path to hardware parameters.

    Returns
    -------
    list
        Results for each gamma value.
    """
    if gamma_values is None:
        # Default sweep [0, 0.5] for consistency with coherence mode
        gamma_values = np.linspace(0, 0.5, 11).tolist()

    results = []
    for gamma in gamma_values:
        print(f"  gamma = {gamma:.3f}...")
        result = run_with_collapse_model(
            mu=mu,
            collapse_model=collapse_model,
            collapse_gamma=gamma,
            shots=shots,
            add_hardware_noise=add_hardware_noise,
            hw_params_path=hw_params_path,
        )
        results.append(result)

    return results


def compute_detectability_threshold(
    gamma_sweep_results: List[dict],
    baseline_V: float,
    baseline_V_err: float,
    device_noise_band: float = 0.0,
) -> dict:
    """
    Compute the detectability threshold for collapse effects.

    The threshold is the smallest gamma such that |V(gamma) - V(0)| exceeds:
        1) The shot-noise CI for V
        2) An empirically estimated device-noise uncertainty band

    Parameters
    ----------
    gamma_sweep_results : list
        Results from run_gamma_sweep.
    baseline_V : float
        Baseline visibility at gamma=0.
    baseline_V_err : float
        Standard error of baseline visibility.
    device_noise_band : float
        Additional uncertainty from device noise (e.g., from opt-level sweep).

    Returns
    -------
    dict
        Threshold information including gamma_threshold and confidence intervals.
    """
    # Combine uncertainties
    total_uncertainty = np.sqrt(baseline_V_err**2 + device_noise_band**2)

    # Find threshold
    threshold_gamma = None
    for result in gamma_sweep_results:
        gamma = result['collapse_gamma']
        V = result['visibility']
        V_err = result['visibility_error']

        # Combined uncertainty for this point
        combined_err = np.sqrt(total_uncertainty**2 + V_err**2)

        # Check if deviation exceeds threshold (2-sigma)
        if abs(V - baseline_V) > 2 * combined_err:
            threshold_gamma = gamma
            break

    return {
        'threshold_gamma': threshold_gamma,
        'baseline_visibility': baseline_V,
        'baseline_error': baseline_V_err,
        'device_noise_band': device_noise_band,
        'total_uncertainty': total_uncertainty,
        'confidence_level': 0.95,  # 2-sigma
    }


def save_sweep_results(
    results: List[dict],
    output_dir: Path,
    prefix: str = 'collapse_sweep'
) -> Path:
    """Save gamma sweep results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if results:
        model = results[0].get('collapse_model', 'unknown')
        noise = 'noisy' if results[0].get('add_hardware_noise', False) else 'ideal'
    else:
        model = 'unknown'
        noise = 'unknown'

    filename = f"{prefix}_{timestamp}_{model}_{noise}.json"
    filepath = output_dir / filename

    # Extract gamma-visibility curve
    gamma_values = [r['collapse_gamma'] for r in results]
    visibility_values = [r['visibility'] for r in results]
    visibility_errors = [r['visibility_error'] for r in results]

    output = {
        'timestamp': timestamp,
        'collapse_model': model,
        'add_hardware_noise': noise == 'noisy',
        'gamma_values': gamma_values,
        'visibility_values': visibility_values,
        'visibility_errors': visibility_errors,
        'full_results': results,
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    return filepath


def save_coherence_sweep_results(
    results: List[dict],
    output_dir: Path,
    prefix: str = 'coherence_sweep'
) -> Path:
    """Save coherence gamma sweep results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if results:
        model = results[0].get('collapse_model', 'unknown')
        noise = 'noisy' if results[0].get('add_hardware_noise', False) else 'ideal'
        basis = results[0].get('measurement_basis', 'X')
    else:
        model = 'unknown'
        noise = 'unknown'
        basis = 'X'

    filename = f"{prefix}_{timestamp}_{model}_{noise}_{basis}.json"
    filepath = output_dir / filename

    # Extract gamma-coherence curve
    gamma_values = [r['collapse_gamma'] for r in results]
    W_values = [r.get(f'W_{basis}', 0) for r in results]
    W_errors = [r.get(f'W_{basis}_error', 0) for r in results]
    W_tilde_values = [r.get(f'W_{basis}_tilde', 0) for r in results]
    W_tilde_errors = [r.get(f'W_{basis}_tilde_error', 0) for r in results]

    output = {
        'timestamp': timestamp,
        'collapse_model': model,
        'measurement_basis': basis,
        'add_hardware_noise': noise == 'noisy',
        'gamma_values': gamma_values,
        f'W_{basis}_values': W_values,
        f'W_{basis}_errors': W_errors,
        f'W_{basis}_tilde_values': W_tilde_values,
        f'W_{basis}_tilde_errors': W_tilde_errors,
        'full_results': results,
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    return filepath


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run collapse model forecast analysis'
    )
    parser.add_argument(
        '--mu', type=int, choices=[0, 1], default=1,
        help='Message bit (0 or 1, default: 1)'
    )
    parser.add_argument(
        '--shots', type=int, default=20000,
        help='Number of shots (default: 20000)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='artifacts/branch_transfer',
        help='Output directory for results'
    )
    parser.add_argument(
        '--collapse-model', type=str, default='dephase',
        choices=['none', 'dephase', 'grw'],
        help='Collapse model type (default: dephase)'
    )
    parser.add_argument(
        '--collapse-gamma', type=float, default=0.0,
        help='Collapse strength parameter (default: 0.0)'
    )
    parser.add_argument(
        '--gamma-sweep', action='store_true',
        help='Run gamma sweep from 0 to 1'
    )
    parser.add_argument(
        '--add-hardware-noise', action='store_true',
        help='Add IBM hardware noise model'
    )
    parser.add_argument(
        '--hw-params', type=str, default='data/ibm_hardware_params_2026.json',
        help='Path to hardware parameters JSON'
    )
    # Coherence witness mode arguments
    parser.add_argument(
        '--mode', type=str, choices=['rp_z', 'coherence_witness'], default='rp_z',
        help='Measurement mode: rp_z (visibility) or coherence_witness (W_X/W_Y)'
    )
    parser.add_argument(
        '--basis', type=str, choices=['X', 'Y'], default='X',
        help='Measurement basis for coherence witness (default: X)'
    )
    parser.add_argument(
        '--include-y-basis', action='store_true',
        help='Also run Y-basis measurement in coherence_witness mode'
    )
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Branch-Transfer Experiment: Collapse Model Forecasts")
    print("=" * 70)
    print(f"  mu = {args.mu}")
    print(f"  mode = {args.mode}")
    print(f"  collapse_model = {args.collapse_model}")
    print(f"  collapse_gamma = {args.collapse_gamma}")
    print(f"  add_hardware_noise = {args.add_hardware_noise}")
    print(f"  shots = {args.shots}")
    if args.mode == 'coherence_witness':
        print(f"  basis = {args.basis}")
        print(f"  include_y_basis = {args.include_y_basis}")
    print()

    if args.mode == 'coherence_witness':
        # Coherence witness mode - probes off-diagonal coherence
        if args.gamma_sweep:
            print("Running coherence gamma sweep (X basis)...")
            results_x = run_coherence_gamma_sweep(
                mu=args.mu,
                collapse_model=args.collapse_model,
                shots=args.shots,
                add_hardware_noise=args.add_hardware_noise,
                hw_params_path=args.hw_params,
                basis='X',
            )

            # Save X basis results
            filepath_x = save_coherence_sweep_results(results_x, output_dir)
            print(f"\nSaved X basis: {filepath_x}")

            # Print X basis summary
            print("\nCoherence gamma sweep summary (X basis):")
            print(f"{'gamma':>8} {'W_X':>8} {'W_tilde':>8}")
            print("-" * 28)
            for r in results_x:
                print(f"{r['collapse_gamma']:8.3f} {r.get('W_X', 0):8.4f} {r.get('W_X_tilde', 0):8.4f}")

            # Compute detectability threshold
            threshold_x = compute_coherence_detectability_threshold(
                results_x,
                device_noise_band=0.05,  # Higher for coherence due to hardware sensitivity
                basis='X',
            )
            print(f"\nDetectability threshold (X): gamma = {threshold_x['threshold_gamma']}")

            # Optional Y basis
            if args.include_y_basis:
                print("\nRunning coherence gamma sweep (Y basis)...")
                results_y = run_coherence_gamma_sweep(
                    mu=args.mu,
                    collapse_model=args.collapse_model,
                    shots=args.shots,
                    add_hardware_noise=args.add_hardware_noise,
                    hw_params_path=args.hw_params,
                    basis='Y',
                )

                filepath_y = save_coherence_sweep_results(results_y, output_dir)
                print(f"\nSaved Y basis: {filepath_y}")

                print("\nCoherence gamma sweep summary (Y basis):")
                print(f"{'gamma':>8} {'W_Y':>8} {'W_tilde':>8}")
                print("-" * 28)
                for r in results_y:
                    print(f"{r['collapse_gamma']:8.3f} {r.get('W_Y', 0):8.4f} {r.get('W_Y_tilde', 0):8.4f}")

        else:
            # Single coherence run
            result = run_coherence_with_collapse_model(
                mu=args.mu,
                collapse_model=args.collapse_model,
                collapse_gamma=args.collapse_gamma,
                shots=args.shots,
                add_hardware_noise=args.add_hardware_noise,
                hw_params_path=args.hw_params,
                basis=args.basis,
            )

            basis = args.basis
            print(f"\nCoherence Results ({basis} basis):")
            print(f"  W_{basis}: {result.get(f'W_{basis}', 0):.4f} +/- {result.get(f'W_{basis}_error', 0):.4f}")
            print(f"  W_{basis}_ideal: {result.get(f'W_{basis}_ideal', 0):.4f}")
            print(f"  W_{basis}_tilde: {result.get(f'W_{basis}_tilde', 0):.4f} +/- {result.get(f'W_{basis}_tilde_error', 0):.4f}")

    else:
        # Original rp_z mode (visibility-based)
        if args.gamma_sweep:
            print("Running gamma sweep (rp_z mode - visibility)...")
            results = run_gamma_sweep(
                mu=args.mu,
                collapse_model=args.collapse_model,
                shots=args.shots,
                add_hardware_noise=args.add_hardware_noise,
                hw_params_path=args.hw_params,
            )

            # Save results
            filepath = save_sweep_results(results, output_dir)
            print(f"\nSaved: {filepath}")

            # Print summary
            print("\nGamma sweep summary (visibility):")
            print(f"{'gamma':>8} {'V':>8} {'V_err':>8}")
            print("-" * 28)
            for r in results:
                print(f"{r['collapse_gamma']:8.3f} {r['visibility']:8.4f} {r['visibility_error']:8.4f}")

            # Compute detectability threshold
            baseline = results[0]
            threshold = compute_detectability_threshold(
                results,
                baseline['visibility'],
                baseline['visibility_error'],
                device_noise_band=0.02
            )
            print(f"\nDetectability threshold: gamma = {threshold['threshold_gamma']}")
            print("\nNOTE: V is insensitive to dephasing! Use --mode coherence_witness for collapse detection.")

        else:
            # Single run
            result = run_with_collapse_model(
                mu=args.mu,
                collapse_model=args.collapse_model,
                collapse_gamma=args.collapse_gamma,
                shots=args.shots,
                add_hardware_noise=args.add_hardware_noise,
                hw_params_path=args.hw_params,
            )

            print(f"\nResults (visibility):")
            print(f"  Visibility: {result['visibility']:.4f} +/- {result['visibility_error']:.4f}")
            print(f"  Probabilities: {result['probabilities']}")

    print("\n" + "=" * 70)
    print("Collapse Model Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
