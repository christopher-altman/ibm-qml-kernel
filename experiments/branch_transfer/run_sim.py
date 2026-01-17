"""
Simulator baselines for the branch-conditioned message transfer experiment.

Runs the protocol on:
    - Ideal (statevector) simulator
    - Aer noisy simulator with IBM hardware calibration

Date: 2026-01-17
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, thermal_relaxation_error,
    depolarizing_error, ReadoutError
)

# Optional: IBM Runtime for backend-matched noise
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False

from circuit import (
    build_branch_transfer_circuit,
    build_control_circuit,
    build_coherence_witness_circuit,
    get_circuit_stats,
    get_expected_ideal_distribution,
    get_expected_ideal_coherence,
    compute_visibility_from_counts,
    compute_coherence_witness_from_counts,
    compute_normalized_coherence,
    compute_coherence_magnitude,
)


def load_hardware_params(path: str = 'data/ibm_hardware_params_2026.json') -> dict:
    """Load IBM hardware calibration parameters."""
    with open(path, 'r') as f:
        return json.load(f)


def build_noise_model_from_params(hw_params: dict, num_qubits: int = 5) -> NoiseModel:
    """
    Build a noise model from hardware calibration parameters.

    Parameters
    ----------
    hw_params : dict
        Hardware parameters (loaded from JSON).
    num_qubits : int
        Number of qubits in the circuit.

    Returns
    -------
    NoiseModel
        Aer noise model configured with hardware-calibrated errors.
    """
    noise_model = NoiseModel()

    # Extract parameters
    T1 = hw_params['coherence_times']['T1_median_us'] * 1e-6  # to seconds
    T2 = hw_params['coherence_times']['T2_median_us'] * 1e-6

    sq_error = hw_params['gate_errors']['single_qubit']['median_percent'] / 100
    tq_error = hw_params['gate_errors']['two_qubit_ECR']['median_percent'] / 100
    ro_error = hw_params['readout_errors']['median_percent'] / 100

    # Gate times
    t_single = 35e-9   # 35 ns for SX, RZ
    t_two = 600e-9     # 600 ns for ECR

    # Thermal relaxation errors
    thermal_single = thermal_relaxation_error(T1, T2, t_single)
    thermal_two = thermal_relaxation_error(T1, T2, t_two)

    # Depolarizing errors
    depol_single = depolarizing_error(sq_error, 1)
    depol_two = depolarizing_error(tq_error, 2)

    # Combined errors
    sq_combined = thermal_single.compose(depol_single)
    tq_combined = thermal_two.compose(depol_two)

    # Add to single-qubit gates
    for gate in ['sx', 'rz', 'x', 'h', 'id']:
        noise_model.add_all_qubit_quantum_error(sq_combined, gate)

    # Add to two-qubit gates
    noise_model.add_all_qubit_quantum_error(tq_combined, 'ecr')
    noise_model.add_all_qubit_quantum_error(tq_combined, 'cx')

    # Readout errors
    ro_matrix = [
        [1 - ro_error, ro_error],
        [ro_error, 1 - ro_error]
    ]
    readout_error = ReadoutError(ro_matrix)

    for qubit in range(num_qubits):
        noise_model.add_readout_error(readout_error, [qubit])

    return noise_model


def build_noise_model_from_backend(backend_name: str, output_dir: Path) -> tuple:
    """
    Build a noise model from IBM Quantum backend properties.

    Parameters
    ----------
    backend_name : str
        IBM backend name (e.g., 'ibm_fez').
    output_dir : Path
        Directory to save calibration snapshot.

    Returns
    -------
    tuple
        (NoiseModel, properties_dict, snapshot_path)
        Returns (None, None, None) if failed.
    """
    if not IBM_RUNTIME_AVAILABLE:
        print("ERROR: qiskit-ibm-runtime not installed, cannot fetch backend noise")
        return None, None, None

    try:
        # Connect to IBM Quantum
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)

        # Build noise model from backend
        noise_model = NoiseModel.from_backend(backend)

        # Save calibration snapshot
        snapshot_dir = output_dir / 'noise_snapshots'
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_path = snapshot_dir / f'{backend_name}_{timestamp}_properties.json'

        # Gather properties for snapshot
        properties = {
            'backend_name': backend_name,
            'timestamp': timestamp,
            'num_qubits': backend.num_qubits,
            'version': str(backend.version),
            'basis_gates': backend.operation_names,
        }

        # Try to include calibration data if available
        try:
            props = backend.properties()
            if props:
                properties['properties_timestamp'] = str(props.last_update_date) if hasattr(props, 'last_update_date') else 'unknown'
        except Exception:
            pass

        # Save snapshot
        with open(snapshot_path, 'w') as f:
            json.dump(properties, f, indent=2)

        print(f"  Noise model built from backend: {backend_name}")
        print(f"  Calibration snapshot saved: {snapshot_path}")

        return noise_model, properties, snapshot_path

    except Exception as e:
        print(f"ERROR: Failed to build noise from backend {backend_name}: {e}")
        return None, None, None


def run_ideal_simulation(
    mu: int = 1,
    include_memory_erase: bool = True,
    shots: int = 20000,
) -> dict:
    """
    Run the circuit on an ideal (noiseless) simulator.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether to include memory erasure step.
    shots : int
        Number of shots.

    Returns
    -------
    dict
        Results including counts, probabilities, visibility.
    """
    # Build circuit
    if include_memory_erase:
        qc = build_branch_transfer_circuit(mu=mu, include_memory_erase=True)
        mode = 'main'
    else:
        qc = build_control_circuit(mu=mu)
        mode = 'control'

    # Ideal simulator
    backend = AerSimulator(method='statevector')
    qc_transpiled = transpile(qc, backend, optimization_level=0)

    # Run
    job = backend.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Compute probabilities
    probs = {k: v / shots for k, v in counts.items()}

    # Compute visibility
    V, V_err, cond_probs = compute_visibility_from_counts(counts, shots)

    # Expected distribution
    expected = get_expected_ideal_distribution(mu, include_memory_erase)

    # Circuit stats
    stats = get_circuit_stats(qc)

    return {
        'backend': 'aer_simulator_statevector',
        'backend_type': 'ideal',
        'mode': mode,
        'mu': mu,
        'shots': shots,
        'counts': counts,
        'probabilities': probs,
        'visibility': V,
        'visibility_error': V_err,
        'conditional_probabilities': cond_probs,
        'expected_distribution': expected,
        'circuit_stats': stats,
        'transpiled_depth': qc_transpiled.depth(),
        'transpiled_size': qc_transpiled.size(),
        'optimization_level': 0,
    }


def run_noisy_simulation(
    mu: int = 1,
    include_memory_erase: bool = True,
    shots: int = 20000,
    optimization_level: int = 1,
    hw_params_path: str = 'data/ibm_hardware_params_2026.json',
    noise_from_backend: str = None,
    output_dir: Path = Path('artifacts/branch_transfer'),
) -> dict:
    """
    Run the circuit on a noisy Aer simulator with hardware calibration.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether to include memory erasure step.
    shots : int
        Number of shots.
    optimization_level : int
        Transpiler optimization level (0-3).
    hw_params_path : str
        Path to hardware parameters JSON.
    noise_from_backend : str, optional
        Backend name to fetch noise model from. Overrides hw_params_path.
    output_dir : Path
        Output directory for noise snapshots.

    Returns
    -------
    dict
        Results including counts, probabilities, visibility.
    """
    # Build circuit
    if include_memory_erase:
        qc = build_branch_transfer_circuit(mu=mu, include_memory_erase=True)
        mode = 'main'
    else:
        qc = build_control_circuit(mu=mu)
        mode = 'control'

    # Build noise model
    noise_backend_name = None
    noise_snapshot_path = None

    if noise_from_backend:
        # Backend-matched noise
        noise_model, noise_props, noise_snapshot_path = build_noise_model_from_backend(
            noise_from_backend, output_dir
        )
        if noise_model is None:
            raise RuntimeError(f"Failed to build noise model from backend {noise_from_backend}")
        noise_backend_name = noise_from_backend
        hw_params = {'backend_name': noise_from_backend}
    else:
        # Parameter-based noise (legacy)
        hw_params = load_hardware_params(hw_params_path)
        noise_model = build_noise_model_from_params(hw_params, num_qubits=5)

    # Noisy simulator
    backend = AerSimulator(noise_model=noise_model)

    # Transpile
    qc_transpiled = transpile(
        qc,
        backend,
        optimization_level=optimization_level,
    )

    # Run
    job = backend.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Compute probabilities
    probs = {k: v / shots for k, v in counts.items()}

    # Compute visibility
    V, V_err, cond_probs = compute_visibility_from_counts(counts, shots)

    # Expected distribution (ideal reference)
    expected = get_expected_ideal_distribution(mu, include_memory_erase)

    # Circuit stats
    stats = get_circuit_stats(qc)

    result = {
        'backend': f'aer_simulator_noisy_{hw_params["backend_name"]}',
        'backend_type': 'noisy_simulator',
        'noise_source': hw_params['backend_name'],
        'mode': mode,
        'mu': mu,
        'shots': shots,
        'counts': counts,
        'probabilities': probs,
        'visibility': V,
        'visibility_error': V_err,
        'conditional_probabilities': cond_probs,
        'expected_distribution': expected,
        'circuit_stats': stats,
        'transpiled_depth': qc_transpiled.depth(),
        'transpiled_size': qc_transpiled.size(),
        'optimization_level': optimization_level,
    }

    # Add backend-matched noise metadata if used
    if noise_backend_name:
        result['noise_model_backend'] = noise_backend_name
        if noise_snapshot_path:
            result['noise_snapshot_path'] = str(noise_snapshot_path)
    else:
        # Legacy noise params
        result['noise_params'] = {
            'T1_us': hw_params['coherence_times']['T1_median_us'],
            'T2_us': hw_params['coherence_times']['T2_median_us'],
            'single_qubit_error_pct': hw_params['gate_errors']['single_qubit']['median_percent'],
            'two_qubit_error_pct': hw_params['gate_errors']['two_qubit_ECR']['median_percent'],
            'readout_error_pct': hw_params['readout_errors']['median_percent'],
        }

    return result


def run_optimization_level_sweep(
    mu: int = 1,
    include_memory_erase: bool = True,
    shots: int = 20000,
    hw_params_path: str = 'data/ibm_hardware_params_2026.json',
) -> list:
    """
    Sweep optimization levels (0-3) for transpilation sensitivity analysis.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether to include memory erasure step.
    shots : int
        Number of shots per configuration.
    hw_params_path : str
        Path to hardware parameters JSON.

    Returns
    -------
    list
        Results for each optimization level.
    """
    results = []
    for opt_level in [0, 1, 2, 3]:
        print(f"  Running optimization_level={opt_level}...")
        result = run_noisy_simulation(
            mu=mu,
            include_memory_erase=include_memory_erase,
            shots=shots,
            optimization_level=opt_level,
            hw_params_path=hw_params_path,
        )
        results.append(result)
    return results


# =============================================================================
# Coherence Witness Simulation Functions
# =============================================================================


def run_coherence_ideal_simulation(
    mu: int = 1,
    include_memory_erase: bool = True,
    basis: str = 'X',
    shots: int = 20000,
) -> dict:
    """
    Run coherence witness measurement on ideal simulator.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether to include memory erasure step.
    basis : {'X', 'Y'}
        Measurement basis for coherence witness.
    shots : int
        Number of shots.

    Returns
    -------
    dict
        Results including counts, coherence witness W, and normalized W_tilde.
    """
    # Build coherence witness circuit
    qc = build_coherence_witness_circuit(
        mu=mu,
        include_memory_erase=include_memory_erase,
        basis=basis,
        barrier=True
    )
    mode = 'main' if include_memory_erase else 'control'

    # Ideal simulator
    backend = AerSimulator(method='statevector')
    qc_transpiled = transpile(qc, backend, optimization_level=0)

    # Run
    job = backend.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Compute coherence witness
    W, W_err, parity_counts = compute_coherence_witness_from_counts(counts, shots)

    # Get ideal reference
    W_ideal = get_expected_ideal_coherence(mu, include_memory_erase, basis)

    # Compute normalized coherence
    W_tilde, W_tilde_err = compute_normalized_coherence(W, W_ideal, W_err, 0.0)

    # Circuit stats
    stats = get_circuit_stats(qc)

    return {
        'backend': 'aer_simulator_statevector',
        'backend_type': 'ideal',
        'experiment_mode': 'coherence_witness',
        'measurement_basis': basis,
        'circuit_mode': mode,
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
        'transpiled_depth': qc_transpiled.depth(),
        'transpiled_size': qc_transpiled.size(),
        'optimization_level': 0,
    }


def run_coherence_noisy_simulation(
    mu: int = 1,
    include_memory_erase: bool = True,
    basis: str = 'X',
    shots: int = 20000,
    optimization_level: int = 1,
    hw_params_path: str = 'data/ibm_hardware_params_2026.json',
    W_ideal_reference: float = None,
    noise_from_backend: str = None,
    output_dir: Path = Path('artifacts/branch_transfer'),
) -> dict:
    """
    Run coherence witness measurement on noisy simulator.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether to include memory erasure step.
    basis : {'X', 'Y'}
        Measurement basis for coherence witness.
    shots : int
        Number of shots.
    optimization_level : int
        Transpiler optimization level.
    hw_params_path : str
        Path to hardware parameters JSON.
    W_ideal_reference : float, optional
        Pre-computed ideal W value for normalization.
        If None, uses theoretical expectation.
    noise_from_backend : str, optional
        Backend name to fetch noise model from.
    output_dir : Path
        Output directory for noise snapshots.

    Returns
    -------
    dict
        Results including counts, coherence witness W, and normalized W_tilde.
    """
    # Build coherence witness circuit
    qc = build_coherence_witness_circuit(
        mu=mu,
        include_memory_erase=include_memory_erase,
        basis=basis,
        barrier=False  # No barriers for transpilation
    )
    mode = 'main' if include_memory_erase else 'control'

    # Build noise model
    noise_backend_name = None
    noise_snapshot_path = None

    if noise_from_backend:
        noise_model, noise_props, noise_snapshot_path = build_noise_model_from_backend(
            noise_from_backend, output_dir
        )
        if noise_model is None:
            raise RuntimeError(f"Failed to build noise model from backend {noise_from_backend}")
        noise_backend_name = noise_from_backend
        hw_params = {'backend_name': noise_from_backend}
    else:
        hw_params = load_hardware_params(hw_params_path)
        noise_model = build_noise_model_from_params(hw_params, num_qubits=5)

    # Noisy simulator
    backend = AerSimulator(noise_model=noise_model)

    # Transpile
    qc_transpiled = transpile(qc, backend, optimization_level=optimization_level)

    # Run
    job = backend.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Compute coherence witness
    W, W_err, parity_counts = compute_coherence_witness_from_counts(counts, shots)

    # Get ideal reference (use provided or theoretical)
    if W_ideal_reference is not None:
        W_ideal = W_ideal_reference
    else:
        W_ideal = get_expected_ideal_coherence(mu, include_memory_erase, basis)

    # Compute normalized coherence
    W_tilde, W_tilde_err = compute_normalized_coherence(W, W_ideal, W_err, 0.0)

    # Circuit stats
    stats = get_circuit_stats(qc)

    result = {
        'backend': f'aer_simulator_noisy_{hw_params["backend_name"]}',
        'backend_type': 'noisy_simulator',
        'noise_source': hw_params['backend_name'],
        'experiment_mode': 'coherence_witness',
        'measurement_basis': basis,
        'circuit_mode': mode,
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
        'transpiled_depth': qc_transpiled.depth(),
        'transpiled_size': qc_transpiled.size(),
        'optimization_level': optimization_level,
    }

    # Add backend-matched noise metadata if used
    if noise_backend_name:
        result['noise_model_backend'] = noise_backend_name
        if noise_snapshot_path:
            result['noise_snapshot_path'] = str(noise_snapshot_path)
    else:
        result['noise_params'] = {
            'T1_us': hw_params['coherence_times']['T1_median_us'],
            'T2_us': hw_params['coherence_times']['T2_median_us'],
            'single_qubit_error_pct': hw_params['gate_errors']['single_qubit']['median_percent'],
            'two_qubit_error_pct': hw_params['gate_errors']['two_qubit_ECR']['median_percent'],
            'readout_error_pct': hw_params['readout_errors']['median_percent'],
        }

    return result


def run_coherence_full(
    mu: int = 1,
    include_memory_erase: bool = True,
    shots: int = 20000,
    optimization_level: int = 1,
    hw_params_path: str = 'data/ibm_hardware_params_2026.json',
    include_Y_basis: bool = True,
) -> dict:
    """
    Run full coherence witness measurement (X and optionally Y basis).

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether to include memory erasure step.
    shots : int
        Number of shots per basis.
    optimization_level : int
        Transpiler optimization level.
    hw_params_path : str
        Path to hardware parameters JSON.
    include_Y_basis : bool
        If True, also measure in Y basis for phase-robust magnitude.

    Returns
    -------
    dict
        Combined results with W_X, W_Y (optional), and |C| (optional).
    """
    # First get ideal reference from simulator
    ideal_X = run_coherence_ideal_simulation(
        mu=mu, include_memory_erase=include_memory_erase, basis='X', shots=shots
    )
    W_X_ideal = ideal_X['W_X']

    # Run noisy X basis
    noisy_X = run_coherence_noisy_simulation(
        mu=mu, include_memory_erase=include_memory_erase, basis='X',
        shots=shots, optimization_level=optimization_level,
        hw_params_path=hw_params_path, W_ideal_reference=W_X_ideal
    )

    result = {
        'experiment_mode': 'coherence_witness_full',
        'mu': mu,
        'circuit_mode': 'main' if include_memory_erase else 'control',
        'shots': shots,
        'optimization_level': optimization_level,
        # X basis results
        'W_X_ideal': W_X_ideal,
        'W_X_ideal_error': ideal_X['W_X_error'],
        'W_X_noisy': noisy_X['W_X'],
        'W_X_noisy_error': noisy_X['W_X_error'],
        'W_X_tilde': noisy_X['W_X_tilde'],
        'W_X_tilde_error': noisy_X['W_X_tilde_error'],
    }

    if include_Y_basis:
        ideal_Y = run_coherence_ideal_simulation(
            mu=mu, include_memory_erase=include_memory_erase, basis='Y', shots=shots
        )
        W_Y_ideal = ideal_Y['W_Y']

        noisy_Y = run_coherence_noisy_simulation(
            mu=mu, include_memory_erase=include_memory_erase, basis='Y',
            shots=shots, optimization_level=optimization_level,
            hw_params_path=hw_params_path, W_ideal_reference=W_Y_ideal if abs(W_Y_ideal) > 1e-10 else 1.0
        )

        result['W_Y_ideal'] = W_Y_ideal
        result['W_Y_ideal_error'] = ideal_Y['W_Y_error']
        result['W_Y_noisy'] = noisy_Y['W_Y']
        result['W_Y_noisy_error'] = noisy_Y['W_Y_error']
        result['W_Y_tilde'] = noisy_Y['W_Y_tilde']
        result['W_Y_tilde_error'] = noisy_Y['W_Y_tilde_error']

        # Compute phase-robust magnitude
        C_ideal, C_ideal_err = compute_coherence_magnitude(
            W_X_ideal, W_Y_ideal, ideal_X['W_X_error'], ideal_Y['W_Y_error']
        )
        C_noisy, C_noisy_err = compute_coherence_magnitude(
            noisy_X['W_X'], noisy_Y['W_Y'], noisy_X['W_X_error'], noisy_Y['W_Y_error']
        )

        result['C_magnitude_ideal'] = C_ideal
        result['C_magnitude_noisy'] = C_noisy
        result['C_magnitude_noisy_error'] = C_noisy_err

        # Normalized magnitude
        if C_ideal > 1e-10:
            C_tilde = C_noisy / C_ideal
            C_tilde_err = C_noisy_err / C_ideal if C_ideal > 0 else 0
        else:
            C_tilde = C_noisy
            C_tilde_err = C_noisy_err

        result['C_tilde'] = C_tilde
        result['C_tilde_error'] = C_tilde_err

    return result


def save_result(result: dict, output_dir: Path, prefix: str = '') -> Path:
    """
    Save a result to JSON with deterministic naming.

    Parameters
    ----------
    result : dict
        Result data to save.
    output_dir : Path
        Output directory.
    prefix : str
        Optional prefix for filename.

    Returns
    -------
    Path
        Path to saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backend = result.get('backend', 'unknown').replace(' ', '_')
    mode = result.get('mode', 'main')
    mu = result.get('mu', 1)
    shots = result.get('shots', 0)
    opt_level = result.get('optimization_level', 0)

    filename = f"{prefix}{timestamp}_{backend}_{mode}_mu-{mu}_shots-{shots}_opt-{opt_level}.json"
    filepath = output_dir / filename

    # Add metadata
    result_with_meta = {
        **result,
        'timestamp': timestamp,
        'qiskit_version': _get_qiskit_version(),
    }

    with open(filepath, 'w') as f:
        json.dump(result_with_meta, f, indent=2)

    return filepath


def _get_qiskit_version() -> dict:
    """Get Qiskit package versions."""
    versions = {}
    try:
        import qiskit
        versions['qiskit'] = qiskit.__version__
    except Exception:
        pass
    try:
        import qiskit_aer
        versions['qiskit_aer'] = qiskit_aer.__version__
    except Exception:
        pass
    return versions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run branch-transfer experiment on simulators'
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
        '--ideal-only', action='store_true',
        help='Run only ideal simulation (skip noisy)'
    )
    parser.add_argument(
        '--noisy-only', action='store_true',
        help='Run only noisy simulation (skip ideal)'
    )
    parser.add_argument(
        '--control', action='store_true',
        help='Run control circuit (no memory erase) instead of main'
    )
    parser.add_argument(
        '--opt-sweep', action='store_true',
        help='Sweep optimization levels 0-3'
    )
    parser.add_argument(
        '--hw-params', type=str, default='data/ibm_hardware_params_2026.json',
        help='Path to hardware parameters JSON'
    )
    # Coherence witness mode
    parser.add_argument(
        '--mode', type=str, choices=['rp_z', 'coherence_witness'], default='rp_z',
        help='Experiment mode: rp_z (measure R,P in Z) or coherence_witness (measure Q,R,F,P in X/Y)'
    )
    parser.add_argument(
        '--include-y-basis', action='store_true',
        help='Include Y-basis measurement for phase-robust coherence magnitude'
    )
    parser.add_argument(
        '--noise-from-backend', type=str, default=None,
        help='Backend name to fetch noise model from (e.g., ibm_fez). Overrides --hw-params.'
    )
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Branch-Transfer Experiment: Simulator Baselines")
    print("=" * 70)
    print(f"  mu = {args.mu}")
    print(f"  shots = {args.shots}")
    print(f"  circuit_mode = {'control (no memory erase)' if args.control else 'main'}")
    print(f"  experiment_mode = {args.mode}")
    print(f"  output_dir = {output_dir}")
    print()

    include_memory_erase = not args.control
    saved_files = []

    if args.mode == 'coherence_witness':
        # Coherence witness mode
        print("=" * 70)
        print("Mode: Coherence Witness (W_X, W_Y measurement)")
        print("=" * 70)
        print()

        # Run X-basis coherence
        print("1. Running X-basis coherence witness (ideal)...")
        result_ideal_X = run_coherence_ideal_simulation(
            mu=args.mu, include_memory_erase=include_memory_erase,
            basis='X', shots=args.shots
        )
        filepath = save_result(result_ideal_X, output_dir, prefix='coherence_')
        saved_files.append(filepath)
        print(f"   W_X (ideal): {result_ideal_X['W_X']:.4f} +/- {result_ideal_X['W_X_error']:.4f}")
        print()

        print("2. Running X-basis coherence witness (noisy)...")
        result_noisy_X = run_coherence_noisy_simulation(
            mu=args.mu, include_memory_erase=include_memory_erase,
            basis='X', shots=args.shots, optimization_level=1,
            hw_params_path=args.hw_params,
            W_ideal_reference=result_ideal_X['W_X'],
            noise_from_backend=args.noise_from_backend,
            output_dir=output_dir,
        )
        filepath = save_result(result_noisy_X, output_dir, prefix='coherence_')
        saved_files.append(filepath)
        print(f"   W_X (noisy): {result_noisy_X['W_X']:.4f} +/- {result_noisy_X['W_X_error']:.4f}")
        print(f"   W_X_tilde:   {result_noisy_X['W_X_tilde']:.4f} +/- {result_noisy_X['W_X_tilde_error']:.4f}")
        print()

        if args.include_y_basis:
            print("3. Running Y-basis coherence witness (ideal)...")
            result_ideal_Y = run_coherence_ideal_simulation(
                mu=args.mu, include_memory_erase=include_memory_erase,
                basis='Y', shots=args.shots
            )
            filepath = save_result(result_ideal_Y, output_dir, prefix='coherence_')
            saved_files.append(filepath)
            print(f"   W_Y (ideal): {result_ideal_Y['W_Y']:.4f} +/- {result_ideal_Y['W_Y_error']:.4f}")
            print()

            print("4. Running Y-basis coherence witness (noisy)...")
            W_Y_ref = result_ideal_Y['W_Y'] if abs(result_ideal_Y['W_Y']) > 1e-10 else 1.0
            result_noisy_Y = run_coherence_noisy_simulation(
                mu=args.mu, include_memory_erase=include_memory_erase,
                basis='Y', shots=args.shots, optimization_level=1,
                hw_params_path=args.hw_params,
                W_ideal_reference=W_Y_ref,
                noise_from_backend=args.noise_from_backend,
                output_dir=output_dir,
            )
            filepath = save_result(result_noisy_Y, output_dir, prefix='coherence_')
            saved_files.append(filepath)
            print(f"   W_Y (noisy): {result_noisy_Y['W_Y']:.4f} +/- {result_noisy_Y['W_Y_error']:.4f}")
            print()

            # Compute and display coherence magnitude
            C_ideal, _ = compute_coherence_magnitude(
                result_ideal_X['W_X'], result_ideal_Y['W_Y']
            )
            C_noisy, C_err = compute_coherence_magnitude(
                result_noisy_X['W_X'], result_noisy_Y['W_Y'],
                result_noisy_X['W_X_error'], result_noisy_Y['W_Y_error']
            )
            C_tilde = C_noisy / C_ideal if C_ideal > 1e-10 else C_noisy

            print("5. Coherence magnitude:")
            print(f"   |C| (ideal): {C_ideal:.4f}")
            print(f"   |C| (noisy): {C_noisy:.4f} +/- {C_err:.4f}")
            print(f"   |C|_tilde:   {C_tilde:.4f}")
            print()

    else:
        # Original rp_z mode
        # Run ideal simulation
        if not args.noisy_only:
            print("1. Running ideal simulation...")
            result_ideal = run_ideal_simulation(
                mu=args.mu,
                include_memory_erase=include_memory_erase,
                shots=args.shots,
            )
            filepath = save_result(result_ideal, output_dir, prefix='sim_')
            saved_files.append(filepath)
            print(f"   Visibility: {result_ideal['visibility']:.4f} +/- {result_ideal['visibility_error']:.4f}")
            print(f"   Probabilities: {result_ideal['probabilities']}")
            print(f"   Saved: {filepath}")
            print()

        # Run noisy simulation
        if not args.ideal_only:
            if args.opt_sweep:
                print("2. Running noisy simulation (optimization level sweep)...")
                results_noisy = run_optimization_level_sweep(
                    mu=args.mu,
                    include_memory_erase=include_memory_erase,
                    shots=args.shots,
                    hw_params_path=args.hw_params,
                )
                for result in results_noisy:
                    filepath = save_result(result, output_dir, prefix='sim_')
                    saved_files.append(filepath)
                    opt = result['optimization_level']
                    print(f"   opt_level={opt}: V={result['visibility']:.4f} +/- {result['visibility_error']:.4f}, depth={result['transpiled_depth']}")
                print()
            else:
                print("2. Running noisy simulation (optimization_level=1)...")
                result_noisy = run_noisy_simulation(
                    mu=args.mu,
                    include_memory_erase=include_memory_erase,
                    shots=args.shots,
                    optimization_level=1,
                    noise_from_backend=args.noise_from_backend,
                    output_dir=output_dir,
                    hw_params_path=args.hw_params,
                )
                filepath = save_result(result_noisy, output_dir, prefix='sim_')
                saved_files.append(filepath)
                print(f"   Visibility: {result_noisy['visibility']:.4f} +/- {result_noisy['visibility_error']:.4f}")
                print(f"   Probabilities: {result_noisy['probabilities']}")
                print(f"   Saved: {filepath}")
                print()

    print("=" * 70)
    print("Simulator Baselines Complete")
    print("=" * 70)
    print("\nSaved files:")
    for f in saved_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
