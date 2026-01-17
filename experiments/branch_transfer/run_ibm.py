"""
IBM Quantum Runtime execution for the branch-conditioned message transfer experiment.

Runs the protocol on IBM Quantum hardware using Qiskit Runtime Sampler.

Date: 2026-01-17
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from circuit import (
    build_branch_transfer_circuit,
    build_control_circuit,
    build_coherence_witness_circuit,
    get_circuit_stats,
    get_expected_ideal_distribution,
    compute_visibility_from_counts,
    compute_coherence_witness_from_counts,
    compute_normalized_coherence,
    get_expected_ideal_coherence,
    COHERENCE_QUBITS,
)

# IBM Runtime imports (may not be available in all environments)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit_ibm_runtime.options import SamplerOptions
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False
    print("Warning: qiskit-ibm-runtime not available. Hardware execution disabled.")


def get_available_backends(service) -> List[str]:
    """Get list of available IBM Quantum backends."""
    backends = service.backends()
    return [b.name for b in backends if b.status().operational]


def select_backend(service, preferred: Optional[str] = None) -> tuple:
    """
    Select an IBM Quantum backend.

    Parameters
    ----------
    service : QiskitRuntimeService
        Authenticated runtime service.
    preferred : str, optional
        Preferred backend name. If None, selects least busy.

    Returns
    -------
    backend
        Selected backend object.
    name : str
        Backend name.
    """
    if preferred:
        backend = service.backend(preferred)
        return backend, preferred

    # Select least busy operational backend
    backends = service.backends(
        simulator=False,
        operational=True,
        min_num_qubits=5
    )
    if not backends:
        raise RuntimeError("No operational backends with 5+ qubits available")

    # Sort by queue depth
    backends_sorted = sorted(backends, key=lambda b: b.status().pending_jobs)
    backend = backends_sorted[0]
    return backend, backend.name


def run_on_hardware(
    mu: int = 1,
    include_memory_erase: bool = True,
    shots: int = 20000,
    optimization_level: int = 1,
    backend_name: Optional[str] = None,
    channel: str = 'ibm_quantum',
) -> dict:
    """
    Run the circuit on IBM Quantum hardware.

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
    backend_name : str, optional
        Specific backend to use. If None, selects least busy.
    channel : str
        IBM Quantum channel ('ibm_quantum' or 'ibm_cloud').

    Returns
    -------
    dict
        Results including counts, probabilities, visibility.
    """
    if not IBM_RUNTIME_AVAILABLE:
        raise RuntimeError("qiskit-ibm-runtime not installed")

    # Connect to service
    print("  Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel=channel)

    # Select backend
    backend, backend_name = select_backend(service, backend_name)
    print(f"  Selected backend: {backend_name}")
    print(f"  Pending jobs: {backend.status().pending_jobs}")

    # Build circuit
    if include_memory_erase:
        qc = build_branch_transfer_circuit(mu=mu, include_memory_erase=True, barrier=False)
        mode = 'main'
    else:
        qc = build_control_circuit(mu=mu, barrier=False)
        mode = 'control'

    # Get circuit stats before transpilation
    stats_original = get_circuit_stats(qc)

    # Transpile for target backend
    print(f"  Transpiling (optimization_level={optimization_level})...")
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend
    )
    qc_transpiled = pm.run(qc)

    # Get transpiled stats
    transpiled_depth = qc_transpiled.depth()
    transpiled_size = qc_transpiled.size()

    # Count 2Q gates in transpiled circuit
    two_qubit_gates = sum(
        1 for inst in qc_transpiled.data
        if len(inst.qubits) == 2 and inst.operation.name not in ['barrier', 'measure']
    )

    # Get physical qubit layout
    layout = None
    if qc_transpiled.layout is not None:
        try:
            layout = {
                'initial_layout': str(qc_transpiled.layout.initial_layout),
                'final_layout': str(qc_transpiled.layout.final_layout) if qc_transpiled.layout.final_layout else None,
            }
        except Exception:
            layout = {'info': 'layout extraction failed'}

    # Run using Sampler
    print(f"  Submitting job ({shots} shots)...")
    sampler = Sampler(mode=backend)
    job = sampler.run([qc_transpiled], shots=shots)
    print(f"  Job ID: {job.job_id()}")

    # Wait for result
    print("  Waiting for result...")
    result = job.result()

    # Extract counts from SamplerV2 result
    pub_result = result[0]
    counts_raw = pub_result.data.c.get_counts()

    # Convert counts to standard format (may need bit-order adjustment)
    counts = {}
    for bitstring, count in counts_raw.items():
        # Ensure 2-bit format
        bs = bitstring.zfill(2)[-2:]  # Take last 2 bits
        counts[bs] = counts.get(bs, 0) + count

    total_shots = sum(counts.values())

    # Compute probabilities
    probs = {k: v / total_shots for k, v in counts.items()}

    # Compute visibility
    V, V_err, cond_probs = compute_visibility_from_counts(counts, total_shots)

    # Expected distribution (ideal reference)
    expected = get_expected_ideal_distribution(mu, include_memory_erase)

    # Get backend configuration
    backend_config = {
        'backend_name': backend_name,
        'num_qubits': backend.num_qubits,
        'processor_type': getattr(backend, 'processor_type', {}).get('family', 'unknown') if hasattr(backend, 'processor_type') else 'unknown',
    }

    return {
        'backend': backend_name,
        'backend_type': 'hardware',
        'backend_config': backend_config,
        'job_id': job.job_id(),
        'mode': mode,
        'mu': mu,
        'shots': shots,
        'actual_shots': total_shots,
        'counts': counts,
        'probabilities': probs,
        'visibility': V,
        'visibility_error': V_err,
        'conditional_probabilities': cond_probs,
        'expected_distribution': expected,
        'circuit_stats': stats_original,
        'transpiled_depth': transpiled_depth,
        'transpiled_size': transpiled_size,
        'transpiled_two_qubit_gates': two_qubit_gates,
        'optimization_level': optimization_level,
        'physical_layout': layout,
    }


def run_coherence_on_hardware(
    mu: int = 1,
    include_memory_erase: bool = True,
    shots: int = 20000,
    optimization_level: int = 1,
    backend_name: Optional[str] = None,
    channel: str = 'ibm_quantum',
    basis: str = 'X',
) -> dict:
    """
    Run coherence witness measurement on IBM Quantum hardware.

    Measures ⟨X_Q X_R X_F X_P⟩ (or Y) to probe off-diagonal coherence,
    which is sensitive to dephasing/collapse effects unlike visibility.

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
    backend_name : str, optional
        Specific backend to use. If None, selects least busy.
    channel : str
        IBM Quantum channel ('ibm_quantum' or 'ibm_cloud').
    basis : {'X', 'Y'}
        Measurement basis for coherence witness.

    Returns
    -------
    dict
        Results including W, W_tilde (normalized coherence).
    """
    if not IBM_RUNTIME_AVAILABLE:
        raise RuntimeError("qiskit-ibm-runtime not installed")

    # Connect to service
    print("  Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel=channel)

    # Select backend
    backend, backend_name = select_backend(service, backend_name)
    print(f"  Selected backend: {backend_name}")
    print(f"  Pending jobs: {backend.status().pending_jobs}")

    # Build coherence witness circuit
    qc = build_coherence_witness_circuit(
        mu=mu,
        include_memory_erase=include_memory_erase,
        basis=basis,
        barrier=False,
    )
    mode = 'coherence_witness'

    # Get circuit stats before transpilation
    stats_original = get_circuit_stats(qc)

    # Get ideal reference for normalization
    W_ideal = get_expected_ideal_coherence(mu, include_memory_erase, basis)

    # Transpile for target backend
    print(f"  Transpiling (optimization_level={optimization_level})...")
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend
    )
    qc_transpiled = pm.run(qc)

    # Get transpiled stats
    transpiled_depth = qc_transpiled.depth()
    transpiled_size = qc_transpiled.size()

    # Count 2Q gates in transpiled circuit
    two_qubit_gates = sum(
        1 for inst in qc_transpiled.data
        if len(inst.qubits) == 2 and inst.operation.name not in ['barrier', 'measure']
    )

    # Get physical qubit layout
    layout = None
    if qc_transpiled.layout is not None:
        try:
            layout = {
                'initial_layout': str(qc_transpiled.layout.initial_layout),
                'final_layout': str(qc_transpiled.layout.final_layout) if qc_transpiled.layout.final_layout else None,
            }
        except Exception:
            layout = {'info': 'layout extraction failed'}

    # Run using Sampler
    print(f"  Submitting job ({shots} shots, {basis} basis)...")
    sampler = Sampler(mode=backend)
    job = sampler.run([qc_transpiled], shots=shots)
    print(f"  Job ID: {job.job_id()}")

    # Wait for result
    print("  Waiting for result...")
    result = job.result()

    # Extract counts from SamplerV2 result
    pub_result = result[0]
    counts_raw = pub_result.data.c.get_counts()

    # Convert counts to standard format (4-bit for QRFP)
    n_measure = len(COHERENCE_QUBITS)
    counts = {}
    for bitstring, count in counts_raw.items():
        bs = bitstring.zfill(n_measure)[-n_measure:]
        counts[bs] = counts.get(bs, 0) + count

    total_shots = sum(counts.values())

    # Compute coherence witness
    W, W_err, parity_counts = compute_coherence_witness_from_counts(counts, total_shots)

    # Compute normalized coherence
    W_tilde, W_tilde_err = compute_normalized_coherence(W, W_ideal, W_err, 0.0)

    # Get backend configuration
    backend_config = {
        'backend_name': backend_name,
        'num_qubits': backend.num_qubits,
        'processor_type': getattr(backend, 'processor_type', {}).get('family', 'unknown') if hasattr(backend, 'processor_type') else 'unknown',
    }

    return {
        'backend': backend_name,
        'backend_type': 'hardware',
        'backend_config': backend_config,
        'job_id': job.job_id(),
        'mode': mode,
        'measurement_basis': basis,
        'mu': mu,
        'shots': shots,
        'actual_shots': total_shots,
        'counts': counts,
        f'W_{basis}': W,
        f'W_{basis}_error': W_err,
        f'W_{basis}_ideal': W_ideal,
        f'W_{basis}_tilde': W_tilde,
        f'W_{basis}_tilde_error': W_tilde_err,
        'parity_counts': parity_counts,
        'circuit_stats': stats_original,
        'transpiled_depth': transpiled_depth,
        'transpiled_size': transpiled_size,
        'transpiled_two_qubit_gates': two_qubit_gates,
        'optimization_level': optimization_level,
        'physical_layout': layout,
    }


def run_coherence_full_hardware(
    mu: int = 1,
    include_memory_erase: bool = True,
    shots: int = 20000,
    optimization_level: int = 1,
    backend_name: Optional[str] = None,
    channel: str = 'ibm_quantum',
    include_y_basis: bool = False,
) -> dict:
    """
    Run complete coherence witness measurement on hardware.

    Runs X basis, and optionally Y basis for phase-robust magnitude |C|.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether to include memory erasure step.
    shots : int
        Number of shots per circuit.
    optimization_level : int
        Transpiler optimization level.
    backend_name : str, optional
        Backend to use.
    channel : str
        IBM Quantum channel.
    include_y_basis : bool
        Also run Y basis measurement.

    Returns
    -------
    dict
        Combined results with W_X, W_Y (if requested), and |C|.
    """
    print(f"\n  Running coherence witness (X basis)...")
    result_x = run_coherence_on_hardware(
        mu=mu,
        include_memory_erase=include_memory_erase,
        shots=shots,
        optimization_level=optimization_level,
        backend_name=backend_name,
        channel=channel,
        basis='X',
    )

    combined = {
        'backend': result_x['backend'],
        'backend_type': 'hardware',
        'mode': 'coherence_witness_full',
        'mu': mu,
        'shots': shots,
        'W_X': result_x['W_X'],
        'W_X_error': result_x['W_X_error'],
        'W_X_ideal': result_x['W_X_ideal'],
        'W_X_tilde': result_x['W_X_tilde'],
        'W_X_tilde_error': result_x['W_X_tilde_error'],
        'x_basis_result': result_x,
    }

    if include_y_basis:
        print(f"\n  Running coherence witness (Y basis)...")
        result_y = run_coherence_on_hardware(
            mu=mu,
            include_memory_erase=include_memory_erase,
            shots=shots,
            optimization_level=optimization_level,
            backend_name=backend_name,
            channel=channel,
            basis='Y',
        )

        combined['W_Y'] = result_y['W_Y']
        combined['W_Y_error'] = result_y['W_Y_error']
        combined['W_Y_ideal'] = result_y['W_Y_ideal']
        combined['W_Y_tilde'] = result_y['W_Y_tilde']
        combined['W_Y_tilde_error'] = result_y['W_Y_tilde_error']
        combined['y_basis_result'] = result_y

        # Compute phase-robust magnitude
        W_X = result_x['W_X']
        W_Y = result_y['W_Y']
        C_magnitude = np.sqrt(W_X**2 + W_Y**2)
        # Error propagation
        C_err = np.sqrt(
            (W_X * result_x['W_X_error'])**2 +
            (W_Y * result_y['W_Y_error'])**2
        ) / C_magnitude if C_magnitude > 0 else 0

        combined['C_magnitude'] = C_magnitude
        combined['C_magnitude_error'] = C_err

    return combined


def run_hardware_opt_sweep(
    mu: int = 1,
    include_memory_erase: bool = True,
    shots: int = 20000,
    backend_name: Optional[str] = None,
    channel: str = 'ibm_quantum',
    opt_levels: List[int] = [0, 1, 2, 3],
) -> List[dict]:
    """
    Sweep optimization levels on hardware.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether to include memory erasure step.
    shots : int
        Number of shots per configuration.
    backend_name : str, optional
        Backend to use.
    channel : str
        IBM Quantum channel.
    opt_levels : list
        Optimization levels to sweep.

    Returns
    -------
    list
        Results for each optimization level.
    """
    results = []
    for opt_level in opt_levels:
        print(f"\n  Running optimization_level={opt_level}...")
        try:
            result = run_on_hardware(
                mu=mu,
                include_memory_erase=include_memory_erase,
                shots=shots,
                optimization_level=opt_level,
                backend_name=backend_name,
                channel=channel,
            )
            results.append(result)
        except Exception as e:
            print(f"  Error at opt_level={opt_level}: {e}")
            results.append({
                'optimization_level': opt_level,
                'error': str(e),
            })
    return results


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
    try:
        import qiskit_ibm_runtime
        versions['qiskit_ibm_runtime'] = qiskit_ibm_runtime.__version__
    except Exception:
        pass
    return versions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run branch-transfer experiment on IBM Quantum hardware'
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
        '--backend', type=str, default=None,
        help='IBM Quantum backend name (default: least busy)'
    )
    parser.add_argument(
        '--optimization-level', type=int, choices=[0, 1, 2, 3], default=1,
        help='Transpiler optimization level (default: 1)'
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
        '--channel', type=str, default='ibm_quantum',
        choices=['ibm_quantum', 'ibm_cloud'],
        help='IBM Quantum channel (default: ibm_quantum)'
    )
    parser.add_argument(
        '--list-backends', action='store_true',
        help='List available backends and exit'
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

    if not IBM_RUNTIME_AVAILABLE:
        print("ERROR: qiskit-ibm-runtime is not installed.")
        print("Install with: pip install qiskit-ibm-runtime")
        return

    print("=" * 70)
    print("Branch-Transfer Experiment: IBM Quantum Hardware")
    print("=" * 70)

    # List backends mode
    if args.list_backends:
        print("\nConnecting to IBM Quantum...")
        service = QiskitRuntimeService(channel=args.channel)
        backends = get_available_backends(service)
        print(f"\nAvailable backends ({len(backends)}):")
        for name in sorted(backends):
            backend = service.backend(name)
            status = backend.status()
            print(f"  {name}: {backend.num_qubits} qubits, {status.pending_jobs} pending jobs")
        return

    print(f"  mu = {args.mu}")
    print(f"  shots = {args.shots}")
    print(f"  mode = {args.mode}")
    if args.mode == 'coherence_witness':
        print(f"  basis = {args.basis}")
        print(f"  include_y_basis = {args.include_y_basis}")
    else:
        print(f"  circuit = {'control (no memory erase)' if args.control else 'main'}")
    print(f"  output_dir = {output_dir}")
    print(f"  channel = {args.channel}")
    print()

    include_memory_erase = not args.control
    saved_files = []

    try:
        if args.mode == 'coherence_witness':
            # Coherence witness mode - probes off-diagonal coherence
            if args.include_y_basis:
                print("Running coherence witness (X and Y basis)...")
                result = run_coherence_full_hardware(
                    mu=args.mu,
                    include_memory_erase=include_memory_erase,
                    shots=args.shots,
                    optimization_level=args.optimization_level,
                    backend_name=args.backend,
                    channel=args.channel,
                    include_y_basis=True,
                )
                filepath = save_result(result, output_dir, prefix='hw_coherence_')
                saved_files.append(filepath)
                print(f"\n   Backend: {result['backend']}")
                print(f"   W_X: {result['W_X']:.4f} +/- {result['W_X_error']:.4f}")
                print(f"   W_X_tilde: {result['W_X_tilde']:.4f}")
                print(f"   W_Y: {result['W_Y']:.4f} +/- {result['W_Y_error']:.4f}")
                print(f"   W_Y_tilde: {result['W_Y_tilde']:.4f}")
                print(f"   |C|: {result['C_magnitude']:.4f} +/- {result['C_magnitude_error']:.4f}")
            else:
                print(f"Running coherence witness ({args.basis} basis)...")
                result = run_coherence_on_hardware(
                    mu=args.mu,
                    include_memory_erase=include_memory_erase,
                    shots=args.shots,
                    optimization_level=args.optimization_level,
                    backend_name=args.backend,
                    channel=args.channel,
                    basis=args.basis,
                )
                filepath = save_result(result, output_dir, prefix='hw_coherence_')
                saved_files.append(filepath)
                basis = args.basis
                print(f"\n   Backend: {result['backend']}")
                print(f"   Job ID: {result['job_id']}")
                print(f"   W_{basis}: {result[f'W_{basis}']:.4f} +/- {result[f'W_{basis}_error']:.4f}")
                print(f"   W_{basis}_ideal: {result[f'W_{basis}_ideal']:.4f}")
                print(f"   W_{basis}_tilde: {result[f'W_{basis}_tilde']:.4f}")
                print(f"   Transpiled depth: {result['transpiled_depth']}")
                print(f"   Two-qubit gates: {result['transpiled_two_qubit_gates']}")

        elif args.opt_sweep:
            # Original rp_z mode with optimization sweep
            print("Running hardware execution (optimization level sweep)...")
            results = run_hardware_opt_sweep(
                mu=args.mu,
                include_memory_erase=include_memory_erase,
                shots=args.shots,
                backend_name=args.backend,
                channel=args.channel,
            )
            for result in results:
                if 'error' not in result:
                    filepath = save_result(result, output_dir, prefix='hw_')
                    saved_files.append(filepath)
                    opt = result['optimization_level']
                    print(f"   opt_level={opt}: V={result['visibility']:.4f}, depth={result['transpiled_depth']}, 2Q={result['transpiled_two_qubit_gates']}")

        else:
            # Original rp_z mode single run
            print("Running hardware execution (visibility mode)...")
            result = run_on_hardware(
                mu=args.mu,
                include_memory_erase=include_memory_erase,
                shots=args.shots,
                optimization_level=args.optimization_level,
                backend_name=args.backend,
                channel=args.channel,
            )
            filepath = save_result(result, output_dir, prefix='hw_')
            saved_files.append(filepath)
            print(f"\n   Backend: {result['backend']}")
            print(f"   Job ID: {result['job_id']}")
            print(f"   Visibility: {result['visibility']:.4f} +/- {result['visibility_error']:.4f}")
            print(f"   Probabilities: {result['probabilities']}")
            print(f"   Transpiled depth: {result['transpiled_depth']}")
            print(f"   Two-qubit gates: {result['transpiled_two_qubit_gates']}")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPossible causes:")
        print("  - Not authenticated (run: ibmq_account.save_account(TOKEN))")
        print("  - No available backends")
        print("  - Network issues")
        raise

    print("\n" + "=" * 70)
    print("Hardware Execution Complete")
    print("=" * 70)
    print("\nSaved files:")
    for f in saved_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
