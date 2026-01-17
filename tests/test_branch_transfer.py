"""
Tests for the branch-conditioned message transfer experiment.

These are fast, simulator-only tests to verify circuit construction
and expected behavior.
"""

import numpy as np
import sys
sys.path.insert(0, 'experiments/branch_transfer')

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
    Q_IDX, R_IDX, F_IDX, M_IDX, P_IDX,
    COHERENCE_QUBITS,
)


def test_circuit_construction():
    """Test that the main circuit can be constructed."""
    qc = build_branch_transfer_circuit(mu=1, include_memory_erase=True)

    assert qc.num_qubits == 5
    assert qc.num_clbits == 2
    assert qc.depth() > 0

    # Check gate composition
    stats = get_circuit_stats(qc)
    assert 'h' in stats['gate_counts']
    assert 'cx' in stats['gate_counts']
    assert 'x' in stats['gate_counts']


def test_control_circuit_construction():
    """Test that the control circuit (no memory erase) can be constructed."""
    qc_main = build_branch_transfer_circuit(mu=1, include_memory_erase=True)
    qc_ctrl = build_control_circuit(mu=1)

    # Control should have fewer gates (one less CNOT)
    stats_main = get_circuit_stats(qc_main)
    stats_ctrl = get_circuit_stats(qc_ctrl)

    assert stats_ctrl['gate_counts']['cx'] == stats_main['gate_counts']['cx'] - 1


def test_expected_distribution_mu1():
    """Test expected ideal distribution for mu=1."""
    expected = get_expected_ideal_distribution(mu=1, include_memory_erase=True)

    # Should only have "01" and "10" with 50% each
    assert set(expected.keys()) == {'01', '10'}
    assert np.isclose(expected['01'], 0.5)
    assert np.isclose(expected['10'], 0.5)


def test_expected_distribution_mu0():
    """Test expected ideal distribution for mu=0 (no message)."""
    expected = get_expected_ideal_distribution(mu=0, include_memory_erase=True)

    # Should only have "00" and "01" with 50% each
    assert set(expected.keys()) == {'00', '01'}
    assert np.isclose(expected['00'], 0.5)
    assert np.isclose(expected['01'], 0.5)


def test_visibility_computation_ideal():
    """Test visibility computation with ideal counts."""
    # Ideal mu=1 outcome: 50% "01", 50% "10"
    counts = {'01': 5000, '10': 5000}
    shots = 10000

    V, V_err, cond_probs = compute_visibility_from_counts(counts, shots)

    # V = P(P=1|R=0) - P(P=1|R=1)
    # P(P=1|R=0) = counts["10"] / (counts["00"] + counts["10"]) = 5000/5000 = 1.0
    # P(P=1|R=1) = counts["11"] / (counts["01"] + counts["11"]) = 0/5000 = 0.0
    # V = 1.0 - 0.0 = 1.0
    assert np.isclose(V, 1.0)


def test_visibility_computation_noisy():
    """Test visibility computation with noisy counts."""
    # Noisy case: some errors
    counts = {'00': 200, '01': 4500, '10': 4500, '11': 800}
    shots = 10000

    V, V_err, cond_probs = compute_visibility_from_counts(counts, shots)

    # V should be positive but less than 1.0
    assert 0 < V < 1.0
    assert V_err > 0


def test_visibility_computation_random():
    """Test visibility computation with random (uniform) counts."""
    # Uniform distribution: no correlation
    counts = {'00': 2500, '01': 2500, '10': 2500, '11': 2500}
    shots = 10000

    V, V_err, cond_probs = compute_visibility_from_counts(counts, shots)

    # V should be close to 0 for uniform distribution
    assert np.isclose(V, 0.0, atol=0.01)


def test_ideal_simulation():
    """Test that ideal simulation produces expected distribution."""
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError:
        print("Qiskit/Aer not available - skipping ideal simulation test")
        return

    qc = build_branch_transfer_circuit(mu=1, include_memory_erase=True)
    backend = AerSimulator(method='statevector')
    qc_t = transpile(qc, backend, optimization_level=0)

    job = backend.run(qc_t, shots=10000)
    result = job.result()
    counts = result.get_counts()

    # Should only have "01" and "10" outcomes
    assert set(counts.keys()) == {'01', '10'}

    # Should be approximately 50/50
    total = sum(counts.values())
    for bitstring in ['01', '10']:
        prob = counts[bitstring] / total
        assert 0.45 < prob < 0.55, f"Expected ~50% for {bitstring}, got {prob:.2%}"


def test_mu0_produces_different_distribution():
    """Test that mu=0 produces a different distribution than mu=1."""
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError:
        print("Qiskit/Aer not available - skipping mu=0 test")
        return

    # mu=1
    qc_mu1 = build_branch_transfer_circuit(mu=1)
    # mu=0
    qc_mu0 = build_branch_transfer_circuit(mu=0)

    backend = AerSimulator(method='statevector')

    job1 = backend.run(transpile(qc_mu1, backend), shots=10000)
    job0 = backend.run(transpile(qc_mu0, backend), shots=10000)

    counts1 = job1.result().get_counts()
    counts0 = job0.result().get_counts()

    # mu=1 should have "01", "10"
    assert set(counts1.keys()) == {'01', '10'}
    # mu=0 should have "00", "01"
    assert set(counts0.keys()) == {'00', '01'}


def test_control_circuit_same_signature():
    """Test that control circuit (no erase) has same ideal signature as main."""
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError:
        print("Qiskit/Aer not available - skipping control test")
        return

    # Main circuit with erase
    qc_main = build_branch_transfer_circuit(mu=1, include_memory_erase=True)
    # Control circuit without erase
    qc_ctrl = build_control_circuit(mu=1)

    backend = AerSimulator(method='statevector')

    job_main = backend.run(transpile(qc_main, backend), shots=10000)
    job_ctrl = backend.run(transpile(qc_ctrl, backend), shots=10000)

    counts_main = job_main.result().get_counts()
    counts_ctrl = job_ctrl.result().get_counts()

    # Both should have same outcome set in ideal case
    assert set(counts_main.keys()) == {'01', '10'}
    assert set(counts_ctrl.keys()) == {'01', '10'}


def test_circuit_stats():
    """Test circuit statistics extraction."""
    qc = build_branch_transfer_circuit(mu=1)
    stats = get_circuit_stats(qc)

    assert stats['num_qubits'] == 5
    assert stats['num_clbits'] == 2
    assert stats['depth'] > 0
    assert stats['size'] > 0
    assert 'gate_counts' in stats
    assert stats['two_qubit_gate_count'] > 0


def test_qubit_indices():
    """Test qubit index constants."""
    assert Q_IDX == 0
    assert R_IDX == 1
    assert F_IDX == 2
    assert M_IDX == 3
    assert P_IDX == 4


# ============================================================================
# Coherence Witness Tests
# ============================================================================

def test_coherence_witness_circuit_construction():
    """Test that the coherence witness circuit can be constructed."""
    qc = build_coherence_witness_circuit(mu=1, include_memory_erase=True, basis='X')

    assert qc.num_qubits == 5
    assert qc.num_clbits == 4  # Measure Q, R, F, P
    assert qc.depth() > 0

    # Check gate composition - should have Hadamards for X-basis measurement
    stats = get_circuit_stats(qc)
    assert 'h' in stats['gate_counts']


def test_coherence_witness_circuit_y_basis():
    """Test Y-basis coherence witness circuit construction."""
    qc_x = build_coherence_witness_circuit(mu=1, basis='X')
    qc_y = build_coherence_witness_circuit(mu=1, basis='Y')

    # Y basis should have S-dagger gates
    stats_y = get_circuit_stats(qc_y)
    assert 'sdg' in stats_y['gate_counts'] or 's' in stats_y['gate_counts']


def test_coherence_witness_computation_ideal():
    """Test coherence witness computation with ideal counts."""
    # For ideal state (|1110⟩ + |0001⟩)/√2 measured in X basis:
    # All even parity outcomes
    counts = {'0000': 5000, '1111': 5000}
    shots = 10000

    W, W_err, parity_counts = compute_coherence_witness_from_counts(counts, shots)

    # W = P(even) - P(odd) = 1.0 - 0.0 = 1.0
    assert np.isclose(W, 1.0)
    # Note: W_err is 0 for perfect counts (all even parity), which is expected
    assert W_err >= 0


def test_coherence_witness_computation_decohered():
    """Test coherence witness computation with fully decohered counts."""
    # Uniform distribution: no coherence
    counts = {
        '0000': 625, '0001': 625, '0010': 625, '0011': 625,
        '0100': 625, '0101': 625, '0110': 625, '0111': 625,
        '1000': 625, '1001': 625, '1010': 625, '1011': 625,
        '1100': 625, '1101': 625, '1110': 625, '1111': 625,
    }
    shots = 10000

    W, W_err, parity_counts = compute_coherence_witness_from_counts(counts, shots)

    # W should be close to 0 for uniform distribution
    assert np.isclose(W, 0.0, atol=0.01)


def test_normalized_coherence_computation():
    """Test normalized coherence computation."""
    W_measured = 0.8
    W_ideal = 1.0
    W_err = 0.05
    W_ideal_err = 0.0

    W_tilde, W_tilde_err = compute_normalized_coherence(
        W_measured, W_ideal, W_err, W_ideal_err
    )

    assert np.isclose(W_tilde, 0.8)
    assert W_tilde_err > 0


def test_expected_ideal_coherence_mu1():
    """Test expected ideal coherence for mu=1 with erase."""
    W_ideal = get_expected_ideal_coherence(mu=1, include_memory_erase=True, basis='X')

    # For mu=1 with erase, W_X should be 1.0
    assert np.isclose(W_ideal, 1.0)


def test_expected_ideal_coherence_mu0():
    """Test expected ideal coherence for mu=0."""
    W_ideal = get_expected_ideal_coherence(mu=0, include_memory_erase=True, basis='X')

    # For mu=0, no message is written, so no branch-dependent information
    # The state has different structure and W_X = 0 (no coherence to detect)
    assert np.isclose(W_ideal, 0.0)


def test_coherence_qubits_constant():
    """Test COHERENCE_QUBITS constant."""
    # Should be Q, R, F, P indices
    assert set(COHERENCE_QUBITS) == {Q_IDX, R_IDX, F_IDX, P_IDX}
    assert M_IDX not in COHERENCE_QUBITS


def test_coherence_simulation_ideal():
    """Test that ideal coherence simulation produces expected W_X=1."""
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError:
        print("Qiskit/Aer not available - skipping coherence simulation test")
        return

    qc = build_coherence_witness_circuit(mu=1, include_memory_erase=True, basis='X')
    backend = AerSimulator(method='statevector')
    qc_t = transpile(qc, backend, optimization_level=0)

    job = backend.run(qc_t, shots=20000)
    result = job.result()
    counts = result.get_counts()

    W, W_err, parity_counts = compute_coherence_witness_from_counts(counts, 20000)

    # W should be close to 1.0 for ideal coherent state
    assert W > 0.95, f"Expected W > 0.95, got {W:.4f}"


def test_coherence_vs_visibility_distinction():
    """Test that coherence and visibility use different measurements."""
    qc_rp = build_branch_transfer_circuit(mu=1)  # Measures R, P
    qc_cw = build_coherence_witness_circuit(mu=1, basis='X')  # Measures Q, R, F, P

    # Different number of classical bits
    assert qc_rp.num_clbits == 2
    assert qc_cw.num_clbits == 4


if __name__ == "__main__":
    print("Running branch-transfer tests...")
    print()

    print("=== Visibility (rp_z) Tests ===")
    test_circuit_construction()
    print("  test_circuit_construction")

    test_control_circuit_construction()
    print("  test_control_circuit_construction")

    test_expected_distribution_mu1()
    print("  test_expected_distribution_mu1")

    test_expected_distribution_mu0()
    print("  test_expected_distribution_mu0")

    test_visibility_computation_ideal()
    print("  test_visibility_computation_ideal")

    test_visibility_computation_noisy()
    print("  test_visibility_computation_noisy")

    test_visibility_computation_random()
    print("  test_visibility_computation_random")

    test_ideal_simulation()
    print("  test_ideal_simulation")

    test_mu0_produces_different_distribution()
    print("  test_mu0_produces_different_distribution")

    test_control_circuit_same_signature()
    print("  test_control_circuit_same_signature")

    test_circuit_stats()
    print("  test_circuit_stats")

    test_qubit_indices()
    print("  test_qubit_indices")

    print()
    print("=== Coherence Witness Tests ===")
    test_coherence_witness_circuit_construction()
    print("  test_coherence_witness_circuit_construction")

    test_coherence_witness_circuit_y_basis()
    print("  test_coherence_witness_circuit_y_basis")

    test_coherence_witness_computation_ideal()
    print("  test_coherence_witness_computation_ideal")

    test_coherence_witness_computation_decohered()
    print("  test_coherence_witness_computation_decohered")

    test_normalized_coherence_computation()
    print("  test_normalized_coherence_computation")

    test_expected_ideal_coherence_mu1()
    print("  test_expected_ideal_coherence_mu1")

    test_expected_ideal_coherence_mu0()
    print("  test_expected_ideal_coherence_mu0")

    test_coherence_qubits_constant()
    print("  test_coherence_qubits_constant")

    test_coherence_simulation_ideal()
    print("  test_coherence_simulation_ideal")

    test_coherence_vs_visibility_distinction()
    print("  test_coherence_vs_visibility_distinction")

    print()
    print("All 22 branch-transfer tests passed!")
