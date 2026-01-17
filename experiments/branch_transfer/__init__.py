"""
Branch-conditioned message transfer experiment.

Implements the 5-qubit "partial branch-swap" protocol to test whether
information written into a branch can be recovered after a partial reversal
operation, providing operational constraints on decoherence/collapse models.

Registers:
    Q - measured qubit (initial superposition)
    R - branch/room record
    F - friend (observer proxy)
    M - memory (message buffer)
    P - paper (persistent record)

Protocol:
    1. Prepare Q in |+>
    2. Correlate friend measurement: CNOT(Q -> F)
    3. Record branch: CNOT(F -> R)
    4. Write message (if mu=1): CNOT(F -> M)
    5. Copy to paper: CNOT(M -> P)
    6. Erase memory: CNOT(P -> M)
    7. Partial branch-swap: X on (Q, R, F)
    8. Measure (R, P) -> bitstring "PR"
"""

from .circuit import (
    build_branch_transfer_circuit,
    build_control_circuit,
    build_coherence_witness_circuit,
    compute_coherence_witness_from_counts,
    compute_normalized_coherence,
    compute_coherence_magnitude,
    get_expected_ideal_coherence,
    COHERENCE_QUBITS,
)
