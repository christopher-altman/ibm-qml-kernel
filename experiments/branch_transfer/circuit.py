"""
Quantum circuits for the branch-conditioned message transfer experiment.

Builds the 5-qubit protocol circuit implementing partial branch-swap
with message encoding and erasure.

Date: 2026-01-17
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import Literal, List
import numpy as np


# Qubit indices (fixed mapping for consistency)
Q_IDX = 0  # Measured qubit (initial superposition)
R_IDX = 1  # Branch/room record
F_IDX = 2  # Friend (observer proxy)
M_IDX = 3  # Memory (message buffer)
P_IDX = 4  # Paper (persistent record)

QUBIT_NAMES = ['Q', 'R', 'F', 'M', 'P']


def build_branch_transfer_circuit(
    mu: Literal[0, 1] = 1,
    include_memory_erase: bool = True,
    barrier: bool = True
) -> QuantumCircuit:
    """
    Build the branch-conditioned message transfer circuit.

    Parameters
    ----------
    mu : {0, 1}
        Message bit. If mu=1, writes message via CNOT(F->M).
        If mu=0, no message is written (blank).
    include_memory_erase : bool
        If True (default), includes the CNOT(P->M) step that erases memory.
        Set False for the control circuit that omits erasure.
    barrier : bool
        If True, insert barriers between logical steps for visualization.

    Returns
    -------
    QuantumCircuit
        The 5-qubit circuit with 2 classical bits for measuring (R, P).

    Notes
    -----
    Register layout:
        q[0] = Q (measured qubit)
        q[1] = R (branch record)
        q[2] = F (friend)
        q[3] = M (memory)
        q[4] = P (paper)
        c[0] = measurement of R
        c[1] = measurement of P

    Protocol steps:
        1. H on Q -> |+>
        2. CNOT(Q -> F) -> friend "measurement" correlation
        3. CNOT(F -> R) -> record branch
        4. if mu=1: CNOT(F -> M) -> write message
        5. CNOT(M -> P) -> copy to paper
        6. if include_memory_erase: CNOT(P -> M) -> erase memory
        7. X on Q, R, F -> partial branch-swap (P untouched)
        8. Measure R, P
    """
    # Create registers
    qreg = QuantumRegister(5, 'q')
    creg = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qreg, creg, name=f'branch_transfer_mu{mu}')

    # Step 1: Prepare Q in |+>
    qc.h(Q_IDX)
    if barrier:
        qc.barrier(label='prep')

    # Step 2: Friend "measurement" correlation
    qc.cx(Q_IDX, F_IDX)
    if barrier:
        qc.barrier(label='corr')

    # Step 3: Record branch
    qc.cx(F_IDX, R_IDX)
    if barrier:
        qc.barrier(label='rec')

    # Step 4: Write message (conditional on mu)
    if mu == 1:
        qc.cx(F_IDX, M_IDX)
    if barrier:
        qc.barrier(label='msg')

    # Step 5: Copy memory to paper
    qc.cx(M_IDX, P_IDX)
    if barrier:
        qc.barrier(label='copy')

    # Step 6: Erase memory (if enabled)
    if include_memory_erase:
        qc.cx(P_IDX, M_IDX)
        if barrier:
            qc.barrier(label='erase')

    # Step 7: Partial branch-swap (X on Q, R, F; P untouched)
    qc.x(Q_IDX)
    qc.x(R_IDX)
    qc.x(F_IDX)
    if barrier:
        qc.barrier(label='swap')

    # Step 8: Measure R and P
    qc.measure(R_IDX, 0)  # c[0] = R
    qc.measure(P_IDX, 1)  # c[1] = P

    return qc


def build_control_circuit(mu: Literal[0, 1] = 1, barrier: bool = True) -> QuantumCircuit:
    """
    Build the control circuit that omits memory erasure.

    This is the "no-wipe" control variant where CNOT(P->M) is skipped,
    demonstrating that memory erasure is essential for the signature.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    barrier : bool
        If True, insert barriers between logical steps.

    Returns
    -------
    QuantumCircuit
        Control circuit without memory erasure step.
    """
    return build_branch_transfer_circuit(mu=mu, include_memory_erase=False, barrier=barrier)


def get_circuit_stats(qc: QuantumCircuit) -> dict:
    """
    Extract circuit statistics for metadata.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to analyze.

    Returns
    -------
    dict
        Statistics including depth, gate counts, etc.
    """
    # Count gate types
    gate_counts = {}
    for instruction in qc.data:
        gate_name = instruction.operation.name
        if gate_name not in ['barrier', 'measure']:
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

    two_qubit_gates = sum(
        count for gate, count in gate_counts.items()
        if gate in ['cx', 'cz', 'ecr', 'rzz', 'swap', 'iswap']
    )

    return {
        'num_qubits': qc.num_qubits,
        'num_clbits': qc.num_clbits,
        'depth': qc.depth(),
        'size': qc.size(),
        'gate_counts': gate_counts,
        'two_qubit_gate_count': two_qubit_gates,
    }


def get_expected_ideal_distribution(mu: Literal[0, 1], include_memory_erase: bool = True) -> dict:
    """
    Compute the expected ideal (noiseless) probability distribution.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether memory erasure step is included.

    Returns
    -------
    dict
        Mapping from bitstring "PR" to probability.
        Bitstring format: c[1]c[0] = P,R (Qiskit convention: LSB first).

    Notes
    -----
    For the main circuit with mu=1 and memory erase:
        - Initial: |00000> (Q,R,F,M,P)
        - After H: |+0000> = (|00000> + |10000>)/sqrt(2)
        - After CNOT(Q->F): (|00000> + |10100>)/sqrt(2)
        - After CNOT(F->R): (|00000> + |11100>)/sqrt(2)
        - After CNOT(F->M) [mu=1]: (|00000> + |11110>)/sqrt(2)
        - After CNOT(M->P): (|00000> + |11111>)/sqrt(2)
        - After CNOT(P->M) [erase]: (|00000> + |11101>)/sqrt(2)
        - After X on Q,R,F: (|11100> + |00001>)/sqrt(2)
        - R values: 1, 0; P values: 0, 1
        - Output bitstrings (PR = c[1]c[0]): "01" and "10" with 50% each.
    """
    if mu == 1 and include_memory_erase:
        # Main protocol: P=0,R=1 or P=1,R=0
        # Bitstring = c[1]c[0] = PR -> "01" or "10"
        return {'01': 0.5, '10': 0.5}

    elif mu == 1 and not include_memory_erase:
        # No memory erase: different final state
        # After CNOT(M->P): (|00000> + |11111>)/sqrt(2)
        # After X on Q,R,F: (|11100> + |00011>)/sqrt(2)
        # R values: 1, 0; P values: 0, 1
        # But M is now 0, 1 respectively (not erased)
        # Actually trace through again:
        # - After H: |+0000>
        # - After CNOT(Q->F): (|00000> + |10100>)/sqrt(2)
        # - After CNOT(F->R): (|00000> + |11100>)/sqrt(2)
        # - After CNOT(F->M): (|00000> + |11110>)/sqrt(2)
        # - After CNOT(M->P): (|00000> + |11111>)/sqrt(2)
        # - NO erase step
        # - After X on Q,R,F: (|11100> + |00011>)/sqrt(2)
        # Measuring R,P: (R=1,P=0) and (R=0,P=1) -> "01" and "10"
        # Same as main? Let me verify more carefully...
        # State |11100>: Q=1,R=1,F=1,M=0,P=0 -> R=1, P=0 -> "01"
        # State |00011>: Q=0,R=0,F=0,M=1,P=1 -> R=0, P=1 -> "10"
        # So still "01" and "10"! The difference is M is entangled with output.
        # The signature appears the same but M is not in |0> state.
        return {'01': 0.5, '10': 0.5}

    elif mu == 0 and include_memory_erase:
        # No message written (mu=0)
        # - After H: |+0000>
        # - After CNOT(Q->F): (|00000> + |10100>)/sqrt(2)
        # - After CNOT(F->R): (|00000> + |11100>)/sqrt(2)
        # - NO CNOT(F->M) step
        # - After CNOT(M->P): M=0 always, so P stays 0
        #   (|00000> + |11100>)/sqrt(2)
        # - After CNOT(P->M): P=0, so M stays 0
        #   (|00000> + |11100>)/sqrt(2)
        # - After X on Q,R,F: (|11100> + |00000>)/sqrt(2)
        # Measuring R,P: (R=1,P=0) and (R=0,P=0) -> "01" and "00"
        return {'00': 0.5, '01': 0.5}

    elif mu == 0 and not include_memory_erase:
        # Same as mu=0 with erase (M,P both 0 throughout)
        return {'00': 0.5, '01': 0.5}

    else:
        raise ValueError(f"Invalid mu={mu}")


def compute_visibility_from_counts(counts: dict, shots: int) -> tuple:
    """
    Compute visibility metric from measurement counts.

    V = P(P=1 | R=0) - P(P=1 | R=1)

    For ideal mu=1: V should be 1.0 (perfect anti-correlation).
    For mu=0: V should be 0.0 (no correlation).

    Parameters
    ----------
    counts : dict
        Measurement counts with keys like "01", "10", etc.
        Bitstring format: c[1]c[0] = PR.
    shots : int
        Total number of shots.

    Returns
    -------
    V : float
        Visibility metric.
    V_err : float
        Standard error estimate (multinomial approximation).
    conditional_probs : dict
        Conditional probabilities used in computation.
    """
    # Parse counts by R value
    # Bitstring format: "PR" where P=c[1], R=c[0]
    # R=0: bitstrings ending in '0': "00", "10"
    # R=1: bitstrings ending in '1': "01", "11"

    n_R0 = counts.get('00', 0) + counts.get('10', 0)
    n_R1 = counts.get('01', 0) + counts.get('11', 0)

    # P(P=1 | R=0) = counts["10"] / (counts["00"] + counts["10"])
    n_P1_R0 = counts.get('10', 0)
    # P(P=1 | R=1) = counts["11"] / (counts["01"] + counts["11"])
    n_P1_R1 = counts.get('11', 0)

    # Compute conditional probabilities (with handling for zero denominator)
    p_P1_given_R0 = n_P1_R0 / n_R0 if n_R0 > 0 else 0.0
    p_P1_given_R1 = n_P1_R1 / n_R1 if n_R1 > 0 else 0.0

    V = p_P1_given_R0 - p_P1_given_R1

    # Standard error estimate (delta method / propagation of binomial errors)
    # Var(p) = p(1-p)/n for binomial
    if n_R0 > 0:
        var_p0 = p_P1_given_R0 * (1 - p_P1_given_R0) / n_R0
    else:
        var_p0 = 0.0
    if n_R1 > 0:
        var_p1 = p_P1_given_R1 * (1 - p_P1_given_R1) / n_R1
    else:
        var_p1 = 0.0

    V_err = (var_p0 + var_p1) ** 0.5

    conditional_probs = {
        'P(P=1|R=0)': p_P1_given_R0,
        'P(P=1|R=1)': p_P1_given_R1,
        'n_R0': n_R0,
        'n_R1': n_R1,
    }

    return V, V_err, conditional_probs


# =============================================================================
# Coherence Witness Circuits and Computation
# =============================================================================

# Qubits involved in the coherence witness (Q, R, F, P - excludes M)
COHERENCE_QUBITS = [Q_IDX, R_IDX, F_IDX, P_IDX]


def build_coherence_witness_circuit(
    mu: Literal[0, 1] = 1,
    include_memory_erase: bool = True,
    basis: Literal['X', 'Y'] = 'X',
    barrier: bool = True
) -> QuantumCircuit:
    """
    Build circuit for coherence witness measurement.

    This circuit measures Q, R, F, P in the X basis (for W_X) or Y basis (for W_Y)
    to probe off-diagonal coherence between the two branches of the superposition.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether to include memory erasure step.
    basis : {'X', 'Y'}
        Measurement basis. 'X' for W_X, 'Y' for W_Y.
    barrier : bool
        If True, insert barriers between logical steps.

    Returns
    -------
    QuantumCircuit
        Circuit with 4 classical bits for measuring Q, R, F, P in rotated basis.

    Notes
    -----
    For X basis: Apply H before measurement (H rotates X eigenbasis to Z readout).
    For Y basis: Apply S†H before measurement (S†H rotates Y eigenbasis to Z readout).

    The coherence witness is computed from the 4-qubit parity:
        W = sum_b (-1)^parity(b) * P(b)
    where parity(b) = b_Q XOR b_R XOR b_F XOR b_P.
    """
    # Create registers - 4 classical bits for Q, R, F, P
    qreg = QuantumRegister(5, 'q')
    creg = ClassicalRegister(4, 'c')  # c[0]=Q, c[1]=R, c[2]=F, c[3]=P
    qc = QuantumCircuit(qreg, creg, name=f'coherence_{basis}_mu{mu}')

    # Step 1: Prepare Q in |+>
    qc.h(Q_IDX)
    if barrier:
        qc.barrier(label='prep')

    # Step 2: Friend "measurement" correlation
    qc.cx(Q_IDX, F_IDX)
    if barrier:
        qc.barrier(label='corr')

    # Step 3: Record branch
    qc.cx(F_IDX, R_IDX)
    if barrier:
        qc.barrier(label='rec')

    # Step 4: Write message (conditional on mu)
    if mu == 1:
        qc.cx(F_IDX, M_IDX)
    if barrier:
        qc.barrier(label='msg')

    # Step 5: Copy memory to paper
    qc.cx(M_IDX, P_IDX)
    if barrier:
        qc.barrier(label='copy')

    # Step 6: Erase memory (if enabled)
    if include_memory_erase:
        qc.cx(P_IDX, M_IDX)
        if barrier:
            qc.barrier(label='erase')

    # Step 7: Partial branch-swap (X on Q, R, F; P untouched)
    qc.x(Q_IDX)
    qc.x(R_IDX)
    qc.x(F_IDX)
    if barrier:
        qc.barrier(label='swap')

    # Step 8: Rotate to measurement basis for coherence qubits (Q, R, F, P)
    if barrier:
        qc.barrier(label='rotate')

    for qubit in COHERENCE_QUBITS:
        if basis == 'X':
            # H rotates X eigenbasis to Z
            qc.h(qubit)
        elif basis == 'Y':
            # S†H rotates Y eigenbasis to Z (S† = Sdg)
            qc.sdg(qubit)
            qc.h(qubit)
        else:
            raise ValueError(f"Invalid basis: {basis}")

    if barrier:
        qc.barrier(label='meas')

    # Step 9: Measure Q, R, F, P
    qc.measure(Q_IDX, 0)  # c[0] = Q
    qc.measure(R_IDX, 1)  # c[1] = R
    qc.measure(F_IDX, 2)  # c[2] = F
    qc.measure(P_IDX, 3)  # c[3] = P

    return qc


def compute_coherence_witness_from_counts(counts: dict, shots: int) -> tuple:
    """
    Compute coherence witness W from measurement counts.

    W = sum_b (-1)^parity(b) * P(b)

    where parity(b) = b_Q XOR b_R XOR b_F XOR b_P (4-bit parity).

    For the ideal coherent superposition (mu=1, main circuit), W_X is near ±1.
    For a decohered mixture, W_X approaches 0.

    Parameters
    ----------
    counts : dict
        Measurement counts with 4-bit keys like "0000", "0101", etc.
        Bitstring format: c[3]c[2]c[1]c[0] = P,F,R,Q (Qiskit LSB convention).
    shots : int
        Total number of shots.

    Returns
    -------
    W : float
        Coherence witness value in [-1, 1].
    W_err : float
        Standard error estimate.
    parity_counts : dict
        Counts grouped by parity (even/odd).
    """
    n_even = 0  # parity = 0
    n_odd = 0   # parity = 1

    for bitstring, count in counts.items():
        # Ensure 4-bit format
        bs = bitstring.zfill(4)[-4:]

        # Compute parity: XOR of all 4 bits
        # Bitstring order: c[3]c[2]c[1]c[0] = P,F,R,Q
        parity = 0
        for bit in bs:
            parity ^= int(bit)

        if parity == 0:
            n_even += count
        else:
            n_odd += count

    total = n_even + n_odd
    if total == 0:
        return 0.0, 0.0, {'n_even': 0, 'n_odd': 0}

    p_even = n_even / total
    p_odd = n_odd / total

    # W = P(even) - P(odd) = (+1)*P(even) + (-1)*P(odd)
    W = p_even - p_odd

    # Standard error: Var(W) = Var(p_even - p_odd)
    # For binomial: Var(p) = p(1-p)/n
    # p_odd = 1 - p_even, so Var(W) = 4 * p_even * (1-p_even) / n
    W_var = 4 * p_even * p_odd / total if total > 0 else 0.0
    W_err = np.sqrt(W_var)

    parity_counts = {
        'n_even': n_even,
        'n_odd': n_odd,
        'p_even': p_even,
        'p_odd': p_odd,
    }

    return W, W_err, parity_counts


def compute_normalized_coherence(
    W_measured: float,
    W_ideal: float,
    W_measured_err: float = 0.0,
    W_ideal_err: float = 0.0
) -> tuple:
    """
    Compute normalized coherence witness.

    W_tilde = W_measured / W_ideal

    This normalization accounts for sign/phase conventions and ensures
    the coherence fraction lies in [-1, 1].

    Parameters
    ----------
    W_measured : float
        Measured coherence witness.
    W_ideal : float
        Ideal reference coherence witness (from simulator).
    W_measured_err : float
        Error in measured W.
    W_ideal_err : float
        Error in ideal W.

    Returns
    -------
    W_tilde : float
        Normalized coherence witness.
    W_tilde_err : float
        Propagated error.
    """
    if abs(W_ideal) < 1e-10:
        # Avoid division by zero
        return 0.0, 0.0

    W_tilde = W_measured / W_ideal

    # Error propagation: d(a/b) = |a/b| * sqrt((da/a)^2 + (db/b)^2)
    if abs(W_measured) > 1e-10:
        rel_err_m = abs(W_measured_err / W_measured) if W_measured != 0 else 0
        rel_err_i = abs(W_ideal_err / W_ideal) if W_ideal != 0 else 0
        W_tilde_err = abs(W_tilde) * np.sqrt(rel_err_m**2 + rel_err_i**2)
    else:
        W_tilde_err = abs(W_measured_err / W_ideal) if W_ideal != 0 else 0

    return W_tilde, W_tilde_err


def compute_coherence_magnitude(
    W_X: float,
    W_Y: float,
    W_X_err: float = 0.0,
    W_Y_err: float = 0.0
) -> tuple:
    """
    Compute phase-robust coherence magnitude.

    |C| = sqrt(W_X^2 + W_Y^2)

    This is robust to unknown relative phases between branches.

    Parameters
    ----------
    W_X : float
        X-basis coherence witness.
    W_Y : float
        Y-basis coherence witness.
    W_X_err : float
        Error in W_X.
    W_Y_err : float
        Error in W_Y.

    Returns
    -------
    C_mag : float
        Coherence magnitude in [0, 1].
    C_mag_err : float
        Propagated error.
    """
    C_mag_sq = W_X**2 + W_Y**2
    C_mag = np.sqrt(C_mag_sq)

    if C_mag > 1e-10:
        # Error propagation for sqrt(a^2 + b^2)
        C_mag_err = np.sqrt((W_X * W_X_err)**2 + (W_Y * W_Y_err)**2) / C_mag
    else:
        C_mag_err = np.sqrt(W_X_err**2 + W_Y_err**2)

    return C_mag, C_mag_err


def get_expected_ideal_coherence(
    mu: Literal[0, 1],
    include_memory_erase: bool = True,
    basis: Literal['X', 'Y'] = 'X'
) -> float:
    """
    Compute expected ideal coherence witness value.

    For mu=1 main circuit, the ideal final state is:
        (|11100> + |00001>)/sqrt(2)  [Q,R,F,M,P ordering]

    After X-basis rotation (H on Q,R,F,P), measuring in Z basis:
    The coherence witness W_X probes the off-diagonal element.

    Parameters
    ----------
    mu : {0, 1}
        Message bit.
    include_memory_erase : bool
        Whether memory erasure step is included.
    basis : {'X', 'Y'}
        Measurement basis.

    Returns
    -------
    float
        Expected coherence witness value.

    Notes
    -----
    For mu=1 with erase, the two-branch superposition is:
        |psi> = (|11100> + |00001>)/sqrt(2)

    The relevant Pauli string is X_Q X_R X_F X_P (ignoring M).
    For this state:
        <X_Q X_R X_F X_P> = Re(<11100| X^4 |00001>) * 2
                         = Re(<11100| |11110>) * 2  (X flips each bit)
                         = 0 (orthogonal states)

    Wait, let me recalculate. The states differ on Q,R,F,P:
        |11100> has Q=1,R=1,F=1,P=0
        |00001> has Q=0,R=0,F=0,P=1

    X_Q X_R X_F X_P |11100> = |00010> (flipping Q,R,F,P but not M)
    X_Q X_R X_F X_P |00001> = |11110>

    So <psi| X_Q X_R X_F X_P |psi> = (1/2)[<11100|00010> + <00001|11110> + cross terms]

    The cross terms: <11100| X^4 |00001> + <00001| X^4 |11100>
    = <11100|11110> + <00001|00010>
    = 0 + 0 = 0

    Hmm, this suggests W_X = 0 for this state, which would be wrong.

    Let me reconsider. The issue is that M is also part of the state and differs.
    The full state is over 5 qubits, but we're only measuring 4.

    For the mu=1 with erase case:
        |psi> = (|11100> + |00001>)/sqrt(2)  where ordering is Q,R,F,M,P

    Let me trace out M and compute the reduced density matrix for Q,R,F,P.
    |11100> -> Q=1,R=1,F=1,M=0,P=0 -> reduced |1110>_{QRFP}
    |00001> -> Q=0,R=0,F=0,M=0,P=1 -> reduced |0001>_{QRFP}

    Both branches have M=0! So:
        |psi>_{QRFP} = (|1110> + |0001>)/sqrt(2)

    Now: X_Q X_R X_F X_P |1110> = |0001>
         X_Q X_R X_F X_P |0001> = |1110>

    So <X_Q X_R X_F X_P> = (1/2)[<1110|0001> + <0001|1110> + <1110|1110> + <0001|0001>]

    Wait, I need to compute this correctly:
    <psi| X^4 |psi> = (1/2)[<1110| + <0001|] X^4 [|1110> + |0001>]
                    = (1/2)[<1110|X^4|1110> + <1110|X^4|0001> + <0001|X^4|1110> + <0001|X^4|0001>]
                    = (1/2)[<1110|0001> + <1110|1110> + <0001|0001> + <0001|1110>]
                    = (1/2)[0 + 1 + 1 + 0]
                    = 1

    So W_X = 1 for the ideal coherent state! This makes sense.

    For mu=0, the state is (|1110> + |0000>)/sqrt(2) in QRFP (both have M=0, P=0).
    X^4 |1110> = |0001>, X^4 |0000> = |1111>
    <X^4> = (1/2)[<1110|0001> + <1110|1111> + <0000|0001> + <0000|1111>] = 0

    So W_X = 0 for mu=0, which makes sense (no message = no coherence to detect).
    """
    if mu == 1 and include_memory_erase:
        # Coherent superposition: W_X = +1 (or -1 depending on phase convention)
        if basis == 'X':
            return 1.0
        elif basis == 'Y':
            # For Y basis, depends on relative phase. For this circuit, Y gives 0.
            return 0.0
    elif mu == 1 and not include_memory_erase:
        # Control circuit: M is entangled, reduces coherence visibility
        # |psi> = (|11100> + |00011>)/sqrt(2) where M differs between branches
        # Tracing out M: mixed state, coherence partially lost
        # After trace over M, the reduced state is mixed.
        # This is actually still pure when we trace out M since M is correlated with P.
        # Let me recalculate...
        # |11100> = |Q=1,R=1,F=1,M=0,P=0>
        # |00011> = |Q=0,R=0,F=0,M=1,P=1>
        # Tracing out M gives: (1/2)[|1110><1110| + |0001><0001|] - a mixed state!
        # For a mixed state, <X^4> = Tr(rho X^4) = (1/2)[<1110|X^4|1110> + <0001|X^4|0001>]
        #                         = (1/2)[<1110|0001> + <0001|1110>] = 0
        if basis == 'X':
            return 0.0
        elif basis == 'Y':
            return 0.0
    elif mu == 0:
        # No message: no branch-dependent information
        if basis == 'X':
            return 0.0
        elif basis == 'Y':
            return 0.0
    else:
        return 0.0
