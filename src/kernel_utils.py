"""
Kernel utility functions for quantum kernel estimation.

Provides numerical conditioning utilities for quantum kernel matrices,
particularly useful when dealing with finite-shot noise and hardware
imperfections that can cause loss of positive semidefiniteness.

Date: 2026-01-15
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import numpy as np
from typing import Optional, Dict, Any


def ensure_psd_kernel(
    K: np.ndarray,
    epsilon: float = 1e-10,
    preserve_trace: bool = True,
    return_diagnostics: bool = False
) -> np.ndarray | tuple[np.ndarray, Dict[str, Any]]:
    """
    Project a kernel matrix to the nearest positive semidefinite (PSD) matrix.

    Quantum kernel matrices computed from finite shots or noisy hardware may
    lose positive semidefiniteness due to statistical fluctuations. This function
    applies a numerically stable projection to restore PSD structure while
    preserving symmetry and (optionally) the original trace.

    Algorithm:
        1. Symmetrize: K <- (K + K^T) / 2
        2. Eigen-decomposition: K = Q * Lambda * Q^T
        3. Clamp eigenvalues: Lambda_i <- max(Lambda_i, epsilon)
        4. Reconstruct: K_psd = Q * Lambda_clamped * Q^T
        5. (Optional) Rescale to preserve original trace

    Parameters
    ----------
    K : np.ndarray
        Input kernel matrix of shape (n, n). Should be approximately symmetric.
    epsilon : float, optional
        Minimum eigenvalue threshold. Negative eigenvalues are clamped to this
        value. Default is 1e-10.
    preserve_trace : bool, optional
        If True, rescale the projected kernel to match the original trace.
        This maintains comparability across ideal/noisy/hardware runs.
        Default is True.
    return_diagnostics : bool, optional
        If True, return a tuple (K_psd, diagnostics) where diagnostics is a
        dict containing projection statistics. Default is False.

    Returns
    -------
    K_psd : np.ndarray
        The PSD-projected kernel matrix.
    diagnostics : dict, optional
        Only returned if return_diagnostics=True. Contains:
        - 'min_eigenvalue_before': float, minimum eigenvalue before clamping
        - 'min_eigenvalue_after': float, minimum eigenvalue after clamping
        - 'num_clamped': int, number of eigenvalues that were clamped
        - 'trace_original': float, trace of the symmetrized input
        - 'trace_projected': float, trace after projection (before rescaling)
        - 'trace_final': float, trace of returned matrix
        - 'scale_factor': float, rescaling factor applied (1.0 if not rescaled)

    Notes
    -----
    This projection is purely linear algebra / numerical conditioning.
    It does not modify the underlying kernel computation, only the matrix
    passed to downstream classifiers (e.g., SVM with kernel='precomputed').

    For valid kernel matrices, this operation should be nearly a no-op
    (no clamping required). Significant clamping indicates substantial
    noise or shot-count limitations.

    Examples
    --------
    >>> K_noisy = compute_quantum_kernel(X)  # may have small negative eigenvalues
    >>> K_psd = ensure_psd_kernel(K_noisy)
    >>> svm.fit(K_psd, y)

    >>> K_psd, diag = ensure_psd_kernel(K_noisy, return_diagnostics=True)
    >>> print(f"Clamped {diag['num_clamped']} eigenvalues")
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {K.shape}")

    n = K.shape[0]

    # Step 1: Symmetrize
    K_sym = (K + K.T) / 2.0

    trace_original = np.trace(K_sym)

    # Step 2: Eigen-decomposition (eigh for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(K_sym)

    min_eigenvalue_before = float(eigenvalues.min())

    # Step 3: Clamp eigenvalues
    num_clamped = int(np.sum(eigenvalues < epsilon))
    eigenvalues_clamped = np.maximum(eigenvalues, epsilon)

    min_eigenvalue_after = float(eigenvalues_clamped.min())

    # Step 4: Reconstruct kernel
    # K_psd = Q * diag(lambda_clamped) * Q^T
    K_psd = eigenvectors @ np.diag(eigenvalues_clamped) @ eigenvectors.T

    trace_projected = np.trace(K_psd)

    # Step 5: Optionally rescale to preserve trace
    scale_factor = 1.0
    if preserve_trace and trace_projected > 0 and trace_original > 0:
        scale_factor = trace_original / trace_projected
        K_psd = K_psd * scale_factor

    trace_final = np.trace(K_psd)

    # Ensure output remains symmetric (numerical safeguard)
    K_psd = (K_psd + K_psd.T) / 2.0

    if return_diagnostics:
        diagnostics = {
            'min_eigenvalue_before': min_eigenvalue_before,
            'min_eigenvalue_after': min_eigenvalue_after,
            'num_clamped': num_clamped,
            'trace_original': float(trace_original),
            'trace_projected': float(trace_projected),
            'trace_final': float(trace_final),
            'scale_factor': scale_factor
        }
        return K_psd, diagnostics

    return K_psd
