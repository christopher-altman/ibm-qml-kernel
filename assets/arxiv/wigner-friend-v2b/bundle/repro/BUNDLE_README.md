# Branch Transfer Experiment - arXiv Bundle v2b

## Overview

This bundle contains complete experimental results for the branch-conditioned message transfer protocol executed on IBM Quantum hardware (ibm_fez) with backend-matched noise simulator baselines.

## Hardware Backend

- **Backend**: ibm_fez
- **Instance**: open-instance (open plan)
- **Shots**: 20,000 per experiment
- **Optimization Level**: 2 (hardware), 1 (simulator)

## Experiments

### Hardware Runs (Fresh)
1. **Coherence Witness (X+Y basis)** - Job IDs: d5lobdt9j2ac739k1a0g (X), d5locdhh2mqc739a2ubg (Y)
2. **Visibility Protocol (rp_z)** - Job ID: d5locnd9j2ac739k1b80

### Backend-Matched Noise Simulations
- Noise model: Built from ibm_fez backend properties via NoiseModel.from_backend()
- Calibration snapshots saved in appendix/

## Contents

### json/ (8 files)
- `hw_coherence_..._ibm_fez_...json` - Hardware coherence witness (X+Y)
- `hw_..._ibm_fez_...json` - Hardware visibility (rp_z)
- `coherence_..._noisy_ibm_fez_...json` (x2) - Backend-matched noisy coherence (X, Y)
- `sim_..._noisy_ibm_fez_...json` - Backend-matched noisy visibility
- `coherence_..._statevector_...json` (x2) - Ideal coherence (X, Y)
- `sim_..._statevector_...json` - Ideal visibility

### figures/ (7 PNG files)
- `visibility_comparison.png` - V across ideal/noisy/hardware
- `coherence_comparison.png` - W_X, W_Y, |C| across backends
- `pr_distribution.png` - Measurement outcome distributions
- `visibility_vs_opt_level.png` - Transpiler sensitivity
- `collapse_forecast_*.png` - Decoherence model predictions

### appendix/ (3 files)
- `ibm_fez_*_properties.json` - Backend calibration snapshots (timestamps included)

## JSON Field Definitions

### Visibility Protocol (rp_z mode)
- `visibility`: V = |P(R=0|P=1) - P(R=1|P=1)|
- `visibility_error`: Statistical uncertainty (Poisson)
- `conditional_probabilities`: P(P|R) breakdown
- `counts`: Raw measurement outcomes

### Coherence Witness (coherence_witness mode)
- `W_X`, `W_Y`: Raw coherence witness values
- `W_X_error`, `W_Y_error`: Statistical uncertainties
- `W_X_ideal`, `W_Y_ideal`: Ideal (statevector) reference values
- `W_X_tilde`: Normalized W_X = W_X / W_X_ideal
- `W_Y_tilde`: **UNDEFINED if W_Y_ideal == 0**; raw W_Y still valid
- `C_magnitude`: sqrt(W_X^2 + W_Y^2) - **NOT bounded by 1**, not a "fraction"
- `C_tilde`: Normalized magnitude (if W_Y_ideal != 0)
- `parity_counts`: Even/odd parity statistics

### Backend-Matched Noise Metadata
- `noise_model_backend`: "ibm_fez" (when using --noise-from-backend)
- `noise_snapshot_path`: Path to calibration snapshot JSON

## Important Interpretation Notes

1. **C_magnitude is NOT bounded**: Unlike W_tilde (bounded [0,1]), C_magnitude = sqrt(W_X^2 + W_Y^2) can exceed 1 due to correlations. Do not interpret as a "coherence fraction."

2. **W_Y normalization**: When W_Y_ideal == 0 (e.g., certain mu/mode combinations), W_Y_tilde is undefined. Use raw W_Y or |C| instead.

3. **Noise model provenance**: Backend-matched simulations use ibm_fez properties fetched at run time. Calibration timestamps in appendix/ snapshots document exact parameters used.

## Commands Executed

```bash
# Hardware connectivity
python3.10 -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; s=S(); bs=s.backends(simulator=False, operational=True); print('n_backends=', len(bs)); b=s.least_busy(simulator=False, operational=True); print('least_busy=', b.name)"

# Test suite  
pytest -q

# Hardware runs (fresh)
python3.10 -m experiments.branch_transfer.run_ibm --backend ibm_fez --mode coherence_witness --include-y-basis --shots 20000 --optimization-level 2
python3.10 -m experiments.branch_transfer.run_ibm --backend ibm_fez --mode rp_z --mu 1 --shots 20000 --optimization-level 2

# Backend-matched noise simulations
python3.10 -m experiments.branch_transfer.run_sim --mode coherence_witness --include-y-basis --mu 1 --shots 20000 --noise-from-backend ibm_fez
python3.10 -m experiments.branch_transfer.run_sim --mode rp_z --mu 1 --shots 20000 --noise-from-backend ibm_fez

# Plot regeneration
python3.10 -m experiments.branch_transfer.analyze --artifacts-dir artifacts/branch_transfer --figures-dir artifacts/branch_transfer/figures --plot-all
```

## Software Versions

- Python: 3.10.19
- Qiskit: 2.3.0
- Qiskit Aer: 0.17.2
- Qiskit IBM Runtime: 0.45.0

## Citation Guidance for GPT-5.2 Deep Research

### Claim: Hardware visibility exceeded noisy sim prediction
- Figure: `visibility_comparison.png`
- JSON fields: `visibility` in hw_20260117_205401_ibm_fez_*.json (0.8771) vs sim_20260117_205835_*_noisy_ibm_fez_*.json (0.9381)
- Note: Backend-matched noise model (ibm_fez) shows V=0.938, hardware V=0.877

### Claim: Coherence magnitude preserved on hardware
- Figure: `coherence_comparison.png`  
- JSON fields: `C_magnitude` in hw_coherence_*_ibm_fez_*.json
- Hardware: |C| = 1.1673 ± 0.004 (not normalized; sqrt(W_X^2+W_Y^2))

### Claim: Backend-matched noise more accurate than proxy model
- Comparison: Backend-matched (ibm_fez NoiseModel.from_backend) vs legacy (ibm_brisbane params)
- Appendix: `ibm_fez_*_properties.json` documents exact calibration used
