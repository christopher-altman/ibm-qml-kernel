"""
Quantum Kernel Estimation with Realistic IBM Quantum Noise
Models hardware imperfections from 127-qubit Eagle processors

Implementation: Noisy simulator with hardware-calibrated parameters
Date: 2026-01-10
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import argparse
import numpy as np
import json
from pathlib import Path

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, thermal_relaxation_error,
    depolarizing_error, pauli_error, ReadoutError
)
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import BackendSamplerV2 as BackendSampler

# Qiskit Machine Learning
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Classical ML
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Local utilities
from kernel_utils import ensure_psd_kernel
from path_utils import resolve_output_paths

# Visualization
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

class NoisyQuantumKernelEstimator:
    """Quantum kernel with realistic IBM hardware noise"""
    
    def __init__(self, hardware_params_path='data/ibm_hardware_params_2026.json',
                 num_features=2, reps=2, shots=1024, psd_project=False, psd_epsilon=1e-10):
        self.num_features = num_features
        self.reps = reps
        self.shots = shots
        # PSD projection: off by default. Enable to improve robustness under
        # finite-shot noise or hardware imperfections that cause loss of
        # positive semidefiniteness in quantum kernel matrices.
        self.psd_project = psd_project
        self.psd_epsilon = psd_epsilon
        
        # Load hardware parameters
        with open(hardware_params_path, 'r') as f:
            self.hw_params = json.load(f)
        
        print(f"Loaded hardware parameters for: {self.hw_params['backend_name']}")
        print(f"Processor: {self.hw_params['processor_family']}")
        
        # Feature map and ansatz
        self.feature_map = zz_feature_map(
            feature_dimension=num_features,
            reps=reps,
            entanglement='linear',
            insert_barriers=False
        )
        
        # Build noise model
        self.noise_model = self._build_noise_model()
        
        # Noisy simulator
        self.backend = AerSimulator(noise_model=self.noise_model)

        self.sampler = BackendSampler(backend=self.backend)

        # BackendSamplerV2 uses an Options object (no 'shots' kwarg). Set shots via options when available.

        if hasattr(self.sampler, 'options'):

            if hasattr(self.sampler.options, 'default_shots'):

                self.sampler.options.default_shots = self.shots

            elif hasattr(self.sampler.options, 'shots'):

                self.sampler.options.shots = self.shots
        fidelity = ComputeUncompute(sampler=self.sampler, shots=self.shots)
        self.kernel = FidelityQuantumKernel(feature_map=self.feature_map, fidelity=fidelity)
        # Results
        self.train_kernel = None
        self.test_kernel = None
        self.psd_diagnostics = {'train': None, 'test': None}
        
    def _build_noise_model(self):
        """Construct realistic noise model from hardware parameters"""
        print("\nBuilding noise model...")
        
        noise_model = NoiseModel()
        
        # Extract parameters
        T1 = self.hw_params['coherence_times']['T1_median_us'] * 1e-6  # Convert to seconds
        T2 = self.hw_params['coherence_times']['T2_median_us'] * 1e-6
        
        sq_error = self.hw_params['gate_errors']['single_qubit']['median_percent'] / 100
        tq_error = self.hw_params['gate_errors']['two_qubit_ECR']['median_percent'] / 100
        ro_error = self.hw_params['readout_errors']['median_percent'] / 100
        
        print(f"  T1 = {T1*1e6:.1f} µs, T2 = {T2*1e6:.1f} µs")
        print(f"  Single-qubit gate error: {sq_error*100:.2f}%")
        print(f"  Two-qubit gate error: {tq_error*100:.2f}%")
        print(f"  Readout error: {ro_error*100:.2f}%")
        
        # Gate times (typical for IBM)
        t_single = 35e-9  # 35 ns for SX, RZ
        t_two = 600e-9    # 600 ns for ECR
        
        # 1. Thermal relaxation (T1/T2 errors)
        thermal_single = thermal_relaxation_error(T1, T2, t_single)
        thermal_two = thermal_relaxation_error(T1, T2, t_two)
        
        # 2. Depolarizing errors (gate imperfections)
        depol_single = depolarizing_error(sq_error, 1)
        depol_two = depolarizing_error(tq_error, 2)
        
        # Combine errors for single-qubit gates
        sq_combined = thermal_single.compose(depol_single)
        
        # Combine errors for two-qubit gates
        tq_combined = thermal_two.compose(depol_two)
        
        # Add to all single-qubit gates
        for gate in ['sx', 'rz', 'x', 'id']:
            noise_model.add_all_qubit_quantum_error(sq_combined, gate)
        
        # Add to two-qubit gates (ECR is IBM's native gate)
        noise_model.add_all_qubit_quantum_error(tq_combined, 'ecr')
        noise_model.add_all_qubit_quantum_error(tq_combined, 'cx')  # CNOT fallback
        
        # 3. Readout errors
        # Model as bit-flip probability
        ro_matrix = [
            [1 - ro_error, ro_error],      # P(measure 0 | state was 0), P(measure 1 | state was 0)
            [ro_error, 1 - ro_error]       # P(measure 0 | state was 1), P(measure 1 | state was 1)
        ]
        readout_error = ReadoutError(ro_matrix)
        
        for qubit in range(self.num_features):
            noise_model.add_readout_error(readout_error, [qubit])
        
        print(f"Noise model constructed with {len(noise_model.basis_gates)} basis gates")
        
        return noise_model
    
    def generate_dataset(self, n_samples=100, noise=0.1, test_size=0.3):
        """Generate synthetic dataset"""
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        y = 2 * y - 1
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def compute_kernel_matrix(self, X1, X2=None):
        """Compute noisy quantum kernel matrix"""
        if X2 is None:
            X2 = X1
        sampler = BackendSampler(backend=self.backend)
        if hasattr(sampler, 'options'):
            if hasattr(sampler.options, 'default_shots'):
                sampler.options.default_shots = self.shots
            elif hasattr(sampler.options, 'shots'):
                sampler.options.shots = self.shots
        fidelity = ComputeUncompute(sampler=sampler, shots=self.shots)
        kernel = FidelityQuantumKernel(feature_map=self.feature_map, fidelity=fidelity)
        kernel_matrix = kernel.evaluate(x_vec=X1, y_vec=X2)
        return kernel_matrix
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train SVM with noisy quantum kernel"""
        print("\nComputing noisy quantum kernel matrices...")

        # Compute kernels
        self.train_kernel = self.compute_kernel_matrix(X_train)
        self.test_kernel = self.compute_kernel_matrix(X_test, X_train)

        # Optional PSD projection before SVM training
        train_kernel_for_svm = self.train_kernel
        test_kernel_for_pred = self.test_kernel

        if self.psd_project:
            # Note: PSD projection only applies to square (Gram) matrices.
            # The test kernel K(X_test, X_train) is rectangular and does not
            # require PSD projection - it's used directly for prediction.
            print("Applying PSD projection to training kernel...")
            train_kernel_for_svm, diag_train = ensure_psd_kernel(
                self.train_kernel,
                epsilon=self.psd_epsilon,
                preserve_trace=True,
                return_diagnostics=True
            )
            self.psd_diagnostics['train'] = diag_train
            print(f"  Train kernel - min eigenvalue: {diag_train['min_eigenvalue_before']:.2e} -> {diag_train['min_eigenvalue_after']:.2e}")
            print(f"  Train kernel - clamped eigenvalues: {diag_train['num_clamped']}")
            # Test kernel is rectangular (n_test x n_train), no PSD projection needed

        # Train SVM
        print("Training SVM with noisy kernel...")
        svm = SVC(kernel='precomputed')
        svm.fit(train_kernel_for_svm, y_train)

        # Evaluate
        train_pred = svm.predict(train_kernel_for_svm)
        test_pred = svm.predict(test_kernel_for_pred)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        results = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'hardware_backend': self.hw_params['backend_name'],
            'noise_params': {
                'T1_us': self.hw_params['coherence_times']['T1_median_us'],
                'T2_us': self.hw_params['coherence_times']['T2_median_us'],
                'single_qubit_error_pct': self.hw_params['gate_errors']['single_qubit']['median_percent'],
                'two_qubit_error_pct': self.hw_params['gate_errors']['two_qubit_ECR']['median_percent'],
                'readout_error_pct': self.hw_params['readout_errors']['median_percent']
            },
            'shots': self.shots
        }
        
        return results, svm
    
    def visualize_kernels(self, save_dir='plots'):
        """Visualize noisy kernel matrices"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training kernel
        im1 = axes[0].imshow(self.train_kernel, cmap='plasma', aspect='auto')
        axes[0].set_title('Training Kernel (Noisy)', fontsize=14)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Sample Index')
        plt.colorbar(im1, ax=axes[0])
        
        # Test kernel
        im2 = axes[1].imshow(self.test_kernel, cmap='plasma', aspect='auto')
        axes[1].set_title('Test Kernel (Noisy)', fontsize=14)
        axes[1].set_xlabel('Training Sample Index')
        axes[1].set_ylabel('Test Sample Index')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/kernel_matrices_noisy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Noisy kernel visualization saved")
    
    def compare_with_ideal(self, save_dir='plots', results_dir='results'):
        """Compare noisy vs ideal kernels"""
        # Load ideal kernels
        try:
            ideal_train = np.load(f'{results_dir}/train_kernel_ideal.npy')
            ideal_test = np.load(f'{results_dir}/test_kernel_ideal.npy')
            
            # Compute differences
            train_diff = np.abs(self.train_kernel - ideal_train)
            test_diff = np.abs(self.test_kernel - ideal_test)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Ideal kernels
            axes[0,0].imshow(ideal_train, cmap='viridis', aspect='auto')
            axes[0,0].set_title('Ideal Training Kernel')
            
            axes[0,1].imshow(ideal_test, cmap='viridis', aspect='auto')
            axes[0,1].set_title('Ideal Test Kernel')
            
            # Noise impact
            im1 = axes[1,0].imshow(train_diff, cmap='hot', aspect='auto')
            axes[1,0].set_title(f'Noise Impact (Train) - Mean: {train_diff.mean():.4f}')
            plt.colorbar(im1, ax=axes[1,0])
            
            im2 = axes[1,1].imshow(test_diff, cmap='hot', aspect='auto')
            axes[1,1].set_title(f'Noise Impact (Test) - Mean: {test_diff.mean():.4f}')
            plt.colorbar(im2, ax=axes[1,1])
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/noise_impact_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Noise impact comparison saved")
            
            # Statistics
            stats = {
                'train_kernel_difference': {
                    'mean': float(train_diff.mean()),
                    'std': float(train_diff.std()),
                    'max': float(train_diff.max())
                },
                'test_kernel_difference': {
                    'mean': float(test_diff.mean()),
                    'std': float(test_diff.std()),
                    'max': float(test_diff.max())
                }
            }
            
            return stats
            
        except FileNotFoundError:
            print("Ideal kernel files not found - run qke_model.py first")
            return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Quantum Kernel Estimation - Noisy Simulator (IBM Hardware)'
    )
    parser.add_argument(
        '--output-tag',
        type=str,
        default=None,
        help='Tag for output directories (e.g., raw, psd). Creates results_<tag>/, plots_<tag>/'
    )
    parser.add_argument(
        '--psd-project',
        action='store_true',
        help='Enable PSD projection for kernel matrices before SVM training'
    )
    parser.add_argument(
        '--psd-epsilon',
        type=float,
        default=1e-10,
        help='Minimum eigenvalue threshold for PSD projection (default: 1e-10)'
    )
    return parser.parse_args()


def main():
    """Main execution pipeline"""
    args = parse_args()
    paths = resolve_output_paths(args.output_tag)
    results_dir = paths['results_dir']
    plots_dir = paths['plots_dir']

    print("=" * 70)
    print("Quantum Kernel Estimation - Noisy Simulator (IBM Hardware)")
    print("=" * 70)
    if args.output_tag:
        print(f"Output tag: {args.output_tag}")
    if args.psd_project:
        print(f"PSD projection: ENABLED (epsilon={args.psd_epsilon})")

    # Initialize with hardware parameters
    qke = NoisyQuantumKernelEstimator(
        hardware_params_path='data/ibm_hardware_params_2026.json',
        num_features=2,
        reps=2,
        shots=1024,
        psd_project=args.psd_project,
        psd_epsilon=args.psd_epsilon
    )

    # Generate dataset
    print("\n1. Generating dataset...")
    X_train, X_test, y_train, y_test = qke.generate_dataset(n_samples=100, noise=0.1)

    # Train and evaluate
    print("\n2. Training with noisy quantum kernel...")
    results, model = qke.train_and_evaluate(X_train, X_test, y_train, y_test)

    print(f"\n   Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"   Test Accuracy: {results['test_accuracy']:.4f}")

    # Add PSD diagnostics to results if enabled
    if args.psd_project:
        results['psd_projection'] = {
            'enabled': True,
            'epsilon': args.psd_epsilon,
            'diagnostics': qke.psd_diagnostics
        }

    # Save results
    print("\n3. Saving results...")
    results_dir.mkdir(exist_ok=True)

    np.save(results_dir / 'train_kernel_noisy.npy', qke.train_kernel)
    np.save(results_dir / 'test_kernel_noisy.npy', qke.test_kernel)

    with open(results_dir / 'metrics_noisy.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Visualizations
    print("\n4. Generating visualizations...")
    qke.visualize_kernels(save_dir=str(plots_dir))

    # Compare with ideal
    print("\n5. Comparing with ideal simulation...")
    noise_stats = qke.compare_with_ideal(save_dir=str(plots_dir), results_dir=str(results_dir))

    if noise_stats:
        with open(results_dir / 'noise_impact_stats.json', 'w') as f:
            json.dump(noise_stats, f, indent=2)

    print("\n" + "=" * 70)
    print("Noisy Simulation Complete!")
    print("=" * 70)
    print("\nOutputs:")
    print(f"  - {results_dir}/train_kernel_noisy.npy")
    print(f"  - {results_dir}/test_kernel_noisy.npy")
    print(f"  - {results_dir}/metrics_noisy.json")
    print(f"  - {results_dir}/noise_impact_stats.json")
    print(f"  - {plots_dir}/kernel_matrices_noisy.png")
    print(f"  - {plots_dir}/noise_impact_comparison.png")


if __name__ == "__main__":
    main()
