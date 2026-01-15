"""
Quantum Kernel Estimation with Realistic IBM Quantum Noise
Models hardware imperfections from 127-qubit Eagle processors

Implementation: Noisy simulator with hardware-calibrated parameters
Date: 2026-01-10
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import numpy as np
import json
from pathlib import Path

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, thermal_relaxation_error,
    depolarizing_error, pauli_error, ReadoutError
)
from qiskit.primitives import BackendSampler

# Qiskit Machine Learning
from qiskit_machine_learning.kernels import QuantumKernel

# Classical ML
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Visualization
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

class NoisyQuantumKernelEstimator:
    """Quantum kernel with realistic IBM hardware noise"""
    
    def __init__(self, hardware_params_path='data/ibm_hardware_params_2026.json',
                 num_features=2, reps=2, shots=1024):
        self.num_features = num_features
        self.reps = reps
        self.shots = shots
        
        # Load hardware parameters
        with open(hardware_params_path, 'r') as f:
            self.hw_params = json.load(f)
        
        print(f"Loaded hardware parameters for: {self.hw_params['backend_name']}")
        print(f"Processor: {self.hw_params['processor_family']}")
        
        # Feature map and ansatz
        self.feature_map = ZZFeatureMap(
            feature_dimension=num_features,
            reps=reps,
            entanglement='linear'
        )
        
        # Build noise model
        self.noise_model = self._build_noise_model()
        
        # Noisy simulator
        self.backend = AerSimulator(noise_model=self.noise_model)
        
        # Results
        self.train_kernel = None
        self.test_kernel = None
        
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
        
        sampler = BackendSampler(backend=self.backend, options={"shots": self.shots})
        kernel = QuantumKernel(feature_map=self.feature_map, quantum_instance=sampler)
        
        kernel_matrix = kernel.evaluate(x_vec=X1, y_vec=X2)
        return kernel_matrix
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train SVM with noisy quantum kernel"""
        print("\nComputing noisy quantum kernel matrices...")
        
        # Compute kernels
        self.train_kernel = self.compute_kernel_matrix(X_train)
        self.test_kernel = self.compute_kernel_matrix(X_test, X_train)
        
        # Train SVM
        print("Training SVM with noisy kernel...")
        svm = SVC(kernel='precomputed')
        svm.fit(self.train_kernel, y_train)
        
        # Evaluate
        train_pred = svm.predict(self.train_kernel)
        test_pred = svm.predict(self.test_kernel)
        
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
    
    def compare_with_ideal(self, save_dir='plots'):
        """Compare noisy vs ideal kernels"""
        # Load ideal kernels
        try:
            ideal_train = np.load('results/train_kernel_ideal.npy')
            ideal_test = np.load('results/test_kernel_ideal.npy')
            
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

def main():
    """Main execution pipeline"""
    print("=" * 70)
    print("Quantum Kernel Estimation - Noisy Simulator (IBM Hardware)")
    print("=" * 70)
    
    # Initialize with hardware parameters
    qke = NoisyQuantumKernelEstimator(
        hardware_params_path='data/ibm_hardware_params_2026.json',
        num_features=2,
        reps=2,
        shots=1024
    )
    
    # Generate dataset
    print("\n1. Generating dataset...")
    X_train, X_test, y_train, y_test = qke.generate_dataset(n_samples=100, noise=0.1)
    
    # Train and evaluate
    print("\n2. Training with noisy quantum kernel...")
    results, model = qke.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print(f"\n   Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
    
    # Save results
    print("\n3. Saving results...")
    Path('results').mkdir(exist_ok=True)
    
    np.save('results/train_kernel_noisy.npy', qke.train_kernel)
    np.save('results/test_kernel_noisy.npy', qke.test_kernel)
    
    with open('results/metrics_noisy.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualizations
    print("\n4. Generating visualizations...")
    qke.visualize_kernels(save_dir='plots')
    
    # Compare with ideal
    print("\n5. Comparing with ideal simulation...")
    noise_stats = qke.compare_with_ideal(save_dir='plots')
    
    if noise_stats:
        with open('results/noise_impact_stats.json', 'w') as f:
            json.dump(noise_stats, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Noisy Simulation Complete!")
    print("=" * 70)
    print("\nOutputs:")
    print("  - results/train_kernel_noisy.npy")
    print("  - results/test_kernel_noisy.npy")
    print("  - results/metrics_noisy.json")
    print("  - results/noise_impact_stats.json")
    print("  - plots/kernel_matrices_noisy.png")
    print("  - plots/noise_impact_comparison.png")

if __name__ == "__main__":
    main()
