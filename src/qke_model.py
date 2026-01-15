"""
Quantum Kernel Estimation for Binary Classification
Uses ZZFeatureMap with RealAmplitudes ansatz for quantum kernels

Implementation: Ideal simulator (no noise)
Date: 2026-01-10
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler, BackendEstimator

# Qiskit Machine Learning
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel

# Classical ML
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# Visualization
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Non-interactive backend for server

class QuantumKernelEstimator:
    """Quantum kernel-based classifier with kernel estimation"""
    
    def __init__(self, num_features=2, reps=2, shots=1024):
        self.num_features = num_features
        self.reps = reps
        self.shots = shots
        
        # Feature map for encoding data
        self.feature_map = ZZFeatureMap(
            feature_dimension=num_features,
            reps=reps,
            entanglement='linear',
            insert_barriers=True
        )
        
        # Ansatz for trainable parameters
        self.ansatz = RealAmplitudes(
            num_qubits=num_features,
            reps=1,
            insert_barriers=True
        )
        
        # Ideal quantum simulator
        self.backend = AerSimulator(method='statevector')
        
        # Quantum kernel
        self.kernel = QuantumKernel(
            feature_map=self.feature_map,
            quantum_instance=self.backend
        )
        
        # Results storage
        self.train_kernel = None
        self.test_kernel = None
        self.convergence_history = []
        
    def generate_dataset(self, n_samples=100, noise=0.1, test_size=0.3):
        """Generate synthetic binary classification dataset"""
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        
        # Convert labels to +1/-1 for SVM
        y = 2 * y - 1
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def compute_kernel_matrix(self, X1, X2=None):
        """Compute quantum kernel matrix"""
        if X2 is None:
            X2 = X1
        
        # Use BackendSampler for kernel estimation
        sampler = BackendSampler(backend=self.backend, options={"shots": self.shots})
        kernel = QuantumKernel(feature_map=self.feature_map, quantum_instance=sampler)
        
        kernel_matrix = kernel.evaluate(x_vec=X1, y_vec=X2)
        return kernel_matrix
    
    def train_vqc(self, X_train, y_train, max_iter=100):
        """Train Variational Quantum Classifier"""
        print("Training VQC with quantum kernel estimation...")
        
        # Optimizer
        optimizer = SPSA(maxiter=max_iter)
        
        # Callback for convergence tracking
        def callback(eval_count, params, value, meta):
            self.convergence_history.append({
                'iteration': eval_count,
                'loss': value,
                'timestamp': datetime.now().isoformat()
            })
            if eval_count % 10 == 0:
                print(f"Iteration {eval_count}: Loss = {value:.4f}")
        
        # VQC with kernel
        sampler = BackendSampler(backend=self.backend, options={"shots": self.shots})
        
        vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=optimizer,
            sampler=sampler,
            callback=callback
        )
        
        # Fit model
        vqc.fit(X_train, y_train)
        
        return vqc
    
    def train_kernel_svm(self, X_train, y_train):
        """Train classical SVM with quantum kernel"""
        print("Computing quantum kernel matrix for training...")
        self.train_kernel = self.compute_kernel_matrix(X_train)
        
        # Train SVM with precomputed kernel
        svm = SVC(kernel='precomputed')
        svm.fit(self.train_kernel, y_train)
        
        return svm
    
    def evaluate(self, model, X_train, X_test, y_train, y_test, model_type='svm'):
        """Evaluate model performance"""
        if model_type == 'svm':
            # Compute test kernel
            self.test_kernel = self.compute_kernel_matrix(X_test, X_train)
            
            # Predictions
            train_pred = model.predict(self.train_kernel)
            test_pred = model.predict(self.test_kernel)
        else:  # VQC
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
        
        # Metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        results = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_report': classification_report(y_train, train_pred, output_dict=True),
            'test_report': classification_report(y_test, test_pred, output_dict=True)
        }
        
        return results
    
    def visualize_kernels(self, save_dir='plots'):
        """Visualize kernel matrices"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training kernel
        im1 = axes[0].imshow(self.train_kernel, cmap='viridis', aspect='auto')
        axes[0].set_title('Training Kernel Matrix (Ideal)', fontsize=14)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Sample Index')
        plt.colorbar(im1, ax=axes[0])
        
        # Test kernel
        im2 = axes[1].imshow(self.test_kernel, cmap='viridis', aspect='auto')
        axes[1].set_title('Test Kernel Matrix (Ideal)', fontsize=14)
        axes[1].set_xlabel('Training Sample Index')
        axes[1].set_ylabel('Test Sample Index')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/kernel_matrices_ideal.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Kernel visualization saved to {save_dir}/kernel_matrices_ideal.png")
    
    def plot_convergence(self, save_dir='plots'):
        """Plot training convergence"""
        if not self.convergence_history:
            print("No convergence history available")
            return
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        iterations = [item['iteration'] for item in self.convergence_history]
        losses = [item['loss'] for item in self.convergence_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, losses, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('VQC Training Convergence (Ideal Simulator)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/convergence_ideal.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Convergence plot saved to {save_dir}/convergence_ideal.png")

def main():
    """Main execution pipeline"""
    print("=" * 70)
    print("Quantum Kernel Estimation - Ideal Simulator")
    print("=" * 70)
    
    # Initialize estimator
    qke = QuantumKernelEstimator(num_features=2, reps=2, shots=1024)
    
    # Generate dataset
    print("\n1. Generating dataset...")
    X_train, X_test, y_train, y_test = qke.generate_dataset(n_samples=100, noise=0.1)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train kernel SVM
    print("\n2. Training Kernel SVM...")
    svm_model = qke.train_kernel_svm(X_train, y_train)
    
    # Evaluate
    print("\n3. Evaluating model...")
    results = qke.evaluate(svm_model, X_train, X_test, y_train, y_test, model_type='svm')
    
    print(f"\n   Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
    
    # Save results
    print("\n4. Saving results...")
    
    # Kernel matrices
    Path('results').mkdir(exist_ok=True)
    np.save('results/train_kernel_ideal.npy', qke.train_kernel)
    np.save('results/test_kernel_ideal.npy', qke.test_kernel)
    
    # Metrics
    with open('results/metrics_ideal.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualizations
    print("\n5. Generating visualizations...")
    qke.visualize_kernels(save_dir='plots')
    
    print("\n" + "=" * 70)
    print("Workflow Complete!")
    print("=" * 70)
    print("\nOutputs:")
    print("  - results/train_kernel_ideal.npy")
    print("  - results/test_kernel_ideal.npy")
    print("  - results/metrics_ideal.json")
    print("  - plots/kernel_matrices_ideal.png")

if __name__ == "__main__":
    main()
