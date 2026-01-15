"""
Comprehensive Analysis of Quantum Kernel Estimation Results
Compares ideal, noisy, and hardware implementations

Date: 2026-01-10
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

class KernelAnalyzer:
    """Analyze and compare quantum kernel estimation results"""
    
    def __init__(self, results_dir='results', plots_dir='plots'):
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load all results
        self.kernels = self._load_kernels()
        self.metrics = self._load_metrics()
        
    def _load_kernels(self):
        """Load all kernel matrices"""
        kernels = {}
        
        kernel_files = {
            'ideal_train': 'train_kernel_ideal.npy',
            'ideal_test': 'test_kernel_ideal.npy',
            'noisy_train': 'train_kernel_noisy.npy',
            'noisy_test': 'test_kernel_noisy.npy',
            'hardware_train': 'train_kernel_hardware.npy',
            'hardware_test': 'test_kernel_hardware.npy'
        }
        
        for key, filename in kernel_files.items():
            filepath = self.results_dir / filename
            if filepath.exists():
                kernels[key] = np.load(filepath)
                print(f"✓ Loaded {filename}")
            else:
                print(f"  Missing: {filename}")
        
        return kernels
    
    def _load_metrics(self):
        """Load all metrics"""
        metrics = {}
        
        metric_files = ['metrics_ideal.json', 'metrics_noisy.json', 'metrics_hardware.json']
        
        for filename in metric_files:
            filepath = self.results_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    key = filename.replace('metrics_', '').replace('.json', '')
                    metrics[key] = json.load(f)
                    print(f"✓ Loaded {filename}")
        
        return metrics
    
    def compute_kernel_alignment(self, K1, K2):
        """Compute kernel alignment between two kernels"""
        # Frobenius inner product
        alignment = np.sum(K1 * K2) / np.sqrt(np.sum(K1 ** 2) * np.sum(K2 ** 2))
        return alignment
    
    def compute_kernel_stats(self, kernel, name):
        """Compute statistical properties of kernel"""
        stats = {
            'name': name,
            'mean': float(np.mean(kernel)),
            'std': float(np.std(kernel)),
            'min': float(np.min(kernel)),
            'max': float(np.max(kernel)),
            'trace': float(np.trace(kernel)) if kernel.shape[0] == kernel.shape[1] else None,
            'frobenius_norm': float(np.linalg.norm(kernel, 'fro'))
        }
        return stats
    
    def analyze_all(self):
        """Perform comprehensive analysis"""
        analysis = {
            'kernel_statistics': {},
            'kernel_alignments': {},
            'accuracy_comparison': {},
            'noise_impact': {}
        }
        
        # 1. Kernel statistics
        print("\n" + "=" * 60)
        print("KERNEL STATISTICS")
        print("=" * 60)
        
        for key, kernel in self.kernels.items():
            stats = self.compute_kernel_stats(kernel, key)
            analysis['kernel_statistics'][key] = stats
            print(f"\n{key}:")
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            if stats['trace']:
                print(f"  Trace: {stats['trace']:.4f}")
        
        # 2. Kernel alignment
        print("\n" + "=" * 60)
        print("KERNEL ALIGNMENT ANALYSIS")
        print("=" * 60)
        
        if 'ideal_train' in self.kernels and 'noisy_train' in self.kernels:
            align_train = self.compute_kernel_alignment(
                self.kernels['ideal_train'],
                self.kernels['noisy_train']
            )
            analysis['kernel_alignments']['ideal_vs_noisy_train'] = float(align_train)
            print(f"\nIdeal vs Noisy (Train): {align_train:.4f}")
        
        if 'ideal_train' in self.kernels and 'hardware_train' in self.kernels:
            align_hw = self.compute_kernel_alignment(
                self.kernels['ideal_train'],
                self.kernels['hardware_train']
            )
            analysis['kernel_alignments']['ideal_vs_hardware_train'] = float(align_hw)
            print(f"Ideal vs Hardware (Train): {align_hw:.4f}")
        
        # 3. Accuracy comparison
        print("\n" + "=" * 60)
        print("ACCURACY COMPARISON")
        print("=" * 60)
        
        for key, metric in self.metrics.items():
            train_acc = metric.get('train_accuracy', 0)
            test_acc = metric.get('test_accuracy', 0)
            analysis['accuracy_comparison'][key] = {
                'train': float(train_acc),
                'test': float(test_acc)
            }
            print(f"\n{key.upper()}:")
            print(f"  Train: {train_acc:.4f}")
            print(f"  Test:  {test_acc:.4f}")
        
        # 4. Noise impact
        if 'ideal_train' in self.kernels and 'noisy_train' in self.kernels:
            print("\n" + "=" * 60)
            print("NOISE IMPACT ANALYSIS")
            print("=" * 60)
            
            diff_train = np.abs(self.kernels['ideal_train'] - self.kernels['noisy_train'])
            diff_test = np.abs(self.kernels['ideal_test'] - self.kernels['noisy_test'])
            
            analysis['noise_impact'] = {
                'train_kernel_mean_diff': float(diff_train.mean()),
                'train_kernel_max_diff': float(diff_train.max()),
                'test_kernel_mean_diff': float(diff_test.mean()),
                'test_kernel_max_diff': float(diff_test.max())
            }
            
            print(f"\nTraining Kernel Difference:")
            print(f"  Mean: {diff_train.mean():.4f}")
            print(f"  Max:  {diff_train.max():.4f}")
            print(f"\nTest Kernel Difference:")
            print(f"  Mean: {diff_test.mean():.4f}")
            print(f"  Max:  {diff_test.max():.4f}")
        
        return analysis
    
    def create_comparison_plots(self):
        """Generate comprehensive comparison visualizations"""
        
        # 1. Kernel heatmap comparison
        if all(k in self.kernels for k in ['ideal_train', 'noisy_train', 'hardware_train']):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            kernels_to_plot = [
                ('ideal_train', 'Ideal', 'viridis'),
                ('noisy_train', 'Noisy', 'plasma'),
                ('hardware_train', 'Hardware', 'inferno')
            ]
            
            for idx, (key, title, cmap) in enumerate(kernels_to_plot):
                im = axes[idx].imshow(self.kernels[key], cmap=cmap, aspect='auto')
                axes[idx].set_title(f'{title} Training Kernel', fontsize=14)
                axes[idx].set_xlabel('Sample Index')
                axes[idx].set_ylabel('Sample Index')
                plt.colorbar(im, ax=axes[idx])
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'kernel_comparison_all.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved kernel_comparison_all.png")
        
        # 2. Accuracy bar chart
        if self.metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            implementations = list(self.metrics.keys())
            train_accs = [self.metrics[k].get('train_accuracy', 0) for k in implementations]
            test_accs = [self.metrics[k].get('test_accuracy', 0) for k in implementations]
            
            x = np.arange(len(implementations))
            width = 0.35
            
            ax.bar(x - width/2, train_accs, width, label='Training', color='steelblue')
            ax.bar(x + width/2, test_accs, width, label='Test', color='coral')
            
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Accuracy Comparison: Ideal vs Noisy vs Hardware', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([k.capitalize() for k in implementations])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.1])
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved accuracy_comparison.png")
        
        # 3. Kernel difference heatmap
        if 'ideal_train' in self.kernels and 'noisy_train' in self.kernels:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            diff_train = np.abs(self.kernels['ideal_train'] - self.kernels['noisy_train'])
            diff_test = np.abs(self.kernels['ideal_test'] - self.kernels['noisy_test'])
            
            im1 = axes[0].imshow(diff_train, cmap='hot', aspect='auto')
            axes[0].set_title(f'Training Kernel Error\nMean: {diff_train.mean():.4f}', fontsize=12)
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(diff_test, cmap='hot', aspect='auto')
            axes[1].set_title(f'Test Kernel Error\nMean: {diff_test.mean():.4f}', fontsize=12)
            plt.colorbar(im2, ax=axes[1])
            
            plt.suptitle('Noise Impact: |Ideal - Noisy|', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'kernel_error_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved kernel_error_heatmap.png")
    
    def generate_summary_report(self, analysis):
        """Generate markdown summary report"""
        report = f"""# Quantum Kernel Estimation Analysis Report

**Date:** {Path('.').resolve().name}  
**Generated:** 2026-01-10

## Executive Summary

This report analyzes quantum kernel estimation across three implementations:
- **Ideal**: Statevector simulator (no noise)
- **Noisy**: Realistic IBM hardware noise model
- **Hardware**: IBM Quantum platform (or high-fidelity fallback)

## Kernel Statistics

"""
        
        for key, stats in analysis['kernel_statistics'].items():
            report += f"\n### {key.replace('_', ' ').title()}\n\n"
            report += f"- Mean: {stats['mean']:.4f}\n"
            report += f"- Std: {stats['std']:.4f}\n"
            report += f"- Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
            if stats['trace']:
                report += f"- Trace: {stats['trace']:.4f}\n"
            report += f"- Frobenius Norm: {stats['frobenius_norm']:.4f}\n"
        
        report += "\n## Kernel Alignment\n\n"
        
        for key, value in analysis['kernel_alignments'].items():
            report += f"- {key.replace('_', ' ').title()}: {value:.4f}\n"
        
        report += "\n## Classification Accuracy\n\n"
        report += "| Implementation | Training | Test |\n"
        report += "|---------------|----------|------|\n"
        
        for impl, accs in analysis['accuracy_comparison'].items():
            report += f"| {impl.capitalize()} | {accs['train']:.4f} | {accs['test']:.4f} |\n"
        
        if analysis['noise_impact']:
            report += "\n## Noise Impact\n\n"
            ni = analysis['noise_impact']
            report += f"- Training Kernel Mean Difference: {ni['train_kernel_mean_diff']:.4f}\n"
            report += f"- Training Kernel Max Difference: {ni['train_kernel_max_diff']:.4f}\n"
            report += f"- Test Kernel Mean Difference: {ni['test_kernel_mean_diff']:.4f}\n"
            report += f"- Test Kernel Max Difference: {ni['test_kernel_max_diff']:.4f}\n"
        
        report += "\n## Key Findings\n\n"
        report += "1. Quantum kernels show sensitivity to hardware noise\n"
        report += "2. Kernel alignment between ideal and noisy implementations indicates noise tolerance\n"
        report += "3. Classification accuracy degradation quantifies practical impact\n"
        report += "4. IBM hardware parameters calibrated from real 127-qubit systems\n"
        
        report += "\n## Recommendations\n\n"
        report += "- **Error Mitigation**: Implement zero-noise extrapolation for hardware runs\n"
        report += "- **Circuit Optimization**: Reduce circuit depth through transpilation\n"
        report += "- **Ensemble Methods**: Average multiple quantum kernel estimates\n"
        report += "- **Adaptive Features**: Optimize feature map based on hardware characteristics\n"
        
        return report

def main():
    """Main analysis pipeline"""
    print("=" * 70)
    print("Quantum Kernel Estimation - Comprehensive Analysis")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = KernelAnalyzer(results_dir='results', plots_dir='plots')
    
    # Perform analysis
    print("\nPerforming comprehensive analysis...")
    analysis = analyzer.analyze_all()
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    analyzer.create_comparison_plots()
    
    # Save analysis
    print("\n" + "=" * 60)
    print("SAVING ANALYSIS RESULTS")
    print("=" * 60)
    
    with open('results/comprehensive_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print("✓ Saved comprehensive_analysis.json")
    
    # Generate report
    report = analyzer.generate_summary_report(analysis)
    with open('docs/analysis_report.md', 'w') as f:
        f.write(report)
    print("✓ Saved docs/analysis_report.md")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - results/comprehensive_analysis.json")
    print("  - docs/analysis_report.md")
    print("  - plots/kernel_comparison_all.png")
    print("  - plots/accuracy_comparison.png")
    print("  - plots/kernel_error_heatmap.png")

if __name__ == "__main__":
    main()
