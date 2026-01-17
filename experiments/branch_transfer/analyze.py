"""
Analysis and visualization for the branch-conditioned message transfer experiment.

Loads artifacts, computes visibility metrics, generates publication-quality plots.

Date: 2026-01-17
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import glob

import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def load_results(artifacts_dir: Path, pattern: str = '*.json') -> List[dict]:
    """
    Load all JSON result files from artifacts directory.

    Parameters
    ----------
    artifacts_dir : Path
        Directory containing result JSON files.
    pattern : str
        Glob pattern for files.

    Returns
    -------
    list
        List of loaded result dictionaries.
    """
    files = sorted(artifacts_dir.glob(pattern))
    results = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                data['_source_file'] = str(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    return results


def filter_results(
    results: List[dict],
    backend_type: Optional[str] = None,
    mode: Optional[str] = None,
    mu: Optional[int] = None,
) -> List[dict]:
    """
    Filter results by criteria.

    Parameters
    ----------
    results : list
        List of result dictionaries.
    backend_type : str, optional
        Filter by backend type ('ideal', 'noisy_simulator', 'hardware').
    mode : str, optional
        Filter by mode ('main', 'control').
    mu : int, optional
        Filter by message bit.

    Returns
    -------
    list
        Filtered results.
    """
    filtered = results
    if backend_type:
        filtered = [r for r in filtered if r.get('backend_type') == backend_type]
    if mode:
        filtered = [r for r in filtered if r.get('mode') == mode]
    if mu is not None:
        filtered = [r for r in filtered if r.get('mu') == mu]
    return filtered


def compute_visibility_summary(results: List[dict]) -> Dict[str, Any]:
    """
    Compute summary statistics for visibility across results.

    Parameters
    ----------
    results : list
        List of result dictionaries.

    Returns
    -------
    dict
        Summary statistics including mean, std, min, max.
    """
    if not results:
        return {}

    visibilities = [r['visibility'] for r in results if 'visibility' in r]
    errors = [r.get('visibility_error', 0) for r in results if 'visibility' in r]

    return {
        'count': len(visibilities),
        'mean': float(np.mean(visibilities)),
        'std': float(np.std(visibilities)),
        'min': float(np.min(visibilities)),
        'max': float(np.max(visibilities)),
        'mean_error': float(np.mean(errors)),
    }


def compute_bootstrap_ci(
    counts: dict,
    shots: int,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> tuple:
    """
    Compute bootstrap confidence interval for visibility.

    Parameters
    ----------
    counts : dict
        Measurement counts.
    shots : int
        Total shots.
    n_bootstrap : int
        Number of bootstrap samples.
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI).

    Returns
    -------
    V_mean : float
        Mean visibility.
    V_ci_low : float
        Lower CI bound.
    V_ci_high : float
        Upper CI bound.
    """
    from circuit import compute_visibility_from_counts

    # Convert counts to samples
    samples = []
    for bitstring, count in counts.items():
        samples.extend([bitstring] * count)
    samples = np.array(samples)

    # Bootstrap
    V_bootstrap = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(samples, size=len(samples), replace=True)
        # Reconstruct counts
        resampled_counts = {}
        for s in resampled:
            resampled_counts[s] = resampled_counts.get(s, 0) + 1
        V, _, _ = compute_visibility_from_counts(resampled_counts, len(samples))
        V_bootstrap.append(V)

    V_bootstrap = np.array(V_bootstrap)
    alpha = (1 - ci_level) / 2
    V_ci_low = np.percentile(V_bootstrap, 100 * alpha)
    V_ci_high = np.percentile(V_bootstrap, 100 * (1 - alpha))
    V_mean = np.mean(V_bootstrap)

    return V_mean, V_ci_low, V_ci_high


def plot_pr_distribution(
    results: List[dict],
    output_path: Path,
    title: str = 'PR Distribution'
):
    """
    Create bar plot of PR distribution for multiple backends/modes.

    Parameters
    ----------
    results : list
        List of result dictionaries.
    output_path : Path
        Path to save plot.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # All possible bitstrings
    bitstrings = ['00', '01', '10', '11']
    x = np.arange(len(bitstrings))
    width = 0.8 / max(len(results), 1)

    for i, result in enumerate(results):
        label = f"{result.get('backend_type', 'unknown')}"
        if 'optimization_level' in result:
            label += f" (opt={result['optimization_level']})"

        probs = result.get('probabilities', {})
        heights = [probs.get(bs, 0) for bs in bitstrings]

        ax.bar(x + i * width, heights, width, label=label, alpha=0.8)

    ax.set_xlabel('Bitstring PR (c[1]c[0] = Paper, Room)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(bitstrings)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Add ideal reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Ideal 50%')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_visibility_comparison(
    results: List[dict],
    output_path: Path,
    group_by: str = 'backend_type',
    title: str = 'Visibility Comparison'
):
    """
    Create bar plot comparing visibility across backends/configurations.

    Parameters
    ----------
    results : list
        List of result dictionaries.
    output_path : Path
        Path to save plot.
    group_by : str
        Key to group results by.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    visibilities = []
    errors = []

    for result in results:
        label = str(result.get(group_by, 'unknown'))
        if 'optimization_level' in result:
            label += f"\n(opt={result['optimization_level']})"
        labels.append(label)
        visibilities.append(result.get('visibility', 0))
        errors.append(result.get('visibility_error', 0))

    x = np.arange(len(labels))
    bars = ax.bar(x, visibilities, yerr=errors, capsize=5, alpha=0.8)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Visibility (V)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Add ideal reference line
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal V=1.0')
    ax.legend()

    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_visibility_vs_opt_level(
    results: List[dict],
    output_path: Path,
    title: str = 'Visibility vs Optimization Level'
):
    """
    Plot visibility as a function of transpiler optimization level.

    Parameters
    ----------
    results : list
        List of result dictionaries (should have different opt levels).
    output_path : Path
        Path to save plot.
    title : str
        Plot title.
    """
    # Group by optimization level
    opt_data = {}
    for result in results:
        opt = result.get('optimization_level', 0)
        if opt not in opt_data:
            opt_data[opt] = {'V': [], 'depth': [], 'error': []}
        opt_data[opt]['V'].append(result.get('visibility', 0))
        opt_data[opt]['depth'].append(result.get('transpiled_depth', 0))
        opt_data[opt]['error'].append(result.get('visibility_error', 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    opt_levels = sorted(opt_data.keys())
    V_means = [np.mean(opt_data[o]['V']) for o in opt_levels]
    V_stds = [np.std(opt_data[o]['V']) if len(opt_data[o]['V']) > 1 else np.mean(opt_data[o]['error']) for o in opt_levels]
    depths = [np.mean(opt_data[o]['depth']) for o in opt_levels]

    # Visibility vs opt level
    ax1.errorbar(opt_levels, V_means, yerr=V_stds, marker='o', capsize=5, linewidth=2, markersize=8)
    ax1.set_xlabel('Optimization Level', fontsize=12)
    ax1.set_ylabel('Visibility (V)', fontsize=12)
    ax1.set_title('Visibility vs Optimization Level', fontsize=14)
    ax1.set_xticks(opt_levels)
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal V=1.0')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.2)

    # Depth vs opt level
    ax2.bar(opt_levels, depths, alpha=0.8)
    ax2.set_xlabel('Optimization Level', fontsize=12)
    ax2.set_ylabel('Transpiled Circuit Depth', fontsize=12)
    ax2.set_title('Circuit Depth vs Optimization Level', fontsize=14)
    ax2.set_xticks(opt_levels)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_collapse_forecast(
    sweep_data: dict,
    output_path: Path,
    title: str = 'Collapse Model Forecast'
):
    """
    Plot V(gamma) forecast curve from collapse model sweep.

    Parameters
    ----------
    sweep_data : dict
        Data from collapse_models.py gamma sweep.
    output_path : Path
        Path to save plot.
    title : str
        Plot title.
    """
    gamma = sweep_data.get('gamma_values', [])
    V = sweep_data.get('visibility_values', [])
    V_err = sweep_data.get('visibility_errors', [])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(gamma, V, yerr=V_err, marker='o', capsize=3, linewidth=2, markersize=6)
    ax.set_xlabel(r'Collapse Strength ($\gamma$)', fontsize=12)
    ax.set_ylabel('Visibility (V)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Reference lines
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal V=1.0')
    ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5, label='Complete dephasing')

    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_coherence_forecast(
    sweep_data: dict,
    output_path: Path,
    title: str = 'Coherence Witness Forecast'
):
    """
    Plot W_tilde(gamma) forecast curve from coherence sweep.

    This is the correct plot for detecting collapse effects, as W_X probes
    off-diagonal coherence and is sensitive to dephasing (unlike V).

    Parameters
    ----------
    sweep_data : dict
        Data from collapse_models.py coherence gamma sweep.
    output_path : Path
        Path to save plot.
    title : str
        Plot title.
    """
    gamma = sweep_data.get('gamma_values', [])
    basis = sweep_data.get('measurement_basis', 'X')

    W = sweep_data.get(f'W_{basis}_values', [])
    W_err = sweep_data.get(f'W_{basis}_errors', [])
    W_tilde = sweep_data.get(f'W_{basis}_tilde_values', [])
    W_tilde_err = sweep_data.get(f'W_{basis}_tilde_errors', [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw W values
    ax1.errorbar(gamma, W, yerr=W_err, marker='o', capsize=3, linewidth=2, markersize=6, color='blue')
    ax1.set_xlabel(r'Collapse Strength ($\gamma$)', fontsize=12)
    ax1.set_ylabel(f'$W_{{{basis}}}$ (raw coherence witness)', fontsize=12)
    ax1.set_title(f'Raw Coherence Witness ({basis} basis)', fontsize=14)
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal coherent')
    ax1.axhline(y=0.0, color='red', linestyle=':', alpha=0.5, label='Fully decohered')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.2, 1.2)

    # Normalized W_tilde values
    ax2.errorbar(gamma, W_tilde, yerr=W_tilde_err, marker='s', capsize=3, linewidth=2, markersize=6, color='purple')
    ax2.set_xlabel(r'Collapse Strength ($\gamma$)', fontsize=12)
    ax2.set_ylabel(r'$\tilde{W}_X = W_X / W_X^{\mathrm{ideal}}$', fontsize=12)
    ax2.set_title(f'Normalized Coherence ({basis} basis)', fontsize=14)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal (no collapse)')
    ax2.axhline(y=0.0, color='red', linestyle=':', alpha=0.5, label='Complete decoherence')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.2, 1.2)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_coherence_comparison(
    results: List[dict],
    output_path: Path,
    title: str = 'Coherence Witness Comparison'
):
    """
    Plot coherence witness comparison across configurations.

    Parameters
    ----------
    results : list
        List of result dictionaries with coherence data.
    output_path : Path
        Path to save plot.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    W_X_values = []
    W_X_errors = []
    W_X_tilde_values = []

    for result in results:
        label = f"{result.get('backend_type', 'unknown')}"
        if 'optimization_level' in result:
            label += f" (opt={result['optimization_level']})"
        labels.append(label)
        W_X_values.append(result.get('W_X', 0))
        W_X_errors.append(result.get('W_X_error', 0))
        W_X_tilde_values.append(result.get('W_X_tilde', 0))

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, W_X_values, width, yerr=W_X_errors,
                   capsize=5, alpha=0.8, label='$W_X$ (raw)')
    bars2 = ax.bar(x + width/2, W_X_tilde_values, width,
                   alpha=0.8, label=r'$\tilde{W}_X$ (normalized)')

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Coherence Witness', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.2, 1.4)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_v_vs_coherence(
    results: List[dict],
    output_path: Path,
    title: str = 'Visibility vs Coherence Witness'
):
    """
    Scatter plot comparing V (visibility) and W_X (coherence witness).

    This visualization shows why W_X is a better collapse detector than V.

    Parameters
    ----------
    results : list
        List of result dictionaries with both V and W_X.
    output_path : Path
        Path to save plot.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Filter for results that have both V and W_X
    filtered = [r for r in results if 'visibility' in r and 'W_X' in r]

    if not filtered:
        print("No results with both V and W_X found")
        return

    V_values = [r['visibility'] for r in filtered]
    W_X_values = [r['W_X'] for r in filtered]

    # Color by gamma if available
    if all('collapse_gamma' in r for r in filtered):
        gamma_values = [r['collapse_gamma'] for r in filtered]
        scatter = ax.scatter(V_values, W_X_values, c=gamma_values, cmap='viridis',
                            s=80, alpha=0.8, edgecolors='black')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'Collapse strength $\gamma$', fontsize=12)
    else:
        ax.scatter(V_values, W_X_values, s=80, alpha=0.8, edgecolors='black')

    ax.set_xlabel('Visibility (V)', fontsize=12)
    ax.set_ylabel('$W_X$ (Coherence Witness)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='V = W_X')

    # Add reference lines
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.3)
    ax.axvline(x=1.0, color='green', linestyle=':', alpha=0.3)

    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def compute_coherence_summary(results: List[dict]) -> Dict[str, Any]:
    """
    Compute summary statistics for coherence witness across results.

    Parameters
    ----------
    results : list
        List of result dictionaries.

    Returns
    -------
    dict
        Summary statistics including mean, std for W_X and W_X_tilde.
    """
    if not results:
        return {}

    W_X = [r['W_X'] for r in results if 'W_X' in r]
    W_X_tilde = [r.get('W_X_tilde', 0) for r in results if 'W_X' in r]

    if not W_X:
        return {}

    return {
        'count': len(W_X),
        'W_X_mean': float(np.mean(W_X)),
        'W_X_std': float(np.std(W_X)),
        'W_X_tilde_mean': float(np.mean(W_X_tilde)),
        'W_X_tilde_std': float(np.std(W_X_tilde)),
    }


def generate_summary_table(results: List[dict]) -> str:
    """
    Generate a markdown summary table of results.

    Parameters
    ----------
    results : list
        List of result dictionaries.

    Returns
    -------
    str
        Markdown table string.
    """
    lines = [
        "| Backend | Mode | μ | Shots | V | V_err | Depth | 2Q Gates |",
        "|---------|------|---|-------|---|-------|-------|----------|",
    ]

    for r in results:
        backend = r.get('backend_type', r.get('backend', 'unknown'))
        mode = r.get('mode', '-')
        mu = r.get('mu', '-')
        shots = r.get('shots', '-')
        V = r.get('visibility', 0)
        V_err = r.get('visibility_error', 0)
        depth = r.get('transpiled_depth', '-')
        two_q = r.get('circuit_stats', {}).get('two_qubit_gate_count', '-')

        lines.append(
            f"| {backend} | {mode} | {mu} | {shots} | {V:.4f} | {V_err:.4f} | {depth} | {two_q} |"
        )

    return '\n'.join(lines)


def generate_analysis_report(
    artifacts_dir: Path,
    output_path: Path,
) -> dict:
    """
    Generate comprehensive analysis report.

    Parameters
    ----------
    artifacts_dir : Path
        Directory containing result artifacts.
    output_path : Path
        Path to save analysis JSON.

    Returns
    -------
    dict
        Analysis results.
    """
    # Load all results
    results = load_results(artifacts_dir)
    if not results:
        print("No results found")
        return {}

    # Separate by type
    ideal_results = filter_results(results, backend_type='ideal')
    noisy_results = filter_results(results, backend_type='noisy_simulator')
    hardware_results = filter_results(results, backend_type='hardware')

    analysis = {
        'timestamp': datetime.now().isoformat(),
        'artifacts_dir': str(artifacts_dir),
        'total_results': len(results),
        'ideal_summary': compute_visibility_summary(ideal_results),
        'noisy_summary': compute_visibility_summary(noisy_results),
        'hardware_summary': compute_visibility_summary(hardware_results),
        'results_table': generate_summary_table(results),
    }

    # Compute visibility degradation
    if ideal_results and noisy_results:
        ideal_V = np.mean([r['visibility'] for r in ideal_results])
        noisy_V = np.mean([r['visibility'] for r in noisy_results])
        analysis['visibility_degradation'] = {
            'ideal_to_noisy': ideal_V - noisy_V,
            'relative_loss': (ideal_V - noisy_V) / ideal_V if ideal_V > 0 else 0,
        }

    # Save
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Saved analysis: {output_path}")
    return analysis


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze branch-transfer experiment results'
    )
    parser.add_argument(
        '--artifacts-dir', type=str, default='artifacts/branch_transfer',
        help='Directory containing result artifacts'
    )
    parser.add_argument(
        '--figures-dir', type=str, default='artifacts/branch_transfer/figures',
        help='Directory to save figures'
    )
    parser.add_argument(
        '--plot-all', action='store_true',
        help='Generate all plots'
    )
    parser.add_argument(
        '--summary', action='store_true',
        help='Generate summary analysis'
    )
    return parser.parse_args()


def main():
    """Main analysis pipeline."""
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Branch-Transfer Experiment: Analysis")
    print("=" * 70)
    print(f"  artifacts_dir = {artifacts_dir}")
    print(f"  figures_dir = {figures_dir}")
    print()

    # Load results
    results = load_results(artifacts_dir)
    print(f"Loaded {len(results)} result files")

    if not results:
        print("No results found. Run run_sim.py first.")
        return

    # Filter by type
    ideal_results = filter_results(results, backend_type='ideal')
    noisy_results = filter_results(results, backend_type='noisy_simulator')
    hardware_results = filter_results(results, backend_type='hardware')

    print(f"  Ideal: {len(ideal_results)}")
    print(f"  Noisy simulator: {len(noisy_results)}")
    print(f"  Hardware: {len(hardware_results)}")
    print()

    # Generate plots
    if args.plot_all or not args.summary:
        # PR distribution comparison
        if ideal_results or noisy_results:
            combined = ideal_results + noisy_results
            plot_pr_distribution(
                combined[:4],  # Limit to avoid crowding
                figures_dir / 'pr_distribution.png',
                title='PR Distribution: Ideal vs Noisy Simulator'
            )

        # Visibility comparison
        if ideal_results or noisy_results or hardware_results:
            all_results = ideal_results[:1] + noisy_results[:3] + hardware_results[:2]
            if all_results:
                plot_visibility_comparison(
                    all_results,
                    figures_dir / 'visibility_comparison.png',
                    title='Visibility Across Configurations'
                )

        # Opt level analysis (noisy)
        if len(noisy_results) > 1:
            plot_visibility_vs_opt_level(
                noisy_results,
                figures_dir / 'visibility_vs_opt_level.png',
                title='Visibility vs Transpiler Optimization (Noisy Sim)'
            )

        # Load and plot collapse sweep if available (visibility-based)
        collapse_files = list(artifacts_dir.glob('collapse_sweep_*.json'))
        for cf in collapse_files:
            with open(cf, 'r') as f:
                sweep_data = json.load(f)
            model = sweep_data.get('collapse_model', 'unknown')
            noise = 'noisy' if sweep_data.get('add_hardware_noise') else 'ideal'
            plot_collapse_forecast(
                sweep_data,
                figures_dir / f'collapse_forecast_{model}_{noise}.png',
                title=f'V(γ) Forecast: {model} model ({noise} baseline)'
            )

        # Load and plot coherence sweep if available (coherence witness-based)
        coherence_files = list(artifacts_dir.glob('coherence_sweep_*.json'))
        for cf in coherence_files:
            with open(cf, 'r') as f:
                sweep_data = json.load(f)
            model = sweep_data.get('collapse_model', 'unknown')
            noise = 'noisy' if sweep_data.get('add_hardware_noise') else 'ideal'
            basis = sweep_data.get('measurement_basis', 'X')
            plot_coherence_forecast(
                sweep_data,
                figures_dir / f'coherence_forecast_{model}_{noise}_{basis}.png',
                title=f'$W_{{{basis}}}(\gamma)$ Forecast: {model} ({noise} baseline)'
            )

        # Plot coherence comparison if coherence results available
        coherence_results = [r for r in results if 'W_X' in r]
        if coherence_results:
            plot_coherence_comparison(
                coherence_results[:6],  # Limit to avoid crowding
                figures_dir / 'coherence_comparison.png',
                title='Coherence Witness Across Configurations'
            )

    # Generate summary
    if args.summary or args.plot_all:
        analysis = generate_analysis_report(
            artifacts_dir,
            artifacts_dir / 'comprehensive_analysis.json'
        )

        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)

        if analysis.get('ideal_summary'):
            s = analysis['ideal_summary']
            print(f"\nIdeal Simulator (n={s['count']}):")
            print(f"  V = {s['mean']:.4f} ± {s['std']:.4f}")

        if analysis.get('noisy_summary'):
            s = analysis['noisy_summary']
            print(f"\nNoisy Simulator (n={s['count']}):")
            print(f"  V = {s['mean']:.4f} ± {s['std']:.4f}")

        if analysis.get('hardware_summary') and analysis['hardware_summary'].get('count'):
            s = analysis['hardware_summary']
            print(f"\nHardware (n={s['count']}):")
            print(f"  V = {s['mean']:.4f} ± {s['std']:.4f}")

        if analysis.get('visibility_degradation'):
            d = analysis['visibility_degradation']
            print(f"\nVisibility degradation (ideal→noisy): {d['ideal_to_noisy']:.4f} ({d['relative_loss']*100:.1f}%)")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
