"""
Output path routing utilities for quantum kernel estimation experiments.

Provides consistent directory resolution for RAW vs PSD experiment outputs.
"""

from pathlib import Path
from typing import Optional, Dict


def resolve_output_paths(output_tag: Optional[str] = None) -> Dict[str, Path]:
    """
    Resolve output directories based on an optional output tag.

    When output_tag is None, returns default paths (current behavior).
    When output_tag is provided (e.g., 'raw', 'psd'), appends suffix to directories.

    Parameters
    ----------
    output_tag : str, optional
        Tag to append to output directories. Common values: 'raw', 'psd'.
        If None, uses default paths without suffix.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'results_dir': Path to results directory
        - 'plots_dir': Path to plots directory
        - 'docs_dir': Path to docs directory
        - 'tag': The output tag (or None)

    Examples
    --------
    >>> paths = resolve_output_paths()
    >>> paths['results_dir']
    PosixPath('results')

    >>> paths = resolve_output_paths('psd')
    >>> paths['results_dir']
    PosixPath('results_psd')
    """
    if output_tag:
        results_dir = Path(f'results_{output_tag}')
        plots_dir = Path(f'plots_{output_tag}')
    else:
        results_dir = Path('results')
        plots_dir = Path('plots')

    return {
        'results_dir': results_dir,
        'plots_dir': plots_dir,
        'docs_dir': Path('docs'),
        'tag': output_tag
    }


def get_analysis_report_path(output_tag: Optional[str] = None) -> Path:
    """
    Get the path for the analysis report markdown file.

    Parameters
    ----------
    output_tag : str, optional
        Tag to append to filename.

    Returns
    -------
    Path
        Path to the analysis report file.
    """
    if output_tag:
        return Path('docs') / f'analysis_report_{output_tag}.md'
    return Path('docs') / 'analysis_report.md'


def get_analysis_json_path(output_tag: Optional[str] = None) -> Path:
    """
    Get the path for the comprehensive analysis JSON file.

    Parameters
    ----------
    output_tag : str, optional
        Tag for output directory.

    Returns
    -------
    Path
        Path to the comprehensive analysis JSON file.
    """
    paths = resolve_output_paths(output_tag)
    return paths['results_dir'] / 'comprehensive_analysis.json'
