"""
Paper-quality matplotlib style configuration and export utilities.

Provides consistent journal-friendly styling for all manuscript figures:
- Light theme (white background, no gradients/shadows)
- Consistent typography (titles ~16pt, axes ~13-14pt, ticks ~11-12pt)
- Thin spines, faint grid, readable line widths
- Dual export: PNG @600 DPI + PDF vector

Design Rationale:
- White backgrounds ensure print compatibility and accessibility
- Typography hierarchy (suptitle > axes title > label > tick) guides reader attention
- Line widths (1.8pt) balance visibility in PDF with clean appearance
- Marker size (6pt) avoids visual clutter while maintaining clarity
- DPI (600) exceeds journal requirements (typically 300+ for raster)
- Vector PDF export preferred for arXiv/journal submission

Date: 2026-01-18
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Paper style constants - centralized for maintainability
PAPER_FONT_SIZES = {
    'base': 11,           # Default font size
    'tick': 10,           # Tick labels (smallest)
    'legend': 10,         # Legend text
    'label': 12,          # Axis labels
    'axes_title': 12,     # Subplot/axes titles (subordinate)
    'suptitle': 16,       # Figure suptitle (dominant)
}

PAPER_LINE_STYLES = {
    'linewidth': 1.8,           # Main plot lines
    'markersize': 6,            # Scatter/line markers
    'markeredgewidth': 0.8,     # Marker outline thickness
    'axes_linewidth': 0.8,      # Spine thickness
    'grid_linewidth': 0.6,      # Grid line thickness
}

PAPER_COLORS = {
    'background': 'white',
    'axes_edge': '#333333',     # Dark gray for spines/ticks
    'text': '#000000',          # Pure black for all text
    'grid': '#888888',          # Medium gray for gridlines
    'legend_edge': '#CCCCCC',   # Light gray for legend border
}

PAPER_EXPORT = {
    'dpi': 600,                 # PNG resolution (exceeds journal minimum of 300)
    'pad_inches': 0.05,         # Minimal padding around figure
}


def set_paper_style():
    """
    Configure matplotlib rcParams for paper-quality figures.

    Call once at the start of the analysis script to set global style.
    Light theme only - no dark mode, no shadows, no gradients.

    Typography hierarchy:
    - Figure suptitle: 16pt semibold (dominant)
    - Axes titles: 12pt normal (subordinate)
    - Axis labels: 12pt
    - Tick labels: 10pt
    - Legend: 10pt

    Uses centralized constants from PAPER_FONT_SIZES, PAPER_LINE_STYLES,
    PAPER_COLORS, and PAPER_EXPORT for consistency across all figures.
    """
    plt.rcParams.update({
        # Figure background
        'figure.facecolor': PAPER_COLORS['background'],
        'axes.facecolor': PAPER_COLORS['background'],
        'savefig.facecolor': PAPER_COLORS['background'],

        # Font sizes - establish clear hierarchy
        'font.size': PAPER_FONT_SIZES['base'],
        'axes.titlesize': PAPER_FONT_SIZES['axes_title'],
        'axes.labelsize': PAPER_FONT_SIZES['label'],
        'xtick.labelsize': PAPER_FONT_SIZES['tick'],
        'ytick.labelsize': PAPER_FONT_SIZES['tick'],
        'legend.fontsize': PAPER_FONT_SIZES['legend'],
        'figure.titlesize': PAPER_FONT_SIZES['suptitle'],

        # Font family
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],

        # Axes styling
        'axes.linewidth': PAPER_LINE_STYLES['axes_linewidth'],
        'axes.edgecolor': PAPER_COLORS['axes_edge'],
        'axes.labelcolor': PAPER_COLORS['text'],
        'axes.titlecolor': PAPER_COLORS['text'],
        'axes.titleweight': 'normal',      # Axes titles: normal weight
        'axes.labelweight': 'normal',
        'axes.titlepad': 8,                # Space between title and plot

        # Grid
        'axes.grid': False,  # Turn off by default; enable selectively
        'grid.alpha': 0.2,
        'grid.linewidth': PAPER_LINE_STYLES['grid_linewidth'],
        'grid.color': PAPER_COLORS['grid'],

        # Ticks
        'xtick.major.width': PAPER_LINE_STYLES['axes_linewidth'],
        'ytick.major.width': PAPER_LINE_STYLES['axes_linewidth'],
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.color': PAPER_COLORS['axes_edge'],
        'ytick.color': PAPER_COLORS['axes_edge'],

        # Lines and markers
        'lines.linewidth': PAPER_LINE_STYLES['linewidth'],
        'lines.markersize': PAPER_LINE_STYLES['markersize'],
        'lines.markeredgewidth': PAPER_LINE_STYLES['markeredgewidth'],

        # Error bars
        'errorbar.capsize': 4,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': False,
        'legend.edgecolor': PAPER_COLORS['legend_edge'],
        'legend.shadow': False,

        # Savefig
        'savefig.dpi': PAPER_EXPORT['dpi'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': PAPER_EXPORT['pad_inches'],

        # Layout - disable constrained_layout globally; we'll manage manually
        'figure.constrained_layout.use': False,
    })


def export_fig(fig: plt.Figure, output_path_stem: str, dpi: int = None):
    """
    Export figure to both PNG and PDF with consistent quality settings.

    Generates two output files:
    - PNG (raster): High-resolution bitmap for presentations/web
    - PDF (vector): Scalable format for LaTeX/arXiv submission

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to export.
    output_path_stem : str
        Path without extension (e.g., 'figures/my_plot').
        Will generate 'figures/my_plot.png' and 'figures/my_plot.pdf'.
    dpi : int, optional
        DPI for PNG export. Defaults to PAPER_EXPORT['dpi'] (600).
        Minimum 300 recommended for journal quality.

    Notes
    -----
    - PDF is preferred for submission as it's resolution-independent
    - PNG is useful for quick previews and presentations
    - Both formats use 'tight' bbox to minimize whitespace
    """
    if dpi is None:
        dpi = PAPER_EXPORT['dpi']

    path = Path(output_path_stem)
    path.parent.mkdir(parents=True, exist_ok=True)

    # PNG export (raster)
    png_path = path.with_suffix('.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight',
                facecolor=PAPER_COLORS['background'])

    # PDF export (vector)
    pdf_path = path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight',
                facecolor=PAPER_COLORS['background'])

    print(f"Exported: {png_path} + {pdf_path}")


def clean_spine(ax, spines_to_remove: Optional[list] = None):
    """
    Remove top and right spines for a cleaner look.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify.
    spines_to_remove : list of str, optional
        Spine names to remove (default: ['top', 'right']).
    """
    if spines_to_remove is None:
        spines_to_remove = ['top', 'right']

    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)


def add_faint_grid(ax, axis: str = 'y', alpha: float = 0.2):
    """
    Add faint horizontal or vertical grid lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify.
    axis : str
        'y' for horizontal lines, 'x' for vertical, 'both' for both.
    alpha : float
        Grid transparency (default 0.2).
    """
    ax.grid(True, axis=axis, alpha=alpha, linewidth=0.6, color='#888888')
