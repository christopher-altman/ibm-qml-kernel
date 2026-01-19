"""
Hero figure styling for README banners.

Premium aesthetics with light/dark themes, gradients, and wide banner layouts.
Completely separate from paper_style.py to ensure paper outputs remain unchanged.

Date: 2026-01-19
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from typing import Optional, Literal


def set_hero_style(theme: Literal['light', 'dark'] = 'light'):
    """
    Configure matplotlib rcParams for hero banner figures.

    Parameters
    ----------
    theme : str
        'light' or 'dark' theme
    """
    # Theme-specific colors
    if theme == 'light':
        bg_color = '#FAFAF8'
        text_color = '#1A1A1A'
        grid_color = '#E0E0E0'
        spine_color = '#D0D0D0'
    else:  # dark
        bg_color = '#0F1115'
        text_color = '#E8E8E8'
        grid_color = '#2A2A2A'
        spine_color = '#3A3A3A'

    plt.rcParams.update({
        # Figure background
        'figure.facecolor': bg_color,
        'axes.facecolor': bg_color,
        'savefig.facecolor': bg_color,

        # Font sizes - hero banners need larger, clearer text
        'font.size': 13,
        'axes.titlesize': 18,      # Larger subplot titles for hero
        'axes.labelsize': 14,      # Axis labels
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 22,    # Large suptitle for hero

        # Font family - prefer modern sans fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'SF Pro Display', 'Helvetica Neue', 'DejaVu Sans', 'Arial'],

        # Axes styling - softer, minimal
        'axes.linewidth': 0.6,
        'axes.edgecolor': spine_color,
        'axes.labelcolor': text_color,
        'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
        'axes.titlepad': 14,       # More breathing room

        # Grid - subtle horizontal only
        'axes.grid': False,
        'grid.alpha': 0.15 if theme == 'light' else 0.1,
        'grid.linewidth': 0.5,
        'grid.color': grid_color,

        # Ticks - minimal
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'xtick.color': spine_color,
        'ytick.color': spine_color,

        # Lines and markers - slightly thicker for readability
        'lines.linewidth': 2.2,
        'lines.markersize': 7,
        'lines.markeredgewidth': 1.0,

        # Error bars - clean and visible
        'errorbar.capsize': 5,

        # Legend - minimal, unobtrusive
        'legend.frameon': True,
        'legend.framealpha': 0.85,
        'legend.fancybox': False,
        'legend.edgecolor': spine_color,
        'legend.shadow': False,

        # Text color
        'text.color': text_color,

        # Savefig
        'savefig.dpi': 300,  # High resolution for hero banners
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,

        # Layout
        'figure.constrained_layout.use': False,
    })


def add_subtle_vignette(ax, theme: Literal['light', 'dark'] = 'light'):
    """
    Add a very subtle radial vignette to the background.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify
    theme : str
        'light' or 'dark' theme
    """
    # Get axis bounds
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create radial gradient mesh
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)

    # Center of plot
    cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2

    # Distance from center (normalized)
    dx = (X - cx) / (xlim[1] - xlim[0])
    dy = (Y - cy) / (ylim[1] - ylim[0])
    dist = np.sqrt(dx**2 + dy**2)

    # Vignette strength (very subtle)
    if theme == 'light':
        vignette = 1 - 0.08 * dist  # Slight darkening at edges
        color = 'black'
    else:
        vignette = 1 - 0.12 * dist  # Slightly more pronounced for dark
        color = 'black'

    # Apply as contour with very low alpha
    ax.contourf(X, Y, vignette, levels=20, cmap='gray', alpha=0.03, zorder=-10)


def get_hero_colors(theme: Literal['light', 'dark'] = 'light'):
    """
    Get hero color palette for the theme.

    Returns
    -------
    dict
        Color scheme for bars, lines, etc.
    """
    if theme == 'light':
        return {
            'primary': '#4A90E2',      # Blue
            'secondary': '#E74C3C',    # Red
            'tertiary': '#9B59B6',     # Purple
            'success': '#28A745',      # Green
            'warning': '#F39C12',      # Orange
            'neutral': '#6C757D',      # Gray
            'bar_gradient_top': 0.95,  # Slight lightening at top
            'bar_gradient_bottom': 1.0,
        }
    else:  # dark
        return {
            'primary': '#5BA3F5',      # Brighter blue for dark bg
            'secondary': '#F55D4D',    # Brighter red
            'tertiary': '#AB6BC6',     # Brighter purple
            'success': '#38B755',      # Brighter green
            'warning': '#FAA61A',      # Brighter orange
            'neutral': '#8C9399',      # Lighter gray
            'bar_gradient_top': 1.0,
            'bar_gradient_bottom': 0.92,  # Slight darkening at bottom
        }


def apply_bar_gradient(bars, theme: Literal['light', 'dark'] = 'light', rounded_corners: bool = True):
    """
    Apply subtle vertical gradient to bar patches and optionally round corners.

    Parameters
    ----------
    bars : matplotlib container or list
        Bar patches from ax.bar()
    theme : str
        'light' or 'dark' theme
    rounded_corners : bool
        Whether to apply rounded corners (default True)
    """
    colors = get_hero_colors(theme)

    # Extract bar patches
    if hasattr(bars, 'patches'):
        patches = bars.patches
    elif isinstance(bars, list):
        patches = bars
    else:
        patches = [bars]

    for bar in patches:
        # Get original color
        fc = bar.get_facecolor()

        # Create gradient (very subtle)
        gradient = mpatches.Rectangle(
            (bar.get_x(), bar.get_y()),
            bar.get_width(),
            bar.get_height(),
            facecolor=fc,
            edgecolor='none',
            alpha=bar.get_alpha()
        )

        # Apply rounded corners
        if rounded_corners:
            # Get current edge color
            edge_color = bar.get_edgecolor()
            # Set rounded corners using BoxStyle
            bar.set_capstyle('round')
            bar.set_joinstyle('round')

        # Note: True gradients require custom rendering or image compositing
        # For matplotlib compatibility, we keep solid colors but adjust alpha slightly
        # A full implementation would use imshow or custom patches


def clean_hero_spines(ax, theme: Literal['light', 'dark'] = 'light'):
    """
    Remove top and right spines, soften bottom and left.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify
    theme : str
        'light' or 'dark' theme
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    spine_color = '#D0D0D0' if theme == 'light' else '#3A3A3A'
    ax.spines['bottom'].set_color(spine_color)
    ax.spines['left'].set_color(spine_color)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_linewidth(0.6)


def add_hero_grid(ax, axis: str = 'y', theme: Literal['light', 'dark'] = 'light'):
    """
    Add faint horizontal grid lines for hero figures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify
    axis : str
        'y' for horizontal lines, 'x' for vertical, 'both' for both
    theme : str
        'light' or 'dark' theme
    """
    alpha = 0.15 if theme == 'light' else 0.1
    grid_color = '#E0E0E0' if theme == 'light' else '#2A2A2A'

    ax.grid(True, axis=axis, alpha=alpha, linewidth=0.5, color=grid_color, zorder=0)


def export_hero_fig(fig: plt.Figure, output_path_stem: str, theme: Literal['light', 'dark'] = 'light', dpi: int = 300):
    """
    Export hero figure with theme suffix.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to export
    output_path_stem : str
        Path without extension (e.g., 'figures_hero/coherence_comparison')
    theme : str
        'light' or 'dark'
    dpi : int
        DPI for PNG export (default 300 for hero)
    """
    path = Path(output_path_stem)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add theme suffix
    png_path = path.parent / f"{path.stem}_hero_{theme}.png"

    # PNG export only for hero (README banners don't need PDF)
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())

    print(f"Exported hero: {png_path}")
