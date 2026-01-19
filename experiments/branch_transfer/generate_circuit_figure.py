"""
Generate branch-transfer circuit diagram for manuscript.

Produces publication-quality circuit diagram (PNG + PDF) using matplotlib drawer.

Date: 2026-01-18
Author: Christopher Altman
Contact: x@christopheraltman.com
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.branch_transfer.circuit import build_branch_transfer_circuit
from experiments.branch_transfer.paper_style import export_fig, set_paper_style
from experiments.branch_transfer.hero_style import export_hero_fig, set_hero_style


def main():
    parser = argparse.ArgumentParser(description='Generate circuit diagram figure')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='docs/arXiv-package/figures',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--export-hero',
        action='store_true',
        help='Export hero banner figures to figures_hero/ (in addition to paper figures)'
    )
    parser.add_argument(
        '--hero-theme',
        type=str,
        default='both',
        choices=['light', 'dark', 'both'],
        help='Hero theme: light, dark, or both (default: both)'
    )
    parser.add_argument(
        '--hero-dir',
        type=str,
        default=None,
        help='Directory for hero figures (default: <output-dir>/../figures_hero)'
    )
    parser.add_argument(
        '--mu',
        type=int,
        default=1,
        choices=[0, 1],
        help='Message bit value'
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup hero export if enabled
    hero_dir = None
    hero_themes = []
    if args.export_hero:
        if args.hero_dir:
            hero_dir = Path(args.hero_dir)
        else:
            hero_dir = output_dir.parent / 'figures_hero'
        hero_dir.mkdir(parents=True, exist_ok=True)

        if args.hero_theme == 'both':
            hero_themes = ['light', 'dark']
        else:
            hero_themes = [args.hero_theme]

    # Build circuit
    qc = build_branch_transfer_circuit(mu=args.mu, barrier=True, include_memory_erase=True)

    # Draw circuit with matplotlib (requires pylatexenc: pip install pylatexenc)
    try:
        # Paper figure (unchanged) - ensure clean state
        import matplotlib
        matplotlib.rcdefaults()  # Reset to matplotlib defaults first
        set_paper_style()
        fig = qc.draw(output='mpl', style='iqp', fold=-1, scale=0.8)
        output_stem = str(output_dir / 'branch_transfer_circuit')
        export_fig(fig.figure if hasattr(fig, 'figure') else fig, output_stem, dpi=600)
        print(f"Circuit diagram generated: {output_stem}.png + .pdf")

        # Hero figures if enabled
        if hero_dir and hero_themes:
            for theme in hero_themes:
                set_hero_style(theme=theme)

                # Force larger figure size for >=2400px width at 300 DPI
                # 2400px / 300 DPI = 8 inches minimum width
                matplotlib.rcParams['figure.figsize'] = (14, 6)  # Wide banner format
                matplotlib.rcParams['font.family'] = 'sans-serif'
                matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                matplotlib.rcParams['savefig.dpi'] = 300
                matplotlib.rcParams['text.antialiased'] = True
                matplotlib.rcParams['lines.antialiased'] = True

                fig = qc.draw(output='mpl', style='iqp', fold=-1, scale=1.0)

                # Apply theme background colors (Qiskit overrides rcParams)
                actual_fig = fig.figure if hasattr(fig, 'figure') else fig
                bg_color = '#FFFFFF' if theme == 'light' else '#1A1D23'  # Pure white or charcoal
                text_color = '#1A1A1A' if theme == 'light' else '#EDEDED'  # Off-white for dark
                line_color = '#2A2A2A' if theme == 'light' else '#D0D3D8'  # Specified colors

                actual_fig.set_facecolor(bg_color)
                for ax in actual_fig.get_axes():
                    # Add gradient + vignette for dark theme (Keynote aesthetic)
                    if theme == 'dark':
                        # Create radial gradient background
                        import numpy as np
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()

                        # Create mesh
                        x = np.linspace(xlim[0], xlim[1], 200)
                        y = np.linspace(ylim[0], ylim[1], 200)
                        X, Y = np.meshgrid(x, y)

                        # Center point
                        cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2

                        # Distance from center (normalized)
                        dx = (X - cx) / (xlim[1] - xlim[0])
                        dy = (Y - cy) / (ylim[1] - ylim[0])
                        dist = np.sqrt(dx**2 + dy**2)

                        # Gradient: center lighter (#232830), edges darker (#141619)
                        # Vignette intensity increases with distance
                        vignette = 1 - 0.25 * np.clip(dist, 0, 1)

                        # Create gradient image (RGB)
                        center_color = np.array([0x23, 0x28, 0x30]) / 255.0
                        edge_color = np.array([0x14, 0x16, 0x19]) / 255.0

                        gradient = np.zeros((200, 200, 3))
                        for i in range(3):
                            gradient[:, :, i] = edge_color[i] + (center_color[i] - edge_color[i]) * vignette

                        # Draw gradient as background (lowest z-order)
                        ax.imshow(gradient, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                                 aspect='auto', zorder=-100, interpolation='bilinear')
                    else:
                        ax.set_facecolor(bg_color)

                    # Update wire lines and other line elements
                    wire_linewidth = 2.2 if theme == 'light' else 2.4
                    for line in ax.lines:
                        line.set_color(line_color)
                        line.set_linewidth(wire_linewidth)

                    # Update patches (gates, boxes, etc.)
                    import matplotlib.patches as mpatches
                    for patch in ax.patches:
                        # Update rectangle backgrounds (qubit label backgrounds)
                        if isinstance(patch, mpatches.Rectangle):
                            current_face = patch.get_facecolor()
                            # If it's white/light colored, change to theme background
                            if current_face[0] > 0.9 and current_face[1] > 0.9 and current_face[2] > 0.9:
                                # Make gradient/vignette visible through transparent qubit label backgrounds
                                if theme == 'dark':
                                    patch.set_facecolor('none')
                                    patch.set_alpha(0)
                                else:
                                    patch.set_facecolor(bg_color)

                            # Premium gate colors for dark theme
                            if theme == 'dark':
                                # H gate: muted premium red
                                if 0.7 < current_face[0] < 1.0 and current_face[1] < 0.3 and current_face[2] < 0.3:
                                    patch.set_facecolor('#C75450')  # Muted red, not neon
                                # X gates: desaturated "Apple blue"
                                elif current_face[2] > 0.6 and current_face[0] < 0.5:
                                    patch.set_facecolor('#5B8DBE')  # Desaturated blue

                        # Keep gate face colors but update edges
                        patch.set_edgecolor(line_color)

                    # First pass: identify all Circle patches (control points)
                    control_circles = []
                    for patch in ax.patches:
                        if isinstance(patch, mpatches.Circle):
                            control_circles.append(patch)

                    # Update all text colors and sizes (typography hierarchy)
                    # Do this AFTER patches so text on top of colored circles gets correct color
                    for text in ax.texts:
                        content = text.get_text()

                        # Check if text is positioned on a control point circle
                        is_on_control_circle = False
                        if control_circles:
                            text_pos = text.get_position()
                            for circle in control_circles:
                                circle_center = circle.center
                                circle_radius = circle.radius
                                # Check if text position is within circle
                                dx = text_pos[0] - circle_center[0]
                                dy = text_pos[1] - circle_center[1]
                                dist = (dx**2 + dy**2)**0.5
                                if dist < circle_radius * 1.5:  # Within 1.5x radius for safety
                                    is_on_control_circle = True
                                    break

                        # Typography hierarchy (B requirement)
                        # Title: fontsize 20 (semibold) - circuit title if present
                        if 'Branch' in content or 'Circuit' in content:
                            text.set_fontsize(20)
                            text.set_fontweight('semibold')
                            text.set_color(text_color)

                        # Qubit labels: fontsize 17 (medium) - q0, q1, q2, q3, q4, c
                        elif content in ['$q_{0}$', '$q_{1}$', '$q_{2}$', '$q_{3}$', '$q_{4}$', '$c$'] or \
                             content.startswith('q') or content == 'c':
                            text.set_fontsize(17)
                            text.set_fontweight('medium')
                            text.set_color(text_color)

                        # Stage labels: fontsize 13 (regular) - prep, corr, rec, msg, copy, erase, swap
                        elif content in ['prep', 'corr', 'rec', 'msg', 'copy', 'erase', 'swap']:
                            text.set_fontsize(13)
                            text.set_fontweight('normal')
                            text.set_color(text_color)
                            text.set_alpha(0.75)  # Visually subordinate

                        # Gate labels (H, X, etc.) - always white on colored gates
                        elif content in ['H', 'X', 'Y', 'Z']:
                            text.set_fontsize(14)
                            text.set_fontweight('semibold')
                            text.set_color('white')  # White on both light/dark for visibility on colored gates

                        # Control points and symbols on colored backgrounds: use white
                        # Simple approach: any single non-alphanumeric character is likely a control symbol
                        elif content in ['⊕', '+', '⊗', '•', '○', '◦'] or \
                             (len(content) == 1 and not content.isalnum() and content not in ['$']) or \
                             is_on_control_circle:
                            text.set_color('white')
                            text.set_fontweight('bold')

                        # Default text
                        else:
                            text.set_color(text_color)

                    # Update spines
                    for spine in ax.spines.values():
                        spine.set_color(line_color)

                    # Update tick parameters
                    ax.tick_params(colors=text_color)

                hero_stem = str(hero_dir / 'branch_transfer_circuit')
                export_hero_fig(actual_fig, hero_stem, theme=theme)

    except Exception as e:
        print(f"Note: Circuit diagram generation requires pylatexenc")
        print(f"Install with: pip install pylatexenc")
        print(f"Error: {e}")
        print(f"\nCircuit already exists at: {output_dir}/branch_transfer_circuit.png")
        print("If you need to regenerate, install pylatexenc first.")


if __name__ == '__main__':
    main()
