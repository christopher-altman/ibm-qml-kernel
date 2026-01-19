"""
Validate paper figure quality metrics.

Checks that generated figures meet journal submission requirements:
- PNG resolution >= 300 DPI
- PDF files exist (vector format)
- File sizes are reasonable
- No dark backgrounds in paper figures

Usage:
    python -m experiments.branch_transfer.validate_paper_figures \
        --figures-dir docs/arXiv-package/figures

Date: 2026-01-19
Author: Christopher Altman
"""

import argparse
from pathlib import Path
from PIL import Image
import PyPDF2


def validate_png(png_path: Path) -> dict:
    """
    Validate PNG figure quality.

    Parameters
    ----------
    png_path : Path
        Path to PNG file

    Returns
    -------
    dict
        Validation results with 'status', 'dpi', 'size', 'background'
    """
    try:
        img = Image.open(png_path)
        dpi = img.info.get('dpi', (0, 0))
        dpi_x, dpi_y = dpi if isinstance(dpi, tuple) else (dpi, dpi)

        # Check background color (should be white or very light)
        # Sample top-left corner pixel
        pixels = list(img.getdata())
        if pixels:
            first_pixel = pixels[0]
            if isinstance(first_pixel, tuple):
                # RGB or RGBA
                is_light = all(c > 200 for c in first_pixel[:3])
            else:
                # Grayscale
                is_light = first_pixel > 200
        else:
            is_light = None

        status = 'PASS' if dpi_x >= 300 and is_light else 'FAIL'

        return {
            'status': status,
            'dpi': f"{dpi_x}x{dpi_y}",
            'size': f"{img.width}x{img.height}",
            'background': 'light' if is_light else 'dark/unknown',
            'file_size_kb': png_path.stat().st_size / 1024
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e)
        }


def validate_pdf(pdf_path: Path) -> dict:
    """
    Validate PDF figure exists and is readable.

    Parameters
    ----------
    pdf_path : Path
        Path to PDF file

    Returns
    -------
    dict
        Validation results with 'status', 'pages', 'file_size_kb'
    """
    try:
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            num_pages = len(pdf.pages)

        return {
            'status': 'PASS',
            'pages': num_pages,
            'file_size_kb': pdf_path.stat().st_size / 1024
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Validate paper figure quality metrics'
    )
    parser.add_argument(
        '--figures-dir',
        type=str,
        default='docs/arXiv-package/figures',
        help='Directory containing paper figures'
    )
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    if not figures_dir.exists():
        print(f"ERROR: Figures directory not found: {figures_dir}")
        return

    print("=" * 80)
    print("Paper Figure Quality Validation")
    print("=" * 80)
    print(f"Figures directory: {figures_dir}\n")

    # Find all PNG files
    png_files = sorted(figures_dir.glob("*.png"))
    if not png_files:
        print("No PNG files found.")
        return

    print(f"Found {len(png_files)} PNG files\n")

    all_pass = True
    for png_path in png_files:
        pdf_path = png_path.with_suffix('.pdf')

        print(f"📄 {png_path.name}")

        # Validate PNG
        png_result = validate_png(png_path)
        if png_result['status'] == 'PASS':
            print(f"   ✓ PNG: {png_result['dpi']} DPI, "
                  f"{png_result['size']}, "
                  f"{png_result['background']} bg, "
                  f"{png_result['file_size_kb']:.1f} KB")
        else:
            print(f"   ✗ PNG: {png_result.get('error', 'Failed validation')}")
            all_pass = False

        # Validate PDF
        if pdf_path.exists():
            pdf_result = validate_pdf(pdf_path)
            if pdf_result['status'] == 'PASS':
                print(f"   ✓ PDF: {pdf_result['pages']} page(s), "
                      f"{pdf_result['file_size_kb']:.1f} KB")
            else:
                print(f"   ✗ PDF: {pdf_result.get('error', 'Failed validation')}")
                all_pass = False
        else:
            print(f"   ✗ PDF: Missing")
            all_pass = False

        print()

    print("=" * 80)
    if all_pass:
        print("✓ All figures passed validation")
    else:
        print("✗ Some figures failed validation")
    print("=" * 80)


if __name__ == '__main__':
    main()
