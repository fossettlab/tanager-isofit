#!/usr/bin/env python
"""
Example: Validate Tanager surface reflectance against EMIT.

This script demonstrates:
1. Finding coincident EMIT data for a Tanager scene
2. Downloading EMIT L2A reflectance
3. Comparing the two datasets
4. Generating a validation report

Usage:
    python validate_vs_emit.py <tanager_refl> <tanager_wavelengths> --emit <emit_path>

    # Or search for coincident EMIT:
    python validate_vs_emit.py <tanager_h5> --search --output-dir <dir>
"""

import argparse
from pathlib import Path


def search_and_validate(tanager_h5: str, output_dir: str, time_window: float = 24.0):
    """Search for EMIT data and perform validation."""
    from tanager_isofit.convert import get_scene_bounds, get_acquisition_time
    from tanager_isofit.validate import find_coincident_emit, download_emit_l2a

    print("Getting Tanager scene info...")
    bounds = get_scene_bounds(tanager_h5)
    acq_time = get_acquisition_time(tanager_h5)

    print(f"  Acquisition time: {acq_time}")
    print(f"  Bounds: {bounds}")

    print(f"\nSearching for EMIT data (±{time_window} hours)...")
    granules = find_coincident_emit(bounds, acq_time, time_window)

    if not granules:
        print("No coincident EMIT data found.")
        return None

    print(f"Found {len(granules)} EMIT granules:")
    for i, g in enumerate(granules):
        print(f"  [{i + 1}] {g['granule_id']}")

    # Download first granule
    print("\nDownloading first EMIT granule...")
    emit_path = download_emit_l2a(granules[0], output_dir)

    if emit_path:
        print(f"Downloaded: {emit_path}")
        return emit_path
    else:
        print("Download failed")
        return None


def run_validation(
    tanager_refl: str,
    wavelength_file: str,
    emit_path: str,
    output_dir: str,
):
    """Run validation comparison."""
    from tanager_isofit.validate import run_full_validation

    print("\nRunning validation...")
    results = run_full_validation(
        tanager_refl,
        wavelength_file,
        emit_path,
        output_dir,
    )

    # Print results
    print("\n" + "=" * 50)
    print("Validation Results")
    print("=" * 50)

    overall = results.get("comparison", {}).get("overall", {})
    print(f"  RMSE: {overall.get('rmse', 'N/A'):.4f}")
    print(f"  MAE: {overall.get('mae', 'N/A'):.4f}")
    print(f"  Correlation: {overall.get('correlation', 'N/A'):.4f}")
    print(f"  Bias: {overall.get('bias', 'N/A'):.4f}")

    spectral = results.get("comparison", {}).get("spectral", {})
    if spectral:
        print(f"  Mean Spectral Angle: {spectral.get('mean_angle', 'N/A'):.2f}°")

    water = results.get("water_validation", {})
    if water:
        print(f"  Water pixels valid: {water.get('valid', False)}")

    print(f"\n  Report: {results.get('report_path', 'N/A')}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate Tanager reflectance against EMIT"
    )
    parser.add_argument("input", help="Tanager reflectance file or HDF5 for search")
    parser.add_argument("--wavelengths", "-w", help="Wavelength file")
    parser.add_argument("--emit", "-e", help="EMIT NetCDF file")
    parser.add_argument(
        "--search", action="store_true", help="Search for coincident EMIT data"
    )
    parser.add_argument(
        "--time-window",
        "-t",
        type=float,
        default=24.0,
        help="Search time window in hours",
    )
    parser.add_argument(
        "--output-dir", "-o", default="./validation", help="Output directory"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.search:
        # Search for and download EMIT data
        emit_path = search_and_validate(args.input, str(output_dir), args.time_window)
        if not emit_path:
            print("No EMIT data available for validation")
            return
    else:
        if not args.emit:
            print("Error: Provide --emit path or use --search")
            return
        emit_path = args.emit

    if args.wavelengths:
        # Run validation
        run_validation(
            args.input,
            args.wavelengths,
            str(emit_path),
            str(output_dir),
        )


if __name__ == "__main__":
    main()
