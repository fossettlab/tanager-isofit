#!/usr/bin/env python
"""
Example: Process a Tanager scene through the full ISOFIT pipeline.

This script demonstrates:
1. Converting Tanager HDF5 to ENVI format
2. Running ISOFIT atmospheric correction (if available)
3. Basic output validation

Usage:
    python process_scene.py <input_h5> <output_dir> [--subset 0,100,0,100]
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Process Tanager HDF5 through ISOFIT pipeline"
    )
    parser.add_argument("input_h5", help="Input Tanager HDF5 file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--subset",
        "-s",
        type=str,
        default=None,
        help="Subset as 'row_start,row_end,col_start,col_end'",
    )
    parser.add_argument(
        "--n-cores", "-n", type=int, default=4, help="Number of CPU cores"
    )
    parser.add_argument(
        "--skip-isofit", action="store_true", help="Only convert, skip ISOFIT"
    )

    args = parser.parse_args()

    # Import here to avoid import errors if dependencies not installed
    from tanager_isofit.isofit_runner import run_isofit_pipeline
    from tanager_isofit.utils import validate_envi_files

    # Parse subset
    subset = None
    if args.subset:
        parts = [int(x) for x in args.subset.split(",")]
        if len(parts) == 4:
            subset = tuple(parts)
        else:
            print("Warning: Invalid subset format, ignoring")

    # Run pipeline
    print(f"\n{'=' * 60}")
    print("Tanager ISOFIT Pipeline")
    print(f"{'=' * 60}")
    print(f"Input: {args.input_h5}")
    print(f"Output: {args.output_dir}")
    if subset:
        print(f"Subset: {subset}")
    print()

    results = run_isofit_pipeline(
        args.input_h5,
        args.output_dir,
        n_cores=args.n_cores,
        subset=subset,
        skip_isofit=args.skip_isofit,
    )

    # Validate ENVI files
    print("\nValidating ENVI files...")
    envi_files = results.get("envi_files", {})
    if envi_files:
        validation = validate_envi_files(
            envi_files.get("radiance", ""),
            envi_files.get("loc", ""),
            envi_files.get("obs", ""),
        )
        if validation["valid"]:
            print("  ✓ ENVI files valid")
        else:
            print("  ✗ ENVI file issues:")
            for issue in validation["issues"]:
                print(f"    - {issue}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"ENVI files: {envi_files.get('radiance', 'N/A')}")

    if results.get("isofit_outputs"):
        print(f"Reflectance: {results['isofit_outputs'].get('rfl', 'N/A')}")
    elif results.get("isofit_error"):
        print(f"ISOFIT error: {results['isofit_error']}")
    else:
        print("ISOFIT: Skipped")


if __name__ == "__main__":
    main()
