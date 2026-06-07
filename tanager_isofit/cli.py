"""
Command-line interface for Tanager ISOFIT pipeline.

Provides commands for:
- inspect: Examine HDF5 file structure
- convert: Convert HDF5 to ENVI format
- process: Full pipeline (convert + ISOFIT)
- find-emit: Search for coincident EMIT data
- validate: Compare with EMIT reflectance
"""

import sys
from pathlib import Path
from typing import Optional

import click

from tanager_isofit import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Tanager ISOFIT Pipeline - Convert Planet Tanager data to surface reflectance."""
    pass


@main.command()
@click.argument("input_h5", type=click.Path(exists=True))
def inspect(input_h5: str):
    """
    Inspect HDF5 file structure.

    Prints all datasets, groups, and attributes in the file.
    """
    from tanager_isofit.convert import print_hdf5_structure, validate_hdf5_structure

    input_path = Path(input_h5)

    print(f"\nInspecting: {input_path}")

    # Validate structure
    is_valid, issues = validate_hdf5_structure(input_path)

    if is_valid:
        click.echo(click.style("\n✓ Valid Tanager HDF5 structure", fg="green"))
    else:
        click.echo(click.style("\n✗ Structure validation issues:", fg="yellow"))
        for issue in issues:
            click.echo(f"  - {issue}")

    # Print full structure
    print_hdf5_structure(input_path)


@main.command()
@click.argument("input_h5", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--subset",
    "-s",
    type=str,
    default=None,
    help="Subset as 'row_start,row_end,col_start,col_end'",
)
@click.option(
    "--fast-solar/--no-fast-solar",
    default=True,
    help="Use fast grid-interpolated solar geometry",
)
def convert(input_h5: str, output_dir: str, subset: Optional[str], fast_solar: bool):
    """
    Convert HDF5 to ENVI format (no ISOFIT).

    Creates radiance, location, and observation ENVI files.
    """
    from tanager_isofit.convert import convert_tanager_to_envi

    # Parse subset
    subset_tuple = None
    if subset:
        try:
            parts = [int(x) for x in subset.split(",")]
            if len(parts) != 4:
                raise ValueError("Subset must have 4 values")
            subset_tuple = tuple(parts)
        except ValueError as e:
            click.echo(f"Error parsing subset: {e}", err=True)
            sys.exit(1)

    try:
        result = convert_tanager_to_envi(
            input_h5,
            output_dir,
            subset=subset_tuple,
            use_fast_solar=fast_solar,
        )

        click.echo(click.style("\n✓ Conversion complete", fg="green"))
        click.echo(f"  Radiance: {result['radiance']}")
        click.echo(f"  Location: {result['loc']}")
        click.echo(f"  Observation: {result['obs']}")
        click.echo(f"  Wavelengths: {result['wavelength_file']}")

    except Exception as e:
        click.echo(click.style(f"\n✗ Conversion failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.argument("input_h5", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--n-cores", "-n", type=int, default=4, help="Number of CPU cores for ISOFIT"
)
@click.option(
    "--empirical-line/--no-empirical-line",
    default=True,
    help="Use empirical line correction",
)
@click.option(
    "--subset",
    "-s",
    type=str,
    default=None,
    help="Subset as 'row_start,row_end,col_start,col_end'",
)
@click.option(
    "--skip-isofit", is_flag=True, help="Only convert, skip ISOFIT processing"
)
@click.option(
    "--sensor", type=str, default="tanager", help="ISOFIT sensor configuration name"
)
@click.option(
    "--surface-path",
    type=click.Path(),
    default=None,
    help="Path to surface model file (.mat). Auto-generates if not provided.",
)
@click.option(
    "--emulator-base",
    type=click.Path(),
    default="auto",
    help="Path to sRTMnet emulator .h5 file. 'auto' uses default location, 'none' uses MODTRAN",
)
def process(
    input_h5: str,
    output_dir: str,
    n_cores: int,
    empirical_line: bool,
    subset: Optional[str],
    skip_isofit: bool,
    sensor: str,
    surface_path: Optional[str],
    emulator_base: str,
):
    """
    Run full pipeline: convert HDF5 and run ISOFIT.

    This is the main command for processing Tanager data to surface reflectance.
    """
    from tanager_isofit.isofit_runner import run_isofit_pipeline

    # Parse subset
    subset_tuple = None
    if subset:
        try:
            parts = [int(x) for x in subset.split(",")]
            if len(parts) != 4:
                raise ValueError("Subset must have 4 values")
            subset_tuple = tuple(parts)
        except ValueError as e:
            click.echo(f"Error parsing subset: {e}", err=True)
            sys.exit(1)

    # Handle emulator_base: "none" means use MODTRAN (None value)
    emu_base = None if emulator_base.lower() == "none" else emulator_base

    try:
        result = run_isofit_pipeline(
            input_h5,
            output_dir,
            n_cores=n_cores,
            empirical_line=empirical_line,
            subset=subset_tuple,
            skip_isofit=skip_isofit,
            sensor=sensor,
            surface_path=surface_path,
            emulator_base=emu_base,
        )

        click.echo(click.style("\n✓ Pipeline complete", fg="green"))
        click.echo(f"  Output directory: {result['output_dir']}")

        if result.get("isofit_outputs"):
            click.echo("  ISOFIT outputs:")
            for key, path in result["isofit_outputs"].items():
                click.echo(f"    {key}: {path}")
        elif result.get("isofit_error"):
            click.echo(
                click.style(f"  ISOFIT error: {result['isofit_error']}", fg="yellow")
            )

    except Exception as e:
        click.echo(click.style(f"\n✗ Pipeline failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command("find-emit")
@click.argument("input_h5", type=click.Path(exists=True))
@click.option(
    "--time-window",
    "-t",
    type=float,
    default=24.0,
    help="Search window in hours (+/- from acquisition)",
)
def find_emit(input_h5: str, time_window: float):
    """
    Search for coincident EMIT data.

    Queries NASA CMR for EMIT L2A scenes overlapping the Tanager footprint.
    """
    from tanager_isofit.convert import get_scene_bounds, get_acquisition_time
    from tanager_isofit.validate import find_coincident_emit

    input_path = Path(input_h5)

    try:
        # Get scene info
        bounds = get_scene_bounds(input_path)
        acq_time = get_acquisition_time(input_path)

        click.echo("\nTanager scene info:")
        click.echo(f"  Time: {acq_time}")
        click.echo(f"  Bounds: {bounds}")

        # Search for EMIT
        granules = find_coincident_emit(bounds, acq_time, time_window)

        if not granules:
            click.echo(click.style("\nNo coincident EMIT data found", fg="yellow"))
        else:
            click.echo(
                click.style(f"\n✓ Found {len(granules)} EMIT granules:", fg="green")
            )
            for i, g in enumerate(granules):
                click.echo(f"\n  [{i + 1}] {g['granule_id']}")
                click.echo(f"      Time: {g['time_start']} - {g['time_end']}")

    except Exception as e:
        click.echo(click.style(f"\n✗ Search failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.argument("tanager_refl", type=click.Path(exists=True))
@click.argument("emit_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="validation_report.html",
    help="Output report path",
)
@click.option(
    "--wavelength-file",
    "-w",
    type=click.Path(exists=True),
    help="Tanager wavelength file",
)
def validate(
    tanager_refl: str, emit_path: str, output: str, wavelength_file: Optional[str]
):
    """
    Validate Tanager reflectance against EMIT L2A.

    Computes comparison metrics and generates an HTML report.
    """
    from tanager_isofit.validate import run_full_validation

    # Find wavelength file if not provided
    if wavelength_file is None:
        tanager_dir = Path(tanager_refl).parent
        wavelength_candidates = [
            tanager_dir / "wavelengths.txt",
            tanager_dir / "wavelength.txt",
        ]
        for candidate in wavelength_candidates:
            if candidate.exists():
                wavelength_file = str(candidate)
                break

        if wavelength_file is None:
            click.echo(
                "Error: Could not find wavelength file. Use --wavelength-file option.",
                err=True,
            )
            sys.exit(1)

    output_dir = Path(output).parent or Path(".")

    try:
        result = run_full_validation(
            tanager_refl,
            wavelength_file,
            emit_path,
            output_dir,
        )

        # Print summary
        summary = result.get("summary", {})
        comparison = result.get("comparison", {}).get("overall", {})

        click.echo("\n" + "=" * 50)
        click.echo("Validation Summary")
        click.echo("=" * 50)

        rmse = comparison.get("rmse", float("nan"))
        corr = comparison.get("correlation", float("nan"))

        rmse_status = "✓" if summary.get("rmse_pass") else "✗"
        corr_status = "✓" if summary.get("correlation_pass") else "✗"
        water_status = "✓" if summary.get("water_valid") else "✗"

        click.echo(f"  {rmse_status} RMSE: {rmse:.4f}")
        click.echo(f"  {corr_status} Correlation: {corr:.4f}")
        click.echo(
            f"  {water_status} Water validation: {'pass' if summary.get('water_valid') else 'fail'}"
        )
        click.echo(f"\n  Report: {result.get('report_path')}")

    except Exception as e:
        click.echo(click.style(f"\n✗ Validation failed: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.argument("envi_file", type=click.Path(exists=True))
def check(envi_file: str):
    """
    Check ENVI file validity and print statistics.

    Useful for verifying conversion output before running ISOFIT.
    """
    from tanager_isofit.utils import read_envi_file, read_envi_header
    import numpy as np

    envi_path = Path(envi_file)

    try:
        # Read header
        header_path = envi_path.with_suffix(".hdr")
        if not header_path.exists():
            header_path = envi_path.parent / (envi_path.name + ".hdr")

        header = read_envi_header(header_path)

        click.echo(f"\nENVI file: {envi_path}")
        click.echo("-" * 40)
        click.echo(f"  Lines: {header.get('lines')}")
        click.echo(f"  Samples: {header.get('samples')}")
        click.echo(f"  Bands: {header.get('bands')}")
        click.echo(f"  Data type: {header.get('data type')}")
        click.echo(f"  Interleave: {header.get('interleave')}")

        # Read data and compute stats
        data, _ = read_envi_file(envi_path)

        click.echo("\nStatistics:")
        click.echo(f"  Min: {np.nanmin(data):.6f}")
        click.echo(f"  Max: {np.nanmax(data):.6f}")
        click.echo(f"  Mean: {np.nanmean(data):.6f}")
        click.echo(f"  NaN fraction: {np.isnan(data).mean():.2%}")

        # Check for wavelengths
        if "wavelength" in header:
            wl = header["wavelength"]
            if isinstance(wl, list) and len(wl) > 0:
                click.echo(f"\nWavelength range: {min(wl):.1f} - {max(wl):.1f} nm")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command("check-deps")
def check_deps():
    """
    Check if all dependencies are installed.

    Verifies ISOFIT, sRTMnet emulator, and reflectance library are available.
    """
    from tanager_isofit.isofit_runner import (
        check_isofit_available,
        check_isofit_data_available,
    )
    from tanager_isofit.config import DEFAULT_SRTMNET_PATH, DEFAULT_REFLECTANCE_LIBRARY

    all_ok = True

    # Check ISOFIT
    if check_isofit_available():
        click.echo(click.style("✓ ISOFIT installed", fg="green"))
    else:
        click.echo(click.style("✗ ISOFIT not installed", fg="red"))
        click.echo("  Install with: pip install isofit")
        all_ok = False

    # Check sRTMnet emulator
    if DEFAULT_SRTMNET_PATH.exists():
        click.echo(click.style("✓ sRTMnet emulator found", fg="green"))
        click.echo(f"  Path: {DEFAULT_SRTMNET_PATH}")
    else:
        click.echo(click.style("✗ sRTMnet emulator not found", fg="yellow"))
        click.echo(f"  Expected: {DEFAULT_SRTMNET_PATH}")
        click.echo("  Download with: isofit download --sRTMnet")
        click.echo("  (ISOFIT can fall back to MODTRAN if available)")

    # Check reflectance library for surface model generation
    isofit_data = check_isofit_data_available()
    if isofit_data["reflectance_library"]:
        click.echo(click.style("✓ Reflectance library found", fg="green"))
        click.echo(f"  Path: {isofit_data['reflectance_library_path']}")
    else:
        click.echo(click.style("✗ Reflectance library not found", fg="yellow"))
        click.echo(f"  Expected: {DEFAULT_REFLECTANCE_LIBRARY}")
        click.echo("  Download with: isofit download data")
        click.echo("  (Required for auto-generating surface models)")

    if all_ok:
        click.echo(click.style("\nAll dependencies ready!", fg="green"))
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
