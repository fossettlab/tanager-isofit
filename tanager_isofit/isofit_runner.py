"""
ISOFIT wrapper for running atmospheric correction on Tanager data.

This module provides a simplified interface to ISOFIT's apply_oe function
with sensible defaults for Tanager data.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import warnings

from tanager_isofit.config import (
    ISOFIT_SENSOR,
    ISOFIT_FALLBACK_SENSORS,
    DEFAULT_N_CORES,
    ISOFIT_WORKING_DIR_NAME,
    DEFAULT_SRTMNET_PATH,
    DEFAULT_REFLECTANCE_LIBRARY,
    SURFACE_MODEL_CONFIG,
)
from tanager_isofit.utils import ensure_directory, validate_envi_files
from tanager_isofit.convert import convert_tanager_to_envi


def check_isofit_available() -> bool:
    """
    Check if ISOFIT is available for import.

    Returns:
        True if ISOFIT can be imported
    """
    try:
        from isofit.utils import apply_oe  # noqa: F401  (import itself is the availability probe)

        return True
    except ImportError:
        return False


def get_available_sensors() -> list:
    """
    Get list of available ISOFIT sensor configurations.

    Returns:
        List of available sensor names
    """
    try:
        from isofit.configs import configs

        return list(configs.keys()) if hasattr(configs, "keys") else []
    except ImportError:
        return []


def check_isofit_data_available() -> Dict[str, Any]:
    """
    Check if ISOFIT data files are available.

    Returns:
        Dictionary with availability status for each data component:
        - reflectance_library: bool indicating if library exists
        - reflectance_library_path: Path to library if found, None otherwise
    """
    result = {
        "reflectance_library": False,
        "reflectance_library_path": None,
    }

    if DEFAULT_REFLECTANCE_LIBRARY.exists():
        result["reflectance_library"] = True
        result["reflectance_library_path"] = DEFAULT_REFLECTANCE_LIBRARY

    return result


def generate_surface_model(
    output_path: Union[str, Path],
    wavelength_file: Union[str, Path],
    reflectance_library: Optional[Path] = None,
) -> Path:
    """
    Generate ISOFIT surface model for Tanager wavelengths.

    Uses ISOFIT's surface_model CLI utility to create a surface model
    based on the sensor's wavelengths and a reflectance library.

    Args:
        output_path: Path where the surface model (.mat) will be saved
        wavelength_file: Path to the wavelength file for the sensor
        reflectance_library: Path to reflectance library directory.
            If None, uses DEFAULT_REFLECTANCE_LIBRARY.

    Returns:
        Path to the generated surface model file

    Raises:
        FileNotFoundError: If reflectance library not found
        RuntimeError: If surface model generation fails
    """
    import json
    import subprocess
    import tempfile

    output_path = Path(output_path)
    wavelength_file = Path(wavelength_file)

    # Check if output already exists (caching)
    if output_path.exists():
        print(f"Surface model already exists: {output_path}")
        return output_path

    # Resolve reflectance library path
    if reflectance_library is None:
        reflectance_library = DEFAULT_REFLECTANCE_LIBRARY
    else:
        reflectance_library = Path(reflectance_library).expanduser()

    # Validate reflectance library exists
    if not reflectance_library.exists():
        raise FileNotFoundError(
            f"Reflectance library not found: {reflectance_library}\n"
            "Run 'isofit download data' to install ISOFIT data files."
        )

    # Build config with absolute paths (ISOFIT doesn't expand ~)
    config = {
        "output_model_file": str(output_path.absolute()),
        "reflectance_library": str(reflectance_library.absolute()),
        **SURFACE_MODEL_CONFIG,
    }

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name

    try:
        print("Generating surface model...")
        print(f"  Wavelength file: {wavelength_file}")
        print(f"  Reflectance library: {reflectance_library}")
        print(f"  Output: {output_path}")

        # Run ISOFIT surface_model command
        result = subprocess.run(
            [
                "isofit",
                "surface_model",
                config_path,
                "--wavelength_path",
                str(wavelength_file.absolute()),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Surface model generation failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        if not output_path.exists():
            raise RuntimeError(
                f"Surface model was not created at {output_path}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        print("  Surface model generated successfully")
        return output_path

    finally:
        # Clean up temp config file
        Path(config_path).unlink(missing_ok=True)


def run_isofit(
    input_radiance: Union[str, Path],
    input_loc: Union[str, Path],
    input_obs: Union[str, Path],
    working_directory: Union[str, Path],
    sensor: str = ISOFIT_SENSOR,
    n_cores: int = DEFAULT_N_CORES,
    empirical_line: bool = True,
    wavelength_file: Optional[Union[str, Path]] = None,
    surface_path: Optional[Union[str, Path]] = None,
    emulator_base: Optional[Union[str, Path]] = "auto",
    **kwargs: Any,
) -> Dict[str, Path]:
    """
    Run ISOFIT atmospheric correction on ENVI files.

    Args:
        input_radiance: Path to radiance ENVI file
        input_loc: Path to location ENVI file
        input_obs: Path to observation ENVI file
        working_directory: Directory for ISOFIT working files
        sensor: Sensor name for ISOFIT config
        n_cores: Number of CPU cores to use
        empirical_line: Whether to use empirical line correction
        wavelength_file: Path to wavelength file (auto-detected if None)
        surface_path: Path to surface reflectance prior (optional)
        emulator_base: Path to sRTMnet emulator .h5 file. Use "auto" to use
            default location (~/.isofit/srtmnet/sRTMnet_v120.h5), None to use
            MODTRAN (requires separate installation).
        **kwargs: Additional arguments passed to apply_oe

    Returns:
        Dictionary with paths to output files
    """
    if not check_isofit_available():
        raise ImportError("ISOFIT is not installed. Install with: pip install isofit")

    from isofit.utils import apply_oe

    input_radiance = Path(input_radiance)
    input_loc = Path(input_loc)
    input_obs = Path(input_obs)
    working_directory = ensure_directory(working_directory)

    # Auto-detect wavelength file
    if wavelength_file is None:
        # Look in same directory as radiance file
        radiance_dir = input_radiance.parent
        wavelength_candidates = [
            radiance_dir / "wavelengths.txt",
            radiance_dir / "wavelength.txt",
            radiance_dir / "wl.txt",
        ]
        for candidate in wavelength_candidates:
            if candidate.exists():
                wavelength_file = candidate
                break

    # Validate input files
    validation = validate_envi_files(input_radiance, input_loc, input_obs)
    if not validation["valid"]:
        warnings.warn(f"Input file validation issues: {validation['issues']}")

    # Check if sensor config exists
    available_sensors = get_available_sensors()
    if sensor not in available_sensors:
        # Try fallback sensors
        for fallback in ISOFIT_FALLBACK_SENSORS:
            if fallback in available_sensors:
                warnings.warn(f"Sensor '{sensor}' not found, using '{fallback}' config")
                sensor = fallback
                break
        else:
            warnings.warn(
                f"Sensor '{sensor}' not found, available: {available_sensors}"
            )

    # Resolve emulator_base path
    resolved_emulator = None
    if emulator_base == "auto":
        # Use default sRTMnet location
        if DEFAULT_SRTMNET_PATH.exists():
            resolved_emulator = DEFAULT_SRTMNET_PATH
            print("Using sRTMnet emulator (auto-detected)")
        else:
            warnings.warn(
                f"sRTMnet not found at {DEFAULT_SRTMNET_PATH}. "
                "ISOFIT will attempt to use MODTRAN. "
                "Install sRTMnet with: isofit download --sRTMnet"
            )
    elif emulator_base is not None:
        resolved_emulator = Path(emulator_base)
        if not resolved_emulator.exists():
            raise FileNotFoundError(f"Emulator not found: {resolved_emulator}")
        print("Using sRTMnet emulator (user-specified)")

    print("Running ISOFIT atmospheric correction...")
    print(f"  Sensor: {sensor}")
    print(f"  Cores: {n_cores}")
    print(f"  Empirical line: {empirical_line}")
    print(f"  Working directory: {working_directory}")
    if resolved_emulator:
        print(f"  Emulator: {resolved_emulator}")
    else:
        print("  Emulator: MODTRAN (requires separate installation)")

    # Build apply_oe arguments
    oe_args = {
        "input_radiance": str(input_radiance),
        "input_loc": str(input_loc),
        "input_obs": str(input_obs),
        "working_directory": str(working_directory),
        "sensor": sensor,
        "n_cores": n_cores,
        "empirical_line": empirical_line,
    }

    if wavelength_file is not None:
        oe_args["wavelength_path"] = str(wavelength_file)

    # Auto-generate surface model if not provided
    if surface_path is None:
        if wavelength_file is None:
            raise ValueError(
                "wavelength_file is required for auto surface model generation. "
                "Either provide a wavelength file or specify --surface-path explicitly."
            )

        isofit_data = check_isofit_data_available()
        if not isofit_data["reflectance_library"]:
            raise FileNotFoundError(
                "Cannot auto-generate surface model: reflectance library not found at "
                f"{DEFAULT_REFLECTANCE_LIBRARY}\n"
                "Run 'isofit download data' or provide --surface-path explicitly."
            )

        surface_path = generate_surface_model(
            output_path=working_directory / "surface_model.mat",
            wavelength_file=wavelength_file,
        )

    oe_args["surface_path"] = str(surface_path)  # Always set now

    if resolved_emulator is not None:
        oe_args["emulator_base"] = str(resolved_emulator)

    # Add any additional kwargs
    oe_args.update(kwargs)

    # Run ISOFIT
    try:
        apply_oe.apply_oe(**oe_args)
    except Exception as e:
        print(f"ISOFIT error: {e}")
        raise

    # Find output files
    output_dir = working_directory / "output"
    outputs = {}

    # Look for standard ISOFIT output patterns
    for output_type in ["rfl", "atm", "unc"]:
        pattern = f"*{output_type}*"
        matches = list(output_dir.glob(pattern)) if output_dir.exists() else []
        if matches:
            outputs[output_type] = matches[0]

    # Also check working_directory directly
    if not outputs:
        for output_type in ["rfl", "atm", "unc"]:
            pattern = f"*{output_type}*"
            matches = list(working_directory.glob(pattern))
            if matches:
                outputs[output_type] = matches[0]

    print(f"  ISOFIT complete. Outputs: {list(outputs.keys())}")

    return outputs


def run_isofit_pipeline(
    input_h5: Union[str, Path],
    output_dir: Union[str, Path],
    n_cores: int = DEFAULT_N_CORES,
    empirical_line: bool = True,
    subset: Optional[tuple] = None,
    skip_isofit: bool = False,
    surface_path: Optional[Union[str, Path]] = None,
    emulator_base: Optional[Union[str, Path]] = "auto",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run the complete Tanager ISOFIT pipeline.

    Steps:
    1. Convert HDF5 to ENVI format
    2. Run ISOFIT atmospheric correction

    Args:
        input_h5: Path to input Tanager HDF5 file
        output_dir: Directory for all outputs
        n_cores: Number of CPU cores for ISOFIT
        empirical_line: Use empirical line correction
        subset: Optional (row_start, row_end, col_start, col_end) for subsetting
        skip_isofit: If True, only do conversion (skip ISOFIT)
        surface_path: Path to surface model file
        emulator_base: Path to sRTMnet emulator .h5 file. Use "auto" to use
            default location (~/.isofit/srtmnet/sRTMnet_v120.h5), None to use
            MODTRAN (requires separate installation).
        **kwargs: Additional arguments for convert or isofit

    Returns:
        Dictionary with all output paths and metadata
    """
    input_h5 = Path(input_h5)
    output_dir = ensure_directory(output_dir)

    results = {
        "input": str(input_h5),
        "output_dir": str(output_dir),
        "subset": subset,
    }

    # Step 1: Convert HDF5 to ENVI
    print(f"\n{'=' * 60}")
    print("Step 1: Converting HDF5 to ENVI format")
    print(f"{'=' * 60}")

    convert_kwargs = {
        k: v for k, v in kwargs.items() if k in ["use_fast_solar", "dem_path"]
    }

    envi_files = convert_tanager_to_envi(
        input_h5,
        output_dir,
        subset=subset,
        **convert_kwargs,
    )

    results["envi_files"] = {k: str(v) for k, v in envi_files.items()}

    if skip_isofit:
        print("\nSkipping ISOFIT (--skip-isofit flag)")
        results["isofit_outputs"] = None
        return results

    # Step 2: Run ISOFIT
    print(f"\n{'=' * 60}")
    print("Step 2: Running ISOFIT atmospheric correction")
    print(f"{'=' * 60}")

    if not check_isofit_available():
        warnings.warn("ISOFIT not available, skipping atmospheric correction")
        results["isofit_outputs"] = None
        return results

    isofit_working = output_dir / ISOFIT_WORKING_DIR_NAME

    isofit_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["use_fast_solar", "dem_path"]
    }

    try:
        isofit_outputs = run_isofit(
            input_radiance=envi_files["radiance"],
            input_loc=envi_files["loc"],
            input_obs=envi_files["obs"],
            working_directory=isofit_working,
            n_cores=n_cores,
            empirical_line=empirical_line,
            wavelength_file=envi_files["wavelength_file"],
            surface_path=surface_path,
            emulator_base=emulator_base,
            **isofit_kwargs,
        )
        results["isofit_outputs"] = {k: str(v) for k, v in isofit_outputs.items()}
    except Exception as e:
        warnings.warn(f"ISOFIT failed: {e}")
        results["isofit_outputs"] = None
        results["isofit_error"] = str(e)

    print(f"\n{'=' * 60}")
    print("Pipeline complete")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")

    return results


def validate_reflectance_output(
    reflectance_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Validate ISOFIT reflectance output.

    Checks:
    - File exists and is readable
    - Values are in physically reasonable range
    - No excessive NaN values

    Args:
        reflectance_path: Path to reflectance ENVI file

    Returns:
        Dictionary with validation results
    """
    import numpy as np
    from tanager_isofit.utils import read_envi_file
    from tanager_isofit.config import REFLECTANCE_MIN, REFLECTANCE_MAX

    reflectance_path = Path(reflectance_path)

    results = {
        "path": str(reflectance_path),
        "valid": True,
        "issues": [],
        "stats": {},
    }

    if not reflectance_path.exists():
        results["valid"] = False
        results["issues"].append(f"File not found: {reflectance_path}")
        return results

    try:
        data, header = read_envi_file(reflectance_path)
    except Exception as e:
        results["valid"] = False
        results["issues"].append(f"Error reading file: {e}")
        return results

    # Compute statistics
    results["stats"]["shape"] = data.shape
    results["stats"]["dtype"] = str(data.dtype)
    results["stats"]["min"] = float(np.nanmin(data))
    results["stats"]["max"] = float(np.nanmax(data))
    results["stats"]["mean"] = float(np.nanmean(data))
    results["stats"]["nan_fraction"] = float(np.isnan(data).mean())

    # Check physical range
    min_val = results["stats"]["min"]
    max_val = results["stats"]["max"]

    if min_val < REFLECTANCE_MIN:
        results["issues"].append(
            f"Minimum value {min_val:.4f} below expected range ({REFLECTANCE_MIN})"
        )

    if max_val > REFLECTANCE_MAX:
        results["issues"].append(
            f"Maximum value {max_val:.4f} above expected range ({REFLECTANCE_MAX})"
        )

    # Check NaN fraction
    nan_frac = results["stats"]["nan_fraction"]
    if nan_frac > 0.5:
        results["issues"].append(f"High NaN fraction: {nan_frac:.2%}")
        results["valid"] = False

    if results["issues"]:
        # Issues don't necessarily mean invalid, just warnings
        pass

    return results
