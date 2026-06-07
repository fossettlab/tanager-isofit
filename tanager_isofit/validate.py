"""
Validation utilities for comparing Tanager surface reflectance to EMIT data.

This module provides:
- EMIT data search and download via earthaccess
- Cross-sensor comparison metrics
- Validation report generation
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import warnings

import numpy as np

from tanager_isofit.config import (
    VALIDATION_RMSE_TARGET,
    VALIDATION_CORRELATION_TARGET,
    WATER_NDWI_THRESHOLD,
)
from tanager_isofit.utils import read_envi_file


def find_coincident_emit(
    tanager_bounds: Dict[str, float],
    tanager_time: datetime,
    time_window_hours: float = 24.0,
) -> List[Dict[str, Any]]:
    """
    Search for EMIT scenes coincident with Tanager acquisition.

    Args:
        tanager_bounds: Dict with min_lat, max_lat, min_lon, max_lon
        tanager_time: UTC acquisition time
        time_window_hours: Search window in hours (+/- from tanager_time)

    Returns:
        List of EMIT granule metadata dictionaries
    """
    try:
        import earthaccess
    except ImportError:
        raise ImportError(
            "earthaccess not installed. Install with: pip install earthaccess"
        )

    # Define temporal window
    start_time = tanager_time - timedelta(hours=time_window_hours)
    end_time = tanager_time + timedelta(hours=time_window_hours)

    # Define spatial bounds
    bbox = (
        tanager_bounds["min_lon"],
        tanager_bounds["min_lat"],
        tanager_bounds["max_lon"],
        tanager_bounds["max_lat"],
    )

    print("Searching for EMIT data...")
    print(f"  Bounding box: {bbox}")
    print(f"  Time range: {start_time} to {end_time}")

    # Search for EMIT L2A reflectance
    try:
        results = earthaccess.search_data(
            short_name="EMITL2ARFL",
            bounding_box=bbox,
            temporal=(start_time, end_time),
        )
    except Exception as e:
        warnings.warn(f"EMIT search failed: {e}")
        return []

    granules = []
    for result in results:
        try:
            granule_info = {
                "granule_id": result.get("meta", {}).get("native-id", "unknown"),
                "time_start": result.get("umm", {})
                .get("TemporalExtent", {})
                .get("RangeDateTime", {})
                .get("BeginningDateTime"),
                "time_end": result.get("umm", {})
                .get("TemporalExtent", {})
                .get("RangeDateTime", {})
                .get("EndingDateTime"),
                "download_links": earthaccess.results.DataGranule(result).data_links(),
            }
            granules.append(granule_info)
        except Exception:
            continue

    print(f"  Found {len(granules)} EMIT granules")
    return granules


def download_emit_l2a(
    granule_info: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Optional[Path]:
    """
    Download EMIT L2A reflectance data.

    Args:
        granule_info: Granule metadata from find_coincident_emit
        output_dir: Directory to save downloaded file

    Returns:
        Path to downloaded file, or None if download failed
    """
    try:
        import earthaccess
    except ImportError:
        raise ImportError("earthaccess not installed")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Authenticate if needed
    try:
        earthaccess.login(strategy="environment")
    except Exception:
        try:
            earthaccess.login(strategy="netrc")
        except Exception as e:
            warnings.warn(f"Authentication failed: {e}")
            return None

    # Download the file
    try:
        links = granule_info.get("download_links", [])
        if not links:
            warnings.warn("No download links found")
            return None

        downloaded = earthaccess.download(links[0], str(output_dir))
        if downloaded:
            return (
                Path(downloaded[0])
                if isinstance(downloaded, list)
                else Path(downloaded)
            )
    except Exception as e:
        warnings.warn(f"Download failed: {e}")

    return None


def read_emit_reflectance(
    emit_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read EMIT L2A reflectance NetCDF file.

    Args:
        emit_path: Path to EMIT L2A NetCDF file

    Returns:
        Tuple of (reflectance, wavelengths, latitude, longitude)
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray not installed")

    emit_path = Path(emit_path)

    # Open EMIT NetCDF
    ds = xr.open_dataset(emit_path)

    # Extract reflectance
    if "reflectance" in ds:
        reflectance = ds["reflectance"].values
    elif "rfl" in ds:
        reflectance = ds["rfl"].values
    else:
        raise ValueError("Cannot find reflectance variable in EMIT file")

    # Extract wavelengths
    if "wavelengths" in ds:
        wavelengths = ds["wavelengths"].values
    elif "wavelength" in ds:
        wavelengths = ds["wavelength"].values
    else:
        wavelengths = None

    # Extract geolocation
    latitude = ds["latitude"].values if "latitude" in ds else None
    longitude = ds["longitude"].values if "longitude" in ds else None

    ds.close()

    return reflectance, wavelengths, latitude, longitude


def spectral_resample(
    source_data: np.ndarray,
    source_wavelengths: np.ndarray,
    target_wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Resample spectral data to new wavelength grid.

    Uses linear interpolation between adjacent bands.

    Args:
        source_data: Source reflectance array (..., bands)
        source_wavelengths: Source center wavelengths
        target_wavelengths: Target center wavelengths

    Returns:
        Resampled data on target wavelength grid
    """
    from scipy.interpolate import interp1d

    # Get shape info
    orig_shape = source_data.shape
    n_bands = orig_shape[-1]

    # Flatten spatial dimensions
    flat_data = source_data.reshape(-1, n_bands)

    # Create interpolator
    interpolator = interp1d(
        source_wavelengths,
        flat_data,
        axis=1,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Interpolate to target wavelengths
    resampled = interpolator(target_wavelengths)

    # Reshape back
    new_shape = orig_shape[:-1] + (len(target_wavelengths),)
    return resampled.reshape(new_shape)


def compute_spectral_angle(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
) -> float:
    """
    Compute spectral angle between two spectra.

    Args:
        spectrum1: First spectrum (1D array)
        spectrum2: Second spectrum (1D array)

    Returns:
        Spectral angle in degrees
    """
    # Keep only finite values (drops NaN and +/-inf, e.g. from resampling).
    valid = np.isfinite(spectrum1) & np.isfinite(spectrum2)
    s1 = spectrum1[valid]
    s2 = spectrum2[valid]

    if len(s1) == 0:
        return np.nan

    # Compute angle
    dot = np.dot(s1, s2)
    norm1 = np.linalg.norm(s1)
    norm2 = np.linalg.norm(s2)

    if norm1 == 0 or norm2 == 0:
        return np.nan

    cos_angle = np.clip(dot / (norm1 * norm2), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def compare_reflectance(
    tanager_data: np.ndarray,
    tanager_wavelengths: np.ndarray,
    emit_data: np.ndarray,
    emit_wavelengths: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compare Tanager and EMIT reflectance data.

    Args:
        tanager_data: Tanager reflectance (lines, samples, bands)
        tanager_wavelengths: Tanager wavelengths in nm
        emit_data: EMIT reflectance (lines, samples, bands)
        emit_wavelengths: EMIT wavelengths in nm
        mask: Optional mask (True = valid pixels)

    Returns:
        Dictionary with comparison metrics
    """
    results = {
        "overall": {},
        "per_band": {},
        "spectral": {},
    }

    # Resample EMIT to Tanager wavelengths
    emit_resampled = spectral_resample(emit_data, emit_wavelengths, tanager_wavelengths)

    # Apply mask if provided
    if mask is None:
        mask = np.ones(tanager_data.shape[:2], dtype=bool)

    # Flatten for statistics
    tanager_flat = tanager_data[mask]
    emit_flat = emit_resampled[mask]

    # Overall statistics. Use isfinite so infinities (e.g. from resampling out
    # of range or divide-by-zero upstream) cannot corrupt RMSE/correlation.
    valid = np.isfinite(tanager_flat) & np.isfinite(emit_flat)
    valid_tanager = tanager_flat[valid]
    valid_emit = emit_flat[valid]

    if len(valid_tanager) > 0:
        diff = valid_tanager - valid_emit

        results["overall"]["rmse"] = float(np.sqrt(np.mean(diff**2)))
        results["overall"]["mae"] = float(np.mean(np.abs(diff)))
        results["overall"]["bias"] = float(np.mean(diff))
        results["overall"]["n_pixels"] = int(np.sum(mask))
        results["overall"]["correlation"] = float(
            np.corrcoef(valid_tanager.ravel(), valid_emit.ravel())[0, 1]
        )

    # Per-band statistics
    n_bands = tanager_data.shape[-1]
    band_rmse = np.zeros(n_bands)
    band_correlation = np.zeros(n_bands)
    band_bias = np.zeros(n_bands)

    for b in range(n_bands):
        tanager_band = tanager_data[:, :, b][mask]
        emit_band = emit_resampled[:, :, b][mask]

        valid = np.isfinite(tanager_band) & np.isfinite(emit_band)
        if np.sum(valid) > 0:
            t = tanager_band[valid]
            e = emit_band[valid]

            band_rmse[b] = np.sqrt(np.mean((t - e) ** 2))
            band_bias[b] = np.mean(t - e)
            if len(t) > 1:
                band_correlation[b] = np.corrcoef(t, e)[0, 1]

    results["per_band"]["rmse"] = band_rmse.tolist()
    results["per_band"]["bias"] = band_bias.tolist()
    results["per_band"]["correlation"] = band_correlation.tolist()
    results["per_band"]["wavelengths"] = tanager_wavelengths.tolist()

    # Compute mean spectral angle
    spectral_angles = []
    for i in range(tanager_data.shape[0]):
        for j in range(tanager_data.shape[1]):
            if mask[i, j]:
                angle = compute_spectral_angle(
                    tanager_data[i, j],
                    emit_resampled[i, j],
                )
                if not np.isnan(angle):
                    spectral_angles.append(angle)

    if spectral_angles:
        results["spectral"]["mean_angle"] = float(np.mean(spectral_angles))
        results["spectral"]["std_angle"] = float(np.std(spectral_angles))
        results["spectral"]["median_angle"] = float(np.median(spectral_angles))

    return results


def identify_water_pixels(
    reflectance: np.ndarray,
    wavelengths: np.ndarray,
    ndwi_threshold: float = WATER_NDWI_THRESHOLD,
) -> np.ndarray:
    """
    Identify water pixels using NDWI.

    NDWI = (Green - NIR) / (Green + NIR)

    Args:
        reflectance: Reflectance array (lines, samples, bands)
        wavelengths: Wavelengths in nm
        ndwi_threshold: NDWI threshold (>= threshold = water)

    Returns:
        Boolean mask (True = water)
    """
    # Find green (~560nm) and NIR (~860nm) bands
    green_idx = np.argmin(np.abs(wavelengths - 560))
    nir_idx = np.argmin(np.abs(wavelengths - 860))

    green = reflectance[:, :, green_idx]
    nir = reflectance[:, :, nir_idx]

    # Compute NDWI
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (green - nir) / (green + nir)

    # Create mask. Require finite NDWI so divide-by-zero (green + nir == 0)
    # producing +/-inf or NaN cannot be misclassified as water.
    water_mask = ndwi >= ndwi_threshold
    water_mask &= np.isfinite(ndwi)

    return water_mask


def validate_water_spectrum(
    reflectance: np.ndarray,
    wavelengths: np.ndarray,
    water_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Validate water pixel spectra against expected characteristics.

    Expected water characteristics:
    - Low reflectance in NIR/SWIR
    - Higher reflectance in blue/green
    - Absorption features in NIR

    Args:
        reflectance: Reflectance array
        wavelengths: Wavelengths in nm
        water_mask: Water pixel mask

    Returns:
        Validation results
    """
    results = {
        "valid": True,
        "issues": [],
        "stats": {},
    }

    if not np.any(water_mask):
        results["valid"] = False
        results["issues"].append("No water pixels found")
        return results

    # Get mean water spectrum
    water_spectra = reflectance[water_mask]
    mean_spectrum = np.nanmean(water_spectra, axis=0)

    results["stats"]["mean_spectrum"] = mean_spectrum.tolist()
    results["stats"]["n_water_pixels"] = int(np.sum(water_mask))

    # Check expected characteristics
    blue_idx = np.argmin(np.abs(wavelengths - 480))
    green_idx = np.argmin(np.abs(wavelengths - 560))
    red_idx = np.argmin(np.abs(wavelengths - 660))
    nir_idx = np.argmin(np.abs(wavelengths - 860))
    swir_idx = np.argmin(np.abs(wavelengths - 1600))

    blue_refl = mean_spectrum[blue_idx]
    green_refl = mean_spectrum[green_idx]
    red_refl = mean_spectrum[red_idx]
    nir_refl = mean_spectrum[nir_idx]
    swir_refl = mean_spectrum[swir_idx]

    results["stats"]["blue_refl"] = float(blue_refl)
    results["stats"]["green_refl"] = float(green_refl)
    results["stats"]["red_refl"] = float(red_refl)
    results["stats"]["nir_refl"] = float(nir_refl)
    results["stats"]["swir_refl"] = float(swir_refl)

    # Validation checks
    if nir_refl > 0.1:
        results["issues"].append(f"NIR reflectance too high for water: {nir_refl:.3f}")

    if swir_refl > 0.05:
        results["issues"].append(
            f"SWIR reflectance too high for water: {swir_refl:.3f}"
        )

    if blue_refl < nir_refl:
        results["issues"].append(
            "Blue reflectance lower than NIR (unexpected for water)"
        )

    if results["issues"]:
        results["valid"] = False

    return results


def generate_validation_report(
    comparison_results: Dict[str, Any],
    output_path: Union[str, Path],
    tanager_path: Optional[str] = None,
    emit_path: Optional[str] = None,
) -> Path:
    """
    Generate HTML validation report.

    Args:
        comparison_results: Results from compare_reflectance
        output_path: Path for output HTML file
        tanager_path: Path to Tanager file (for metadata)
        emit_path: Path to EMIT file (for metadata)

    Returns:
        Path to generated report
    """
    output_path = Path(output_path)

    # Build HTML content
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Tanager vs EMIT Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .pass { color: green; }
        .fail { color: red; }
        .metric { font-weight: bold; }
    </style>
</head>
<body>
    <h1>Tanager vs EMIT Validation Report</h1>
"""

    # Add metadata
    if tanager_path or emit_path:
        html += "<h2>Data Sources</h2>\n<ul>\n"
        if tanager_path:
            html += f"<li>Tanager: {tanager_path}</li>\n"
        if emit_path:
            html += f"<li>EMIT: {emit_path}</li>\n"
        html += "</ul>\n"

    # Overall metrics
    html += "<h2>Overall Metrics</h2>\n"
    html += "<table>\n<tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th></tr>\n"

    overall = comparison_results.get("overall", {})

    rmse = overall.get("rmse", float("nan"))
    rmse_status = "pass" if rmse <= VALIDATION_RMSE_TARGET else "fail"
    html += f"""<tr>
        <td class="metric">RMSE</td>
        <td>{rmse:.4f}</td>
        <td>≤ {VALIDATION_RMSE_TARGET}</td>
        <td class="{rmse_status}">{rmse_status.upper()}</td>
    </tr>\n"""

    corr = overall.get("correlation", float("nan"))
    corr_status = "pass" if corr >= VALIDATION_CORRELATION_TARGET else "fail"
    html += f"""<tr>
        <td class="metric">Correlation</td>
        <td>{corr:.4f}</td>
        <td>≥ {VALIDATION_CORRELATION_TARGET}</td>
        <td class="{corr_status}">{corr_status.upper()}</td>
    </tr>\n"""

    html += f"""<tr>
        <td class="metric">MAE</td>
        <td>{overall.get("mae", float("nan")):.4f}</td>
        <td>-</td>
        <td>-</td>
    </tr>\n"""

    html += f"""<tr>
        <td class="metric">Bias</td>
        <td>{overall.get("bias", float("nan")):.4f}</td>
        <td>-</td>
        <td>-</td>
    </tr>\n"""

    html += f"""<tr>
        <td class="metric">N Pixels</td>
        <td>{overall.get("n_pixels", 0)}</td>
        <td>-</td>
        <td>-</td>
    </tr>\n"""

    html += "</table>\n"

    # Spectral angle
    spectral = comparison_results.get("spectral", {})
    if spectral:
        html += "<h2>Spectral Metrics</h2>\n"
        html += "<table>\n<tr><th>Metric</th><th>Value</th></tr>\n"
        html += f"<tr><td>Mean Spectral Angle</td><td>{spectral.get('mean_angle', float('nan')):.2f}°</td></tr>\n"
        html += f"<tr><td>Std Spectral Angle</td><td>{spectral.get('std_angle', float('nan')):.2f}°</td></tr>\n"
        html += f"<tr><td>Median Spectral Angle</td><td>{spectral.get('median_angle', float('nan')):.2f}°</td></tr>\n"
        html += "</table>\n"

    # Close HTML
    html += """
    <hr>
    <p><small>Generated by tanager_isofit validation module</small></p>
</body>
</html>
"""

    # Write file
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def run_full_validation(
    tanager_reflectance_path: Union[str, Path],
    tanager_wavelength_file: Union[str, Path],
    emit_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """
    Run complete validation pipeline.

    Args:
        tanager_reflectance_path: Path to Tanager reflectance ENVI file
        tanager_wavelength_file: Path to Tanager wavelength file
        emit_path: Path to EMIT L2A NetCDF file
        output_dir: Directory for validation outputs

    Returns:
        Complete validation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "tanager_path": str(tanager_reflectance_path),
        "emit_path": str(emit_path),
    }

    # Read Tanager data
    print("Reading Tanager reflectance...")
    tanager_data, tanager_header = read_envi_file(tanager_reflectance_path)

    # Read wavelengths. create_wavelength_file writes 3 columns:
    # channel index, wavelength (nm), fwhm (nm). Column 1 is the wavelength.
    wavelength_table = np.loadtxt(tanager_wavelength_file)
    if wavelength_table.ndim != 2 or wavelength_table.shape[1] < 3:
        raise ValueError(
            f"Wavelength file {tanager_wavelength_file} must have 3 columns "
            "(channel, wavelength, fwhm); got shape "
            f"{wavelength_table.shape}"
        )
    tanager_wavelengths = wavelength_table[:, 1]

    # Read EMIT data
    print("Reading EMIT reflectance...")
    emit_data, emit_wavelengths, emit_lat, emit_lon = read_emit_reflectance(emit_path)

    # TODO: Spatial co-registration would go here
    # For now, assume data is already aligned

    # Compare reflectance
    print("Comparing reflectance...")
    comparison = compare_reflectance(
        tanager_data, tanager_wavelengths, emit_data, emit_wavelengths
    )
    results["comparison"] = comparison

    # Identify and validate water pixels
    print("Validating water pixels...")
    water_mask = identify_water_pixels(tanager_data, tanager_wavelengths)
    water_validation = validate_water_spectrum(
        tanager_data, tanager_wavelengths, water_mask
    )
    results["water_validation"] = water_validation

    # Generate report
    print("Generating report...")
    report_path = generate_validation_report(
        comparison,
        output_dir / "validation_report.html",
        str(tanager_reflectance_path),
        str(emit_path),
    )
    results["report_path"] = str(report_path)

    # Summary
    overall_rmse = comparison.get("overall", {}).get("rmse", float("nan"))
    overall_corr = comparison.get("overall", {}).get("correlation", float("nan"))

    results["summary"] = {
        "rmse_pass": overall_rmse <= VALIDATION_RMSE_TARGET,
        "correlation_pass": overall_corr >= VALIDATION_CORRELATION_TARGET,
        "water_valid": water_validation.get("valid", False),
    }

    print(f"Validation complete. Report: {report_path}")

    return results
