"""
Convert Tanager HDF5 radiance data to ENVI format for ISOFIT processing.

This module handles:
- Reading Tanager HDF5 files
- Extracting radiance, geolocation, and metadata
- Writing ENVI-format files (radiance, location, observation)
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
import warnings

import h5py
import numpy as np
from pyproj import CRS, Transformer

from tanager_isofit.config import (
    HDF5_ALT_PATHS,
    WAVELENGTH_ATTR_NAMES,
    FWHM_ATTR_NAMES,
    FILL_VALUE_ATTR_NAMES,
    ENVI_RADIANCE_FILENAME,
    ENVI_LOCATION_FILENAME,
    ENVI_OBSERVATION_FILENAME,
    LOC_BAND_NAMES,
    OBS_BAND_NAMES,
    TANAGER_NUM_BANDS,
    TANAGER_ALTITUDE_KM,
    RADIANCE_CONVERSION_FACTOR,
    get_default_wavelengths,
    get_default_fwhm,
)
from tanager_isofit.utils import (
    write_envi_file,
    ensure_directory,
    create_wavelength_file,
)
from tanager_isofit.geometry import (
    create_observation_array,
    parse_acquisition_time,
)
from tanager_isofit.dem import get_flat_terrain

logger = logging.getLogger(__name__)


def inspect_hdf5(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspect HDF5 file structure and return detailed information.

    Args:
        path: Path to HDF5 file

    Returns:
        Dictionary with datasets, groups, and attributes
    """
    path = Path(path)
    info = {
        "path": str(path),
        "groups": [],
        "datasets": [],
        "attributes": {},
    }

    with h5py.File(path, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                ds_info = {
                    "name": name,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "attrs": dict(obj.attrs),
                }
                info["datasets"].append(ds_info)
            elif isinstance(obj, h5py.Group):
                info["groups"].append(name)

        f.visititems(visitor)

        # Get root attributes
        info["attributes"] = dict(f.attrs)

    return info


def print_hdf5_structure(path: Union[str, Path]) -> None:
    """
    Print HDF5 structure to console for debugging.

    Args:
        path: Path to HDF5 file
    """
    info = inspect_hdf5(path)

    print(f"\nHDF5 File: {info['path']}")
    print("=" * 60)

    print("\nRoot Attributes:")
    for key, val in info["attributes"].items():
        print(f"  {key}: {val}")

    print("\nGroups:")
    for group in sorted(info["groups"]):
        print(f"  /{group}/")

    print("\nDatasets:")
    for ds in info["datasets"]:
        print(f"  {ds['name']}")
        print(f"    shape: {ds['shape']}, dtype: {ds['dtype']}")
        if ds["attrs"]:
            for key, val in list(ds["attrs"].items())[:5]:
                print(f"    @{key}: {val}")


def _find_dataset(f: h5py.File, paths: List[str]) -> Optional[h5py.Dataset]:
    """Try multiple paths to find a dataset."""
    for path in paths:
        if path in f:
            return f[path]
    return None


def _get_attribute(
    obj: Union[h5py.File, h5py.Dataset], names: List[str]
) -> Optional[Any]:
    """Try multiple attribute names to find a value."""
    for name in names:
        if name in obj.attrs:
            return obj.attrs[name]
    return None


def _generate_grids_latlon(
    f: h5py.File,
    subset: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate lat/lon arrays from GRIDS UTM metadata.

    Args:
        f: Open h5py.File object (must be GRIDS format)
        subset: Optional (row_start, row_end, col_start, col_end)

    Returns:
        (latitude, longitude) arrays in WGS84 (EPSG:4326).
        Shape is (rows, cols) with row 0 at top (north).

    Raises:
        ValueError: If StructMetadata.0 is missing or malformed
    """
    # Read and decode StructMetadata.0
    metadata_path = "HDFEOS INFORMATION/StructMetadata.0"
    if metadata_path not in f:
        raise ValueError(f"GRIDS file missing {metadata_path}")

    metadata = f[metadata_path][()]
    if isinstance(metadata, bytes):
        metadata = metadata.decode("utf-8")

    # Parse required fields
    ul = re.search(
        r"UpperLeftPointMtrs=\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\)",
        metadata,
    )
    lr = re.search(
        r"LowerRightMtrs=\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\)", metadata
    )
    xdim = re.search(r"XDim=(\d+)", metadata)
    ydim = re.search(r"YDim=(\d+)", metadata)
    zone = re.search(r"ZoneCode=(-?\d+)", metadata)

    if not all([ul, lr, xdim, ydim, zone]):
        raise ValueError(
            "StructMetadata.0 missing required fields "
            "(UpperLeftPointMtrs, LowerRightMtrs, XDim, YDim, ZoneCode)"
        )

    # Extract values
    ul_x, ul_y = float(ul.group(1)), float(ul.group(2))
    lr_x, lr_y = float(lr.group(1)), float(lr.group(2))
    cols, rows = int(xdim.group(1)), int(ydim.group(1))
    utm_zone = int(zone.group(1))

    # Validate UTM zone
    if utm_zone == 0 or abs(utm_zone) > 60:
        raise ValueError(
            f"Invalid UTM zone: {utm_zone}. Must be 1-60 (negative for south)"
        )

    # Generate UTM grid coordinates
    x_coords = np.linspace(ul_x, lr_x, cols)
    y_coords = np.linspace(ul_y, lr_y, rows)
    easting, northing = np.meshgrid(x_coords, y_coords)

    # Determine EPSG code: 326XX for north, 327XX for south
    epsg = 32600 + utm_zone if utm_zone > 0 else 32700 + abs(utm_zone)

    # Transform UTM -> WGS84
    transformer = Transformer.from_crs(
        CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True
    )
    longitude, latitude = transformer.transform(easting, northing)

    # Validate output
    if np.any(np.isnan(latitude)) or np.any(np.isnan(longitude)):
        raise ValueError(
            "UTM to lat/lon transformation produced NaN values. "
            "Check grid parameters and UTM zone."
        )

    # Apply subset if specified
    if subset is not None:
        row_start, row_end, col_start, col_end = subset
        latitude = latitude[row_start:row_end, col_start:col_end]
        longitude = longitude[row_start:row_end, col_start:col_end]

    return latitude.astype(np.float64), longitude.astype(np.float64)


def read_tanager_hdf5(
    path: Union[str, Path],
    subset: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Read Tanager HDF5 file and extract all needed data.

    Args:
        path: Path to HDF5 file
        subset: Optional (row_start, row_end, col_start, col_end) for subsetting

    Returns:
        Dictionary with:
            - radiance: (lines, samples, bands) array
            - latitude: (lines, samples) array
            - longitude: (lines, samples) array
            - path_length: (lines, samples) array
            - wavelengths: list of center wavelengths
            - fwhm: list of FWHM values
            - acquisition_time: datetime
            - metadata: dict of additional metadata
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        # Find radiance dataset
        radiance_ds = _find_dataset(f, HDF5_ALT_PATHS["radiance"])
        if radiance_ds is None:
            # Try to find any dataset with 'radiance' in name
            for ds_name in f.keys():
                if "radiance" in ds_name.lower():
                    radiance_ds = f[ds_name]
                    break

            # Also search in nested groups
            if radiance_ds is None:

                def find_radiance(name, obj):
                    if isinstance(obj, h5py.Dataset) and "radiance" in name.lower():
                        return obj

                result = None

                def visitor(name, obj):
                    nonlocal result
                    if result is None and isinstance(obj, h5py.Dataset):
                        if "radiance" in name.lower():
                            result = obj

                f.visititems(visitor)
                radiance_ds = result

        if radiance_ds is None:
            raise ValueError(f"Cannot find radiance dataset in {path}")

        # Get full dimensions - handle both (bands, lines, samples) and (lines, samples, bands)
        full_shape = radiance_ds.shape

        # Detect dimension order: Tanager uses (bands, lines, samples) where
        # bands=426. Disambiguate by matching the band count against the band
        # axis only. If both the first and last axes equal the band count, the
        # order is genuinely ambiguous and silently guessing could transpose the
        # cube, so raise instead.
        first_is_bands = full_shape[0] == TANAGER_NUM_BANDS
        last_is_bands = full_shape[2] == TANAGER_NUM_BANDS
        if first_is_bands and last_is_bands:
            raise ValueError(
                f"Ambiguous radiance dimension order {full_shape} in {path}: "
                f"both the first and last axes equal the band count "
                f"({TANAGER_NUM_BANDS}). Cannot determine (bands, lines, samples) "
                "vs (lines, samples, bands)."
            )

        if first_is_bands:
            # Shape is (bands, lines, samples) - need to transpose
            bands_first = True
            lines, samples = full_shape[1], full_shape[2]
        else:
            # Shape is (lines, samples, bands)
            bands_first = False
            lines, samples = full_shape[0], full_shape[1]

        # Apply subset if specified
        if subset is not None:
            row_start, row_end, col_start, col_end = subset
            row_start = max(0, row_start)
            row_end = min(lines, row_end)
            col_start = max(0, col_start)
            col_end = min(samples, col_end)

            if bands_first:
                radiance = radiance_ds[:, row_start:row_end, col_start:col_end]
            else:
                radiance = radiance_ds[row_start:row_end, col_start:col_end, :]
        else:
            radiance = radiance_ds[:]

        radiance = radiance.astype(np.float32)

        # Honor any declared fill / no-data sentinel BEFORE scaling so that fill
        # pixels are never multiplied through and handed to ISOFIT as if they
        # were real radiance. Matching pixels become NaN, which downstream
        # consumers already treat as invalid. The sentinel comes from the file's
        # own metadata; no value is invented.
        fill_value = _get_attribute(radiance_ds, FILL_VALUE_ATTR_NAMES)
        if fill_value is not None:
            fill_value = float(np.asarray(fill_value).ravel()[0])
            fill_mask = radiance == fill_value
            n_fill = int(np.count_nonzero(fill_mask))
            if n_fill > 0:
                warnings.warn(
                    f"Radiance contains {n_fill} fill pixels "
                    f"(_FillValue={fill_value}); setting them to NaN."
                )
                radiance[fill_mask] = np.nan

        # Convert radiance units from Tanager (W/m²/sr/µm) to ISOFIT (µW/cm²/sr/nm)
        # This is a critical unit conversion - ISOFIT expects µW/cm²/sr/nm
        radiance = radiance * RADIANCE_CONVERSION_FACTOR

        # Transpose to (lines, samples, bands) if needed
        if bands_first:
            radiance = np.transpose(radiance, (1, 2, 0))

        # Get geolocation based on format
        if "HDFEOS/GRIDS" in f:
            # GRIDS format: generate lat/lon from UTM metadata
            logger.info("Detected GRIDS format, generating lat/lon from UTM")
            latitude, longitude = _generate_grids_latlon(f, subset)

        elif "HDFEOS/SWATHS" in f:
            # SWATHS format: read explicit lat/lon arrays
            logger.info("Detected SWATHS format, reading explicit lat/lon")
            latitude_ds = _find_dataset(f, HDF5_ALT_PATHS["latitude"])
            longitude_ds = _find_dataset(f, HDF5_ALT_PATHS["longitude"])

            # If standard paths don't work, search for lat/lon
            if latitude_ds is None:

                def find_dataset_by_name(target_name):
                    result = None

                    def visitor(name, obj):
                        nonlocal result
                        if result is None and isinstance(obj, h5py.Dataset):
                            if target_name in name.lower():
                                result = obj

                    f.visititems(visitor)
                    return result

                latitude_ds = find_dataset_by_name("latitude")
                longitude_ds = find_dataset_by_name("longitude")

            if latitude_ds is None or longitude_ds is None:
                raise ValueError(f"Cannot find latitude/longitude datasets in {path}")

            # Read geolocation with same subsetting
            if subset is not None:
                latitude = latitude_ds[row_start:row_end, col_start:col_end]
                longitude = longitude_ds[row_start:row_end, col_start:col_end]
            else:
                latitude = latitude_ds[:]
                longitude = longitude_ds[:]

            latitude = latitude.astype(np.float64)
            longitude = longitude.astype(np.float64)

        else:
            raise ValueError(
                f"Cannot determine HDF5 format for {path}. "
                "Expected SWATHS (basic_radiance) or GRIDS (ortho_radiance) structure."
            )

        # Find path length dataset
        # NOTE: ISOFIT expects path_length in METERS (it converts to km internally)
        path_length_ds = _find_dataset(f, HDF5_ALT_PATHS["path_length"])
        if path_length_ds is not None:
            if subset is not None:
                path_length = path_length_ds[row_start:row_end, col_start:col_end]
            else:
                path_length = path_length_ds[:]
            path_length = path_length.astype(np.float32)
            # Keep in METERS - ISOFIT expects meters and converts internally!
        else:
            # Estimate path length from satellite altitude (convert km to meters)
            warnings.warn(
                "Path length not found, using constant value from satellite altitude"
            )
            current_lines = radiance.shape[0]
            current_samples = radiance.shape[1]
            path_length = np.full(
                (current_lines, current_samples),
                TANAGER_ALTITUDE_KM * 1000.0,
                dtype=np.float32,
            )  # km -> m

        # Read sun/sensor geometry if available in HDF5
        sun_zenith_ds = _find_dataset(f, HDF5_ALT_PATHS.get("sun_zenith", []))
        sun_azimuth_ds = _find_dataset(f, HDF5_ALT_PATHS.get("sun_azimuth", []))
        sensor_zenith_ds = _find_dataset(f, HDF5_ALT_PATHS.get("sensor_zenith", []))
        sensor_azimuth_ds = _find_dataset(f, HDF5_ALT_PATHS.get("sensor_azimuth", []))

        geometry_from_hdf5 = {}
        for name, ds in [
            ("sun_zenith", sun_zenith_ds),
            ("sun_azimuth", sun_azimuth_ds),
            ("sensor_zenith", sensor_zenith_ds),
            ("sensor_azimuth", sensor_azimuth_ds),
        ]:
            if ds is not None:
                if subset is not None:
                    geometry_from_hdf5[name] = ds[
                        row_start:row_end, col_start:col_end
                    ].astype(np.float32)
                else:
                    geometry_from_hdf5[name] = ds[:].astype(np.float32)

        # Get wavelengths from attributes
        wavelengths = _get_attribute(radiance_ds, WAVELENGTH_ATTR_NAMES)
        if wavelengths is None:
            wavelengths = _get_attribute(f, WAVELENGTH_ATTR_NAMES)

        if wavelengths is not None:
            wavelengths = list(wavelengths)
        else:
            warnings.warn(
                "Wavelengths not found in HDF5, using default Tanager wavelengths"
            )
            wavelengths = get_default_wavelengths()

        # Get FWHM from attributes
        fwhm = _get_attribute(radiance_ds, FWHM_ATTR_NAMES)
        if fwhm is None:
            fwhm = _get_attribute(f, FWHM_ATTR_NAMES)

        if fwhm is not None:
            fwhm = list(fwhm)
        else:
            warnings.warn("FWHM not found in HDF5, using default values")
            fwhm = get_default_fwhm()

        # Guard against wavelength/FWHM vectors that do not match the radiance
        # band count. A mismatch would silently misalign every band downstream
        # (wrong wavelength labels in the ENVI header and the ISOFIT wavelength
        # file), so fail closed rather than emit a corrupted cube.
        n_bands = radiance.shape[2]
        if len(wavelengths) != n_bands:
            raise ValueError(
                f"Wavelength count ({len(wavelengths)}) does not match radiance "
                f"band count ({n_bands}) in {path}"
            )
        if len(fwhm) != n_bands:
            raise ValueError(
                f"FWHM count ({len(fwhm)}) does not match radiance band count "
                f"({n_bands}) in {path}"
            )

        # Get acquisition time from Time dataset, strip_id, or filename
        acquisition_time = None

        # Try to get time from the Time dataset (Unix timestamps)
        time_ds = _find_dataset(
            f,
            [
                "HDFEOS/SWATHS/HYP/Geolocation Fields/Time",
                "Time",
            ],
        )
        if time_ds is not None:
            try:
                first_time = float(time_ds[0])
                # Unix timestamp interpretation
                acquisition_time = datetime.utcfromtimestamp(first_time)
            except (ValueError, OverflowError):
                pass

        # Fall back to strip_id attribute
        if acquisition_time is None:
            strip_id = None
            if "strip_id" in f.attrs:
                strip_id = f.attrs["strip_id"]
                if isinstance(strip_id, bytes):
                    strip_id = strip_id.decode()
            elif "strip_id" in radiance_ds.attrs:
                strip_id = radiance_ds.attrs["strip_id"]
                if isinstance(strip_id, bytes):
                    strip_id = strip_id.decode()

            if strip_id is not None:
                try:
                    acquisition_time = parse_acquisition_time(strip_id)
                except ValueError:
                    pass
        else:
            strip_id = None

        # Final fallback to filename parsing
        if acquisition_time is None:
            acquisition_time = _parse_time_from_filename(path)

        # Collect additional metadata
        metadata = {
            "source_file": str(path),
            "full_shape": full_shape,
            "subset": subset,
            "strip_id": strip_id,
        }

        # Copy relevant attributes
        for attr_name in ["processing_level", "sensor", "platform"]:
            if attr_name in f.attrs:
                val = f.attrs[attr_name]
                if isinstance(val, bytes):
                    val = val.decode()
                metadata[attr_name] = val

    return {
        "radiance": radiance,
        "latitude": latitude,
        "longitude": longitude,
        "path_length": path_length,
        "wavelengths": wavelengths,
        "fwhm": fwhm,
        "acquisition_time": acquisition_time,
        "geometry": geometry_from_hdf5,  # Pre-computed sun/sensor angles from HDF5
        "metadata": metadata,
    }


def _parse_time_from_filename(path: Path) -> datetime:
    """
    Parse acquisition time from Tanager filename.

    Expected format: YYYYMMDD_HHMMSS_...

    Args:
        path: Path to file

    Returns:
        datetime object
    """
    filename = path.stem
    parts = filename.split("_")

    if len(parts) >= 2:
        try:
            return parse_acquisition_time(f"{parts[0]}_{parts[1]}")
        except ValueError:
            pass

    # No usable acquisition time anywhere (Time dataset, strip_id, or filename).
    # Falling back to the wall clock would silently fabricate the value that
    # drives the output filename prefix and the observation UTC-time band,
    # making the conversion nonreproducible. Fail closed instead.
    raise ValueError(
        f"Could not determine acquisition time for {path}. Expected a "
        "'YYYYMMDD_HHMMSS_...' filename, a parseable strip_id, or a Time "
        "dataset. Rename the file or supply valid metadata."
    )


def _create_observation_array_from_hdf5(
    lines: int,
    samples: int,
    path_length_m: np.ndarray,
    geometry: Dict[str, np.ndarray],
    acquisition_time: datetime,
    slope: Optional[np.ndarray] = None,
    aspect: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create observation array using pre-computed geometry from HDF5.

    Args:
        lines: Number of lines
        samples: Number of samples
        path_length_m: Path length in METERS (ISOFIT converts to km internally)
        geometry: Dict with sun_zenith, sun_azimuth, sensor_zenith, sensor_azimuth
        acquisition_time: Acquisition time for UTC time band
        slope: Terrain slope (default 0)
        aspect: Terrain aspect (default 0)

    Returns:
        Observation array (lines, samples, 10)
    """
    from tanager_isofit.geometry import (
        calculate_phase_angle,
        calculate_cosine_i,
        time_to_decimal_hours,
    )

    obs = np.zeros((lines, samples, 10), dtype=np.float32)

    # Band 0: Path length in METERS (ISOFIT expects meters, converts internally)
    obs[:, :, 0] = path_length_m

    # Band 1: Sensor azimuth (degrees)
    if "sensor_azimuth" in geometry:
        obs[:, :, 1] = geometry["sensor_azimuth"]

    # Band 2: Sensor zenith (degrees)
    if "sensor_zenith" in geometry:
        obs[:, :, 2] = geometry["sensor_zenith"]

    # Band 3: Sun azimuth (degrees)
    sun_azimuth = geometry.get("sun_azimuth", np.zeros((lines, samples)))
    obs[:, :, 3] = sun_azimuth

    # Band 4: Sun zenith (degrees)
    sun_zenith = geometry.get("sun_zenith", np.zeros((lines, samples)))
    obs[:, :, 4] = sun_zenith

    # Band 5: Phase angle (degrees)
    sensor_zenith = geometry.get("sensor_zenith", np.zeros((lines, samples)))
    sensor_azimuth = geometry.get("sensor_azimuth", np.zeros((lines, samples)))
    phase_angle = calculate_phase_angle(
        sun_zenith, sun_azimuth, sensor_zenith, sensor_azimuth
    )
    obs[:, :, 5] = phase_angle

    # Band 6: Slope (degrees)
    if slope is None:
        slope = np.zeros((lines, samples), dtype=np.float32)
    obs[:, :, 6] = slope

    # Band 7: Aspect (degrees)
    if aspect is None:
        aspect = np.zeros((lines, samples), dtype=np.float32)
    obs[:, :, 7] = aspect

    # Band 8: Cosine of illumination angle
    cosine_i = calculate_cosine_i(sun_zenith, slope, aspect, sun_azimuth)
    obs[:, :, 8] = cosine_i

    # Band 9: UTC time (decimal hours)
    utc_time = time_to_decimal_hours(acquisition_time)
    obs[:, :, 9] = utc_time

    return obs


def convert_tanager_to_envi(
    input_h5: Union[str, Path],
    output_dir: Union[str, Path],
    subset: Optional[Tuple[int, int, int, int]] = None,
    use_fast_solar: bool = True,
    dem_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """
    Convert Tanager HDF5 to ENVI files for ISOFIT.

    Creates three ENVI files:
    - radiance: (lines, samples, 426 bands) - TOA radiance
    - loc: (lines, samples, 3 bands) - lon, lat, elevation
    - obs: (lines, samples, 10 bands) - observation geometry

    Args:
        input_h5: Path to input HDF5 file
        output_dir: Directory for output ENVI files
        subset: Optional (row_start, row_end, col_start, col_end)
        use_fast_solar: Use fast grid-interpolated solar calculation
        dem_path: Path to DEM file (optional, defaults to flat terrain)

    Returns:
        Dictionary with paths to created files:
            - radiance: path to radiance binary
            - loc: path to location binary
            - obs: path to observation binary
            - wavelength_file: path to wavelength text file
    """
    input_h5 = Path(input_h5)
    output_dir = ensure_directory(output_dir)

    print(f"Converting {input_h5.name} to ENVI format...")

    # Read HDF5 data
    print("  Reading HDF5 file...")
    data = read_tanager_hdf5(input_h5, subset=subset)

    radiance = data["radiance"]
    latitude = data["latitude"]
    longitude = data["longitude"]
    path_length = data["path_length"]
    wavelengths = data["wavelengths"]
    fwhm = data["fwhm"]
    acquisition_time = data["acquisition_time"]
    geometry_from_hdf5 = data.get("geometry", {})

    lines, samples, bands = radiance.shape
    print(f"  Data dimensions: {lines} lines x {samples} samples x {bands} bands")

    # Get terrain data (flat terrain for v1)
    print("  Getting terrain data...")
    elevation, slope, aspect = get_flat_terrain(lines, samples)

    # Create observation array
    print("  Computing observation geometry...")
    if geometry_from_hdf5:
        # Use pre-computed geometry from HDF5 file
        print("    Using pre-computed sun/sensor geometry from HDF5")
        obs_array = _create_observation_array_from_hdf5(
            lines=lines,
            samples=samples,
            path_length_m=path_length,
            geometry=geometry_from_hdf5,
            acquisition_time=acquisition_time,
            slope=slope,
            aspect=aspect,
        )
    else:
        # Compute geometry from scratch
        print("    Computing sun/sensor geometry from lat/lon/time")
        obs_array = create_observation_array(
            lines=lines,
            samples=samples,
            path_length_m=path_length,
            acquisition_time=acquisition_time,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            slope=slope,
            aspect=aspect,
            use_fast_solar=use_fast_solar,
        )

    # Create location array (lon, lat, elevation)
    print("  Creating location array...")
    loc_array = np.stack([longitude, latitude, elevation], axis=-1).astype(np.float32)

    # Generate timestamp prefix for ISOFIT compatibility (YYYYMMDD_HHMMSS)
    timestamp_prefix = acquisition_time.strftime("%Y%m%d_%H%M%S")
    print(f"  Using timestamp prefix: {timestamp_prefix}")

    # Write radiance file with timestamp prefix
    print("  Writing radiance ENVI file...")
    radiance_filename = f"{timestamp_prefix}_{ENVI_RADIANCE_FILENAME}"
    radiance_path, radiance_hdr = write_envi_file(
        radiance,
        output_dir / radiance_filename,
        wavelengths=wavelengths,
        fwhm=fwhm,
        interleave="bip",
        description=f"Tanager TOA radiance from {input_h5.name}",
    )

    # Write location file with timestamp prefix
    print("  Writing location ENVI file...")
    loc_filename = f"{timestamp_prefix}_{ENVI_LOCATION_FILENAME}"
    loc_path, loc_hdr = write_envi_file(
        loc_array,
        output_dir / loc_filename,
        band_names=LOC_BAND_NAMES,
        interleave="bip",
        description="Tanager geolocation (lon, lat, elev)",
    )

    # Write observation file with timestamp prefix
    print("  Writing observation ENVI file...")
    obs_filename = f"{timestamp_prefix}_{ENVI_OBSERVATION_FILENAME}"
    obs_path, obs_hdr = write_envi_file(
        obs_array,
        output_dir / obs_filename,
        band_names=OBS_BAND_NAMES,
        interleave="bip",
        description="Tanager observation geometry",
    )

    # Create wavelength file for ISOFIT
    print("  Creating wavelength file...")
    wavelength_file = create_wavelength_file(
        wavelengths, fwhm, output_dir / "wavelengths.txt"
    )

    print(f"  Conversion complete. Output in: {output_dir}")

    return {
        "radiance": radiance_path,
        "loc": loc_path,
        "obs": obs_path,
        "wavelength_file": wavelength_file,
        "radiance_hdr": radiance_hdr,
        "loc_hdr": loc_hdr,
        "obs_hdr": obs_hdr,
        "timestamp_prefix": timestamp_prefix,
        "acquisition_time": acquisition_time,
    }


def validate_hdf5_structure(path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Validate that HDF5 file has expected Tanager structure.

    Supports both SWATHS (basic_radiance) and GRIDS (ortho_radiance) formats.

    Args:
        path: Path to HDF5 file

    Returns:
        Tuple of (is_valid, list of issues)
    """
    path = Path(path)
    issues = []

    if not path.exists():
        return False, [f"File not found: {path}"]

    try:
        with h5py.File(path, "r") as f:
            # Check for valid format
            is_swaths = "HDFEOS/SWATHS" in f
            is_grids = "HDFEOS/GRIDS" in f

            if not is_swaths and not is_grids:
                return False, [
                    "Cannot determine format: expected SWATHS or GRIDS structure"
                ]

            # Check for radiance (required for both formats)
            radiance_ds = _find_dataset(f, HDF5_ALT_PATHS["radiance"])
            if radiance_ds is None:
                issues.append("Missing radiance dataset")
            else:
                if len(radiance_ds.shape) != 3:
                    issues.append(
                        f"Radiance should be 3D, got {len(radiance_ds.shape)}D"
                    )
                else:
                    # Handle both (bands, lines, samples) and (lines, samples, bands)
                    if radiance_ds.shape[0] == TANAGER_NUM_BANDS:
                        n_bands = radiance_ds.shape[0]
                    else:
                        n_bands = radiance_ds.shape[2]
                    if n_bands != TANAGER_NUM_BANDS:
                        issues.append(
                            f"Expected {TANAGER_NUM_BANDS} bands, got {n_bands}"
                        )

            # Format-specific checks
            if is_swaths:
                lat_ds = _find_dataset(f, HDF5_ALT_PATHS["latitude"])
                if lat_ds is None:
                    issues.append("Missing latitude dataset (SWATHS format)")
                lon_ds = _find_dataset(f, HDF5_ALT_PATHS["longitude"])
                if lon_ds is None:
                    issues.append("Missing longitude dataset (SWATHS format)")

            if is_grids:
                if "HDFEOS INFORMATION/StructMetadata.0" not in f:
                    issues.append("Missing StructMetadata.0 (GRIDS format)")

    except Exception as e:
        return False, [f"Error reading HDF5: {e}"]

    return len(issues) == 0, issues


def get_scene_bounds(path: Union[str, Path]) -> Dict[str, float]:
    """
    Get geographic bounds of a Tanager scene.

    Args:
        path: Path to HDF5 file

    Returns:
        Dictionary with min_lat, max_lat, min_lon, max_lon
    """
    with h5py.File(path, "r") as f:
        lat_ds = _find_dataset(f, HDF5_ALT_PATHS["latitude"])
        lon_ds = _find_dataset(f, HDF5_ALT_PATHS["longitude"])

        if lat_ds is None or lon_ds is None:
            raise ValueError("Cannot find lat/lon datasets")

        latitude = lat_ds[:]
        longitude = lon_ds[:]

    return {
        "min_lat": float(np.nanmin(latitude)),
        "max_lat": float(np.nanmax(latitude)),
        "min_lon": float(np.nanmin(longitude)),
        "max_lon": float(np.nanmax(longitude)),
    }


def get_acquisition_time(path: Union[str, Path]) -> datetime:
    """
    Get acquisition time from Tanager HDF5 file.

    Args:
        path: Path to HDF5 file

    Returns:
        UTC datetime
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        strip_id = None
        if "strip_id" in f.attrs:
            strip_id = f.attrs["strip_id"]
            if isinstance(strip_id, bytes):
                strip_id = strip_id.decode()

    if strip_id is not None:
        try:
            return parse_acquisition_time(strip_id)
        except ValueError:
            pass

    return _parse_time_from_filename(path)
