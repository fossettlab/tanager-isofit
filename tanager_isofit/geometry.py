"""
Solar and sensor geometry calculations for Tanager data.
"""

from datetime import datetime
from typing import Tuple, Union, Optional

import numpy as np

from tanager_isofit.config import TANAGER_ALTITUDE_KM


def calculate_solar_geometry(
    acquisition_time: Union[datetime, str],
    latitude: Union[float, np.ndarray],
    longitude: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate solar zenith and azimuth angles using pvlib.

    Args:
        acquisition_time: UTC acquisition time (datetime or ISO string)
        latitude: Latitude in degrees (scalar or array)
        longitude: Longitude in degrees (scalar or array)

    Returns:
        Tuple of (solar_zenith, solar_azimuth) in degrees
    """
    import pvlib
    import pandas as pd

    # Parse time if string
    if isinstance(acquisition_time, str):
        acquisition_time = datetime.fromisoformat(
            acquisition_time.replace("Z", "+00:00")
        )

    # Convert to pandas timestamp for pvlib
    times = pd.DatetimeIndex([acquisition_time], tz="UTC")

    # Handle scalar vs array inputs
    lat_is_array = isinstance(latitude, np.ndarray)
    lon_is_array = isinstance(longitude, np.ndarray)

    if lat_is_array or lon_is_array:
        # For arrays, we need to compute for each pixel
        # pvlib expects scalar lat/lon, so we'll use vectorized approach
        latitude = np.atleast_1d(latitude)
        longitude = np.atleast_1d(longitude)

        # Get original shape for reshaping
        orig_shape = latitude.shape

        # Flatten for computation
        lat_flat = latitude.ravel()
        lon_flat = longitude.ravel()

        solar_zenith = np.zeros_like(lat_flat, dtype=np.float64)
        solar_azimuth = np.zeros_like(lat_flat, dtype=np.float64)

        # Compute for each unique location (or sample for large arrays)
        # For large arrays, we compute at scene center and apply uniformly
        # This is a simplification that's valid for small scenes
        if lat_flat.size > 10000:
            # Use scene center for large arrays
            center_lat = np.nanmean(lat_flat)
            center_lon = np.nanmean(lon_flat)
            solar_pos = pvlib.solarposition.get_solarposition(
                times, center_lat, center_lon
            )
            solar_zenith[:] = solar_pos["zenith"].values[0]
            solar_azimuth[:] = solar_pos["azimuth"].values[0]
            # Do not fabricate angles for pixels with missing geolocation:
            # keep NaN where lat/lon is NaN rather than assigning the scene
            # center value.
            invalid = np.isnan(lat_flat) | np.isnan(lon_flat)
            solar_zenith[invalid] = np.nan
            solar_azimuth[invalid] = np.nan
        else:
            # Compute per-pixel for smaller arrays
            for i, (lat, lon) in enumerate(zip(lat_flat, lon_flat)):
                if np.isnan(lat) or np.isnan(lon):
                    solar_zenith[i] = np.nan
                    solar_azimuth[i] = np.nan
                else:
                    solar_pos = pvlib.solarposition.get_solarposition(times, lat, lon)
                    solar_zenith[i] = solar_pos["zenith"].values[0]
                    solar_azimuth[i] = solar_pos["azimuth"].values[0]

        # Reshape to original shape
        solar_zenith = solar_zenith.reshape(orig_shape)
        solar_azimuth = solar_azimuth.reshape(orig_shape)
    else:
        # Scalar case
        solar_pos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
        solar_zenith = solar_pos["zenith"].values[0]
        solar_azimuth = solar_pos["azimuth"].values[0]

    return solar_zenith, solar_azimuth


def calculate_solar_geometry_fast(
    acquisition_time: Union[datetime, str],
    latitude: np.ndarray,
    longitude: np.ndarray,
    grid_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast solar geometry calculation using grid interpolation.

    Computes solar angles at a coarse grid and interpolates to full resolution.
    This is much faster for large arrays while maintaining good accuracy.

    Args:
        acquisition_time: UTC acquisition time
        latitude: 2D latitude array
        longitude: 2D longitude array
        grid_size: Number of grid points in each dimension

    Returns:
        Tuple of (solar_zenith, solar_azimuth) arrays in degrees
    """
    import pvlib
    import pandas as pd
    from scipy.interpolate import RegularGridInterpolator

    # Parse time if string
    if isinstance(acquisition_time, str):
        acquisition_time = datetime.fromisoformat(
            acquisition_time.replace("Z", "+00:00")
        )

    times = pd.DatetimeIndex([acquisition_time], tz="UTC")

    lines, samples = latitude.shape

    # Create coarse grid indices
    row_indices = np.linspace(0, lines - 1, min(grid_size, lines)).astype(int)
    col_indices = np.linspace(0, samples - 1, min(grid_size, samples)).astype(int)

    # Compute solar geometry at grid points
    grid_zenith = np.zeros((len(row_indices), len(col_indices)))
    grid_azimuth = np.zeros((len(row_indices), len(col_indices)))

    for i, row in enumerate(row_indices):
        for j, col in enumerate(col_indices):
            lat = latitude[row, col]
            lon = longitude[row, col]
            if np.isnan(lat) or np.isnan(lon):
                grid_zenith[i, j] = np.nan
                grid_azimuth[i, j] = np.nan
            else:
                solar_pos = pvlib.solarposition.get_solarposition(times, lat, lon)
                grid_zenith[i, j] = solar_pos["zenith"].values[0]
                grid_azimuth[i, j] = solar_pos["azimuth"].values[0]

    # Create interpolators
    zenith_interp = RegularGridInterpolator(
        (row_indices.astype(float), col_indices.astype(float)),
        grid_zenith,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    azimuth_interp = RegularGridInterpolator(
        (row_indices.astype(float), col_indices.astype(float)),
        grid_azimuth,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Create full resolution grid
    row_full = np.arange(lines)
    col_full = np.arange(samples)
    row_grid, col_grid = np.meshgrid(row_full, col_full, indexing="ij")
    points = np.stack([row_grid.ravel(), col_grid.ravel()], axis=-1)

    # Interpolate to full resolution
    solar_zenith = zenith_interp(points).reshape(lines, samples)
    solar_azimuth = azimuth_interp(points).reshape(lines, samples)

    return solar_zenith, solar_azimuth


def calculate_sensor_geometry(
    path_length_km: Union[float, np.ndarray],
    satellite_altitude_km: float = TANAGER_ALTITUDE_KM,
    satellite_heading_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate sensor zenith and azimuth angles.

    Tanager is approximately nadir-pointing, so sensor zenith is typically small.
    The sensor zenith is estimated from path length and satellite altitude.

    Args:
        path_length_km: Sensor-to-ground path length in km
        satellite_altitude_km: Satellite altitude in km
        satellite_heading_deg: Satellite heading (track azimuth) in degrees

    Returns:
        Tuple of (sensor_zenith, sensor_azimuth) in degrees
    """
    path_length_km = np.atleast_1d(path_length_km)

    # For nadir pointing, sensor zenith ≈ 0
    # Slight off-nadir angle can be estimated from path length vs altitude
    # path_length = altitude / cos(zenith)
    # zenith = arccos(altitude / path_length)

    # Handle edge cases where path_length < altitude (shouldn't happen)
    ratio = np.clip(satellite_altitude_km / path_length_km, -1, 1)
    sensor_zenith = np.degrees(np.arccos(ratio))

    # For polar-orbiting satellite, azimuth is approximately along-track
    # This is a simplification - actual azimuth depends on orbit geometry
    sensor_azimuth = np.full_like(sensor_zenith, satellite_heading_deg)

    return sensor_zenith, sensor_azimuth


def calculate_phase_angle(
    solar_zenith: Union[float, np.ndarray],
    solar_azimuth: Union[float, np.ndarray],
    sensor_zenith: Union[float, np.ndarray],
    sensor_azimuth: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Calculate the phase angle between sun, target, and sensor.

    Uses the spherical law of cosines.

    Args:
        solar_zenith: Solar zenith angle in degrees
        solar_azimuth: Solar azimuth angle in degrees
        sensor_zenith: Sensor zenith angle in degrees
        sensor_azimuth: Sensor azimuth angle in degrees

    Returns:
        Phase angle in degrees
    """
    # Convert to radians
    sz = np.radians(solar_zenith)
    sa = np.radians(solar_azimuth)
    vz = np.radians(sensor_zenith)
    va = np.radians(sensor_azimuth)

    # Spherical law of cosines
    cos_phase = np.cos(sz) * np.cos(vz) + np.sin(sz) * np.sin(vz) * np.cos(sa - va)

    # Clamp to valid range for arccos
    cos_phase = np.clip(cos_phase, -1, 1)

    phase_angle = np.degrees(np.arccos(cos_phase))

    return phase_angle


def calculate_cosine_i(
    solar_zenith: Union[float, np.ndarray],
    slope: Union[float, np.ndarray] = 0.0,
    aspect: Union[float, np.ndarray] = 0.0,
    solar_azimuth: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """
    Calculate cosine of illumination angle.

    For flat terrain (slope=0), this is simply cos(solar_zenith).

    Args:
        solar_zenith: Solar zenith angle in degrees
        slope: Terrain slope in degrees (0 for flat)
        aspect: Terrain aspect in degrees (0 for flat)
        solar_azimuth: Solar azimuth in degrees (needed if slope > 0)

    Returns:
        Cosine of illumination angle
    """
    solar_zenith = np.atleast_1d(solar_zenith)
    slope = np.atleast_1d(slope)

    if np.all(slope == 0):
        # Flat terrain - simple case
        return np.cos(np.radians(solar_zenith))

    # For sloped terrain, use full formula
    if solar_azimuth is None:
        raise ValueError("solar_azimuth required for non-flat terrain")

    aspect = np.atleast_1d(aspect)
    solar_azimuth = np.atleast_1d(solar_azimuth)

    # Convert to radians
    sz = np.radians(solar_zenith)
    sl = np.radians(slope)
    sa = np.radians(solar_azimuth)
    asp = np.radians(aspect)

    # Full cosine_i formula
    cosine_i = np.cos(sz) * np.cos(sl) + np.sin(sz) * np.sin(sl) * np.cos(sa - asp)

    return cosine_i


def parse_acquisition_time(strip_id: str) -> datetime:
    """
    Parse acquisition time from Tanager strip_id.

    Expected format: YYYYMMDD_HHMMSS_...
    Example: 20250511_074311_00_4001

    Args:
        strip_id: Tanager strip identifier

    Returns:
        UTC datetime
    """
    # Extract date and time parts
    parts = strip_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse strip_id: {strip_id}")

    date_str = parts[0]  # YYYYMMDD
    time_str = parts[1]  # HHMMSS

    # Parse components
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(time_str[:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])

    return datetime(year, month, day, hour, minute, second)


def time_to_decimal_hours(dt: datetime) -> float:
    """
    Convert datetime to decimal hours (UTC).

    Args:
        dt: Datetime object

    Returns:
        Time as decimal hours (0-24)
    """
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def estimate_satellite_heading(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> float:
    """
    Estimate satellite heading from lat/lon arrays.

    Assumes satellite is moving roughly along-track (first to last line).

    Args:
        latitude: 2D latitude array (lines x samples)
        longitude: 2D longitude array (lines x samples)

    Returns:
        Estimated heading in degrees (0 = North, 90 = East)
    """
    # Get start and end positions (center of first and last lines)
    mid_col = latitude.shape[1] // 2

    lat_start = latitude[0, mid_col]
    lon_start = longitude[0, mid_col]
    lat_end = latitude[-1, mid_col]
    lon_end = longitude[-1, mid_col]

    # Calculate heading using simple spherical approximation
    delta_lon = np.radians(lon_end - lon_start)
    lat1 = np.radians(lat_start)
    lat2 = np.radians(lat_end)

    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)

    heading = np.degrees(np.arctan2(x, y))

    # Normalize to 0-360
    heading = (heading + 360) % 360

    return heading


def create_observation_array(
    lines: int,
    samples: int,
    path_length_m: np.ndarray,
    acquisition_time: Union[datetime, str],
    latitude: np.ndarray,
    longitude: np.ndarray,
    elevation: Optional[np.ndarray] = None,
    slope: Optional[np.ndarray] = None,
    aspect: Optional[np.ndarray] = None,
    use_fast_solar: bool = True,
) -> np.ndarray:
    """
    Create the full observation array for ISOFIT (10 bands).

    Band layout:
        0: path_length (METERS - ISOFIT converts to km internally)
        1: to_sensor_azimuth (degrees)
        2: to_sensor_zenith (degrees)
        3: to_sun_azimuth (degrees)
        4: to_sun_zenith (degrees)
        5: phase_angle (degrees)
        6: slope (degrees)
        7: aspect (degrees)
        8: cosine_i
        9: utc_time (decimal hours)

    Args:
        lines: Number of lines
        samples: Number of samples
        path_length_m: Path length array in METERS (lines, samples)
        acquisition_time: UTC acquisition time
        latitude: Latitude array (lines, samples)
        longitude: Longitude array (lines, samples)
        elevation: Elevation array in meters (optional, default 0)
        slope: Slope array in degrees (optional, default 0)
        aspect: Aspect array in degrees (optional, default 0)
        use_fast_solar: Use fast grid-interpolated solar calculation

    Returns:
        Observation array (lines, samples, 10)
    """
    # Parse acquisition time
    if isinstance(acquisition_time, str):
        acquisition_time = datetime.fromisoformat(
            acquisition_time.replace("Z", "+00:00")
        )

    # Initialize output array
    obs = np.zeros((lines, samples, 10), dtype=np.float32)

    # Band 0: Path length in METERS (ISOFIT expects meters, converts to km internally)
    obs[:, :, 0] = path_length_m

    # Calculate sensor geometry (requires km, so convert)
    satellite_heading = estimate_satellite_heading(latitude, longitude)
    path_length_km = path_length_m / 1000.0  # Convert m to km for geometry calc
    sensor_zenith, sensor_azimuth = calculate_sensor_geometry(
        path_length_km, satellite_heading_deg=satellite_heading
    )
    obs[:, :, 1] = sensor_azimuth  # to_sensor_azimuth
    obs[:, :, 2] = sensor_zenith  # to_sensor_zenith

    # Calculate solar geometry
    if use_fast_solar:
        solar_zenith, solar_azimuth = calculate_solar_geometry_fast(
            acquisition_time, latitude, longitude
        )
    else:
        solar_zenith, solar_azimuth = calculate_solar_geometry(
            acquisition_time, latitude, longitude
        )

    obs[:, :, 3] = solar_azimuth  # to_sun_azimuth
    obs[:, :, 4] = solar_zenith  # to_sun_zenith

    # Band 5: Phase angle
    phase_angle = calculate_phase_angle(
        solar_zenith, solar_azimuth, sensor_zenith, sensor_azimuth
    )
    obs[:, :, 5] = phase_angle

    # Bands 6-7: Slope and aspect (default to flat terrain)
    if slope is None:
        slope = np.zeros((lines, samples), dtype=np.float32)
    if aspect is None:
        aspect = np.zeros((lines, samples), dtype=np.float32)

    obs[:, :, 6] = slope
    obs[:, :, 7] = aspect

    # Band 8: Cosine of illumination angle
    cosine_i = calculate_cosine_i(solar_zenith, slope, aspect, solar_azimuth)
    obs[:, :, 8] = cosine_i

    # Band 9: UTC time (decimal hours)
    utc_time = time_to_decimal_hours(acquisition_time)
    obs[:, :, 9] = utc_time

    return obs
