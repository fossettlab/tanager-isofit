"""
Digital Elevation Model (DEM) utilities for terrain data.

For v1, this module provides simplified elevation handling:
- Default elevation of 0m (sea level) for coastal/water scenes
- Future support for DEM queries (SRTM, Copernicus DEM)
"""

from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np


def get_elevation_constant(
    lines: int,
    samples: int,
    elevation_m: float = 0.0,
) -> np.ndarray:
    """
    Create a constant elevation array.

    This is the simplest elevation model, suitable for:
    - Coastal/ocean scenes
    - Initial testing
    - Cases where terrain effects are negligible

    Args:
        lines: Number of lines in the image
        samples: Number of samples per line
        elevation_m: Constant elevation in meters (default: sea level)

    Returns:
        2D elevation array (lines, samples) in meters
    """
    return np.full((lines, samples), elevation_m, dtype=np.float32)


def get_elevation_from_dem(
    latitude: np.ndarray,
    longitude: np.ndarray,
    dem_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Get elevation values from a DEM file.

    Currently a placeholder for future DEM integration.
    Falls back to constant elevation if DEM not available.

    Args:
        latitude: 2D latitude array
        longitude: 2D longitude array
        dem_path: Path to DEM file (GeoTIFF or similar)

    Returns:
        2D elevation array in meters
    """
    if dem_path is None:
        # Return sea level
        return get_elevation_constant(latitude.shape[0], latitude.shape[1], 0.0)

    dem_path = Path(dem_path)

    if not dem_path.exists():
        print(f"Warning: DEM file not found: {dem_path}, using sea level")
        return get_elevation_constant(latitude.shape[0], latitude.shape[1], 0.0)

    try:
        import rioxarray as rxr

        # Open DEM
        dem = rxr.open_rasterio(dem_path)

        # Sample DEM at lat/lon points
        # This requires the DEM to be in a geographic CRS
        lines, samples = latitude.shape
        elevation = np.zeros((lines, samples), dtype=np.float32)

        # Get DEM bounds
        dem_bounds = dem.rio.bounds()

        for i in range(lines):
            for j in range(samples):
                lat = latitude[i, j]
                lon = longitude[i, j]

                # Check if point is within DEM bounds
                if (
                    dem_bounds[0] <= lon <= dem_bounds[2]
                    and dem_bounds[1] <= lat <= dem_bounds[3]
                ):
                    # Sample DEM at this location
                    try:
                        val = dem.sel(x=lon, y=lat, method="nearest").values
                        elevation[i, j] = val[0] if val.size > 0 else 0.0
                    except Exception:
                        elevation[i, j] = 0.0
                else:
                    elevation[i, j] = 0.0

        return elevation

    except ImportError:
        print("Warning: rioxarray not installed, using sea level elevation")
        return get_elevation_constant(latitude.shape[0], latitude.shape[1], 0.0)
    except Exception as e:
        print(f"Warning: Error reading DEM: {e}, using sea level")
        return get_elevation_constant(latitude.shape[0], latitude.shape[1], 0.0)


def get_elevation_from_opentopo(
    latitude: np.ndarray,
    longitude: np.ndarray,
    api_key: Optional[str] = None,
) -> np.ndarray:
    """
    Query elevation from OpenTopography API.

    This is a placeholder for future cloud DEM integration.
    Falls back to constant elevation.

    Args:
        latitude: 2D latitude array
        longitude: 2D longitude array
        api_key: OpenTopography API key (optional)

    Returns:
        2D elevation array in meters
    """
    # Placeholder - would need API key and rate limiting
    print("Warning: OpenTopography query not implemented, using sea level")
    return get_elevation_constant(latitude.shape[0], latitude.shape[1], 0.0)


def compute_slope_aspect(
    elevation: np.ndarray,
    pixel_size_m: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope and aspect from elevation data.

    Uses a 3x3 window gradient calculation.

    Args:
        elevation: 2D elevation array in meters
        pixel_size_m: Pixel size in meters (for gradient calculation)

    Returns:
        Tuple of (slope, aspect) arrays in degrees
    """
    # Compute gradients
    dy, dx = np.gradient(elevation, pixel_size_m)

    # Compute slope
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope = np.degrees(slope_rad)

    # Compute aspect (direction of steepest descent)
    aspect_rad = np.arctan2(-dx, dy)  # Note: aspect convention varies
    aspect = np.degrees(aspect_rad)
    aspect = (aspect + 360) % 360  # Normalize to 0-360

    return slope.astype(np.float32), aspect.astype(np.float32)


def get_flat_terrain(
    lines: int,
    samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get flat terrain arrays (elevation=0, slope=0, aspect=0).

    This is the default for v1 which assumes flat terrain.

    Args:
        lines: Number of lines
        samples: Number of samples

    Returns:
        Tuple of (elevation, slope, aspect) arrays
    """
    elevation = get_elevation_constant(lines, samples, 0.0)
    slope = np.zeros((lines, samples), dtype=np.float32)
    aspect = np.zeros((lines, samples), dtype=np.float32)

    return elevation, slope, aspect


def get_mean_elevation(elevation: np.ndarray) -> float:
    """
    Get mean elevation from array, ignoring NaN values.

    Args:
        elevation: Elevation array

    Returns:
        Mean elevation in meters
    """
    return float(np.nanmean(elevation))
