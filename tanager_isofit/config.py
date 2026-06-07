"""
Configuration constants and default paths for Tanager ISOFIT pipeline.
"""

from pathlib import Path
from typing import Dict, Any

# ============================================================================
# TANAGER SENSOR SPECIFICATIONS
# ============================================================================

# Tanager satellite specifications
TANAGER_ALTITUDE_KM = 500.0  # Nominal satellite altitude in km

# ============================================================================
# UNIT CONVERSION CONSTANTS
# ============================================================================

# Radiance unit conversion from Tanager to ISOFIT
# Tanager radiance units: W/(m² sr µm)
# ISOFIT radiance units: µW/(cm² sr nm)
# Conversion: W/(m² sr µm) * 0.1 = µW/(cm² sr nm)
#   W → µW: multiply by 10^6
#   m² → cm²: divide by 10^4
#   µm → nm: divide by 10^3
#   Net: 10^6 / 10^4 / 10^3 = 0.1
RADIANCE_CONVERSION_FACTOR = 0.1  # Tanager to ISOFIT radiance units
TANAGER_NUM_BANDS = 426  # Number of spectral bands
TANAGER_WAVELENGTH_MIN_NM = 400  # Minimum wavelength (nm)
TANAGER_WAVELENGTH_MAX_NM = 2500  # Maximum wavelength (nm)

# Estimated spectral properties (fallback if not in HDF5 metadata)
# Approximate wavelength spacing: (2500-400)/426 ≈ 4.93 nm
TANAGER_WAVELENGTH_SPACING_NM = (
    TANAGER_WAVELENGTH_MAX_NM - TANAGER_WAVELENGTH_MIN_NM
) / TANAGER_NUM_BANDS

# ============================================================================
# HDF5 PATH CONSTANTS
# ============================================================================

# Base paths in HDF5 structure (note: actual files use spaces, not underscores)
HDF5_SWATH_BASE = "/HDFEOS/SWATHS/HYP"
HDF5_DATA_FIELDS = f"{HDF5_SWATH_BASE}/Data Fields"
HDF5_GEO_FIELDS = f"{HDF5_SWATH_BASE}/Geolocation Fields"

# Data field paths
HDF5_RADIANCE_PATH = f"{HDF5_DATA_FIELDS}/toa_radiance"
HDF5_PATH_LENGTH_PATH = f"{HDF5_DATA_FIELDS}/sensor_to_ground_path_length"
HDF5_CIRRUS_MASK_PATH = f"{HDF5_DATA_FIELDS}/beta_cirrus_mask"
HDF5_SENSOR_ZENITH_PATH = f"{HDF5_DATA_FIELDS}/sensor_zenith"
HDF5_SENSOR_AZIMUTH_PATH = f"{HDF5_DATA_FIELDS}/sensor_azimuth"
HDF5_SUN_ZENITH_PATH = f"{HDF5_DATA_FIELDS}/sun_zenith"
HDF5_SUN_AZIMUTH_PATH = f"{HDF5_DATA_FIELDS}/sun_azimuth"

# Geolocation field paths (note: capitalized in actual files)
HDF5_LATITUDE_PATH = f"{HDF5_GEO_FIELDS}/Latitude"
HDF5_LONGITUDE_PATH = f"{HDF5_GEO_FIELDS}/Longitude"
HDF5_TIME_PATH = f"{HDF5_GEO_FIELDS}/Time"

# Alternative paths to try if primary paths don't exist
HDF5_ALT_PATHS = {
    "radiance": [
        f"{HDF5_DATA_FIELDS}/toa_radiance",
        f"{HDF5_SWATH_BASE}/Data_Fields/toa_radiance",
        "/toa_radiance",
    ],
    "latitude": [
        f"{HDF5_GEO_FIELDS}/Latitude",
        f"{HDF5_GEO_FIELDS}/latitude",
        f"{HDF5_SWATH_BASE}/Geolocation_Fields/Latitude",
    ],
    "longitude": [
        f"{HDF5_GEO_FIELDS}/Longitude",
        f"{HDF5_GEO_FIELDS}/longitude",
        f"{HDF5_SWATH_BASE}/Geolocation_Fields/Longitude",
    ],
    "path_length": [
        f"{HDF5_DATA_FIELDS}/sensor_to_ground_path_length",
        f"{HDF5_SWATH_BASE}/Data_Fields/sensor_to_ground_path_length",
    ],
    "sensor_zenith": [
        f"{HDF5_DATA_FIELDS}/sensor_zenith",
    ],
    "sensor_azimuth": [
        f"{HDF5_DATA_FIELDS}/sensor_azimuth",
    ],
    "sun_zenith": [
        f"{HDF5_DATA_FIELDS}/sun_zenith",
    ],
    "sun_azimuth": [
        f"{HDF5_DATA_FIELDS}/sun_azimuth",
    ],
}

# Attribute names for wavelength data
WAVELENGTH_ATTR_NAMES = [
    "wavelengths",
    "wavelength",
    "center_wavelengths",
    "band_centers",
]

FWHM_ATTR_NAMES = [
    "fwhm",
    "FWHM",
    "bandwidth",
    "band_widths",
]

# Attribute names that declare a fill / no-data sentinel on the radiance
# dataset. A declared fill value must never be scaled and passed to ISOFIT as
# if it were real radiance.
FILL_VALUE_ATTR_NAMES = [
    "_FillValue",
    "fill_value",
    "missing_value",
    "no_data_value",
    "nodata",
]

# ============================================================================
# ENVI FILE CONFIGURATION
# ============================================================================

# ENVI data types
ENVI_DTYPE_FLOAT32 = 4  # 32-bit floating point

# Standard ENVI file names for ISOFIT
ENVI_RADIANCE_FILENAME = "radiance"
ENVI_LOCATION_FILENAME = "loc"
ENVI_OBSERVATION_FILENAME = "obs"

# Observation file band definitions (10 bands as required by ISOFIT)
OBS_BAND_NAMES = [
    "path_length",  # Band 0: path length (km)
    "to_sensor_azimuth",  # Band 1: sensor azimuth (degrees)
    "to_sensor_zenith",  # Band 2: sensor zenith (degrees)
    "to_sun_azimuth",  # Band 3: solar azimuth (degrees)
    "to_sun_zenith",  # Band 4: solar zenith (degrees)
    "phase_angle",  # Band 5: phase angle (degrees)
    "slope",  # Band 6: terrain slope (degrees)
    "aspect",  # Band 7: terrain aspect (degrees)
    "cosine_i",  # Band 8: cosine of illumination angle
    "utc_time",  # Band 9: UTC time (decimal hours)
]

# Location file band definitions (3 bands)
LOC_BAND_NAMES = [
    "longitude",  # Band 0: longitude (degrees)
    "latitude",  # Band 1: latitude (degrees)
    "elevation",  # Band 2: elevation (meters)
]

# ============================================================================
# ISOFIT CONFIGURATION
# ============================================================================

# Default ISOFIT sensor name
ISOFIT_SENSOR = "tanager"

# Fallback sensors if tanager config not available
ISOFIT_FALLBACK_SENSORS = ["emit", "aviris_ng"]

# Default number of processing cores
DEFAULT_N_CORES = 4

# Default surface model
DEFAULT_SURFACE_MODEL = "multicomponent_surface"

# sRTMnet emulator path (neural network alternative to MODTRAN)
# Default location follows ISOFIT conventions
DEFAULT_SRTMNET_PATH = Path.home() / ".isofit" / "srtmnet" / "sRTMnet_v120.h5"

# Default reflectance library for surface model generation
DEFAULT_REFLECTANCE_LIBRARY = (
    Path.home() / ".isofit" / "data" / "reflectance" / "surface_model_ucsb"
)

# Surface model configuration for auto-generation
SURFACE_MODEL_CONFIG = {
    "normalize": "Euclidean",
    "n_components": 4,
    "reference_windows": [[400, 1250], [1450, 1800], [2050, 2450]],
    "windows": [
        {"interval": [350, 1330], "regularizer": 1e-6, "correlation": "EM"},
        {"interval": [1330, 1450], "regularizer": 10, "correlation": "decorrelated"},
        {"interval": [1450, 1800], "regularizer": 1e-6, "correlation": "EM"},
        {"interval": [1800, 1950], "regularizer": 10, "correlation": "decorrelated"},
        {"interval": [1950, 2500], "regularizer": 1e-6, "correlation": "EM"},
    ],
}

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

# Physical range for surface reflectance
REFLECTANCE_MIN = -0.05  # Allow slight negative due to noise
REFLECTANCE_MAX = 1.5  # Allow slightly > 1 for bright targets

# Water pixel detection threshold (NDWI > threshold)
WATER_NDWI_THRESHOLD = 0.0

# Validation metrics thresholds
VALIDATION_RMSE_TARGET = 0.02  # Target RMSE vs EMIT for water
VALIDATION_CORRELATION_TARGET = 0.9  # Target correlation for clear sky

# ============================================================================
# DEFAULT PATHS
# ============================================================================

# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_TEST_SCENES_DIR = DEFAULT_DATA_DIR / "test_scenes"

# ISOFIT working directory name
ISOFIT_WORKING_DIR_NAME = "isofit_working"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_default_wavelengths() -> list:
    """
    Generate default wavelength array for Tanager sensor.

    Returns:
        List of wavelengths in nm from 400 to 2500 nm (426 bands).
    """
    import numpy as np

    return np.linspace(
        TANAGER_WAVELENGTH_MIN_NM, TANAGER_WAVELENGTH_MAX_NM, TANAGER_NUM_BANDS
    ).tolist()


def get_default_fwhm() -> list:
    """
    Generate default FWHM array for Tanager sensor.

    Returns:
        List of FWHM values in nm (constant ~5nm spacing).
    """
    # Approximate FWHM as slightly larger than band spacing
    fwhm_value = TANAGER_WAVELENGTH_SPACING_NM * 1.2
    return [fwhm_value] * TANAGER_NUM_BANDS


def get_isofit_config_template() -> Dict[str, Any]:
    """
    Get default ISOFIT configuration template for Tanager.

    Returns:
        Dictionary with ISOFIT configuration defaults.
    """
    return {
        "forward_model": {
            "instrument": {
                "wavelength_file": None,  # Set during pipeline run
                "integrations": 1,
                "unknowns": {
                    "channelized_radiometric_uncertainty": 0.01,
                },
            },
            "radiative_transfer": {
                "radiative_transfer_engines": {
                    "vswir": {
                        "engine_name": "sRTMnet",
                        "aerosol_model_file": None,
                        "aerosol_template_file": None,
                    }
                },
            },
            "surface": {
                "surface_category": "multicomponent_surface",
            },
        },
        "implementation": {
            "inversion": {
                "windows": [[400, 1330], [1450, 1780], [2050, 2450]],
            },
            "n_cores": DEFAULT_N_CORES,
        },
    }
