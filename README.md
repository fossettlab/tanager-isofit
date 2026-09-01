# Tanager ISOFIT Pipeline

Convert Planet Tanager hyperspectral HDF5 data (TOA radiance) to surface reflectance using ISOFIT atmospheric correction.

## What it does

- Converts Tanager HDF5 radiance files to ISOFIT-compatible ENVI format
- Reads sun/sensor angles directly from HDF5 metadata
- Runs atmospheric correction via sRTMnet (no MODTRAN license needed)
- Wraps ISOFIT with defaults tuned for Tanager data
- Compares results against NASA EMIT L2A for validation
- Provides a CLI for all operations

## Installation

### Basic installation

```bash
git clone https://github.com/bradleylab/tanager-isofit.git
cd tanager-isofit
pip install -e .
```

### With ISOFIT (for atmospheric correction)

```bash
pip install -e ".[isofit]"
```

### Development install (all extras)

```bash
pip install -e ".[all]"
```

## Prerequisites

If you only need HDF5-to-ENVI conversion, skip this section. For atmospheric correction, you'll need:

### 1. ISOFIT

```bash
pip install isofit>=3.4.0
```

### 2. 6S radiative transfer code

sRTMnet needs 6S, which requires a Fortran compiler.

```bash
# Install gfortran (if not already installed)
# Ubuntu/Debian:
sudo apt-get install gfortran

# macOS:
brew install gcc

# Then download 6S via ISOFIT
isofit download sixs
```

### 3. sRTMnet emulator

```bash
isofit download srtmnet
```

This downloads the emulator to `~/.isofit/srtmnet/sRTMnet_v120.h5`

### 4. Surface model

ISOFIT needs a surface reflectance prior. Two options:

**Option A: Use an existing model** (if wavelengths match):
```bash
# Download ISOFIT examples which include surface models
isofit download examples
# Models in: ~/.isofit/examples/*/surface/surface.mat
```

**Option B: Build a model for Tanager wavelengths** (recommended):
```bash
# First run conversion to get wavelength file
tanager-isofit convert input.h5 output/

# Copy and edit the example config
cp tanager_surface_config.json.example my_surface_config.json
# Edit my_surface_config.json - replace /home/USERNAME with your actual home dir

# Build surface model
isofit surface_model my_surface_config.json \
  --wavelength_path output/wavelengths.txt
```

Note: ISOFIT doesn't expand `~` in JSON configs, so you must use absolute paths like `/home/jane/.isofit/...` in the config file. See `tanager_surface_config.json.example` for details.

### 5. ISOFIT data files

```bash
isofit download data
```

### Verify installation

```bash
# Quick check of all dependencies
tanager-isofit check-deps
```

Or manually:
```bash
ls ~/.isofit/srtmnet/sRTMnet_v120.h5   # sRTMnet emulator
ls ~/.isofit/sixs/sixs                  # 6S executable
ls ~/.isofit/data/                      # Aerosol models, noise files
```

## Quick Start

### Convert Only (No ISOFIT)

Convert Tanager HDF5 to ENVI format without atmospheric correction:

```bash
tanager-isofit convert input.h5 output/
```

Output files:
- `output/radiance` - TOA radiance ENVI file
- `output/loc` - Location (lon, lat, elevation)
- `output/obs` - Observation geometry
- `output/wavelengths.txt` - Wavelength table

### Full Pipeline (Convert + ISOFIT)

Run complete atmospheric correction:

```bash
tanager-isofit process input.h5 output/ \
  --surface-path ~/.isofit/tanager_surface.mat \
  --n-cores 4
```

Output reflectance will be in `output/isofit_working/output/`

### Process a Subset

For testing or memory-constrained systems:

```bash
# Process 100x100 pixel subset
tanager-isofit process input.h5 output/ \
  --subset "0,100,0,100" \
  --surface-path ~/.isofit/tanager_surface.mat \
  --n-cores 2
```

### Skip Empirical Line

For faster processing (less accurate for inhomogeneous scenes):

```bash
tanager-isofit process input.h5 output/ \
  --no-empirical-line \
  --surface-path ~/.isofit/tanager_surface.mat
```

## CLI Reference

### `tanager-isofit inspect`

Examine HDF5 file structure and validate Tanager format.

```bash
tanager-isofit inspect input.h5
```

### `tanager-isofit convert`

Convert HDF5 to ENVI format without ISOFIT processing.

```bash
tanager-isofit convert INPUT_H5 OUTPUT_DIR [OPTIONS]

Options:
  -s, --subset TEXT       Subset as 'row_start,row_end,col_start,col_end'
  --fast-solar/--no-fast-solar  Use fast grid-interpolated solar geometry (default: on)
```

### `tanager-isofit process`

Run full pipeline: conversion + ISOFIT atmospheric correction.

```bash
tanager-isofit process INPUT_H5 OUTPUT_DIR [OPTIONS]

Options:
  -n, --n-cores INTEGER   Number of CPU cores for ISOFIT (default: 4)
  --empirical-line/--no-empirical-line  Use empirical line correction (default: on)
  -s, --subset TEXT       Subset as 'row_start,row_end,col_start,col_end'
  --skip-isofit           Only convert, skip ISOFIT processing
  --sensor TEXT           ISOFIT sensor configuration name (default: tanager)
  --surface-path PATH     Path to surface model file (.mat) - REQUIRED
  --emulator-base PATH    Path to sRTMnet .h5 file ('auto' uses default, 'none' for MODTRAN)
```

### `tanager-isofit check`

Validate ENVI file and print statistics.

```bash
tanager-isofit check output/radiance
```

### `tanager-isofit check-deps`

Verify all dependencies (ISOFIT, sRTMnet) are installed.

```bash
tanager-isofit check-deps
```

### `tanager-isofit find-emit`

Search for coincident NASA EMIT data.

```bash
tanager-isofit find-emit input.h5 --time-window 24
```

### `tanager-isofit validate`

Compare Tanager reflectance against EMIT L2A.

```bash
tanager-isofit validate tanager_refl emit_l2a.nc --output report.html
```

## How it works

### 1. HDF5 to ENVI conversion

```
Tanager HDF5 → Radiance + Location + Observation (ENVI format)
```

- Extracts TOA radiance from `/HDFEOS/SWATHS/HYP/Data Fields/toa_radiance`
- Extracts geolocation (lat/lon) from `/HDFEOS/SWATHS/HYP/Geolocation Fields/`
- Computes observation geometry (solar/sensor angles, path length)
- Writes ISOFIT-compatible ENVI files with proper headers

### 2. ISOFIT atmospheric correction

```
Radiance + Atmosphere Model → Surface Reflectance
```

- Builds look-up tables (LUTs) using 6S radiative transfer
- Uses sRTMnet neural network to emulate atmospheric properties
- Performs optimal estimation inversion per pixel
- Retrieves surface reflectance and atmospheric state

### 3. Output products

| File | Description |
|------|-------------|
| `*_rfl` | Surface reflectance (0-1 scale) |
| `*_state` | Retrieved atmospheric state (H2O, AOD) |
| `*_uncert` | Uncertainty estimates |

## Python API

```python
from tanager_isofit import convert_tanager_to_envi, run_isofit_pipeline

# Convert only
result = convert_tanager_to_envi(
    "input.h5",
    "output/",
    subset=(0, 100, 0, 100)  # Optional subset
)

# Full pipeline
result = run_isofit_pipeline(
    "input.h5",
    "output/",
    n_cores=4,
    empirical_line=True,
    surface_path="/path/to/surface.mat",
    emulator_base="auto"  # Uses default sRTMnet location
)

# Access outputs
print(result['isofit_outputs']['rfl'])  # Reflectance path
```

## Output files

### Conversion output

| File | Bands | Description |
|------|-------|-------------|
| `radiance` | 426 | TOA radiance (uW/cm^2/sr/nm) |
| `loc` | 3 | Longitude, Latitude, Elevation (deg, deg, m) |
| `obs` | 10 | Observation geometry |
| `wavelengths.txt` | - | Wavelength table (index, center, FWHM) |

### Observation file bands

| Band | Name | Units |
|------|------|-------|
| 0 | path_length | meters |
| 1 | to_sensor_azimuth | degrees |
| 2 | to_sensor_zenith | degrees |
| 3 | to_sun_azimuth | degrees |
| 4 | to_sun_zenith | degrees |
| 5 | phase_angle | degrees |
| 6 | slope | degrees |
| 7 | aspect | degrees |
| 8 | cosine_i | - |
| 9 | utc_time | decimal hours |

## Tanager HDF5 structure

Expected HDF5 layout:

```
/HDFEOS/SWATHS/HYP/
├── Data Fields/
│   ├── toa_radiance              (426, lines, samples)
│   ├── sensor_to_ground_path_length
│   ├── sensor_zenith
│   ├── sensor_azimuth
│   ├── sun_zenith
│   └── sun_azimuth
└── Geolocation Fields/
    ├── Latitude
    ├── Longitude
    └── Time
```

Wavelengths and FWHM are read from `toa_radiance` dataset attributes.

## Limitations

- Assumes flat terrain (slope=0, aspect=0). Works fine for water and coastal scenes.
- Surface model wavelengths must match Tanager's 426 bands.
- Large scenes need lots of RAM. Use `--subset` if you're running out.
- Full scenes take hours. Grab coffee.

See [ISSUES.md](tanager-isofit-ISSUES.md) for troubleshooting.

## Test data

Sample scene from Planet's open data program:

- Abu Dhabi coastal scene: `20250511_074311_00_4001_basic_radiance.h5` (558 MB)
- URL: https://storage.googleapis.com/open-cogs/planet-stac/release1-basic-radiance/20250511_074311_00_4001_basic_radiance.h5

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_convert.py -v
```

## Package structure

```
tanager_isofit/
├── __init__.py          # Package init and exports
├── config.py            # Constants, HDF5 paths, sensor specs
├── utils.py             # ENVI I/O helpers
├── convert.py           # HDF5 → ENVI conversion
├── geometry.py          # Solar/sensor geometry calculations
├── dem.py               # Elevation utilities
├── isofit_runner.py     # ISOFIT wrapper
├── validate.py          # EMIT comparison & metrics
└── cli.py               # Command-line interface
```

## License

MIT License

## Acknowledgments

- [ISOFIT](https://github.com/isofit/isofit) - Imaging Spectrometer Optimal FITting
- [Planet](https://www.planet.com/) - Tanager hyperspectral satellite
- [NASA EMIT](https://earth.jpl.nasa.gov/emit/) - Earth Surface Mineral Dust Source Investigation
- [6S](http://6s.ltdri.org/) - Second Simulation of the Satellite Signal in the Solar Spectrum
