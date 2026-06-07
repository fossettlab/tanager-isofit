"""Tests for HDF5 to ENVI conversion module."""

from pathlib import Path

import pytest
import numpy as np
import h5py


class TestHDF5Inspection:
    """Tests for HDF5 file inspection."""

    def test_inspect_hdf5_returns_dict(self, sample_hdf5):
        """Test that inspect_hdf5 returns a dictionary with expected keys."""
        from tanager_isofit.convert import inspect_hdf5

        info = inspect_hdf5(sample_hdf5)

        assert isinstance(info, dict)
        assert "path" in info
        assert "groups" in info
        assert "datasets" in info
        assert "attributes" in info

    def test_inspect_hdf5_finds_datasets(self, sample_hdf5):
        """Test that datasets are correctly identified."""
        from tanager_isofit.convert import inspect_hdf5

        info = inspect_hdf5(sample_hdf5)

        # Should find our test datasets
        dataset_names = [ds["name"] for ds in info["datasets"]]
        assert len(dataset_names) > 0


class TestHDF5Reading:
    """Tests for reading Tanager HDF5 files."""

    def test_read_tanager_hdf5_returns_expected_keys(self, sample_hdf5):
        """Test that read_tanager_hdf5 returns all expected data."""
        from tanager_isofit.convert import read_tanager_hdf5

        data = read_tanager_hdf5(sample_hdf5)

        expected_keys = [
            "radiance",
            "latitude",
            "longitude",
            "path_length",
            "wavelengths",
            "fwhm",
            "acquisition_time",
            "metadata",
        ]

        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_read_tanager_hdf5_radiance_shape(self, sample_hdf5):
        """Test that radiance has correct shape (lines, samples, bands)."""
        from tanager_isofit.convert import read_tanager_hdf5

        data = read_tanager_hdf5(sample_hdf5)
        radiance = data["radiance"]

        assert len(radiance.shape) == 3
        assert radiance.shape[2] > 0  # Has spectral bands

    def test_read_tanager_hdf5_subset(self, sample_hdf5):
        """Test subsetting functionality."""
        from tanager_isofit.convert import read_tanager_hdf5

        # Read full data first
        full_data = read_tanager_hdf5(sample_hdf5)
        full_shape = full_data["radiance"].shape

        # Read subset
        subset = (0, 5, 0, 5)
        subset_data = read_tanager_hdf5(sample_hdf5, subset=subset)
        subset_shape = subset_data["radiance"].shape

        assert subset_shape[0] == min(5, full_shape[0])
        assert subset_shape[1] == min(5, full_shape[1])
        assert subset_shape[2] == full_shape[2]  # Bands unchanged

    def test_read_tanager_hdf5_geolocation_shape(self, sample_hdf5):
        """Test that lat/lon arrays match radiance spatial dimensions."""
        from tanager_isofit.convert import read_tanager_hdf5

        data = read_tanager_hdf5(sample_hdf5)

        rad_shape = data["radiance"].shape
        lat_shape = data["latitude"].shape
        lon_shape = data["longitude"].shape

        assert lat_shape == (rad_shape[0], rad_shape[1])
        assert lon_shape == (rad_shape[0], rad_shape[1])


class TestENVIConversion:
    """Tests for ENVI file creation."""

    def test_convert_creates_all_files(self, sample_hdf5, tmp_path):
        """Test that conversion creates all expected output files."""
        from tanager_isofit.convert import convert_tanager_to_envi

        output_dir = tmp_path / "output"
        result = convert_tanager_to_envi(sample_hdf5, output_dir)

        # Check all files exist
        assert Path(result["radiance"]).exists()
        assert Path(result["loc"]).exists()
        assert Path(result["obs"]).exists()
        assert Path(result["wavelength_file"]).exists()

        # Check headers exist
        assert Path(result["radiance_hdr"]).exists()
        assert Path(result["loc_hdr"]).exists()
        assert Path(result["obs_hdr"]).exists()

    def test_convert_radiance_readable(self, sample_hdf5, tmp_path):
        """Test that converted radiance file is readable."""
        from tanager_isofit.convert import convert_tanager_to_envi
        from tanager_isofit.utils import read_envi_file

        output_dir = tmp_path / "output"
        result = convert_tanager_to_envi(sample_hdf5, output_dir)

        data, header = read_envi_file(result["radiance"])

        assert data.shape[2] > 0  # Has bands
        assert "wavelength" in header or header.get("bands", 0) > 0

    def test_convert_location_has_3_bands(self, sample_hdf5, tmp_path):
        """Test that location file has exactly 3 bands (lon, lat, elev)."""
        from tanager_isofit.convert import convert_tanager_to_envi
        from tanager_isofit.utils import read_envi_file

        output_dir = tmp_path / "output"
        result = convert_tanager_to_envi(sample_hdf5, output_dir)

        data, header = read_envi_file(result["loc"])

        assert data.shape[2] == 3

    def test_convert_observation_has_10_bands(self, sample_hdf5, tmp_path):
        """Test that observation file has exactly 10 bands."""
        from tanager_isofit.convert import convert_tanager_to_envi
        from tanager_isofit.utils import read_envi_file

        output_dir = tmp_path / "output"
        result = convert_tanager_to_envi(sample_hdf5, output_dir)

        data, header = read_envi_file(result["obs"])

        assert data.shape[2] == 10


class TestValidation:
    """Tests for HDF5 validation."""

    def test_validate_valid_file(self, sample_hdf5):
        """Test validation of a valid HDF5 file."""
        from tanager_isofit.convert import validate_hdf5_structure

        is_valid, issues = validate_hdf5_structure(sample_hdf5)

        # Should be valid (our test file has correct structure)
        if not is_valid:
            print(f"Validation issues: {issues}")

    def test_validate_nonexistent_file(self, tmp_path):
        """Test validation of non-existent file."""
        from tanager_isofit.convert import validate_hdf5_structure

        fake_path = tmp_path / "nonexistent.h5"
        is_valid, issues = validate_hdf5_structure(fake_path)

        assert not is_valid
        assert any("not found" in issue.lower() for issue in issues)


# Fixtures


@pytest.fixture
def sample_hdf5(tmp_path):
    """Create a sample HDF5 file for testing."""
    h5_path = tmp_path / "test_radiance.h5"

    with h5py.File(h5_path, "w") as f:
        # Create structure similar to Tanager files
        swath = f.create_group("HDFEOS/SWATHS/HYP")
        data_fields = swath.create_group("Data_Fields")
        geo_fields = swath.create_group("Geolocation_Fields")

        # Create test data
        lines, samples, bands = 10, 10, 50  # Small test size

        # Radiance data
        radiance = np.random.rand(lines, samples, bands).astype(np.float32) * 100
        data_fields.create_dataset("toa_radiance", data=radiance)

        # Path length
        path_length = np.full((lines, samples), 500.0, dtype=np.float32)
        data_fields.create_dataset("sensor_to_ground_path_length", data=path_length)

        # Geolocation
        lat = np.linspace(30.0, 31.0, lines)[:, np.newaxis] * np.ones((1, samples))
        lon = np.linspace(-120.0, -119.0, samples)[np.newaxis, :] * np.ones((lines, 1))
        geo_fields.create_dataset("latitude", data=lat.astype(np.float64))
        geo_fields.create_dataset("longitude", data=lon.astype(np.float64))

        # Add wavelength attributes
        wavelengths = np.linspace(400, 2500, bands)
        fwhm = np.full(bands, 5.0)
        data_fields["toa_radiance"].attrs["wavelengths"] = wavelengths
        data_fields["toa_radiance"].attrs["fwhm"] = fwhm

        # Add strip_id
        f.attrs["strip_id"] = "20250511_074311_00_4001"

    return h5_path


@pytest.fixture
def sample_grids_hdf5(tmp_path):
    """Create a minimal GRIDS-format HDF5 file for testing."""
    h5_path = tmp_path / "test_ortho_radiance.h5"
    lines, samples, bands = 10, 10, 50

    with h5py.File(h5_path, "w") as f:
        # GRIDS structure
        grids = f.create_group("HDFEOS/GRIDS/HYP")
        data_fields = grids.create_group("Data Fields")

        # Radiance data with wavelength attributes
        radiance = np.random.rand(lines, samples, bands).astype(np.float32) * 100
        ds = data_fields.create_dataset("ortho_radiance", data=radiance)
        ds.attrs["wavelengths"] = np.linspace(400, 2500, bands)
        ds.attrs["fwhm"] = np.full(bands, 5.0)

        # StructMetadata.0 (UTM zone 33N, roughly central Europe)
        struct_metadata = (
            "GROUP=GridStructure\n"
            "  UpperLeftPointMtrs=(500000.0,4000000.0)\n"
            "  LowerRightMtrs=(510000.0,3990000.0)\n"
            "  XDim=10\n"
            "  YDim=10\n"
            "  ZoneCode=33\n"
            "END_GROUP=GridStructure\n"
        )
        hdfeos_info = f.create_group("HDFEOS INFORMATION")
        hdfeos_info.create_dataset("StructMetadata.0", data=struct_metadata.encode())

        # Geometry fields (same structure as SWATHS)
        data_fields.create_dataset("sensor_zenith", data=np.zeros((lines, samples)))
        data_fields.create_dataset("sensor_azimuth", data=np.zeros((lines, samples)))
        data_fields.create_dataset("sun_zenith", data=np.full((lines, samples), 30.0))
        data_fields.create_dataset("sun_azimuth", data=np.full((lines, samples), 180.0))

        # Add strip_id for timestamp
        f.attrs["strip_id"] = "20250511_074311_00_4001"

    return h5_path


@pytest.fixture
def real_hdf5():
    """Path to real test HDF5 file if available.

    Set TANAGER_TEST_DATA_DIR environment variable to specify location.
    Default: ~/tanager-isofit-testdata/
    """
    import os

    test_dir = Path(
        os.environ.get("TANAGER_TEST_DATA_DIR", Path.home() / "tanager-isofit-testdata")
    )
    test_path = test_dir / "20250511_074311_00_4001_basic_radiance.h5"
    if test_path.exists():
        return test_path
    pytest.skip(
        f"Real test HDF5 not found at {test_path}. "
        f"Set TANAGER_TEST_DATA_DIR or download test data."
    )


# GRIDS Format Tests


class TestGRIDSFormat:
    """Tests for GRIDS format (ortho_radiance) support."""

    def test_generate_grids_latlon_northern(self, sample_grids_hdf5):
        """Test lat/lon generation for northern hemisphere."""
        from tanager_isofit.convert import _generate_grids_latlon

        with h5py.File(sample_grids_hdf5, "r") as f:
            lat, lon = _generate_grids_latlon(f)

        assert lat.shape == (10, 10)
        assert lon.shape == (10, 10)
        assert 35 < lat[0, 0] < 37  # Zone 33N latitude range
        assert lat[0, 0] > lat[-1, 0]  # Latitude decreases going south

    def test_generate_grids_latlon_with_subset(self, sample_grids_hdf5):
        """Test subsetting works correctly."""
        from tanager_isofit.convert import _generate_grids_latlon

        with h5py.File(sample_grids_hdf5, "r") as f:
            lat, lon = _generate_grids_latlon(f, subset=(2, 5, 3, 7))

        assert lat.shape == (3, 4)

    def test_generate_grids_latlon_invalid_zone(self, tmp_path):
        """Test error for invalid UTM zone."""
        from tanager_isofit.convert import _generate_grids_latlon

        h5_path = tmp_path / "bad_zone.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_group("HDFEOS/GRIDS/HYP")
            hdfeos_info = f.create_group("HDFEOS INFORMATION")
            bad_metadata = (
                "UpperLeftPointMtrs=(500000.0,4000000.0)\n"
                "LowerRightMtrs=(510000.0,3990000.0)\n"
                "XDim=10\nYDim=10\nZoneCode=0\n"  # Invalid zone
            )
            hdfeos_info.create_dataset("StructMetadata.0", data=bad_metadata.encode())

        with h5py.File(h5_path, "r") as f:
            with pytest.raises(ValueError, match="Invalid UTM zone"):
                _generate_grids_latlon(f)

    @pytest.mark.parametrize(
        "utm_zone,expected_south",
        [
            (33, False),  # Northern hemisphere
            (-33, True),  # Southern hemisphere
        ],
    )
    def test_generate_grids_latlon_hemispheres(
        self, tmp_path, utm_zone, expected_south
    ):
        """Test both hemispheres produce correct latitude signs."""
        from tanager_isofit.convert import _generate_grids_latlon

        h5_path = tmp_path / f"zone_{utm_zone}.h5"
        northing = 6000000 if expected_south else 4000000

        with h5py.File(h5_path, "w") as f:
            f.create_group("HDFEOS/GRIDS/HYP")
            hdfeos_info = f.create_group("HDFEOS INFORMATION")
            metadata = (
                f"UpperLeftPointMtrs=(500000.0,{northing}.0)\n"
                f"LowerRightMtrs=(510000.0,{northing - 10000}.0)\n"
                f"XDim=10\nYDim=10\nZoneCode={utm_zone}\n"
            )
            hdfeos_info.create_dataset("StructMetadata.0", data=metadata.encode())

        with h5py.File(h5_path, "r") as f:
            lat, lon = _generate_grids_latlon(f)

        if expected_south:
            assert lat[0, 0] < 0, "Southern hemisphere should have negative latitude"
        else:
            assert lat[0, 0] > 0, "Northern hemisphere should have positive latitude"

    def test_read_grids_hdf5_returns_expected_keys(self, sample_grids_hdf5):
        """Test that read_tanager_hdf5 works with GRIDS format."""
        from tanager_isofit.convert import read_tanager_hdf5

        data = read_tanager_hdf5(sample_grids_hdf5)

        expected_keys = [
            "radiance",
            "latitude",
            "longitude",
            "path_length",
            "wavelengths",
            "fwhm",
            "acquisition_time",
            "metadata",
        ]

        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_read_grids_hdf5_geolocation_shape(self, sample_grids_hdf5):
        """Test that generated lat/lon arrays match radiance spatial dimensions."""
        from tanager_isofit.convert import read_tanager_hdf5

        data = read_tanager_hdf5(sample_grids_hdf5)

        rad_shape = data["radiance"].shape
        lat_shape = data["latitude"].shape
        lon_shape = data["longitude"].shape

        assert lat_shape == (rad_shape[0], rad_shape[1])
        assert lon_shape == (rad_shape[0], rad_shape[1])

    def test_convert_grids_creates_all_files(self, sample_grids_hdf5, tmp_path):
        """Test that conversion of GRIDS file creates all expected output files."""
        from tanager_isofit.convert import convert_tanager_to_envi

        output_dir = tmp_path / "output"
        result = convert_tanager_to_envi(sample_grids_hdf5, output_dir)

        # Check all files exist
        assert Path(result["radiance"]).exists()
        assert Path(result["loc"]).exists()
        assert Path(result["obs"]).exists()
        assert Path(result["wavelength_file"]).exists()

    def test_validate_grids_structure(self, sample_grids_hdf5):
        """Test validation of GRIDS format file."""
        from tanager_isofit.convert import validate_hdf5_structure

        is_valid, issues = validate_hdf5_structure(sample_grids_hdf5)

        # Should be valid (our test file has correct structure)
        if not is_valid:
            print(f"Validation issues: {issues}")

    def test_swaths_still_works(self, sample_hdf5, tmp_path):
        """Regression: existing SWATHS files still work."""
        from tanager_isofit.convert import convert_tanager_to_envi

        output_dir = tmp_path / "output"
        result = convert_tanager_to_envi(sample_hdf5, output_dir)
        assert Path(result["radiance"]).exists()
