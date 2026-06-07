"""Tests for solar and sensor geometry calculations."""

from datetime import datetime

import numpy as np


class TestSolarGeometry:
    """Tests for solar position calculations."""

    def test_calculate_solar_geometry_scalar(self):
        """Test solar geometry with scalar lat/lon."""
        from tanager_isofit.geometry import calculate_solar_geometry

        # Test with known location and time
        acq_time = datetime(2025, 5, 11, 7, 43, 11)  # From test file
        lat = 30.0
        lon = -120.0

        solar_zenith, solar_azimuth = calculate_solar_geometry(acq_time, lat, lon)

        # Basic sanity checks
        assert 0 <= solar_zenith <= 180
        assert 0 <= solar_azimuth <= 360

    def test_calculate_solar_geometry_array(self):
        """Test solar geometry with array lat/lon."""
        from tanager_isofit.geometry import calculate_solar_geometry

        acq_time = datetime(2025, 5, 11, 12, 0, 0)

        # Small array
        lat = np.array([[30.0, 30.1], [30.2, 30.3]])
        lon = np.array([[-120.0, -119.9], [-120.0, -119.9]])

        solar_zenith, solar_azimuth = calculate_solar_geometry(acq_time, lat, lon)

        assert solar_zenith.shape == lat.shape
        assert solar_azimuth.shape == lon.shape

    def test_calculate_solar_geometry_fast(self):
        """Test fast grid-interpolated solar geometry."""
        from tanager_isofit.geometry import calculate_solar_geometry_fast

        acq_time = datetime(2025, 5, 11, 12, 0, 0)

        # Larger array
        lines, samples = 100, 100
        lat = np.linspace(30, 31, lines)[:, np.newaxis] * np.ones((1, samples))
        lon = np.linspace(-120, -119, samples)[np.newaxis, :] * np.ones((lines, 1))

        solar_zenith, solar_azimuth = calculate_solar_geometry_fast(
            acq_time, lat, lon, grid_size=5
        )

        assert solar_zenith.shape == (lines, samples)
        assert solar_azimuth.shape == (lines, samples)

        # Values should be reasonable
        assert np.all(solar_zenith >= 0)
        assert np.all(solar_zenith <= 180)


class TestSensorGeometry:
    """Tests for sensor geometry calculations."""

    def test_calculate_sensor_geometry_nadir(self):
        """Test sensor geometry for near-nadir viewing."""
        from tanager_isofit.geometry import calculate_sensor_geometry

        # Path length approximately equal to satellite altitude = nadir
        path_length_km = 500.0

        sensor_zenith, sensor_azimuth = calculate_sensor_geometry(path_length_km)

        # Nadir viewing should have small zenith angle
        assert sensor_zenith < 10  # Should be close to 0

    def test_calculate_sensor_geometry_off_nadir(self):
        """Test sensor geometry for off-nadir viewing."""
        from tanager_isofit.geometry import calculate_sensor_geometry

        # Longer path length = off-nadir
        path_length_km = 600.0

        sensor_zenith, sensor_azimuth = calculate_sensor_geometry(path_length_km)

        # Off-nadir should have larger zenith angle
        assert sensor_zenith > 0

    def test_calculate_sensor_geometry_array(self):
        """Test sensor geometry with array input."""
        from tanager_isofit.geometry import calculate_sensor_geometry

        path_length = np.array([[500, 510], [520, 530]])

        sensor_zenith, sensor_azimuth = calculate_sensor_geometry(path_length)

        assert sensor_zenith.shape == path_length.shape


class TestPhaseAngle:
    """Tests for phase angle calculations."""

    def test_calculate_phase_angle_backscatter(self):
        """Test phase angle for backscatter geometry (sun behind sensor)."""
        from tanager_isofit.geometry import calculate_phase_angle

        # Sun and sensor in same direction
        solar_zenith = 30.0
        solar_azimuth = 180.0
        sensor_zenith = 0.0
        sensor_azimuth = 180.0

        phase = calculate_phase_angle(
            solar_zenith, solar_azimuth, sensor_zenith, sensor_azimuth
        )

        # Phase angle should be close to solar zenith for nadir viewing
        assert abs(phase - solar_zenith) < 1.0

    def test_calculate_phase_angle_array(self):
        """Test phase angle with array inputs."""
        from tanager_isofit.geometry import calculate_phase_angle

        solar_zenith = np.array([30, 40, 50])
        solar_azimuth = np.array([180, 180, 180])
        sensor_zenith = np.array([0, 5, 10])
        sensor_azimuth = np.array([180, 180, 180])

        phase = calculate_phase_angle(
            solar_zenith, solar_azimuth, sensor_zenith, sensor_azimuth
        )

        assert phase.shape == solar_zenith.shape


class TestCosineI:
    """Tests for cosine of illumination angle."""

    def test_cosine_i_flat_terrain(self):
        """Test cosine_i for flat terrain."""
        from tanager_isofit.geometry import calculate_cosine_i

        solar_zenith = 30.0

        cosine_i = calculate_cosine_i(solar_zenith, slope=0, aspect=0)

        # For flat terrain, cosine_i = cos(solar_zenith)
        expected = np.cos(np.radians(solar_zenith))
        assert abs(cosine_i - expected) < 1e-6

    def test_cosine_i_range(self):
        """Test that cosine_i is in valid range."""
        from tanager_isofit.geometry import calculate_cosine_i

        solar_zenith = np.linspace(0, 90, 10)

        cosine_i = calculate_cosine_i(solar_zenith)

        assert np.all(cosine_i >= 0)
        assert np.all(cosine_i <= 1)


class TestTimeConversion:
    """Tests for time parsing and conversion."""

    def test_parse_acquisition_time(self):
        """Test parsing acquisition time from strip_id."""
        from tanager_isofit.geometry import parse_acquisition_time

        strip_id = "20250511_074311_00_4001"
        dt = parse_acquisition_time(strip_id)

        assert dt.year == 2025
        assert dt.month == 5
        assert dt.day == 11
        assert dt.hour == 7
        assert dt.minute == 43
        assert dt.second == 11

    def test_time_to_decimal_hours(self):
        """Test conversion to decimal hours."""
        from tanager_isofit.geometry import time_to_decimal_hours

        dt = datetime(2025, 5, 11, 12, 30, 0)

        decimal = time_to_decimal_hours(dt)

        assert decimal == 12.5

    def test_time_to_decimal_hours_midnight(self):
        """Test decimal hours at midnight."""
        from tanager_isofit.geometry import time_to_decimal_hours

        dt = datetime(2025, 5, 11, 0, 0, 0)

        decimal = time_to_decimal_hours(dt)

        assert decimal == 0.0


class TestObservationArray:
    """Tests for observation array creation."""

    def test_create_observation_array_shape(self):
        """Test that observation array has correct shape."""
        from tanager_isofit.geometry import create_observation_array

        lines, samples = 10, 10
        path_length = np.full((lines, samples), 500.0)
        lat = np.linspace(30, 31, lines)[:, np.newaxis] * np.ones((1, samples))
        lon = np.linspace(-120, -119, samples)[np.newaxis, :] * np.ones((lines, 1))
        acq_time = datetime(2025, 5, 11, 12, 0, 0)

        obs = create_observation_array(lines, samples, path_length, acq_time, lat, lon)

        assert obs.shape == (lines, samples, 10)

    def test_create_observation_array_bands(self):
        """Test that observation array bands have reasonable values."""
        from tanager_isofit.geometry import create_observation_array

        lines, samples = 5, 5
        path_length = np.full((lines, samples), 500.0)
        lat = np.full((lines, samples), 30.0)
        lon = np.full((lines, samples), -120.0)
        acq_time = datetime(2025, 5, 11, 12, 0, 0)

        obs = create_observation_array(lines, samples, path_length, acq_time, lat, lon)

        # Band 0: path_length (km)
        assert np.allclose(obs[:, :, 0], 500.0)

        # Band 2: sensor_zenith - should be small for nadir
        assert np.all(obs[:, :, 2] < 10)

        # Band 4: solar_zenith - should be in valid range
        assert np.all(obs[:, :, 4] >= 0)
        assert np.all(obs[:, :, 4] <= 180)

        # Band 6: slope - should be 0 for flat terrain
        assert np.allclose(obs[:, :, 6], 0)

        # Band 9: utc_time - should be 12.0 for noon
        assert np.allclose(obs[:, :, 9], 12.0)


class TestSatelliteHeading:
    """Tests for satellite heading estimation."""

    def test_estimate_satellite_heading_northward(self):
        """Test heading estimation for northward track."""
        from tanager_isofit.geometry import estimate_satellite_heading

        lines, samples = 10, 5
        # Moving north (increasing latitude)
        lat = np.linspace(30, 31, lines)[:, np.newaxis] * np.ones((1, samples))
        lon = np.full((lines, samples), -120.0)

        heading = estimate_satellite_heading(lat, lon)

        # Should be approximately north (0 or 360 degrees)
        assert heading < 30 or heading > 330

    def test_estimate_satellite_heading_southward(self):
        """Test heading estimation for southward track."""
        from tanager_isofit.geometry import estimate_satellite_heading

        lines, samples = 10, 5
        # Moving south (decreasing latitude)
        lat = np.linspace(31, 30, lines)[:, np.newaxis] * np.ones((1, samples))
        lon = np.full((lines, samples), -120.0)

        heading = estimate_satellite_heading(lat, lon)

        # Should be approximately south (180 degrees)
        assert 150 < heading < 210
