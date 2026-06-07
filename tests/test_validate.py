"""Tests for validation and EMIT comparison module."""

import numpy as np


class TestSpectralResample:
    """Tests for spectral resampling."""

    def test_resample_identity(self):
        """Test resampling to same wavelengths."""
        from tanager_isofit.validate import spectral_resample

        data = np.random.rand(10, 10, 50)
        wavelengths = np.linspace(400, 2500, 50)

        resampled = spectral_resample(data, wavelengths, wavelengths)

        # Should be nearly identical
        assert np.allclose(resampled, data, rtol=1e-5)

    def test_resample_subset(self):
        """Test resampling to subset of wavelengths."""
        from tanager_isofit.validate import spectral_resample

        data = np.random.rand(5, 5, 100)
        source_wl = np.linspace(400, 2500, 100)
        target_wl = np.linspace(400, 2500, 50)

        resampled = spectral_resample(data, source_wl, target_wl)

        assert resampled.shape == (5, 5, 50)


class TestSpectralAngle:
    """Tests for spectral angle computation."""

    def test_spectral_angle_identical(self):
        """Test spectral angle of identical spectra."""
        from tanager_isofit.validate import compute_spectral_angle

        spectrum = np.array([0.1, 0.2, 0.15, 0.3])

        angle = compute_spectral_angle(spectrum, spectrum)

        assert angle < 1e-5  # Should be essentially zero (allow float precision)

    def test_spectral_angle_orthogonal(self):
        """Test spectral angle of orthogonal spectra."""
        from tanager_isofit.validate import compute_spectral_angle

        s1 = np.array([1, 0, 0, 0])
        s2 = np.array([0, 1, 0, 0])

        angle = compute_spectral_angle(s1, s2)

        assert abs(angle - 90) < 1e-6  # Should be 90 degrees

    def test_spectral_angle_with_nan(self):
        """Test spectral angle handles NaN values."""
        from tanager_isofit.validate import compute_spectral_angle

        s1 = np.array([0.1, np.nan, 0.2, 0.3])
        s2 = np.array([0.1, 0.15, np.nan, 0.3])

        angle = compute_spectral_angle(s1, s2)

        # Should still compute with valid values
        assert not np.isnan(angle)


class TestWaterIdentification:
    """Tests for water pixel identification."""

    def test_identify_water_pixels(self):
        """Test water pixel identification with NDWI."""
        from tanager_isofit.validate import identify_water_pixels

        # Create synthetic data
        lines, samples, bands = 10, 10, 100
        wavelengths = np.linspace(400, 2500, bands)

        # Find green and NIR band indices
        green_idx = np.argmin(np.abs(wavelengths - 560))
        nir_idx = np.argmin(np.abs(wavelengths - 860))

        # Create water-like spectra (high green, low NIR)
        reflectance = np.zeros((lines, samples, bands), dtype=np.float32)
        reflectance[:5, :, green_idx] = 0.1
        reflectance[:5, :, nir_idx] = 0.02

        # Create land-like spectra (high NIR)
        reflectance[5:, :, green_idx] = 0.1
        reflectance[5:, :, nir_idx] = 0.3

        water_mask = identify_water_pixels(reflectance, wavelengths)

        # First half should be water
        assert np.all(water_mask[:5, :])
        # Second half should be land
        assert not np.any(water_mask[5:, :])


class TestCompareReflectance:
    """Tests for reflectance comparison."""

    def test_compare_reflectance_identical(self):
        """Test comparison of identical data."""
        from tanager_isofit.validate import compare_reflectance

        data = np.random.rand(10, 10, 50).astype(np.float32)
        wavelengths = np.linspace(400, 2500, 50)

        results = compare_reflectance(data, wavelengths, data, wavelengths)

        # RMSE should be 0 for identical data
        assert results["overall"]["rmse"] < 1e-6
        assert results["overall"]["correlation"] > 0.999

    def test_compare_reflectance_with_bias(self):
        """Test comparison with systematic bias."""
        from tanager_isofit.validate import compare_reflectance

        data1 = np.random.rand(10, 10, 50).astype(np.float32)
        data2 = data1 + 0.1  # Add constant bias
        wavelengths = np.linspace(400, 2500, 50)

        results = compare_reflectance(data1, wavelengths, data2, wavelengths)

        # Bias should be approximately 0.1
        assert abs(results["overall"]["bias"] + 0.1) < 0.01

    def test_compare_reflectance_with_mask(self):
        """Test comparison with mask."""
        from tanager_isofit.validate import compare_reflectance

        data1 = np.random.rand(10, 10, 50).astype(np.float32)
        data2 = np.random.rand(10, 10, 50).astype(np.float32)
        wavelengths = np.linspace(400, 2500, 50)

        # Only compare first 5 rows
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :] = True

        results = compare_reflectance(data1, wavelengths, data2, wavelengths, mask=mask)

        assert results["overall"]["n_pixels"] == 50  # 5 * 10


class TestWaterValidation:
    """Tests for water spectrum validation."""

    def test_validate_water_spectrum_valid(self):
        """Test validation of valid water spectrum."""
        from tanager_isofit.validate import validate_water_spectrum

        lines, samples, bands = 10, 10, 100
        wavelengths = np.linspace(400, 2500, bands)

        # Create realistic water spectrum
        # Water: higher blue/green, very low NIR/SWIR
        reflectance = np.zeros((lines, samples, bands), dtype=np.float32)

        # Find band indices
        blue_idx = np.argmin(np.abs(wavelengths - 480))
        green_idx = np.argmin(np.abs(wavelengths - 560))
        nir_idx = np.argmin(np.abs(wavelengths - 860))
        swir_idx = np.argmin(np.abs(wavelengths - 1600))

        # Set water-like values
        reflectance[:, :, blue_idx] = 0.05
        reflectance[:, :, green_idx] = 0.04
        reflectance[:, :, nir_idx] = 0.02
        reflectance[:, :, swir_idx] = 0.01

        water_mask = np.ones((lines, samples), dtype=bool)

        result = validate_water_spectrum(reflectance, wavelengths, water_mask)

        assert result["valid"]

    def test_validate_water_spectrum_invalid_nir(self):
        """Test validation fails for high NIR."""
        from tanager_isofit.validate import validate_water_spectrum

        lines, samples, bands = 10, 10, 100
        wavelengths = np.linspace(400, 2500, bands)

        reflectance = np.zeros((lines, samples, bands), dtype=np.float32)

        # High NIR (not water-like)
        nir_idx = np.argmin(np.abs(wavelengths - 860))
        reflectance[:, :, nir_idx] = 0.3

        water_mask = np.ones((lines, samples), dtype=bool)

        result = validate_water_spectrum(reflectance, wavelengths, water_mask)

        assert not result["valid"]
        assert any("NIR" in issue for issue in result["issues"])


class TestValidationMetrics:
    """Tests for validation metric calculations."""

    def test_rmse_calculation(self):
        """Test RMSE is calculated correctly."""
        from tanager_isofit.validate import compare_reflectance

        # Create data with known RMSE
        data1 = np.ones((10, 10, 50), dtype=np.float32)
        data2 = np.ones((10, 10, 50), dtype=np.float32) + 0.1
        wavelengths = np.linspace(400, 2500, 50)

        results = compare_reflectance(data1, wavelengths, data2, wavelengths)

        # RMSE should be 0.1
        assert abs(results["overall"]["rmse"] - 0.1) < 0.001

    def test_mae_calculation(self):
        """Test MAE is calculated correctly."""
        from tanager_isofit.validate import compare_reflectance

        data1 = np.ones((10, 10, 50), dtype=np.float32)
        data2 = np.ones((10, 10, 50), dtype=np.float32) + 0.1
        wavelengths = np.linspace(400, 2500, 50)

        results = compare_reflectance(data1, wavelengths, data2, wavelengths)

        # MAE should be 0.1
        assert abs(results["overall"]["mae"] - 0.1) < 0.001
