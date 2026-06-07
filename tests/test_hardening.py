"""Regression tests for data-integrity and edge-case hardening.

These cover behavior that a radiance->reflectance converter must guarantee:
no invented values, no silently dropped/misaligned bands, no fabricated
acquisition times, and finite-only validation metrics.
"""

from datetime import datetime

import numpy as np
import h5py
import pytest

from tanager_isofit.config import TANAGER_NUM_BANDS


def _write_swath_hdf5(
    path,
    radiance,
    *,
    wavelengths=None,
    fwhm=None,
    fill_value=None,
    strip_id="20250511_074311_00_4001",
    bands_first=False,
):
    """Write a minimal SWATHS-format Tanager HDF5 file for testing.

    radiance is given as (lines, samples, bands); transposed to bands-first
    on disk when requested.
    """
    lines, samples, bands = radiance.shape
    with h5py.File(path, "w") as f:
        swath = f.create_group("HDFEOS/SWATHS/HYP")
        data_fields = swath.create_group("Data_Fields")
        geo_fields = swath.create_group("Geolocation_Fields")

        disk_rad = np.transpose(radiance, (2, 0, 1)) if bands_first else radiance
        ds = data_fields.create_dataset(
            "toa_radiance", data=disk_rad.astype(np.float32)
        )

        if wavelengths is None:
            wavelengths = np.linspace(400, 2500, bands)
        if fwhm is None:
            fwhm = np.full(bands, 5.0)
        ds.attrs["wavelengths"] = np.asarray(wavelengths)
        ds.attrs["fwhm"] = np.asarray(fwhm)
        if fill_value is not None:
            ds.attrs["_FillValue"] = fill_value

        path_length = np.full((lines, samples), 500000.0, dtype=np.float32)
        data_fields.create_dataset("sensor_to_ground_path_length", data=path_length)

        lat = np.linspace(30.0, 31.0, lines)[:, None] * np.ones((1, samples))
        lon = np.linspace(-120.0, -119.0, samples)[None, :] * np.ones((lines, 1))
        geo_fields.create_dataset("latitude", data=lat.astype(np.float64))
        geo_fields.create_dataset("longitude", data=lon.astype(np.float64))

        if strip_id is not None:
            f.attrs["strip_id"] = strip_id
    return path


class TestFillValueHandling:
    """A declared _FillValue must never be scaled and passed through as signal."""

    def test_fill_value_becomes_nan(self, tmp_path):
        from tanager_isofit.convert import read_tanager_hdf5

        bands = 8
        radiance = np.ones((6, 6, bands), dtype=np.float32) * 50.0
        radiance[0, 0, :] = -9999.0  # one fill pixel across all bands
        h5 = _write_swath_hdf5(tmp_path / "fill.h5", radiance, fill_value=-9999.0)

        with pytest.warns(UserWarning, match="fill pixels"):
            data = read_tanager_hdf5(h5)

        out = data["radiance"]
        assert np.all(np.isnan(out[0, 0, :])), "fill pixel must be NaN"
        # Real pixels survive and are scaled (50 * 0.1 = 5.0).
        assert np.allclose(out[1, 1, :], 5.0)
        # Fill value must not appear as scaled signal anywhere.
        assert not np.any(out == -9999.0 * 0.1)

    def test_no_fill_attr_leaves_data_intact(self, tmp_path):
        from tanager_isofit.convert import read_tanager_hdf5

        radiance = np.ones((4, 4, 6), dtype=np.float32) * 20.0
        h5 = _write_swath_hdf5(tmp_path / "nofill.h5", radiance)

        data = read_tanager_hdf5(h5)
        assert not np.any(np.isnan(data["radiance"]))
        assert np.allclose(data["radiance"], 2.0)  # 20 * 0.1


class TestWavelengthBandParity:
    """Wavelength / FWHM vectors must match the radiance band count."""

    def test_wavelength_count_mismatch_raises(self, tmp_path):
        from tanager_isofit.convert import read_tanager_hdf5

        radiance = np.ones((4, 4, 10), dtype=np.float32)
        # Only 9 wavelengths for 10 bands.
        h5 = _write_swath_hdf5(
            tmp_path / "wlmismatch.h5",
            radiance,
            wavelengths=np.linspace(400, 2500, 9),
            fwhm=np.full(10, 5.0),
        )
        with pytest.raises(ValueError, match="Wavelength count"):
            read_tanager_hdf5(h5)

    def test_fwhm_count_mismatch_raises(self, tmp_path):
        from tanager_isofit.convert import read_tanager_hdf5

        radiance = np.ones((4, 4, 10), dtype=np.float32)
        h5 = _write_swath_hdf5(
            tmp_path / "fwhmmismatch.h5",
            radiance,
            wavelengths=np.linspace(400, 2500, 10),
            fwhm=np.full(7, 5.0),
        )
        with pytest.raises(ValueError, match="FWHM count"):
            read_tanager_hdf5(h5)


class TestBandAxisAmbiguity:
    """When both first and last axes equal the band count, refuse to guess."""

    def test_ambiguous_dimension_order_raises(self, tmp_path):
        from tanager_isofit.convert import read_tanager_hdf5

        # Shape (426, lines, 426): both axes equal the band count.
        radiance = np.ones((TANAGER_NUM_BANDS, 5, TANAGER_NUM_BANDS), dtype=np.float32)
        h5_path = tmp_path / "ambiguous.h5"
        with h5py.File(h5_path, "w") as f:
            swath = f.create_group("HDFEOS/SWATHS/HYP")
            data_fields = swath.create_group("Data_Fields")
            geo = swath.create_group("Geolocation_Fields")
            ds = data_fields.create_dataset("toa_radiance", data=radiance)
            ds.attrs["wavelengths"] = np.linspace(400, 2500, TANAGER_NUM_BANDS)
            ds.attrs["fwhm"] = np.full(TANAGER_NUM_BANDS, 5.0)
            geo.create_dataset(
                "latitude", data=np.zeros((TANAGER_NUM_BANDS, TANAGER_NUM_BANDS))
            )
            geo.create_dataset(
                "longitude", data=np.zeros((TANAGER_NUM_BANDS, TANAGER_NUM_BANDS))
            )
            f.attrs["strip_id"] = "20250511_074311_00_4001"

        with pytest.raises(ValueError, match="[Aa]mbiguous"):
            read_tanager_hdf5(h5_path)

    def test_bands_first_layout_transposed(self, tmp_path):
        """A genuine bands-first cube (bands=426) is detected and transposed."""
        from tanager_isofit.convert import read_tanager_hdf5

        lines, samples = 4, 5
        radiance = np.random.rand(lines, samples, TANAGER_NUM_BANDS).astype(np.float32)
        h5 = _write_swath_hdf5(
            tmp_path / "bandsfirst.h5",
            radiance,
            wavelengths=np.linspace(400, 2500, TANAGER_NUM_BANDS),
            fwhm=np.full(TANAGER_NUM_BANDS, 5.0),
            bands_first=True,
        )
        data = read_tanager_hdf5(h5)
        assert data["radiance"].shape == (lines, samples, TANAGER_NUM_BANDS)


class TestAcquisitionTimeNoFabrication:
    """An unparseable acquisition time must fail, not fall back to wall clock."""

    def test_unparseable_time_raises(self, tmp_path):
        from tanager_isofit.convert import read_tanager_hdf5

        radiance = np.ones((4, 4, 6), dtype=np.float32)
        # No strip_id and a filename that carries no YYYYMMDD_HHMMSS.
        h5 = _write_swath_hdf5(
            tmp_path / "noinfo.h5",
            radiance,
            strip_id=None,
        )
        with pytest.raises(ValueError, match="acquisition time"):
            read_tanager_hdf5(h5)


class TestWavelengthFileColumn:
    """run_full_validation must read the wavelength column, not the channel index."""

    def test_loadtxt_column_one_is_wavelength(self, tmp_path):
        from tanager_isofit.utils import create_wavelength_file

        wl = [400.0, 450.0, 500.0, 550.0]
        fwhm = [5.0, 5.0, 5.0, 5.0]
        out = create_wavelength_file(wl, fwhm, tmp_path / "wavelengths.txt")

        table = np.loadtxt(out)
        # Column 0 is the channel index (0..n-1), column 1 the wavelength.
        assert np.allclose(table[:, 0], np.arange(len(wl)))
        assert np.allclose(table[:, 1], wl)

    def test_run_full_validation_uses_wavelength_column(self, tmp_path, monkeypatch):
        """End-to-end: the loaded Tanager wavelengths must be the real nm grid."""
        import tanager_isofit.validate as validate
        from tanager_isofit.utils import create_wavelength_file

        wl = np.linspace(400, 2500, 12)
        wl_file = create_wavelength_file(
            wl.tolist(), [5.0] * 12, tmp_path / "wavelengths.txt"
        )

        tan_data = np.random.rand(4, 4, 12).astype(np.float32)
        emit_wl = np.linspace(400, 2500, 20)
        emit_data = np.random.rand(4, 4, 20).astype(np.float32)

        captured = {}

        def fake_read_envi_file(path):
            return tan_data, {"bands": 12}

        def fake_read_emit(path):
            return emit_data, emit_wl, None, None

        def fake_compare(t_data, t_wl, e_data, e_wl, mask=None):
            captured["tanager_wavelengths"] = np.asarray(t_wl)
            return {"overall": {"rmse": 0.0, "correlation": 1.0}}

        monkeypatch.setattr(validate, "read_envi_file", fake_read_envi_file)
        monkeypatch.setattr(validate, "read_emit_reflectance", fake_read_emit)
        monkeypatch.setattr(validate, "compare_reflectance", fake_compare)

        validate.run_full_validation(
            tmp_path / "rfl", wl_file, tmp_path / "emit.nc", tmp_path / "out"
        )

        loaded = captured["tanager_wavelengths"]
        # Must be the real wavelength grid, NOT the channel index 0..11.
        assert np.allclose(loaded, wl)
        assert loaded.min() > 100  # channel indices would be 0..11


class TestFiniteValidation:
    """Validation metrics and NDWI must ignore inf, not just NaN."""

    def test_compare_reflectance_ignores_inf(self):
        from tanager_isofit.validate import compare_reflectance

        data1 = np.ones((4, 4, 6), dtype=np.float32)
        data2 = np.ones((4, 4, 6), dtype=np.float32) + 0.1
        # Inject an infinity into one pixel/band.
        data2[0, 0, 0] = np.inf
        wl = np.linspace(400, 2500, 6)

        results = compare_reflectance(data1, wl, data2, wl)
        # RMSE/bias finite despite the inf pixel being excluded.
        assert np.isfinite(results["overall"]["rmse"])
        assert abs(results["overall"]["bias"] + 0.1) < 0.01

    def test_identify_water_pixels_no_inf_classified(self):
        from tanager_isofit.validate import identify_water_pixels

        bands = 100
        wl = np.linspace(400, 2500, bands)
        green_idx = int(np.argmin(np.abs(wl - 560)))
        nir_idx = int(np.argmin(np.abs(wl - 860)))

        refl = np.zeros((3, 3, bands), dtype=np.float32)
        # A pixel with green == -nir gives green+nir == 0 -> divide-by-zero.
        refl[0, 0, green_idx] = 0.1
        refl[0, 0, nir_idx] = -0.1

        mask = identify_water_pixels(refl, wl)
        # The divide-by-zero pixel must not be classified as water.
        assert not mask[0, 0]


class TestSolarGeometryNaNPreserved:
    """Large-array scene-center solar approximation must not fabricate angles
    for pixels with missing geolocation."""

    def test_large_array_preserves_nan_geolocation(self):
        from tanager_isofit.geometry import calculate_solar_geometry

        n = 110  # 110*110 = 12100 > 10000 -> scene-center branch
        lat = np.full((n, n), 30.0)
        lon = np.full((n, n), -120.0)
        lat[0, 0] = np.nan  # missing geolocation
        lon[0, 0] = np.nan
        acq_time = datetime(2025, 5, 11, 18, 0, 0)

        sz, sa = calculate_solar_geometry(acq_time, lat, lon)

        assert np.isnan(sz[0, 0]), "missing-geolocation pixel must stay NaN"
        assert np.isnan(sa[0, 0])
        # Valid pixels get a finite scene-center angle.
        assert np.isfinite(sz[1, 1])
