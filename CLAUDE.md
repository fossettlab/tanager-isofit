# Project: tanager-isofit

Python package that converts Planet Tanager hyperspectral HDF5 (TOA radiance, 425 bands, 400-2500 nm, ~5 nm sampling) to surface reflectance using ISOFIT atmospheric correction. Validated against coincident NASA EMIT L2A.

Full design and rationale: `spec.md`. User-facing install + CLI: `README.md`.

## Strategic context

- Supporting tool for the Planet Tanager Open Data Competition. Outputs feed `../TanagerFM/` (pretraining patches) and `../tanager-rocks/` (spectral extraction).
- ISOFIT v3.4.0 (Feb 2025) added native Tanager support via PRs #634/#643. Target instances run ISOFIT 3.6.1.
- Replicates the atmospheric correction Planet runs internally for commercial products.

## Stack

- Python ≥ 3.9 (3.11+ preferred for parity with sibling projects)
- Core: `h5py`, `numpy`, `spectral`, `pvlib`, `scipy`, `xarray`, `rioxarray`, `pyproj`
- CLI: `click`, `tqdm`
- EMIT validation: `earthaccess`
- Atmospheric correction: `isofit ≥ 3.4.0` (optional extra; sRTMnet, no MODTRAN license needed)
- Test: `pytest`

## Commands

```bash
pip install -e ".[all]"          # dev install with isofit + pytest
pytest                           # run test suite
tanager-isofit --help            # CLI entrypoint (see project.scripts in pyproject.toml)
```

## Data flow

```
Tanager HDF5 (basic_radiance, 426 bands)
  → tanager_isofit (HDF5 → ENVI: radiance, location, observation, wavelengths)
  → ISOFIT apply_oe (sensor="tanager", sRTMnet)
  → reflectance, uncertainty, atmospheric state
```

## Conventions

- ENVI output is the package's contract — downstream consumers (TanagerFM, tanager-rocks) read ENVI.
- Sensor angles are read directly from HDF5 metadata, not recomputed.
- 6S radiative transfer code is required for atmospheric correction; users install separately (see `README.md`).
- Surface model is auto-generated when not provided (current `master` HEAD behavior).
- Test data lives under `tests/`; example notebooks under `examples/`.

## Related

- `../TanagerFM/` — consumes ENVI reflectance for pretraining
- `../tanager-rocks/` — consumes ENVI reflectance for mineral classification
- `../entry_optimization.md` — competition rubric context
- Workspace orientation: `../CLAUDE.md`

## Known issues to fix when convenient

- `pyproject.toml` lists author email as `alex.bradley@wustl.edu`; canonical form is `abradley@wustl.edu`.
- `pyproject.toml` `Homepage` / `Repository` URLs point to `alexkbradley/tanager-isofit`; actual remote is `bradleylab/tanager-isofit`.
