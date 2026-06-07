"""
Utility functions for ENVI file I/O and file helpers.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np

from tanager_isofit.config import (
    ENVI_DTYPE_FLOAT32,
)


def write_envi_header(
    header_path: Union[str, Path],
    lines: int,
    samples: int,
    bands: int,
    dtype: int = ENVI_DTYPE_FLOAT32,
    interleave: str = "bip",
    wavelengths: Optional[List[float]] = None,
    fwhm: Optional[List[float]] = None,
    band_names: Optional[List[str]] = None,
    description: str = "",
    byte_order: int = 0,
    **kwargs: Any,
) -> None:
    """
    Write an ENVI header file.

    Args:
        header_path: Path to write header file (.hdr)
        lines: Number of lines (rows) in the image
        samples: Number of samples (columns) per line
        bands: Number of bands
        dtype: ENVI data type (4 = float32)
        interleave: Data interleave format ('bip', 'bil', 'bsq')
        wavelengths: List of center wavelengths in nm
        fwhm: List of FWHM values in nm
        band_names: List of band names
        description: File description
        byte_order: 0 for little-endian, 1 for big-endian
        **kwargs: Additional header fields
    """
    header_path = Path(header_path)

    with open(header_path, "w") as f:
        f.write("ENVI\n")
        if description:
            f.write(f"description = {{{description}}}\n")
        f.write(f"samples = {samples}\n")
        f.write(f"lines = {lines}\n")
        f.write(f"bands = {bands}\n")
        f.write("header offset = 0\n")
        f.write("file type = ENVI Standard\n")
        f.write(f"data type = {dtype}\n")
        f.write(f"interleave = {interleave}\n")
        f.write(f"byte order = {byte_order}\n")

        if wavelengths is not None and len(wavelengths) > 0:
            f.write("wavelength units = Nanometers\n")
            f.write("wavelength = {\n")
            _write_list_field(f, wavelengths)
            f.write("}\n")

        if fwhm is not None and len(fwhm) > 0:
            f.write("fwhm = {\n")
            _write_list_field(f, fwhm)
            f.write("}\n")

        if band_names is not None and len(band_names) > 0:
            f.write("band names = {\n")
            for i, name in enumerate(band_names):
                if i < len(band_names) - 1:
                    f.write(f"  {name},\n")
                else:
                    f.write(f"  {name}\n")
            f.write("}\n")

        # Write additional kwargs
        for key, value in kwargs.items():
            if isinstance(value, list):
                f.write(f"{key} = {{\n")
                _write_list_field(f, value)
                f.write("}\n")
            else:
                f.write(f"{key} = {value}\n")


def _write_list_field(f, values: List, items_per_line: int = 10) -> None:
    """Helper to write a list field with proper formatting."""
    for i in range(0, len(values), items_per_line):
        chunk = values[i : i + items_per_line]
        line = ", ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in chunk)
        if i + items_per_line < len(values):
            f.write(f"  {line},\n")
        else:
            f.write(f"  {line}\n")


def write_envi_file(
    data: np.ndarray,
    output_path: Union[str, Path],
    wavelengths: Optional[List[float]] = None,
    fwhm: Optional[List[float]] = None,
    band_names: Optional[List[str]] = None,
    interleave: str = "bip",
    description: str = "",
    **header_kwargs: Any,
) -> Tuple[Path, Path]:
    """
    Write data array to ENVI format (binary + header).

    Args:
        data: 3D numpy array (lines, samples, bands)
        output_path: Base path for output (without extension)
        wavelengths: Center wavelengths for radiance data
        fwhm: FWHM values for radiance data
        band_names: Names for each band
        interleave: Data interleave ('bip', 'bil', 'bsq')
        description: File description
        **header_kwargs: Additional header fields

    Returns:
        Tuple of (binary_path, header_path)
    """
    output_path = Path(output_path)
    binary_path = output_path.with_suffix("") if output_path.suffix else output_path
    header_path = binary_path.with_suffix(".hdr")

    # Ensure data is float32 and contiguous
    data = np.ascontiguousarray(data, dtype=np.float32)

    lines, samples, bands = data.shape

    # Reorder if needed based on interleave
    if interleave.lower() == "bip":
        # BIP: (lines, samples, bands) - already correct
        write_data = data
    elif interleave.lower() == "bil":
        # BIL: (lines, bands, samples)
        write_data = np.transpose(data, (0, 2, 1))
    elif interleave.lower() == "bsq":
        # BSQ: (bands, lines, samples)
        write_data = np.transpose(data, (2, 0, 1))
    else:
        raise ValueError(f"Unknown interleave format: {interleave}")

    # Write binary file
    write_data.tofile(str(binary_path))

    # Write header file
    write_envi_header(
        header_path,
        lines=lines,
        samples=samples,
        bands=bands,
        interleave=interleave,
        wavelengths=wavelengths,
        fwhm=fwhm,
        band_names=band_names,
        description=description,
        **header_kwargs,
    )

    return binary_path, header_path


def read_envi_header(header_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse an ENVI header file.

    Args:
        header_path: Path to header file (.hdr)

    Returns:
        Dictionary of header fields
    """
    header_path = Path(header_path)
    header = {}

    with open(header_path, "r") as f:
        content = f.read()

    # Remove ENVI marker if present
    if content.startswith("ENVI"):
        content = content[4:].strip()

    # Parse fields
    i = 0
    while i < len(content):
        # Skip whitespace
        while i < len(content) and content[i] in " \t\n":
            i += 1

        if i >= len(content):
            break

        # Find '='
        eq_pos = content.find("=", i)
        if eq_pos == -1:
            break

        key = content[i:eq_pos].strip()
        i = eq_pos + 1

        # Skip whitespace after '='
        while i < len(content) and content[i] in " \t\n":
            i += 1

        if i >= len(content):
            header[key] = ""
            break

        # Check for multi-line value (enclosed in braces)
        if content[i] == "{":
            # Find closing brace
            brace_end = content.find("}", i)
            if brace_end == -1:
                value = content[i + 1 :].strip()
                i = len(content)
            else:
                value = content[i + 1 : brace_end].strip()
                i = brace_end + 1

            # Parse list values
            if "," in value or "\n" in value:
                items = [v.strip() for v in value.replace("\n", ",").split(",")]
                items = [v for v in items if v]  # Remove empty strings
                # Try to convert to numbers
                try:
                    header[key] = [float(v) for v in items]
                except ValueError:
                    header[key] = items
            else:
                header[key] = value
        else:
            # Single-line value
            newline_pos = content.find("\n", i)
            if newline_pos == -1:
                value = content[i:].strip()
                i = len(content)
            else:
                value = content[i:newline_pos].strip()
                i = newline_pos + 1

            # Try to convert to number
            try:
                if "." in value:
                    header[key] = float(value)
                else:
                    header[key] = int(value)
            except ValueError:
                header[key] = value

    return header


def read_envi_file(
    file_path: Union[str, Path],
    header_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read ENVI file (binary + header).

    Args:
        file_path: Path to binary file
        header_path: Path to header file (auto-detected if None)

    Returns:
        Tuple of (data array, header dict)
    """
    file_path = Path(file_path)

    # Find header file
    if header_path is None:
        if file_path.suffix == ".hdr":
            header_path = file_path
            file_path = file_path.with_suffix("")
        else:
            header_path = file_path.with_suffix(".hdr")

    header_path = Path(header_path)

    # Read header
    header = read_envi_header(header_path)

    # Get dimensions
    lines = int(header.get("lines", 0))
    samples = int(header.get("samples", 0))
    bands = int(header.get("bands", 1))
    dtype_code = int(header.get("data type", 4))
    interleave = header.get("interleave", "bip").lower()

    # Map ENVI data type to numpy dtype
    dtype_map = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }
    dtype = dtype_map.get(dtype_code, np.float32)

    # Read binary data
    data = np.fromfile(str(file_path), dtype=dtype)

    # Reshape based on interleave
    if interleave == "bip":
        data = data.reshape((lines, samples, bands))
    elif interleave == "bil":
        data = data.reshape((lines, bands, samples))
        data = np.transpose(data, (0, 2, 1))
    elif interleave == "bsq":
        data = data.reshape((bands, lines, samples))
        data = np.transpose(data, (1, 2, 0))

    return data, header


def create_wavelength_file(
    wavelengths: List[float],
    fwhm: List[float],
    output_path: Union[str, Path],
) -> Path:
    """
    Create a wavelength file for ISOFIT.

    ISOFIT expects 3 columns: channel, wavelength (nm), fwhm (nm)

    Args:
        wavelengths: List of center wavelengths in nm
        fwhm: List of FWHM values in nm
        output_path: Path for output file

    Returns:
        Path to created wavelength file
    """
    output_path = Path(output_path)

    # Create channel numbers (0-indexed)
    channels = np.arange(len(wavelengths))

    # Stack channel, wavelengths, and FWHM (ISOFIT expects 3 columns)
    data = np.column_stack([channels, wavelengths, fwhm])

    # Write as space-separated text file
    np.savetxt(output_path, data, fmt=["%.0f", "%.6f", "%.6f"], delimiter=" ")

    return output_path


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_basename(path: Union[str, Path]) -> str:
    """
    Get the base name of a file without extension.

    Args:
        path: File path

    Returns:
        Base name without extension
    """
    path = Path(path)
    # Handle multiple extensions (e.g., .h5)
    return path.stem


def validate_envi_files(
    radiance_path: Union[str, Path],
    loc_path: Union[str, Path],
    obs_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Validate ENVI files for ISOFIT compatibility.

    Args:
        radiance_path: Path to radiance file
        loc_path: Path to location file
        obs_path: Path to observation file

    Returns:
        Dictionary with validation results and any issues found
    """
    issues = []
    info = {}

    # Check radiance file
    rad_header = read_envi_header(Path(radiance_path).with_suffix(".hdr"))
    info["radiance"] = {
        "lines": rad_header.get("lines"),
        "samples": rad_header.get("samples"),
        "bands": rad_header.get("bands"),
    }

    # Check location file
    loc_header = read_envi_header(Path(loc_path).with_suffix(".hdr"))
    info["location"] = {
        "lines": loc_header.get("lines"),
        "samples": loc_header.get("samples"),
        "bands": loc_header.get("bands"),
    }

    # Check observation file
    obs_header = read_envi_header(Path(obs_path).with_suffix(".hdr"))
    info["observation"] = {
        "lines": obs_header.get("lines"),
        "samples": obs_header.get("samples"),
        "bands": obs_header.get("bands"),
    }

    # Validate dimensions match
    rad_dims = (info["radiance"]["lines"], info["radiance"]["samples"])
    loc_dims = (info["location"]["lines"], info["location"]["samples"])
    obs_dims = (info["observation"]["lines"], info["observation"]["samples"])

    if rad_dims != loc_dims:
        issues.append(f"Dimension mismatch: radiance {rad_dims} != location {loc_dims}")
    if rad_dims != obs_dims:
        issues.append(
            f"Dimension mismatch: radiance {rad_dims} != observation {obs_dims}"
        )

    # Check location has 3 bands
    if info["location"]["bands"] != 3:
        issues.append(
            f"Location file should have 3 bands, has {info['location']['bands']}"
        )

    # Check observation has 10 bands
    if info["observation"]["bands"] != 10:
        issues.append(
            f"Observation file should have 10 bands, has {info['observation']['bands']}"
        )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "info": info,
    }
