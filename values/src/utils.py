"""
Shared utilities for FloodLedger: config loading, unit conversions, and GeoTIFF I/O.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union
import numpy as np
import rasterio
import yaml


def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """
    Load the project configuration file.

    Args:
        config_path (str | Path): Path to config.yaml.

    Returns:
        dict: Parsed YAML configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Create a directory (and parents) if it does not exist.

    Args:
        dir_path (str | Path): Directory path.

    Returns:
        Path: Resolved Path object.
    """
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def linear_to_db(linear_power: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert radar backscatter from LINEAR POWER (unitless, >= 0) to DECIBELS (dB).

    Formula: dB = 10 * log10(linear_power)

    Args:
        linear_power (np.ndarray): Backscatter in linear power units.
        eps (float): Floor value to prevent log10(0) / negative values.

    Returns:
        np.ndarray: Backscatter in decibels (dB), float32. NaNs are preserved.
    """
    arr = np.asarray(linear_power, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    out[valid] = 10.0 * np.log10(np.maximum(arr[valid], eps))
    return out


def db_to_linear(db_values: np.ndarray) -> np.ndarray:
    """
    Convert radar backscatter from DECIBELS (dB) to LINEAR POWER (unitless).

    Formula: linear = 10 ** (dB / 10)

    Args:
        db_values (np.ndarray): Backscatter in dB.

    Returns:
        np.ndarray: Backscatter in linear power units, float32.
    """
    arr = np.asarray(db_values, dtype=np.float32)
    return np.power(10.0, arr / 10.0, dtype=np.float32)


def m2_to_ha(area_m2: float) -> float:
    """
    Convert area from square meters (m²) to hectares (ha).
    1 ha = 10,000 m².

    Args:
        area_m2 (float): Area in m².

    Returns:
        float: Area in hectares (ha).
    """
    return float(area_m2) / 10_000.0


def ha_to_m2(area_ha: float) -> float:
    """
    Convert area from hectares (ha) to square meters (m²).
    1 ha = 10,000 m².

    Args:
        area_ha (float): Area in hectares (ha).

    Returns:
        float: Area in m².
    """
    return float(area_ha) * 10_000.0


def inr_to_crore(amount_inr: float) -> float:
    """
    Convert Indian Rupees (₹) to Rupees Crore (₹ crore).
    1 crore = 10,000,000 ₹ (1e7).

    Args:
        amount_inr (float): Amount in ₹.

    Returns:
        float: Amount in ₹ crore.
    """
    return float(amount_inr) / 1e7


def read_geotiff(path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a GeoTIFF file into a numpy array and its rasterio profile.

    Args:
        path (str | Path): Path to GeoTIFF file.

    Returns:
        tuple[np.ndarray, dict]: (data array of shape (bands, H, W), rasterio profile).
    """
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()
    return data, profile


def save_geotiff(
    path: Union[str, Path],
    data: np.ndarray,
    profile: Dict[str, Any],
    nodata: Union[float, int, None] = None,
) -> Path:
    """
    Save a 2D (H, W) or 3D (bands, H, W) numpy array as a GeoTIFF.

    Args:
        path (str | Path): Output GeoTIFF path.
        data (np.ndarray): Image array.
        profile (dict): Rasterio profile (CRS, transform, etc.).
        nodata (float | int | None): Optional nodata value to set in profile.

    Returns:
        Path: Path to the saved GeoTIFF.
    """
    out_path = Path(path)
    ensure_dir(out_path.parent)

    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=arr.shape[0],
        height=arr.shape[1],
        width=arr.shape[2],
        dtype=arr.dtype,
        compress="lzw",
    )
    if nodata is not None:
        out_profile["nodata"] = nodata

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(arr)

    return out_path