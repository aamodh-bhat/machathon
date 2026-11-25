"""
M3: Exposure split using ESA WorldCover (10m land cover).
Fetches and clips ESA WorldCover to the flood mask grid (nearest neighbour),
computes flooded hectares by class (cropland, built-up, other) for both RF
and baseline masks, and saves cache/exposure.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import planetary_computer as pc
from pystac_client import Client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.utils import ensure_dir, load_config, m2_to_ha, read_geotiff, save_geotiff

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CLASS_CROPLAND = 40
CLASS_BUILTUP = 50


def fetch_worldcover_to_grid(
    aoi_bbox: List[float],
    ref_profile: Dict[str, Any],
    out_path: Path,
) -> np.ndarray:
    """
    Fetch ESA WorldCover (2021 v200 preferred) from Planetary Computer, resample
    to match the flood mask's exact CRS/transform/shape using nearest neighbour,
    and cache as a uint8 GeoTIFF.

    Args:
        aoi_bbox (list[float]): [lon_min, lat_min, lon_max, lat_max] in EPSG:4326.
        ref_profile (dict): Rasterio profile of the flood mask (projected CRS in meters).
        out_path (Path): Cache path for the aligned WorldCover GeoTIFF.

    Returns:
        np.ndarray: 2D uint8 array of ESA WorldCover class codes aligned to flood mask.
    """
    if out_path.exists():
        print(f"Using cached WorldCover: {out_path}")
        wc_data, _ = read_geotiff(out_path)
        return wc_data[0]

    catalog = Client.open(STAC_URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=aoi_bbox,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No ESA WorldCover items found for this AOI.")

    # Prefer 2021 (v200) tiles if available
    items_2021 = [it for it in items if "2021" in it.id]
    selected_items = items_2021 if items_2021 else items

    height = ref_profile["height"]
    width = ref_profile["width"]
    dst_crs = ref_profile["crs"]
    dst_transform = ref_profile["transform"]

    wc_aligned = np.zeros((height, width), dtype=np.uint8)

    for item in selected_items:
        href = pc.sign(item.assets["map"].href)
        print(f"  Fetching ESA WorldCover tile: {item.id}...")
        with rasterio.open(href) as src:
            tmp = np.zeros((height, width), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=tmp,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                dst_nodata=0,
            )
            # Fill any pixels not yet populated by an earlier tile
            fill_mask = (wc_aligned == 0) & (tmp > 0)
            wc_aligned[fill_mask] = tmp[fill_mask]

    save_geotiff(out_path, wc_aligned, ref_profile, nodata=0)
    return wc_aligned


def compute_class_areas_ha(
    flood_mask: np.ndarray,
    worldcover: np.ndarray,
    pixel_area_m2: float,
) -> Dict[str, float]:
    """
    Compute flooded area in HECTARES (ha) split into cropland, built-up, and other.

    Args:
        flood_mask (np.ndarray): 2D binary flood mask (1 = flooded, 0 = not flooded).
        worldcover (np.ndarray): 2D ESA WorldCover class array aligned to flood_mask.
        pixel_area_m2 (float): Area of one pixel in square meters (m²).

    Returns:
        dict[str, float]: Flooded areas in hectares (cropland_ha, builtup_ha, other_ha, total_flooded_ha).
    """
    flooded = flood_mask == 1
    crop_px = int(np.sum(flooded & (worldcover == CLASS_CROPLAND)))
    built_px = int(np.sum(flooded & (worldcover == CLASS_BUILTUP)))
    other_px = int(np.sum(flooded & (worldcover != CLASS_CROPLAND) & (worldcover != CLASS_BUILTUP)))

    crop_ha = m2_to_ha(crop_px * pixel_area_m2)
    built_ha = m2_to_ha(built_px * pixel_area_m2)
    other_ha = m2_to_ha(other_px * pixel_area_m2)

    return {
        "cropland_ha": round(crop_ha, 2),
        "builtup_ha": round(built_ha, 2),
        "other_ha": round(other_ha, 2),
        "total_flooded_ha": round(crop_ha + built_ha + other_ha, 2),
    }


def run_m3(config_path: str = "config.yaml", cache_dir: str = "cache") -> Path:
    """
    Run Module 3: align ESA WorldCover to flood masks, compute exposure in hectares,
    and save cache/exposure.json.

    Args:
        config_path (str): Path to config.yaml.
        cache_dir (str): Path to cache directory.

    Returns:
        Path: Path to cache/exposure.json.
    """
    cfg = load_config(config_path)
    ev = cfg["event"]
    cache = ensure_dir(cache_dir)

    rf_data, profile = read_geotiff(cache / "flood_rf.tif")
    base_data, _ = read_geotiff(cache / "flood_baseline.tif")

    # Pixel area in m² from projected CRS transform (UTM meters)
    transform = profile["transform"]
    pixel_area_m2 = abs(float(transform.a * transform.e))

    wc = fetch_worldcover_to_grid(ev["aoi_bbox"], profile, cache / "scenes" / "worldcover.tif")

    rf_exposure = compute_class_areas_ha(rf_data[0], wc, pixel_area_m2)
    base_exposure = compute_class_areas_ha(base_data[0], wc, pixel_area_m2)

    # Total AOI area per class (hectares)
    aoi_crop_ha = m2_to_ha(int(np.sum(wc == CLASS_CROPLAND)) * pixel_area_m2)
    aoi_built_ha = m2_to_ha(int(np.sum(wc == CLASS_BUILTUP)) * pixel_area_m2)
    aoi_other_ha = m2_to_ha(
        int(np.sum((wc != CLASS_CROPLAND) & (wc != CLASS_BUILTUP))) * pixel_area_m2
    )

    out = {
        "units": "hectares (ha)",
        "pixel_area_m2": round(pixel_area_m2, 2),
        "worldcover_classes": {"cropland": CLASS_CROPLAND, "builtup": CLASS_BUILTUP},
        "rf": rf_exposure,
        "baseline": base_exposure,
        "aoi_total": {
            "cropland_ha": round(aoi_crop_ha, 2),
            "builtup_ha": round(aoi_built_ha, 2),
            "other_ha": round(aoi_other_ha, 2),
            "total_aoi_ha": round(aoi_crop_ha + aoi_built_ha + aoi_other_ha, 2),
        },
        "radar_limit_note": "C-band SAR poorly detects flooded urban areas and dense vegetation; built-up flood estimates are less reliable than cropland estimates.",
    }

    out_path = cache / "exposure.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Flood Exposure (RF, Hectares) ===")
    print(json.dumps(rf_exposure, indent=2))
    print(f"Saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    run_m3()