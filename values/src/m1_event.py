"""
M1: Event selection, Sentinel-1 STAC availability check, and scene caching.
Queries Microsoft Planetary Computer STAC API and saves cache/s1_items.json.
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import planetary_computer as pc
from pystac_client import Client
from shapely.geometry import box, shape

from src.utils import ensure_dir, load_config

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def compute_aoi_coverage(item_geometry: Dict[str, Any], aoi_bbox: List[float]) -> float:
    """
    Compute the fraction of the AOI bounding box covered by a STAC item's footprint.

    Args:
        item_geometry (dict): GeoJSON geometry of the STAC item (EPSG:4326 degrees).
        aoi_bbox (list[float]): [lon_min, lat_min, lon_max, lat_max] in EPSG:4326.

    Returns:
        float: Coverage fraction in [0.0, 1.0].
    """
    aoi_poly = box(*aoi_bbox)
    item_poly = shape(item_geometry)
    if aoi_poly.area == 0:
        return 0.0
    intersection_area = aoi_poly.intersection(item_poly).area
    return float(min(1.0, intersection_area / aoi_poly.area))


def query_sentinel1_items(
    aoi_bbox: List[float],
    onset_date: str,
    window_days_before: int = 30,
    window_days_after: int = 30,
    collection: str = "sentinel-1-rtc",
) -> List[Dict[str, Any]]:
    """
    Query Planetary Computer STAC API for Sentinel-1 items intersecting the AOI.

    Args:
        aoi_bbox (list[float]): [lon_min, lat_min, lon_max, lat_max] in EPSG:4326.
        onset_date (str): Flood onset date in "YYYY-MM-DD" format.
        window_days_before (int): Days before onset to search.
        window_days_after (int): Days after onset to search.
        collection (str): STAC collection name ("sentinel-1-rtc" or "sentinel-1-grd").

    Returns:
        list[dict]: Sorted list of acquisition metadata dictionaries.
    """
    onset_dt = datetime.strptime(onset_date, "%Y-%m-%d")
    start_str = (onset_dt - timedelta(days=window_days_before)).strftime("%Y-%m-%d")
    end_str = (onset_dt + timedelta(days=window_days_after)).strftime("%Y-%m-%d")

    catalog = Client.open(STAC_URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=[collection],
        bbox=aoi_bbox,
        datetime=f"{start_str}/{end_str}",
    )

    items = list(search.items())
    records: List[Dict[str, Any]] = []

    for item in items:
        props = item.properties
        dt_str = props.get("datetime")
        cov = compute_aoi_coverage(item.geometry, aoi_bbox)
        records.append(
            {
                "id": item.id,
                "collection": collection,
                "datetime": dt_str,
                "orbit_direction": props.get("sat:orbit_state", "unknown"),
                "relative_orbit": props.get("sat:relative_orbit", -1),
                "platform": props.get("platform", "unknown"),
                "aoi_coverage": round(cov, 4),
                "assets": {
                    k: v.href for k, v in item.assets.items() if k in ("vv", "vh")
                },
            }
        )

    records.sort(key=lambda r: r["datetime"])
    return records


def run_m1(config_path: str = "config.yaml", cache_dir: str = "cache") -> Path:
    """
    Run Module 1: query STAC, print acquisition table, and save cache/s1_items.json.

    Args:
        config_path (str): Path to config.yaml.
        cache_dir (str): Path to cache directory.

    Returns:
        Path: Path to saved s1_items.json.
    """
    cfg = load_config(config_path)
    ev = cfg["event"]

    records = query_sentinel1_items(
        aoi_bbox=ev["aoi_bbox"],
        onset_date=ev["onset_date"],
        window_days_before=ev.get("window_days_before", 30),
        window_days_after=ev.get("window_days_after", 30),
        collection=ev.get("s1_collection", "sentinel-1-rtc"),
    )

    out_dir = ensure_dir(cache_dir)
    out_path = out_dir / "s1_items.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "event_name": ev["name"],
                "onset_date": ev["onset_date"],
                "aoi_bbox": ev["aoi_bbox"],
                "coverage_threshold": ev.get("coverage_threshold", 0.80),
                "items": records,
            },
            f,
            indent=2,
        )

    if not records:
        print("WARNING: No Sentinel-1 items found for this AOI and date window.")
        return out_path

    df = pd.DataFrame(records)[
        ["datetime", "platform", "orbit_direction", "relative_orbit", "aoi_coverage", "id"]
    ]
    print(f"\n=== Sentinel-1 Acquisitions for {ev['name']} (Onset: {ev['onset_date']}) ===")
    print(df.to_string(index=False))

    thresh = ev.get("coverage_threshold", 0.80)
    usable = df[df["aoi_coverage"] >= thresh]
    print(f"\nTotal items: {len(df)} | Usable (>= {thresh:.0%} AOI coverage): {len(usable)}")
    print(f"Saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    run_m1()