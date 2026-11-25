"""
M2: Scene clipping, threshold baseline, Random Forest training/evaluation on
Sen1Floods11, and flood inference on the target Assam event.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from src.utils import ensure_dir, linear_to_db, load_config, read_geotiff, save_geotiff


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """
    Compute Otsu's threshold on a 1D array of finite radar backscatter values (dB).

    Args:
        values (np.ndarray): Finite backscatter values in dB.
        bins (int): Number of histogram bins.

    Returns:
        float: Optimal threshold in dB separating dark (water) from bright (land).
    """
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return -15.0
    hist, bin_edges = np.histogram(vals, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-10)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1e-10))[::-1]

    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = int(np.argmax(variance))
    return float(bin_centers[idx])


def download_and_clip_scene(
    item: Dict[str, Any],
    aoi_bbox: List[float],
    out_path: Path,
    target_crs: str = "EPSG:32646",
    resolution_m: float = 20.0,
) -> Path:
    """
    Download VV and VH bands from a Planetary Computer Sentinel-1 RTC item,
    clip to the AOI, reproject to a projected CRS (meters), convert from
    LINEAR POWER to DECIBELS (dB), and save as a 2-band GeoTIFF [VV_dB, VH_dB].

    Args:
        item (dict): STAC item metadata dict from cache/s1_items.json.
        aoi_bbox (list[float]): [lon_min, lat_min, lon_max, lat_max] in EPSG:4326.
        out_path (Path): Destination GeoTIFF path.
        target_crs (str): Target projected CRS in meters (default UTM 46N for Assam).
        resolution_m (float): Output pixel size in meters.

    Returns:
        Path: Path to the saved 2-band GeoTIFF (units: dB).
    """
    if out_path.exists():
        print(f"Using cached scene: {out_path}")
        return out_path

    ensure_dir(out_path.parent)
    # Re-sign URLs in case SAS tokens in s1_items.json expired
    vv_url = pc.sign(item["assets"]["vv"])
    vh_url = pc.sign(item["assets"]["vh"])

    # Compute target grid in projected CRS (meters)
    dst_bounds = transform_bounds("EPSG:4326", target_crs, *aoi_bbox)
    width = int(round((dst_bounds[2] - dst_bounds[0]) / resolution_m))
    height = int(round((dst_bounds[3] - dst_bounds[1]) / resolution_m))
    dst_transform = rasterio.transform.from_bounds(*dst_bounds, width, height)

    bands_db = []
    for band_name, url in [("VV", vv_url), ("VH", vh_url)]:
        print(f"  Fetching {band_name} for {item['id']}...")
        with rasterio.open(url) as src:
            dst_arr = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
            # Sentinel-1 RTC on Planetary Computer is linear power -> convert to dB
            bands_db.append(linear_to_db(dst_arr))

    stack = np.stack(bands_db, axis=0)  # shape: (2, H, W), units: dB
    profile = {
        "driver": "GTiff",
        "crs": target_crs,
        "transform": dst_transform,
        "width": width,
        "height": height,
        "count": 2,
        "dtype": "float32",
        "nodata": np.nan,
    }
    return save_geotiff(out_path, stack, profile, nodata=np.nan)


def extract_features(vv_db: np.ndarray, vh_db: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Build per-pixel single-date feature stack matching Sen1Floods11:
    [vv_db, vh_db, vv_minus_vh, local_mean_vv, local_std_vv].

    Args:
        vv_db (np.ndarray): 2D VV backscatter array in dB.
        vh_db (np.ndarray): 2D VH backscatter array in dB.
        window (int): Local spatial window size in pixels (default 3x3).

    Returns:
        np.ndarray: Feature array of shape (H, W, 5), float32.
    """
    vv = np.nan_to_num(vv_db.astype(np.float32), nan=-25.0)
    vh = np.nan_to_num(vh_db.astype(np.float32), nan=-30.0)
    diff = vv - vh

    local_mean = uniform_filter(vv, size=window)
    local_sq_mean = uniform_filter(vv ** 2, size=window)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0.0))

    return np.stack([vv, vh, diff, local_mean, local_std], axis=-1)


def compute_baseline_flood(
    before_vv_db: np.ndarray,
    after_vv_db: np.ndarray,
    change_thresh_db: float = -2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute binary flood mask using Otsu thresholding + before/after change,
    and remove permanent water detected in the before-scene.

    Args:
        before_vv_db (np.ndarray): 2D pre-flood VV backscatter in dB.
        after_vv_db (np.ndarray): 2D post-flood VV backscatter in dB.
        change_thresh_db (float): Minimum drop in dB (after - before) to count as new flood.

    Returns:
        tuple[np.ndarray, np.ndarray]: (flood_mask uint8 [0/1], permanent_water_mask uint8 [0/1]).
    """
    valid = np.isfinite(before_vv_db) & np.isfinite(after_vv_db)

    # Permanent water: dark pixels in the pre-flood scene
    before_thresh = min(otsu_threshold(before_vv_db[valid]), -15.0)
    perm_water = (before_vv_db < before_thresh) & valid

    # Post-flood water: dark in after-scene AND significantly darker than before
    after_thresh = min(otsu_threshold(after_vv_db[valid]), -14.0)
    diff_db = after_vv_db - before_vv_db
    flood_raw = (after_vv_db < after_thresh) & (diff_db < change_thresh_db) & valid

    flood_mask = (flood_raw & ~perm_water).astype(np.uint8)
    return flood_mask, perm_water.astype(np.uint8)


def _load_sen1floods11_split(
    sen1_dir: Path,
    split_csv: Path,
    max_pixels_per_chip: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sampled (X, y, vv_db) pixels from a Sen1Floods11 split CSV.
    Ignores pixels labelled -1 (no data). Sen1Floods11 S1Hand chips are already in dB.

    Args:
        sen1_dir (Path): Root directory containing S1Hand/ and LabelHand/.
        split_csv (Path): Path to split CSV (e.g. flood_train_data.csv).
        max_pixels_per_chip (int): Max valid pixels sampled per chip for fast training.
        seed (int): Random seed.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (X features, y labels [0/1], raw vv_db).
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(split_csv, header=None, names=["s1_file", "label_file"])

    X_list, y_list, vv_list = [], [], []
    for _, row in df.iterrows():
        s1_path = sen1_dir / "S1Hand" / Path(row["s1_file"]).name
        lbl_path = sen1_dir / "LabelHand" / Path(row["label_file"]).name
        if not s1_path.exists() or not lbl_path.exists():
            continue

        with rasterio.open(s1_path) as src_s1, rasterio.open(lbl_path) as src_lbl:
            s1 = src_s1.read()  # Band 1 = VV (dB), Band 2 = VH (dB)
            lbl = src_lbl.read(1)

        valid = (lbl >= 0) & np.isfinite(s1[0]) & np.isfinite(s1[1])
        ys, xs = np.where(valid)
        if ys.size == 0:
            continue

        if ys.size > max_pixels_per_chip:
            idx = rng.choice(ys.size, size=max_pixels_per_chip, replace=False)
            ys, xs = ys[idx], xs[idx]

        feats = extract_features(s1[0], s1[1])
        X_list.append(feats[ys, xs])
        y_list.append(lbl[ys, xs].astype(np.uint8))
        vv_list.append(s1[0][ys, xs])

    if not X_list:
        raise FileNotFoundError(
            f"No valid Sen1Floods11 chips found under {sen1_dir}. "
            "Check that S1Hand/, LabelHand/, and splits/ are downloaded."
        )

    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(vv_list)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute binary flood-class metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "iou": round(float(jaccard_score(y_true, y_pred, zero_division=0)), 4),
    }


def run_m2(
    config_path: str = "config.yaml",
    cache_dir: str = "cache",
    sen1_dir: str = "data/sen1floods11",
) -> Path:
    """
    Run Module 2: clip before/after scenes, compute baseline flood map, train/eval
    Random Forest on Sen1Floods11, run RF inference on Assam, and save metrics.json.

    Args:
        config_path (str): Path to config.yaml.
        cache_dir (str): Path to cache directory.
        sen1_dir (str): Path to Sen1Floods11 directory.

    Returns:
        Path: Path to cache/metrics.json.
    """
    cfg = load_config(config_path)
    ev = cfg["event"]
    fm = cfg["flood_model"]
    cache = ensure_dir(cache_dir)

    # 1. Select before and after scenes from M1 output
    with open(cache / "s1_items.json", "r", encoding="utf-8") as f:
        s1_meta = json.load(f)

    onset_dt = datetime.strptime(ev["onset_date"], "%Y-%m-%d")
    thresh = ev.get("coverage_threshold", 0.80)
    usable = [it for it in s1_meta["items"] if it["aoi_coverage"] >= thresh]

    before_items = [it for it in usable if datetime.fromisoformat(it["datetime"][:10]) < onset_dt]
    after_items = [it for it in usable if datetime.fromisoformat(it["datetime"][:10]) >= onset_dt]

    if not before_items or not after_items:
        raise RuntimeError("Need at least one usable before-scene and one usable after-scene in s1_items.json.")

    before_item = before_items[-1]  # last good pass before onset
    after_item = after_items[0]     # first good pass after onset
    print(f"Selected BEFORE scene: {before_item['datetime']} ({before_item['id']})")
    print(f"Selected AFTER  scene: {after_item['datetime']} ({after_item['id']})")

    before_path = download_and_clip_scene(before_item, ev["aoi_bbox"], cache / "scenes" / "before_s1.tif")
    after_path = download_and_clip_scene(after_item, ev["aoi_bbox"], cache / "scenes" / "after_s1.tif")

    before_data, profile = read_geotiff(before_path)
    after_data, _ = read_geotiff(after_path)

    # 2. Threshold baseline flood map
    baseline_mask, perm_water = compute_baseline_flood(before_data[0], after_data[0])
    save_geotiff(cache / "flood_baseline.tif", baseline_mask, profile, nodata=0)
    print(f"Saved baseline flood map: {int(baseline_mask.sum())} flooded pixels")

    # 3. Train & evaluate Random Forest on Sen1Floods11
    s11_root = Path(sen1_dir)
    print("Loading Sen1Floods11 train and held-out test splits...")
    X_train, y_train, _ = _load_sen1floods11_split(s11_root, s11_root / "splits" / "flood_train_data.csv")
    X_test, y_test, vv_test = _load_sen1floods11_split(
        s11_root, s11_root / "splits" / "flood_test_data.csv", max_pixels_per_chip=5000
    )

    print(f"Training RandomForest on {len(y_train):,} pixels...")
    rf = RandomForestClassifier(
        n_estimators=fm.get("rf_n_estimators", 200),
        class_weight=fm.get("class_weight", "balanced"),
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    model_path = cache / "rf_model.joblib"
    joblib.dump(rf, model_path, compress=3)
    model_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)

    # Evaluate both baseline (Otsu on VV) and RF on the held-out test split
    test_otsu = otsu_threshold(vv_test)
    y_pred_base = (vv_test < test_otsu).astype(np.uint8)
    y_pred_rf = rf.predict(X_test)

    metrics_base = _classification_metrics(y_test, y_pred_base)
    metrics_rf = _classification_metrics(y_test, y_pred_rf)

    # 4. Run RF inference on Assam after-scene (remove permanent water)
    H, W = after_data[0].shape
    feats_assam = extract_features(after_data[0], after_data[1]).reshape(-1, 5)
    valid_assam = np.isfinite(after_data[0]).reshape(-1)

    rf_pred_flat = np.zeros(H * W, dtype=np.uint8)
    rf_pred_flat[valid_assam] = rf.predict(feats_assam[valid_assam])
    rf_mask = (rf_pred_flat.reshape(H, W) & ~perm_water.astype(bool)).astype(np.uint8)

    save_geotiff(cache / "flood_rf.tif", rf_mask, profile, nodata=0)
    print(f"Saved RF flood map: {int(rf_mask.sum())} flooded pixels")

    # Consistency check between baseline and RF on Assam (unlabelled)
    assam_iou = round(float(jaccard_score(baseline_mask.ravel(), rf_mask.ravel(), zero_division=0)), 4)

    metrics_out = {
        "before_scene": before_item["datetime"],
        "after_scene": after_item["datetime"],
        "test_split": "Sen1Floods11 held-out test (Bolivia)",
        "baseline_test": metrics_base,
        "rf_test": metrics_rf,
        "assam_baseline_vs_rf_iou": assam_iou,
        "rf_model_size_mb": model_size_mb,
        "domain_shift_note": "RF trained on Sen1Floods11 global events, not Assam; Assam agreement is a consistency check, not ground-truth accuracy.",
    }

    metrics_path = cache / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print("\n=== Held-Out Test Metrics (Sen1Floods11) ===")
    print("Baseline:", metrics_base)
    print("RF      :", metrics_rf)
    print(f"Assam Baseline vs RF IoU: {assam_iou} | Model size: {model_size_mb} MB")
    return metrics_path


if __name__ == "__main__":
    run_m2()