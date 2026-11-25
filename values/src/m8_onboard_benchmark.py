"""
M8 (Part B): Train & Benchmark Onboard-Edge AI.
Trains a compressed Random Forest (depth-limited for edge/flight memory budgets)
on Sen1Floods11, evaluates it on the held-out Bolivia test split, and compares
size, speed, accuracy, and downlink bandwidth against the full Ground RF.
Saves cache/rf_onboard.joblib and cache/onboard_benchmark.json.
"""

import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.m2_flood import (
    _classification_metrics,
    _load_sen1floods11_split,
    extract_features,
)
from src.utils import ensure_dir, read_geotiff


def run_m8(
    cache_dir: str = "cache",
    sen1_dir: str = "data/sen1floods11",
) -> Path:
    """Train the onboard-edge RF, benchmark against Ground RF, and save results."""
    cache = ensure_dir(cache_dir)
    s11_root = Path(sen1_dir)

    ground_model_path = cache / "rf_model.joblib"
    onboard_model_path = cache / "rf_onboard.joblib"
    scene_path = cache / "scenes" / "after_s1.tif"

    with open(cache / "exposure.json", "r", encoding="utf-8") as f:
        exposure = json.load(f)
    with open(cache / "loss.json", "r", encoding="utf-8") as f:
        loss = json.load(f)
    with open(cache / "observation_gap.json", "r", encoding="utf-8") as f:
        gap = json.load(f)
    with open(cache / "metrics.json", "r", encoding="utf-8") as f:
        m2_metrics = json.load(f)

    # 1. Train compressed Onboard-Edge RF on Sen1Floods11
    print("Loading Sen1Floods11 splits for Onboard-Edge RF training...")
    X_train, y_train, _ = _load_sen1floods11_split(
        s11_root, s11_root / "splits" / "flood_train_data.csv"
    )
    X_test, y_test, _ = _load_sen1floods11_split(
        s11_root, s11_root / "splits" / "flood_test_data.csv", max_pixels_per_chip=5000
    )

    print("Training compressed Onboard-Edge RF (50 trees, max_depth=10)...")
    rf_onboard = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf_onboard.fit(X_train, y_train)
    joblib.dump(rf_onboard, onboard_model_path, compress=3)

    onboard_test_metrics = _classification_metrics(y_test, rf_onboard.predict(X_test))

    # 2. Model footprints (MB)
    ground_size_mb = round(ground_model_path.stat().st_size / (1024 * 1024), 2)
    onboard_size_mb = round(onboard_model_path.stat().st_size / (1024 * 1024), 2)
    size_reduction_pct = round((1.0 - (onboard_size_mb / max(ground_size_mb, 1e-6))) * 100.0, 2)

    # 3. Inference speed comparison on Assam scene (ground CPU proxy)
    rf_ground = joblib.load(ground_model_path)
    after_data, _ = read_geotiff(scene_path)
    valid = np.isfinite(after_data[0]).reshape(-1)
    feats = extract_features(after_data[0], after_data[1]).reshape(-1, 5)
    feats_valid = feats[valid]

    t0 = time.perf_counter()
    _ = rf_ground.predict(feats_valid)
    ground_inf_sec = round(time.perf_counter() - t0, 2)

    t1 = time.perf_counter()
    _ = rf_onboard.predict(feats_valid)
    onboard_inf_sec = round(time.perf_counter() - t1, 2)
    speedup_x = round(ground_inf_sec / max(onboard_inf_sec, 0.01), 2)

    # 4. Downlink compression (raw SAR scene vs. compact JSON flood alert)
    raw_bytes = scene_path.stat().st_size
    raw_scene_mb = round(raw_bytes / (1024 * 1024), 2)

    alert_payload = {
        "onset_date": gap["onset_date"],
        "first_pass_days": gap["first_usable_pass_days"],
        "flooded_ha": exposure["rf"],
        "loss_crore": {
            "crop": loss["rf"]["crop_loss_crore"],
            "builtup": loss["rf"]["built_loss_crore"],
            "total": loss["rf"]["total_loss_crore"],
            "draft_claim": loss["rf"]["draft_claim_crore"],
        },
    }
    alert_bytes = len(json.dumps(alert_payload, separators=(",", ":")).encode("utf-8"))
    alert_payload_kb = round(alert_bytes / 1024.0, 2)
    downlink_reduction_pct = round((1.0 - (alert_bytes / raw_bytes)) * 100.0, 3)

    out = {
        "ground_rf": {
            "model_size_mb": ground_size_mb,
            "inference_sec": ground_inf_sec,
            "test_metrics": m2_metrics["rf_test"],
        },
        "onboard_rf": {
            "model_size_mb": onboard_size_mb,
            "inference_sec": onboard_inf_sec,
            "test_metrics": onboard_test_metrics,
            "hyperparams": "50 trees, max_depth=10, min_samples_leaf=5",
        },
        "size_reduction_pct": size_reduction_pct,
        "speedup_x": speedup_x,
        "raw_scene_mb": raw_scene_mb,
        "alert_payload_kb": alert_payload_kb,
        "downlink_reduction_pct": downlink_reduction_pct,
        "pixels_scored": int(valid.sum()),
        "proxy_note": "Timed on laptop CPU as a ground proxy; not executed on flight hardware.",
    }

    out_path = cache / "onboard_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Part B: Ground RF vs. Trained Onboard-Edge RF ===")
    print(f"Size       : {ground_size_mb} MB -> {onboard_size_mb} MB ({size_reduction_pct}% smaller)")
    print(f"Speed      : {ground_inf_sec} s -> {onboard_inf_sec} s ({speedup_x}x faster)")
    print(f"Bolivia F1 : {m2_metrics['rf_test']['f1']} -> {onboard_test_metrics['f1']}")
    print(f"Downlink   : {raw_scene_mb} MB -> {alert_payload_kb} KB (>99.9% saved)")
    print(f"Saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    run_m8()