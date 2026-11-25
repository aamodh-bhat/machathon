"""
M4: Economic loss in ₹ / ₹ crore and draft crop insurance claim.
Reads flooded areas from cache/exposure.json and ALL economic parameters from
config.yaml (no hard-coded assumptions). Saves cache/loss.json.
"""

import json
from pathlib import Path
from typing import Any, Dict

from src.utils import ensure_dir, ha_to_m2, inr_to_crore, load_config


def compute_loss_inr(
    flooded_cropland_ha: float,
    flooded_builtup_ha: float,
    loss_cfg: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute flood damage and draft insurance claim in INR (₹) and ₹ crore.

    Formulas (all parameters from config.yaml -> loss):
      - crop_loss_inr   = flooded_cropland_ha * crop_value_inr_per_ha * crop_loss_ratio
      - built_loss_inr  = flooded_builtup_m2  * value_inr_per_m2      * builtup_damage_ratio
      - total_loss_inr  = crop_loss_inr + built_loss_inr
      - draft_claim_inr = flooded_cropland_ha * sum_insured_inr_per_ha * insured_fraction

    Args:
        flooded_cropland_ha (float): Flooded cropland area in hectares (ha).
        flooded_builtup_ha (float): Flooded built-up area in hectares (ha).
        loss_cfg (dict): 'loss' section of config.yaml.

    Returns:
        dict[str, float]: Loss and claim figures in ₹ and ₹ crore.
    """
    flooded_builtup_m2 = ha_to_m2(flooded_builtup_ha)

    crop_loss_inr = (
        flooded_cropland_ha
        * float(loss_cfg["crop_value_inr_per_ha"])
        * float(loss_cfg["crop_loss_ratio"])
    )
    built_loss_inr = (
        flooded_builtup_m2
        * float(loss_cfg["value_inr_per_m2"])
        * float(loss_cfg["builtup_damage_ratio"])
    )
    total_loss_inr = crop_loss_inr + built_loss_inr

    draft_claim_inr = (
        flooded_cropland_ha
        * float(loss_cfg["sum_insured_inr_per_ha"])
        * float(loss_cfg["insured_fraction"])
    )

    return {
        "crop_loss_inr": round(crop_loss_inr, 2),
        "built_loss_inr": round(built_loss_inr, 2),
        "total_loss_inr": round(total_loss_inr, 2),
        "draft_claim_inr": round(draft_claim_inr, 2),
        "crop_loss_crore": round(inr_to_crore(crop_loss_inr), 2),
        "built_loss_crore": round(inr_to_crore(built_loss_inr), 2),
        "total_loss_crore": round(inr_to_crore(total_loss_inr), 2),
        "draft_claim_crore": round(inr_to_crore(draft_claim_inr), 2),
    }


def run_m4(config_path: str = "config.yaml", cache_dir: str = "cache") -> Path:
    """
    Run Module 4: compute rupee losses for RF (primary) and baseline (comparison)
    using cache/exposure.json and config.yaml, and save cache/loss.json.

    Args:
        config_path (str): Path to config.yaml.
        cache_dir (str): Path to cache directory.

    Returns:
        Path: Path to cache/loss.json.
    """
    cfg = load_config(config_path)
    loss_cfg = cfg["loss"]
    cache = ensure_dir(cache_dir)

    with open(cache / "exposure.json", "r", encoding="utf-8") as f:
        exposure = json.load(f)

    rf_loss = compute_loss_inr(
        exposure["rf"]["cropland_ha"],
        exposure["rf"]["builtup_ha"],
        loss_cfg,
    )
    base_loss = compute_loss_inr(
        exposure["baseline"]["cropland_ha"],
        exposure["baseline"]["builtup_ha"],
        loss_cfg,
    )

    out = {
        "rf": rf_loss,
        "baseline": base_loss,
        "parameters_used": loss_cfg,
        "caveat": "Flooded does not mean destroyed: crop_loss_ratio and builtup_damage_ratio are < 1 and adjustable.",
    }

    out_path = cache / "loss.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Estimated Flood Loss (RF) ===")
    print(
        f"Cropland Loss : ₹ {rf_loss['crop_loss_crore']} crore\n"
        f"Built-up Loss : ₹ {rf_loss['built_loss_crore']} crore\n"
        f"TOTAL LOSS    : ₹ {rf_loss['total_loss_crore']} crore\n"
        f"Draft Claim   : ₹ {rf_loss['draft_claim_crore']} crore"
    )
    print(f"Saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    run_m4()