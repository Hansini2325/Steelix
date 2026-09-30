from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def iou(box1: List[float], box2: List[float]) -> float:
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def perform_error_analysis(
    predictions_json: Path,
    ground_truth_json: Path,
    output_dir: Path,
    iou_thresh: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:

    predictions_json = Path(predictions_json)
    ground_truth_json = Path(ground_truth_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args_ok(predictions_json, ground_truth_json):
        return {}

    with open(predictions_json, "r") as f:
        preds = json.load(f)
    with open(ground_truth_json, "r") as f:
        gts = json.load(f)

    pred_by_img = {p["image_path"]: p["detections"] for p in preds}
    gt_by_img = {g["image_path"]: g["detections"] for g in gts}
    all_images = set(pred_by_img.keys()) | set(gt_by_img.keys())

    errors: List[Dict[str, Any]] = []

    for img in sorted(all_images):
        p_dets = pred_by_img.get(img, [])
        g_dets = gt_by_img.get(img, [])

        g_matched = [False] * len(g_dets)
        p_matched = [False] * len(p_dets)

        for pi, p in enumerate(p_dets):
            best_iou = 0.0
            best_gi = -1

            for gi, g in enumerate(g_dets):
                if g_matched[gi]:
                    continue
                v_iou = iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if v_iou > best_iou:
                    best_iou = v_iou
                    best_gi = gi

            if best_iou >= iou_thresh:
                g = g_dets[best_gi]
                if p["class_id"] != g["class_id"]:
                    errors.append({
                        "type": "misclassification",
                        "image_path": img,
                        "pred_class": p["class_name"],
                        "true_class": g["class_name"],
                        "confidence": p["confidence"],
                        "iou": best_iou,
                        "pred_box": p["bbox_xyxy"],
                        "true_box": g["bbox_xyxy"],
                    })
                g_matched[best_gi] = True
                p_matched[pi] = True
            else:
                if best_iou > 0.1:
                    g = g_dets[best_gi]
                    errors.append({
                        "type": "localization_error",
                        "image_path": img,
                        "pred_class": p["class_name"],
                        "true_class": g.get("class_name", "unknown"),
                        "confidence": p["confidence"],
                        "iou": best_iou,
                        "pred_box": p["bbox_xyxy"],
                        "true_box": g["bbox_xyxy"],
                    })

        for pi, p_match in enumerate(p_matched):
            if not p_match:
                p = p_dets[pi]
                errors.append({
                    "type": "false_positive",
                    "image_path": img,
                    "pred_class": p["class_name"],
                    "confidence": p["confidence"],
                    "pred_box": p["bbox_xyxy"],
                })

        for gi, g_match in enumerate(g_matched):
            if not g_match:
                g = g_dets[gi]
                errors.append({
                    "type": "false_negative",
                    "image_path": img,
                    "true_class": g["class_name"],
                    "true_box": g["bbox_xyxy"],
                })

    summary = {
        "false_positives": sum(1 for e in errors if e["type"] == "false_positive"),
        "false_negatives": sum(1 for e in errors if e["type"] == "false_negative"),
        "misclassifications": sum(1 for e in errors if e["type"] == "misclassification"),
        "localization_errors": sum(1 for e in errors if e["type"] == "localization_error"),
        "total_errors": len(errors),
    }

    log.info(f"Error Analysis Summary: {summary}")
    out_file = output_dir / "error_analysis.json"
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "errors": errors}, f, indent=2)
    log.info(f"Detailed error analysis saved -> {out_file}")

    return summary


def args_ok(p: Path, g: Path) -> bool:
    if not p.exists():
        log.warning(f"Predictions file not found: {p}")
        return False
    if not g.exists():
        log.warning(f"Ground truth file not found: {g}")
        return False
    return True
