from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

log = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
ANNOTATION_EXTENSIONS = {".xml", ".json", ".txt"}
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def _read_image_meta(path: Path) -> Optional[Dict[str, Any]]:
    try:
        from PIL import Image
        with Image.open(path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "channels": len(img.getbands()),
                "is_grayscale": img.mode in ("L", "LA"),
            }
    except Exception as exc:
        log.warning(f"Corrupted / unreadable image {path.name}: {exc}")
        return None


def _validate_yolo_line(line: str) -> Tuple[bool, str]:
    parts = line.strip().split()
    if len(parts) != 5:
        return False, f"Expected 5 fields, got {len(parts)}"
    try:
        cls_id = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return False, "Non-numeric values"
    if cls_id < 0:
        return False, f"Negative class id ({cls_id})"
    if not (0.0 <= xc <= 1.0):
        return False, f"x_center out of range ({xc})"
    if not (0.0 <= yc <= 1.0):
        return False, f"y_center out of range ({yc})"
    if not (0.0 < w <= 1.0):
        return False, f"width out of range ({w})"
    if not (0.0 < h <= 1.0):
        return False, f"height out of range ({h})"
    return True, "OK"


def validate_dataset(
    data_dir: Path,
    class_names: List[str] = CLASS_NAMES,
    check_duplicates: bool = True,
    output_report: Optional[Path] = None,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    log.info(f"Starting dataset validation for: {data_dir}")

    all_images = sorted(
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )
    all_annotations = sorted(
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in ANNOTATION_EXTENSIONS and p.is_file()
    )

    log.info(f"Found {len(all_images)} image files, {len(all_annotations)} annotation files")

    widths, heights, modes = [], [], []
    corrupted: List[str] = []
    class_counts: Dict[str, int] = defaultdict(int)
    image_class_map: Dict[str, str] = {}

    for img_path in all_images:
        for part in img_path.parts:
            if part.lower() in [c.lower() for c in class_names]:
                matched = next(c for c in class_names if c.lower() == part.lower())
                class_counts[matched] += 1
                image_class_map[img_path.name] = matched
                break

        meta = _read_image_meta(img_path)
        if meta is None:
            corrupted.append(str(img_path))
        else:
            widths.append(meta["width"])
            heights.append(meta["height"])
            modes.append(meta["mode"])

    duplicates: List[List[str]] = []
    if check_duplicates and all_images:
        log.info("Computing MD5 hashes for duplicate detection…")
        hash_map: Dict[str, List[str]] = defaultdict(list)
        for img_path in all_images:
            try:
                h = _md5(img_path)
                hash_map[h].append(str(img_path))
            except OSError:
                pass
        duplicates = [paths for paths in hash_map.values() if len(paths) > 1]
        if duplicates:
            log.warning(f"Found {len(duplicates)} duplicate image group(s).")

    annotation_stats: Dict[str, Any] = {
        "xml_count": 0,
        "json_count": 0,
        "txt_count": 0,
        "missing_image_for_annotation": [],
        "invalid_yolo": [],
        "missing_annotation_for_image": [],
        "total_boxes": 0,
        "boxes_per_class": defaultdict(int),
        "bbox_widths": [],
        "bbox_heights": [],
    }

    image_stems = {p.stem for p in all_images}

    for ann_path in all_annotations:
        ext = ann_path.suffix.lower()
        if ext == ".xml":
            annotation_stats["xml_count"] += 1
        elif ext == ".json":
            annotation_stats["json_count"] += 1
        elif ext == ".txt":
            annotation_stats["txt_count"] += 1

        if ann_path.stem not in image_stems:
            annotation_stats["missing_image_for_annotation"].append(str(ann_path))
        try:
            lines = ann_path.read_text(encoding="utf-8").strip().splitlines()
            lines = [l for l in lines if l.strip()]
            for line in lines:
                valid, reason = _validate_yolo_line(line)
                if valid:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])
                    annotation_stats["total_boxes"] += 1
                    if 0 <= cls_id < len(class_names):
                        annotation_stats["boxes_per_class"][class_names[cls_id]] += 1
                    annotation_stats["bbox_widths"].append(w)
                    annotation_stats["bbox_heights"].append(h)
                else:
                    annotation_stats["invalid_yolo"].append(
                        {"file": str(ann_path), "line": line, "reason": reason}
                    )
        except Exception as exc:
            log.warning(f"Could not read annotation {ann_path}: {exc}")

    ann_stems = {p.stem for p in all_annotations}
    annotation_stats["missing_annotation_for_image"] = [
        str(p) for p in all_images if p.stem not in ann_stems
    ]

    import statistics as stats_lib

    def _safe_stat(values, fn):
        try:
            return round(fn(values), 2) if values else 0
        except Exception:
            return 0

    report: Dict[str, Any] = {
        "data_dir": str(data_dir),
        "total_images": len(all_images),
        "readable_images": len(all_images) - len(corrupted),
        "corrupted_images": len(corrupted),
        "corrupted_files": corrupted,
        "image_widths": {
            "min": _safe_stat(widths, min),
            "max": _safe_stat(widths, max),
            "mean": _safe_stat(widths, stats_lib.mean),
            "unique_count": len(set(widths)),
        },
        "image_heights": {
            "min": _safe_stat(heights, min),
            "max": _safe_stat(heights, max),
            "mean": _safe_stat(heights, stats_lib.mean),
            "unique_count": len(set(heights)),
        },
        "image_modes": dict(
            zip(*[list(x) for x in zip(*[(m, modes.count(m)) for m in set(modes)])])
        ) if modes else {},
        "class_distribution": dict(class_counts),
        "class_names": class_names,
        "total_annotations": annotation_stats["txt_count"] + annotation_stats["xml_count"] + annotation_stats["json_count"],
        "total_xml_annotations": annotation_stats["xml_count"],
        "total_json_annotations": annotation_stats["json_count"],
        "total_yolo_labels": annotation_stats["txt_count"],
        "total_bounding_boxes": annotation_stats["total_boxes"],
        "boxes_per_class": dict(annotation_stats["boxes_per_class"]),
        "invalid_yolo_lines": len(annotation_stats["invalid_yolo"]),
        "invalid_yolo_samples": annotation_stats["invalid_yolo"][:20],
        "images_missing_annotation": len(annotation_stats["missing_annotation_for_image"]),
        "annotations_missing_image": len(annotation_stats["missing_image_for_annotation"]),
        "duplicate_groups": len(duplicates),
        "duplicate_samples": duplicates[:10],
        "has_detection_annotations": (
            annotation_stats["xml_count"] > 0
            or annotation_stats["total_boxes"] > 0
        ),
        "annotation_format_detected": _detect_annotation_format(annotation_stats),
    }

    _print_report(report)

    if output_report:
        output_report = Path(output_report)
        output_report.parent.mkdir(parents=True, exist_ok=True)

        report_json = {
            k: dict(v) if isinstance(v, defaultdict) else v
            for k, v in report.items()
        }
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2, default=str)
        log.info(f"Dataset validation report saved → {output_report}")

    return report


def _detect_annotation_format(ann_stats: Dict[str, Any]) -> str:
    if ann_stats["xml_count"] > 0 and ann_stats["txt_count"] == 0:
        return "pascal_voc_xml"
    if ann_stats["json_count"] > 0:
        return "coco_json_or_other"
    if ann_stats["txt_count"] > 0:
        return "yolo_txt"
    return "none"


def _print_report(report: Dict[str, Any]) -> None:
    sep = "═" * 62
    print(f"\n{sep}")
    print("  DATASET VALIDATION REPORT")
    print(sep)
    print(f"  Directory      : {report['data_dir']}")
    print(f"  Total Images   : {report['total_images']}")
    print(f"  Readable       : {report['readable_images']}")
    print(f"  Corrupted      : {report['corrupted_images']}")

    print("\n  ── Image Dimensions ──────────────────────────────")
    print(f"  Width   min/max/mean : {report['image_widths']['min']} / {report['image_widths']['max']} / {report['image_widths']['mean']}")
    print(f"  Height  min/max/mean : {report['image_heights']['min']} / {report['image_heights']['max']} / {report['image_heights']['mean']}")
    print(f"  Colour modes : {report['image_modes']}")

    print("\n  ── Class Distribution ────────────────────────────")
    total = sum(report["class_distribution"].values()) or 1
    for cls, cnt in sorted(report["class_distribution"].items()):
        bar = "█" * int(30 * cnt / total)
        print(f"  {cls:<20} {cnt:>5}  {bar}")

    print("\n  ── Annotations ───────────────────────────────────")
    print(f"  Format detected : {report['annotation_format_detected']}")
    print(f"  XML files       : {report['total_xml_annotations']}")
    print(f"  YOLO TXT files  : {report['total_yolo_labels']}")
    print(f"  Total boxes     : {report['total_bounding_boxes']}")
    print(f"  Invalid YOLO    : {report['invalid_yolo_lines']}")
    print(f"  Missing labels  : {report['images_missing_annotation']}")
    print(f"  Orphan labels   : {report['annotations_missing_image']}")
    print(f"  Duplicate groups: {report['duplicate_groups']}")

    if not report["has_detection_annotations"]:
        print("\n  ⚠  NO DETECTION ANNOTATIONS FOUND")
        print("     Object detection training requires bounding-box annotations.")
        print("     If Pascal VOC XML files exist, run: python scripts/prepare_dataset.py")
        print("     to convert them to YOLO format.")
    print(f"{sep}\n")
