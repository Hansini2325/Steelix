from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

log = get_logger(__name__)

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]


def validate_yolo_bbox(xc: float, yc: float, w: float, h: float) -> Tuple[bool, str]:
    if not (0.0 <= xc <= 1.0):
        return False, f"x_center={xc:.4f} out of [0,1]"
    if not (0.0 <= yc <= 1.0):
        return False, f"y_center={yc:.4f} out of [0,1]"
    if not (0.0 < w <= 1.0):
        return False, f"width={w:.4f} out of (0,1]"
    if not (0.0 < h <= 1.0):
        return False, f"height={h:.4f} out of (0,1]"
    return True, "OK"


def clip_bbox(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(1e-4, min(1.0, w))
    h = max(1e-4, min(1.0, h))
    return xc, yc, w, h


def _parse_voc_xml(xml_path: Path) -> Optional[Dict]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        log.warning(f"Failed to parse XML {xml_path}: {exc}")
        return None

    size = root.find("size")
    if size is None:
        log.warning(f"No <size> element in {xml_path}")
        return None

    try:
        img_w = int(size.findtext("width", "0"))
        img_h = int(size.findtext("height", "0"))
    except ValueError:
        log.warning(f"Invalid width/height in {xml_path}")
        return None

    if img_w <= 0 or img_h <= 0:
        log.warning(f"Zero or negative image dimensions in {xml_path}: {img_w}x{img_h}")
        return None

    filename = root.findtext("filename", xml_path.stem)
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            x1 = float(bndbox.findtext("xmin", "0"))
            y1 = float(bndbox.findtext("ymin", "0"))
            x2 = float(bndbox.findtext("xmax", "0"))
            y2 = float(bndbox.findtext("ymax", "0"))
        except ValueError:
            continue
        objects.append({"name": name, "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2})

    return {"filename": filename, "width": img_w, "height": img_h, "objects": objects}


def voc_to_yolo_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float,
    img_w: int, img_h: int,
) -> Tuple[float, float, float, float]:
    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return xc, yc, w, h


def convert_voc_to_yolo(
    xml_dir: Path,
    output_dir: Path,
    class_names: List[str] = CLASS_NAMES,
    skip_unknown_classes: bool = True,
) -> Dict[str, int]:
    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.rglob("*.xml"))
    if not xml_files:
        log.warning(f"No XML files found in {xml_dir}")
        return {"converted": 0, "skipped_files": 0, "invalid_boxes": 0, "unknown_classes": 0}

    log.info(f"Converting {len(xml_files)} VOC XML files -> YOLO TXT in {output_dir}")

    name_to_id: Dict[str, int] = {n.lower(): i for i, n in enumerate(class_names)}
    stats = {"converted": 0, "skipped_files": 0, "invalid_boxes": 0, "unknown_classes": 0}

    for xml_path in xml_files:
        parsed = _parse_voc_xml(xml_path)
        if parsed is None:
            stats["skipped_files"] += 1
            continue

        img_w, img_h = parsed["width"], parsed["height"]
        lines: List[str] = []

        for obj in parsed["objects"]:
            cls_name = obj["name"].lower().replace("-", "_")
            if cls_name not in name_to_id:
                stats["unknown_classes"] += 1
                if skip_unknown_classes:
                    log.debug(f"Unknown class '{obj['name']}' in {xml_path.name} - skipped")
                    continue
                else:
                    log.warning(f"Unknown class '{obj['name']}' in {xml_path.name}")
                    continue

            cls_id = name_to_id[cls_name]
            xc, yc, w, h = voc_to_yolo_bbox(
                obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"], img_w, img_h
            )

            valid, reason = validate_yolo_bbox(xc, yc, w, h)
            if not valid:
                log.warning(f"Invalid bbox in {xml_path.name}: {reason} - clipping")
                xc, yc, w, h = clip_bbox(xc, yc, w, h)
                stats["invalid_boxes"] += 1

            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        out_path = output_dir / (xml_path.stem + ".txt")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        stats["converted"] += 1

    log.info(f"VOC conversion complete: {stats}")
    return stats


def convert_coco_to_yolo(
    coco_json_path: Path,
    output_dir: Path,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, int]:
    coco_json_path = Path(coco_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not coco_json_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json_path}")

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images: Dict[int, Dict] = {img["id"]: img for img in coco.get("images", [])}
    coco_cats: Dict[int, str] = {cat["id"]: cat["name"].lower() for cat in coco.get("categories", [])}
    name_to_yolo_id = {n.lower(): i for i, n in enumerate(class_names)}

    from collections import defaultdict
    ann_by_image: Dict[int, List] = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_by_image[ann["image_id"]].append(ann)

    stats = {"converted_images": 0, "total_boxes": 0, "skipped": 0}

    for img_id, img_info in images.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        stem = Path(img_info["file_name"]).stem
        lines: List[str] = []

        for ann in ann_by_image.get(img_id, []):
            cat_name = coco_cats.get(ann["category_id"], "").lower()
            if cat_name not in name_to_yolo_id:
                stats["skipped"] += 1
                continue

            cls_id = name_to_yolo_id[cat_name]
            x, y, bw, bh = ann["bbox"]
            xc = (x + bw / 2.0) / img_w
            yc = (y + bh / 2.0) / img_h
            w = bw / img_w
            h = bh / img_h

            valid, reason = validate_yolo_bbox(xc, yc, w, h)
            if not valid:
                log.warning(f"Invalid bbox (img_id={img_id}): {reason} - clipping")
                xc, yc, w, h = clip_bbox(xc, yc, w, h)

            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            stats["total_boxes"] += 1

        out_path = output_dir / (stem + ".txt")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        stats["converted_images"] += 1

    log.info(f"COCO conversion complete: {stats}")
    return stats


def visualise_annotations(
    image_dir: Path,
    label_dir: Path,
    output_dir: Path,
    class_names: List[str] = CLASS_NAMES,
    n_samples: int = 12,
    seed: int = 42,
) -> None:
    try:
        import cv2
    except ImportError:
        log.error("OpenCV not installed. Cannot visualise annotations.")
        return

    import random
    from src.utils.visualization import get_class_color_bgr

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        log.warning(f"No images found in {image_dir}")
        return

    rng = random.Random(seed)
    selected = rng.sample(images, min(n_samples, len(images)))

    saved = 0
    for img_path in selected:
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_px, w_px = img.shape[:2]

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue

            x1 = int((xc - bw / 2) * w_px)
            y1 = int((yc - bh / 2) * h_px)
            x2 = int((xc + bw / 2) * w_px)
            y2 = int((yc + bh / 2) * h_px)

            color = get_class_color_bgr(cls_id)
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out_path = output_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(out_path), img)
        saved += 1

    log.info(f"Saved {saved} annotation visualisation images -> {output_dir}")
