from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.utils.logger import get_logger
from src.data.convert_annotations import convert_voc_to_yolo, convert_coco_to_yolo

log = get_logger(__name__)

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def detect_dataset_layout(raw_dir: Path) -> Dict:
    raw_dir = Path(raw_dir)
    all_images: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(raw_dir.rglob(f"*{ext}"))

    if not all_images:
        return {"layout": "empty", "image_paths": [], "xml_dir": None, "label_dir": None}

    xml_files = list(raw_dir.rglob("*.xml"))
    txt_files = [p for p in raw_dir.rglob("*.txt")
                 if p.parent.name.lower() in ("labels", "annotations", "label")]
    json_files = list(raw_dir.rglob("*.json"))

    sub_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    class_sub_dirs = [d for d in sub_dirs if d.name.lower() in [c.lower() for c in CLASS_NAMES]]

    if xml_files:
        xml_parent = xml_files[0].parent
        log.info(f"Layout A detected: Pascal VOC XML annotations in {xml_parent}")
        return {
            "layout": "A_voc_xml",
            "image_paths": all_images,
            "xml_dir": xml_parent,
            "label_dir": None,
            "json_path": None,
        }

    if any(p.parent.name.lower() == "labels" for p in txt_files):
        label_dir = next(p.parent for p in txt_files if p.parent.name.lower() == "labels")
        log.info(f"Layout C detected: existing YOLO TXT labels in {label_dir}")
        return {
            "layout": "C_yolo_txt",
            "image_paths": all_images,
            "xml_dir": None,
            "label_dir": label_dir,
            "json_path": None,
        }

    if json_files:
        log.info(f"COCO JSON annotation file detected: {json_files[0]}")
        return {
            "layout": "A_coco_json",
            "image_paths": all_images,
            "xml_dir": None,
            "label_dir": None,
            "json_path": json_files[0],
        }

    if class_sub_dirs:
        log.info("Layout B detected: class sub-directories (classification only, no bboxes)")
        return {
            "layout": "B_classification",
            "image_paths": all_images,
            "xml_dir": None,
            "label_dir": None,
            "json_path": None,
            "class_sub_dirs": class_sub_dirs,
        }

    log.warning("Layout D: found images but no recognisable annotation files.")
    return {"layout": "D_no_annotations", "image_paths": all_images, "xml_dir": None, "label_dir": None}


def _copy_images(src_paths: List[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_paths:
        dst = dst_dir / p.name
        if not dst.exists():
            shutil.copy2(p, dst)


def prepare_dataset(
    raw_dir: Path,
    interim_dir: Path,
    class_names: List[str] = CLASS_NAMES,
) -> Tuple[str, Optional[Path]]:
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    layout_info = detect_dataset_layout(raw_dir)
    layout = layout_info["layout"]
    image_paths: List[Path] = layout_info["image_paths"]

    log.info(f"Dataset layout detected: {layout}, images found: {len(image_paths)}")

    interim_images = interim_dir / "images"
    _copy_images(image_paths, interim_images)
    log.info(f"Copied {len(image_paths)} images -> {interim_images}")

    interim_labels = interim_dir / "labels"

    if layout == "A_voc_xml":
        xml_dir: Path = layout_info["xml_dir"]
        stats = convert_voc_to_yolo(xml_dir, interim_labels, class_names)
        log.info(f"VOC->YOLO conversion: {stats}")
        task = "detect"
        return task, interim_labels

    elif layout == "A_coco_json":
        json_path: Path = layout_info["json_path"]
        stats = convert_coco_to_yolo(json_path, interim_labels, class_names)
        log.info(f"COCO->YOLO conversion: {stats}")
        task = "detect"
        return task, interim_labels

    elif layout == "C_yolo_txt":
        src_labels: Path = layout_info["label_dir"]
        interim_labels.mkdir(parents=True, exist_ok=True)
        for lbl in src_labels.glob("*.txt"):
            dst = interim_labels / lbl.name
            if not dst.exists():
                shutil.copy2(lbl, dst)
        log.info(f"Copied existing YOLO labels -> {interim_labels}")
        task = "detect"
        return task, interim_labels

    elif layout == "B_classification":
        log.warning("Classification layout detected.")

        cls_root = interim_dir / "cls"
        for img_path in image_paths:
            parent_name = img_path.parent.name
            matched = next(
                (c for c in class_names if c.lower() == parent_name.lower()), None
            )
            if matched:
                dst_dir = cls_root / matched
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / img_path.name
                if not dst.exists():
                    shutil.copy2(img_path, dst)
        task = "classify"
        return task, None

    else:
        log.warning("No valid annotations found for preparation.")
        task = "none"
        return task, None


def write_yolo_dataset_yaml(
    processed_dir: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    class_names: List[str] = CLASS_NAMES,
    task: str = "detect",
) -> Path:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "path": str(processed_dir.resolve()),
        "train": str(train_dir.resolve()),
        "val": str(val_dir.resolve()),
        "test": str(test_dir.resolve()),
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
        "task": task,
    }

    yaml_path = processed_dir / "neu_det.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    log.info(f"YOLO dataset YAML written -> {yaml_path}")
    return yaml_path
