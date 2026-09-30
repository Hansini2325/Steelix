from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

log = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _infer_class_from_label(label_path: Path, fallback: str = "unknown") -> str:
    try:
        first_line = label_path.read_text(encoding="utf-8").strip().splitlines()[0]
        return str(int(first_line.split()[0]))
    except Exception:
        return fallback


def split_dataset(
    images_dir: Path,
    labels_dir: Optional[Path],
    output_dir: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    class_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    import random
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "train/val/test ratios must sum to 1.0"

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    rng = random.Random(seed)

    all_images: List[Path] = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )
    if not all_images:
        log.warning(f"No images found in {images_dir}")
        return {"train": 0, "val": 0, "test": 0}

    log.info(f"Splitting {len(all_images)} images (seed={seed}, train={train_ratio}, val={val_ratio}, test={test_ratio})")

    grouped: Dict[str, List[Path]] = defaultdict(list)
    for img_path in all_images:
        parent_name = img_path.parent.name.lower()
        if class_names and any(c.lower() == parent_name for c in class_names):
            key = parent_name
        elif labels_dir:
            lbl = labels_dir / (img_path.stem + ".txt")
            key = _infer_class_from_label(lbl) if lbl.exists() else "unknown"
        else:
            key = parent_name

        grouped[key].append(img_path)

    log.info(f"Stratification groups: { {k: len(v) for k, v in grouped.items()} }")

    train_imgs, val_imgs, test_imgs = [], [], []

    for cls_key, paths in grouped.items():
        rng.shuffle(paths)
        n = len(paths)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            log.warning(f"Class '{cls_key}' has very few images ({n}) — placing all in train.")
            train_imgs.extend(paths)
            continue

        train_imgs.extend(paths[:n_train])
        val_imgs.extend(paths[n_train:n_train + n_val])
        test_imgs.extend(paths[n_train + n_val:])

    train_set = {p.name for p in train_imgs}
    val_set = {p.name for p in val_imgs}
    test_set = {p.name for p in test_imgs}
    assert not (train_set & val_set), "Data leakage: train ∩ val is non-empty!"
    assert not (train_set & test_set), "Data leakage: train ∩ test is non-empty!"
    assert not (val_set & test_set), "Data leakage: val ∩ test is non-empty!"

    split_map = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
    counts: Dict[str, int] = {}

    for split_name, img_list in split_map.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        if labels_dir:
            lbl_out.mkdir(parents=True, exist_ok=True)

        copied = 0
        for img_path in img_list:
            dst_img = img_out / img_path.name
            shutil.copy2(img_path, dst_img)

            if labels_dir:
                lbl_path = labels_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_out / lbl_path.name)
            copied += 1

        counts[split_name] = copied
        log.info(f"  {split_name:<6}: {copied:>5} images -> {img_out}")

    log.info(f"Split complete. Train={counts['train']}, Val={counts['val']}, Test={counts['test']}")
    return counts


def print_split_report(
    output_dir: Path,
    class_names: Optional[List[str]] = None,
) -> None:
    output_dir = Path(output_dir)
    splits = ["train", "val", "test"]
    print("\n" + "─" * 50)
    print("  DATASET SPLIT REPORT")
    print("─" * 50)
    for split in splits:
        img_dir = output_dir / split / "images"
        if img_dir.exists():
            count = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            print(f"  {split:<6}: {count:>5} images")
        else:
            print(f"  {split:<6}: (not found)")
    print("─" * 50 + "\n")
