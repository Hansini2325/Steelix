from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate_dataset import validate_dataset
from src.data.prepare_dataset import prepare_dataset, write_yolo_dataset_yaml
from src.data.split_dataset import split_dataset, print_split_report
from src.utils.logger import get_logger
from src.utils.config import load_yaml as load_dataset_config

log = get_logger("prepare_dataset")

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate and prepare the NEU-DET dataset.")
    p.add_argument("--raw-dir", default="data/raw", help="Raw dataset root directory")
    p.add_argument("--interim-dir", default="data/interim", help="Interim output directory")
    p.add_argument("--processed-dir", default="data/processed", help="Processed YOLO dataset directory")
    p.add_argument("--config", default="configs/dataset.yaml", help="Dataset config YAML")
    p.add_argument("--force", action="store_true", help="Re-process even if processed dataset exists")
    p.add_argument("--skip-eda", action="store_true", help="Skip EDA figure generation")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    interim_dir = PROJECT_ROOT / args.interim_dir
    processed_dir = PROJECT_ROOT / args.processed_dir
    config_path = PROJECT_ROOT / args.config

    log.info("=" * 62)
    log.info("  SteelVision AI — Dataset Preparation")
    log.info("=" * 62)

    try:
        cfg = load_dataset_config(config_path)
        log.info(f"Config loaded: {config_path}")
    except FileNotFoundError:
        log.error(f"Config not found: {config_path}")
        return 1

    train_ratio = cfg.get("train_ratio", 0.80)
    val_ratio = cfg.get("val_ratio", 0.10)
    test_ratio = cfg.get("test_ratio", 0.10)
    seed = cfg.get("split_seed", 42)

    neu_det_dir = raw_dir / "NEU-DET"
    scan_dir = neu_det_dir if neu_det_dir.exists() else raw_dir

    log.info(f"\nValidating raw dataset at: {scan_dir}")
    report = validate_dataset(scan_dir)

    log.info(f"  Total images found     : {report.get('total_images', 0)}")
    log.info(f"  Annotation format      : {report.get('annotation_format_detected', 'none')}")
    log.info(f"  Corrupted images       : {report.get('corrupted_images', 0)}")
    log.info(f"  XML annotations        : {report.get('total_xml_annotations', 0)}")

    if report.get("total_images", 0) == 0:
        log.error(
            "\n" + "=" * 62 + "\n"
            "  ERROR: No images found. Download NEU-DET and place it in:\n"
            f"  {scan_dir}\n"
            + "=" * 62
        )
        return 2

    yolo_yaml = processed_dir / "neu_det.yaml"
    train_dir = processed_dir / "train" / "images"
    if yolo_yaml.exists() and train_dir.exists() and not args.force:
        imgs = list(train_dir.rglob("*.jpg")) + list(train_dir.rglob("*.png"))
        if imgs:
            log.info(f"\n✓ Dataset already prepared at {processed_dir}")
            log.info("  Use --force to re-run preparation.")
            print_split_report(processed_dir)
            log.info(f"\nNext step:")
            log.info(f"  python scripts/train.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16")
            return 0

    log.info(f"\nStep 1/3: Converting annotations (VOC XML → YOLO TXT)…")
    task, label_dir = prepare_dataset(
        raw_dir=scan_dir,
        interim_dir=interim_dir,
        class_names=CLASS_NAMES,
    )

    if task == "none":
        log.error("No annotation files found. Cannot prepare training data.")
        return 1

    if task == "classify":
        log.warning(
            "Classification dataset detected. YOLO bounding box training requires object detection labels. "
            "Proceeding with classification setup."
        )
        return 1

    interim_images = interim_dir / "images"
    interim_labels = interim_dir / "labels"

    n_imgs = len(list(interim_images.rglob("*"))) if interim_images.exists() else 0
    n_lbls = len(list(interim_labels.rglob("*.txt"))) if interim_labels.exists() else 0
    log.info(f"  Interim images : {n_imgs}")
    log.info(f"  Interim labels : {n_lbls}")

    log.info("\nStep 2/3: Verifying converted labels…")
    unknown_count = 0
    invalid_count = 0
    valid_ids = set(range(len(CLASS_NAMES)))

    if interim_labels.exists():
        for lf in interim_labels.rglob("*.txt"):
            for line in lf.read_text(encoding="utf-8").strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    invalid_count += 1
                    continue
                cls_id = int(parts[0])
                if cls_id not in valid_ids:
                    unknown_count += 1
                    continue
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    invalid_count += 1

    log.info(f"  Unknown class IDs  : {unknown_count}")
    log.info(f"  Invalid YOLO lines : {invalid_count}")

    if unknown_count > 0:
        log.error(
            "ERROR: Found unknown class IDs in labels. Check the CLASS_NAMES list "
            "or ensure skipping of unknown classes in conversion."
        )
        return 1

    log.info(f"\nStep 3/3: Splitting dataset ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}, seed={seed})…")

    counts = split_dataset(
        images_dir=interim_images,
        labels_dir=interim_labels if interim_labels.exists() else None,
        output_dir=processed_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        class_names=CLASS_NAMES,
    )

    train_count = counts.get("train", 0)
    val_count = counts.get("val", 0)
    test_count = counts.get("test", 0)

    yolo_yaml_path = write_yolo_dataset_yaml(
        processed_dir=processed_dir,
        train_dir=processed_dir / "train" / "images",
        val_dir=processed_dir / "val" / "images",
        test_dir=processed_dir / "test" / "images",
        class_names=CLASS_NAMES,
        task="detect",
    )

    if not args.skip_eda:
        try:
            _generate_eda(interim_images, interim_labels, CLASS_NAMES)
        except Exception as exc:
            log.warning(f"EDA skipped (non-fatal): {exc}")

    log.info("\n" + "=" * 62)
    log.info("  ✓ DATASET PREPARATION COMPLETE")
    log.info("=" * 62)
    log.info(f"  YOLO YAML      : {yolo_yaml_path}")
    log.info(f"  Train images   : {train_count}")
    log.info(f"  Val   images   : {val_count}")
    log.info(f"  Test  images   : {test_count}")
    log.info(f"  Total          : {train_count + val_count + test_count}")
    log.info(f"  Annotation task: detect")
    log.info(f"  Classes        : {CLASS_NAMES}")
    if not args.skip_eda:
        log.info("  EDA figures    : results/figures/")
    log.info("")
    log.info("Next step:")
    log.info("  python scripts/train.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16")
    log.info("=" * 62)

    print_split_report(processed_dir)

    _validate_final_labels(processed_dir, CLASS_NAMES)

    return 0


def _generate_eda(
    image_dir: Path,
    label_dir: Path,
    class_names: list,
) -> None:
    from collections import Counter
    from src.utils.visualization import plot_class_distribution

    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter = Counter()
    if label_dir.exists():
        for lf in label_dir.rglob("*.txt"):
            for line in lf.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    if 0 <= cid < len(class_names):
                        class_counts[class_names[cid]] += 1

    if class_counts:
        plot_class_distribution(
            dict(class_counts),
            fig_dir / "class_distribution.png",
            title="NEU-DET Defect Class Distribution",
        )
        log.info(f"  EDA class distribution chart → {fig_dir / 'class_distribution.png'}")


def _validate_final_labels(processed_dir: Path, class_names: list) -> None:
    splits = ["train", "val", "test"]
    total_imgs = 0
    total_lbls = 0
    issues = 0

    for split in splits:
        img_dir = processed_dir / split / "images"
        lbl_dir = processed_dir / split / "labels"
        if img_dir.exists():
            n_imgs = sum(1 for _ in img_dir.iterdir() if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            total_imgs += n_imgs
        if lbl_dir.exists():
            n_lbls = sum(1 for _ in lbl_dir.iterdir() if _.suffix == ".txt")
            total_lbls += n_lbls
            for lf in lbl_dir.iterdir():
                if lf.suffix != ".txt":
                    continue
                for line in lf.read_text(encoding="utf-8").strip().splitlines():
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) != 5:
                        issues += 1

    log.info(f"\n  Post-split validation:")
    log.info(f"    Total split images : {total_imgs}")
    log.info(f"    Total split labels : {total_lbls}")
    log.info(f"    Malformed lines    : {issues}")
    if issues == 0:
        log.info("    ✓ All YOLO labels are valid.")
    else:
        log.warning(f"    ⚠  {issues} malformed label lines found — review conversion.")


if __name__ == "__main__":
    sys.exit(main())
