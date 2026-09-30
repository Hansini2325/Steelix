from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

PALETTE_RGB = [
    (229, 20, 0),
    (0, 162, 232),
    (34, 177, 76),
    (255, 174, 201),
    (255, 127, 39),
    (163, 73, 164),
]


def get_class_color_rgb(class_id: int) -> Tuple[int, int, int]:
    if 0 <= class_id < len(PALETTE_RGB):
        return PALETTE_RGB[class_id]

    rng = random.Random(class_id)
    return (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))


def get_class_color_bgr(class_id: int) -> Tuple[int, int, int]:
    r, g, b = get_class_color_rgb(class_id)
    return (b, g, r)


def draw_detections_cv2(
    image: np.ndarray,
    boxes: List[List[float]],
    class_ids: List[int],
    confidences: List[float],
    class_names: List[str],
    thickness: int = 2,
    font_scale: float = 0.55,
    show_conf: bool = True,
) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        log.error("OpenCV (cv2) is not installed — cannot draw detections.")
        return image

    result = image.copy()
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = get_class_color_bgr(cls_id)
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label = f"{name}: {conf:.2f}" if show_conf else name

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        bg_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(result, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            result, label, (x1 + 2, y1 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return result


def draw_fps_overlay(image: np.ndarray, fps: float, latency_ms: float) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return image

    result = image.copy()
    text_fps = f"FPS: {fps:.1f}"
    text_lat = f"Latency: {latency_ms:.1f}ms"
    cv2.putText(result, text_fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(result, text_lat, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return result


def plot_class_distribution(
    class_counts: Dict[str, int],
    output_path: Path,
    title: str = "Class Distribution",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("Matplotlib not installed — skipping class distribution plot.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [
        "#{:02x}{:02x}{:02x}".format(*get_class_color_rgb(i)) for i in range(len(names))
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Defect Class")
    ax.set_ylabel("Number of Images / Annotations")
    ax.tick_params(axis="x", rotation=25)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            str(count),
            ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved class distribution chart → {output_path}")


def plot_image_size_distribution(
    widths: List[int],
    heights: List[int],
    output_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("Matplotlib not installed — skipping image size plot.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(widths, heights, alpha=0.4, s=15, color="#4a90e2")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].set_title("Image Size Scatter")

    axes[1].hist(widths, bins=30, alpha=0.7, label="Width", color="#4a90e2")
    axes[1].hist(heights, bins=30, alpha=0.7, label="Height", color="#e24a90")
    axes[1].set_xlabel("Pixels")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Width / Height Histograms")
    axes[1].legend()

    plt.suptitle("Image Size Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved image size distribution → {output_path}")


def plot_bbox_size_distribution(
    widths_norm: List[float],
    heights_norm: List[float],
    output_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(widths_norm, bins=40, color="#4a90e2", edgecolor="white")
    axes[0].set_xlabel("Normalised Width")
    axes[0].set_ylabel("Count")
    axes[0].set_title("BBox Width Distribution")

    axes[1].hist(heights_norm, bins=40, color="#e24a90", edgecolor="white")
    axes[1].set_xlabel("Normalised Height")
    axes[1].set_ylabel("Count")
    axes[1].set_title("BBox Height Distribution")

    plt.suptitle("Bounding-Box Size Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved BBox size distribution → {output_path}")


def save_sample_images(
    image_paths: List[Path],
    class_labels: List[str],
    output_path: Path,
    n_per_class: int = 4,
    class_names: Optional[List[str]] = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        log.warning("Matplotlib/Pillow not installed — skipping sample image grid.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    class_to_paths: Dict[str, List[Path]] = defaultdict(list)
    for p, label in zip(image_paths, class_labels):
        class_to_paths[label].append(p)

    unique_classes = class_names if class_names else sorted(class_to_paths.keys())
    n_classes = len(unique_classes)
    if n_classes == 0:
        log.warning("No images available for sample grid.")
        return

    fig, axes = plt.subplots(n_per_class, n_classes, figsize=(n_classes * 2.5, n_per_class * 2.5))
    if n_classes == 1:
        axes = np.array(axes).reshape(-1, 1)
    if n_per_class == 1:
        axes = np.array(axes).reshape(1, -1)

    for col, cls_name in enumerate(unique_classes):
        paths = class_to_paths.get(cls_name, [])
        sample_paths = paths[:n_per_class]
        for row in range(n_per_class):
            ax = axes[row][col]
            ax.axis("off")
            if row < len(sample_paths):
                try:
                    img = Image.open(sample_paths[row]).convert("RGB")
                    ax.imshow(np.array(img), cmap="gray" if img.mode == "L" else None)
                except Exception as exc:
                    log.warning(f"Could not open {sample_paths[row]}: {exc}")
            if row == 0:
                ax.set_title(cls_name, fontsize=9, fontweight="bold")

    plt.suptitle("Sample Images per Defect Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Saved sample image grid → {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("Matplotlib not installed — skipping confusion matrix plot.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm.astype(float) / row_sums, 0)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    thresh = cm_plot.max() / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            text = format(val, fmt) if fmt == ".2f" else str(int(val))
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved confusion matrix → {output_path}")
