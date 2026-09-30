from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

GRID_LABELS = [
    "Top-left", "Top-center", "Top-right",
    "Center-left", "Center", "Center-right",
    "Bottom-left", "Bottom-center", "Bottom-right",
]


class HeatmapGenerator:

    def __init__(self, canvas_size: int = 640, sigma: float = 20.0) -> None:
        self.canvas_size = canvas_size
        self.sigma = sigma
        self._coords: List[Tuple[float, float]] = []
        self._class_coords: Dict[str, List[Tuple[float, float]]] = {}

    def add_detection(
        self,
        cx_norm: float,
        cy_norm: float,
        class_name: Optional[str] = None,
    ) -> None:
        self._coords.append((cx_norm, cy_norm))
        if class_name:
            if class_name not in self._class_coords:
                self._class_coords[class_name] = []
            self._class_coords[class_name].append((cx_norm, cy_norm))

    def add_detections(
        self,
        coords: List[Tuple[float, float]],
        class_name: Optional[str] = None,
    ) -> None:
        for cx, cy in coords:
            self.add_detection(cx, cy, class_name)

    def _render_heatmap(self, coords: List[Tuple[float, float]]) -> np.ndarray:
        hmap = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)

        if not coords:
            return hmap

        for cx_norm, cy_norm in coords:
            px = int(np.clip(cx_norm, 0, 1) * (self.canvas_size - 1))
            py = int(np.clip(cy_norm, 0, 1) * (self.canvas_size - 1))
            hmap[py, px] += 1.0

        try:
            import cv2
            kernel_size = max(int(self.sigma * 3) | 1, 3)
            hmap = cv2.GaussianBlur(hmap, (kernel_size, kernel_size), self.sigma)
        except ImportError:
            from scipy.ndimage import gaussian_filter
            hmap = gaussian_filter(hmap, sigma=self.sigma)

        max_val = hmap.max()
        if max_val > 0:
            hmap /= max_val

        return hmap

    def get_heatmap_array(self, class_name: Optional[str] = None) -> np.ndarray:
        coords = self._class_coords.get(class_name, []) if class_name else self._coords
        return self._render_heatmap(coords)

    def get_heatmap_rgb(
        self,
        class_name: Optional[str] = None,
        colormap_name: str = "hot",
    ) -> np.ndarray:
        hmap = self.get_heatmap_array(class_name)

        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap_name)
            rgba = (cmap(hmap) * 255).astype(np.uint8)
            return rgba[:, :, :3]
        except ImportError:
            gray = (hmap * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)

    def save_heatmap(
        self,
        output_path: Path,
        class_name: Optional[str] = None,
        title: str = "Defect Spatial Heatmap",
    ) -> bool:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        coords = self._class_coords.get(class_name, []) if class_name else self._coords

        if not coords:
            log.warning(f"No coordinates for heatmap (class={class_name}). Saving blank.")

        hmap = self._render_heatmap(coords)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 7))
            im = ax.imshow(hmap, cmap="hot", interpolation="bilinear",
                           origin="upper", extent=[0, 1, 1, 0])
            plt.colorbar(im, ax=ax, label="Relative Detection Density")
            ax.set_xlabel("Normalized X Position")
            ax.set_ylabel("Normalized Y Position")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)

            for v in [1 / 3, 2 / 3]:
                ax.axhline(v, color="white", linewidth=0.4, alpha=0.6)
                ax.axvline(v, color="white", linewidth=0.4, alpha=0.6)

            for row in range(3):
                for col in range(3):
                    idx = row * 3 + col
                    ax.text(
                        (col + 0.5) / 3, (row + 0.5) / 3,
                        GRID_LABELS[idx],
                        ha="center", va="center",
                        fontsize=6, color="white", alpha=0.7,
                        transform=ax.transData,
                    )

            n_dets = len(coords)
            ax.set_xlabel(f"Normalized X  |  {n_dets} detections")

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"Heatmap saved → {output_path}")
            return True

        except Exception as exc:
            log.error(f"Could not save heatmap: {exc}")
            return False

    def get_grid_density(self) -> np.ndarray:
        grid = np.zeros((3, 3), dtype=float)
        for cx_norm, cy_norm in self._coords:
            col = min(int(cx_norm * 3), 2)
            row = min(int(cy_norm * 3), 2)
            grid[row, col] += 1.0

        max_val = grid.max()
        if max_val > 0:
            grid /= max_val
        return grid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_detections": len(self._coords),
            "canvas_size": self.canvas_size,
            "sigma": self.sigma,
            "grid_density": self.get_grid_density().tolist(),
            "class_detection_counts": {
                cls: len(coords) for cls, coords in self._class_coords.items()
            },
            "coords_sample": self._coords[:2000],
        }

    def reset(self) -> None:
        self._coords.clear()
        self._class_coords.clear()
