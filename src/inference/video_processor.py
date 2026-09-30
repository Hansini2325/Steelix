from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.logger import get_logger

log = get_logger(__name__)


class VideoProcessor:

    def __init__(
        self,
        predictor_cls: Any,
        predictor_kwargs: Dict[str, Any],
        class_names: List[str],
    ) -> None:
        self.predictor = predictor_cls(**predictor_kwargs)
        self.class_names = class_names

    def process(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        stride: int = 1,
    ) -> Dict[str, Any]:
        """
        Process a video file, optionally applying tracking logic
        and saving an annotated output stream.
        """
        try:
            import cv2
        except ImportError:
            log.error("cv2 is required for video processing.")
            return {}

        video_path = Path(video_path)
        if not video_path.exists():
            log.error(f"Target video not found: {video_path}")
            return {}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.error(f"Cannot read video: {video_path}")
            return {}

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps / stride, (w, h))

        frame_idx = 0
        latencies = []

        log.info(f"VideoProcessor starting on {video_path.name} | {w}x{h} @ {fps} FPS")

        from src.utils.visualization import draw_detections_cv2, draw_fps_overlay

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            t0 = time.perf_counter()

            if hasattr(self.predictor, "model") and hasattr(self.predictor.model, "predict"):
                results = self.predictor.model.predict(
                    frame,
                    imgsz=self.predictor.imgsz,
                    conf=self.predictor.conf,
                    verbose=False,
                )
                pred = results[0]
                boxes, cls_ids, confs = [], [], []
                if pred.boxes is not None:
                    for b in pred.boxes:
                        boxes.append(b.xyxy[0].tolist())
                        cls_ids.append(int(b.cls[0].item()))
                        confs.append(float(b.conf[0].item()))
            else:
                log.error("Predictor model not compatible with VideoProcessor loop.")
                break

            t1 = time.perf_counter()
            lat = (t1 - t0) * 1000.0
            latencies.append(lat)

            cur_fps = 1000.0 / lat if lat > 0 else 0.0
            annotated = draw_detections_cv2(frame, boxes, cls_ids, confs, self.class_names)
            annotated = draw_fps_overlay(annotated, cur_fps, lat)

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("Video Processing", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_lat = sum(latencies) / max(len(latencies), 1)
        res = {
            "total_frames_processed": len(latencies),
            "avg_latency_ms": round(avg_lat, 2),
            "output_path": str(output_path) if output_path else None,
        }
        log.info(f"Video processing finished. Avg latency: {avg_lat:.2f}ms")
        return res
