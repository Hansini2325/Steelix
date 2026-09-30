from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Union

from src.utils.logger import get_logger

log = get_logger(__name__)


def run_webcam_inference(
    model_path: Union[str, Path],
    camera_id: Union[int, str] = 0,
    conf_threshold: float = 0.30,
    imgsz: int = 640,
    device: str = "",
    display: bool = True,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    try:
        import cv2
    except ImportError:
        log.error("cv2 is missing.")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics is missing.")
        return

    from src.utils.visualization import draw_detections_cv2, draw_fps_overlay

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        log.error(f"Cannot open webcam: {camera_id}")
        return

    log.info(f"Started real-time inference on {camera_id}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            results = model.predict(
                frame,
                conf=conf_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            t1 = time.perf_counter()
            lat = (t1 - t0) * 1000.0

            boxes, class_ids, confs = [], [], []
            if results and results[0].boxes is not None:
                for b in results[0].boxes:
                    boxes.append(b.xyxy[0].tolist())
                    class_ids.append(int(b.cls[0].item()))
                    confs.append(float(b.conf[0].item()))

            annotated = draw_detections_cv2(
                frame, boxes, class_ids, confs, list(model.names.values())
            )
            cur_fps = 1000.0 / lat if lat > 0 else 0.0
            annotated = draw_fps_overlay(annotated, cur_fps, lat)

            if callback:
                dets = [
                    {"box": bx, "cls": cid, "conf": cnf}
                    for bx, cid, cnf in zip(boxes, class_ids, confs)
                ]
                callback({"latency_ms": lat, "detections": dets})

            if display:
                cv2.imshow("Webcam Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        log.info("Real-time inference stopped.")
