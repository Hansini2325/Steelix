from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.visualization import draw_detections_cv2, draw_fps_overlay

log = get_logger("realtime")

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live webcam/camera YOLOv8 defect detection.")
    p.add_argument("--weights", required=True, help="Path to .pt weights")
    p.add_argument("--source", default="0", help="Camera index or video path (default: 0)")
    p.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="", help="Device string")
    p.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    p.add_argument("--show", action="store_true", default=True, help="Display window")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import cv2
    except ImportError:
        log.error("OpenCV not installed. Run: pip install opencv-python")
        return 1

    weights = Path(args.weights)
    if not weights.exists():
        log.error(f"Weights not found: {weights}")
        return 1

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open camera/source: {source}")
        return 1

    try:
        from ultralytics import YOLO
        from src.utils.hardware import get_best_device
    except ImportError as exc:
        log.error(f"Import failed: {exc}")
        return 1

    device = args.device or get_best_device()
    log.info(f"Loading model: {weights}  device={device}")
    model = YOLO(str(weights))

    log.info("Starting live inspection. Press 'q' to quit.")
    frame_count = 0
    latencies = []

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Failed to read frame from camera.")
            break

        t0 = time.perf_counter()
        try:
            results = model.track(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                tracker=f"{args.tracker}.yaml",
                persist=True,
                verbose=False,
            )
        except Exception as exc:
            log.warning(f"Inference error: {exc}")
            continue

        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000.0
        latencies.append(lat)
        cur_fps = 1000.0 / lat if lat > 0 else 0.0

        boxes, cls_ids, confs = [], [], []
        if results and results[0].boxes is not None:
            pred = results[0]
            for box in pred.boxes:
                boxes.append(box.xyxy[0].tolist())
                cls_ids.append(int(box.cls[0].item()))
                confs.append(float(box.conf[0].item()))

        annotated = draw_detections_cv2(frame, boxes, cls_ids, confs, CLASS_NAMES)
        annotated = draw_fps_overlay(annotated, cur_fps, lat)

        if args.show:
            cv2.imshow("SteelVision AI — Live Inspection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_count += 1
        if frame_count % 100 == 0:
            avg = sum(latencies) / len(latencies)
            log.info(f"Frame {frame_count}, avg latency={avg:.1f}ms, FPS={1000/avg:.1f}")

    cap.release()
    cv2.destroyAllWindows()

    avg = sum(latencies) / len(latencies) if latencies else 0.0
    log.info(f"Live inspection ended. Total frames: {frame_count}, Avg FPS: {1000/avg:.1f}" if avg > 0 else "No frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
