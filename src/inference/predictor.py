from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logger import get_logger
from src.utils.visualization import draw_detections_cv2, draw_fps_overlay

log = get_logger(__name__)

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]


def make_inspection_decision(
    detections: List[Dict[str, Any]],
    conf_threshold: float = 0.40,
    flag_classes: Optional[List[int]] = None,
) -> str:
    for det in detections:
        if det["confidence"] >= conf_threshold:
            if flag_classes is None or det["class_id"] in flag_classes:
                return "FLAG FOR REVIEW"
    return "PASS"


class UltralyticsPredictor:

    def __init__(
        self,
        weights_path: Union[str, Path],
        conf: float = 0.30,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        class_names: List[str] = CLASS_NAMES,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.class_names = class_names

        if not device:
            from src.utils.hardware import get_best_device
            device = get_best_device()
        self.device = device

        log.info(f"Loading model: {self.weights_path}  device={device}")
        self.model = YOLO(str(self.weights_path))
        log.info("Model loaded.")

    def predict_image(
        self,
        image_path: Union[str, Path],
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        import cv2

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        t0 = time.perf_counter()
        raw = self.model(
            str(image_path),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        detections: List[Dict[str, Any]] = []
        pred = raw[0]
        if pred.boxes is not None:
            for box in pred.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                detections.append({
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                })

        boxes = [d["bbox_xyxy"] for d in detections]
        cls_ids = [d["class_id"] for d in detections]
        confs = [d["confidence"] for d in detections]
        annotated = draw_detections_cv2(img_bgr, boxes, cls_ids, confs, self.class_names)
        annotated = draw_fps_overlay(annotated, 1000.0 / latency_ms, latency_ms)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), annotated)
            log.info(f"Annotated image saved → {save_path}")

        return {
            "image_path": str(image_path),
            "detections": detections,
            "latency_ms": round(latency_ms, 2),
            "num_detections": len(detections),
            "annotated_image": annotated,
            "inspection": make_inspection_decision(detections),
        }

    def predict_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> Dict[str, Any]:
        import cv2

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps_src, (w, h))

        latencies: List[float] = []
        frame_idx = 0

        log.info(f"Processing video: {video_path}  ({w}x{h} @ {fps_src:.1f} fps)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            raw = self.model(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                             device=self.device, verbose=False)
            t1 = time.perf_counter()
            lat = (t1 - t0) * 1000.0
            latencies.append(lat)

            pred = raw[0]
            boxes, cls_ids, confs = [], [], []
            if pred.boxes is not None:
                for box in pred.boxes:
                    boxes.append(box.xyxy[0].tolist())
                    cls_ids.append(int(box.cls[0].item()))
                    confs.append(float(box.conf[0].item()))

            annotated = draw_detections_cv2(frame, boxes, cls_ids, confs, self.class_names)
            cur_fps = 1000.0 / lat if lat > 0 else 0.0
            annotated = draw_fps_overlay(annotated, cur_fps, lat)

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("Steel Defect — Video Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 50 == 0:
                log.info(f"  Frame {frame_idx}, avg latency={sum(latencies)/len(latencies):.1f}ms")

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        avg_fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        log.info(f"Video inference done. Frames={frame_idx}, AvgFPS={avg_fps:.1f}, AvgLatency={avg_lat:.1f}ms")
        if output_path:
            log.info(f"Annotated video saved → {output_path}")

        return {
            "total_frames": frame_idx,
            "avg_latency_ms": round(avg_lat, 2),
            "avg_fps": round(avg_fps, 2),
            "output_path": str(output_path) if output_path else None,
        }


class ONNXPredictor:

    def __init__(
        self,
        onnx_path: Union[str, Path],
        conf: float = 0.30,
        iou: float = 0.45,
        imgsz: int = 640,
        providers: Optional[List[str]] = None,
        class_names: List[str] = CLASS_NAMES,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Run: pip install onnxruntime")

        self.onnx_path = Path(onnx_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.class_names = class_names
        self.input_size = (imgsz, imgsz)

        if providers is None:
            providers = ort.get_available_providers()
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        log.info(f"ONNX model loaded: {self.onnx_path}  providers={providers}")

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2
        img = cv2.resize(image_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _nms(self, boxes, scores, iou_threshold):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou_val = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou_val <= iou_threshold)[0] + 1]
        return keep

    def predict_image(
        self,
        image_path: Union[str, Path],
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        import cv2

        image_path = Path(image_path)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        orig_h, orig_w = img_bgr.shape[:2]
        sx = orig_w / self.imgsz
        sy = orig_h / self.imgsz

        inp = self._preprocess(img_bgr)

        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: inp})
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        preds = outputs[0]
        if preds.ndim == 3:
            preds = preds[0]
        preds = preds.T

        nc = len(self.class_names)
        boxes_xywh = preds[:, :4]
        class_probs = preds[:, 4:4 + nc]
        cls_ids_raw = np.argmax(class_probs, axis=1)
        scores_raw = class_probs[np.arange(len(cls_ids_raw)), cls_ids_raw]

        mask = scores_raw >= self.conf
        boxes_xywh = boxes_xywh[mask]
        cls_ids_raw = cls_ids_raw[mask]
        scores_raw = scores_raw[mask]

        detections: List[Dict[str, Any]] = []
        if len(boxes_xywh) > 0:
            bx = boxes_xywh[:, 0] * sx
            by = boxes_xywh[:, 1] * sy
            bw = boxes_xywh[:, 2] * sx
            bh = boxes_xywh[:, 3] * sy
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = bx + bw / 2
            y2 = by + bh / 2
            xyxy = np.stack([x1, y1, x2, y2], axis=1)

            keep = self._nms(xyxy, scores_raw, self.iou)
            for k in keep:
                cls_id = int(cls_ids_raw[k])
                conf = float(scores_raw[k])
                name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                detections.append({
                    "class_id": cls_id, "class_name": name, "confidence": conf,
                    "bbox_xyxy": xyxy[k].tolist(),
                })

        boxes = [d["bbox_xyxy"] for d in detections]
        c_ids = [d["class_id"] for d in detections]
        confs = [d["confidence"] for d in detections]
        annotated = draw_detections_cv2(img_bgr, boxes, c_ids, confs, self.class_names)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), annotated)

        return {
            "image_path": str(image_path),
            "detections": detections,
            "latency_ms": round(latency_ms, 2),
            "num_detections": len(detections),
            "annotated_image": annotated,
            "inspection": make_inspection_decision(detections),
            "backend": "onnx_runtime",
        }
