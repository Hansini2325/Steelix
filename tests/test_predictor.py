from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.inference.predictor import make_inspection_decision


def test_make_inspection_decision():
    dets_pass = [
        {"confidence": 0.2, "class_id": 0},
        {"confidence": 0.35, "class_id": 1},
    ]
    assert make_inspection_decision(dets_pass, conf_threshold=0.4) == "PASS"

    dets_review = [
        {"confidence": 0.45, "class_id": 2},
    ]
    assert make_inspection_decision(dets_review, conf_threshold=0.4) == "FLAG FOR REVIEW"

    assert make_inspection_decision(dets_review, conf_threshold=0.4, flag_classes=[0, 1]) == "PASS"


@patch("src.inference.predictor.ONNXPredictor")
def test_onnx_predictor_mock(mock_onnx):
    mock_instance = mock_onnx.return_value
    mock_instance.predict_image.return_value = {
        "num_detections": 1,
        "latency_ms": 15.0,
        "inspection": "PASS",
        "detections": [
            {"class_id": 0, "class_name": "crazing", "confidence": 0.5, "bbox_xyxy": [0,0,10,10]}
        ]
    }
    
    predictor = mock_onnx("model.onnx")
    res = predictor.predict_image("dummy.jpg")
    assert res["num_detections"] == 1
    assert res["latency_ms"] == 15.0
