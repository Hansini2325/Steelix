from __future__ import annotations

import numpy as np

from src.utils.visualization import (
    get_class_color_rgb,
    get_class_color_bgr,
    draw_detections_cv2,
)


def test_colors():
    c0 = get_class_color_rgb(0)
    assert len(c0) == 3
    
    c_bgr = get_class_color_bgr(0)
    assert c_bgr == (c0[2], c0[1], c0[0])
    
    c99 = get_class_color_rgb(99)
    assert len(c99) == 3


def test_draw_detections():
    # Only run if cv2 is installed, otherwise skip
    try:
        import cv2
    except ImportError:
        return
        
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [[10, 10, 50, 50]]
    cls_ids = [0]
    confs = [0.8]
    names = ["cls0"]
    
    res = draw_detections_cv2(img, boxes, cls_ids, confs, names)
    assert res.shape == (100, 100, 3)
    # The bounding box should draw some pixels (non-zero)
    assert np.any(res[10:50, 10:50] != 0)
