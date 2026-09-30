from __future__ import annotations

import cv2
import numpy as np

from src.analytics.heatmap import HeatmapGenerator
from src.analytics.recurrence_detector import RecurrenceDetector
from src.analytics.spatial_analysis import SpatialAnalyzer
from src.analytics.temporal_analysis import TemporalAnalyzer


def test_heatmap_generator():
    hg = HeatmapGenerator(canvas_size=100)
    hg.add_detection(0.5, 0.5, class_name="patches")
    hg.add_detection(0.1, 0.1, class_name="crazing")
    hg.add_detection(0.9, 0.9, class_name="patches")
    
    h_rgb = hg.get_heatmap_rgb(class_name="patches")
    assert h_rgb.shape == (100, 100, 3)
    
    h_all = hg.get_heatmap_rgb()
    assert h_all.shape == (100, 100, 3)
    
    stats = hg.to_dict()
    assert stats["total_detections"] == 3
    assert stats["class_detection_counts"]["patches"] == 2
    assert stats["class_detection_counts"]["crazing"] == 1


def test_spatial_analyzer():
    sa = SpatialAnalyzer()
    sa.add_detection(0.1, 0.1, class_name="A")
    sa.add_detection(0.5, 0.5, class_name="A")
    sa.add_detection(0.5, 0.5, class_name="A")
    sa.add_detection(0.9, 0.9, class_name="B")
    
    top = sa.most_affected_region()
    assert top == "Center"
    
    stats = sa.get_summary()
    assert stats["total_detections"] == 4
    assert stats["most_affected_region"] == "Center"
    assert "Center" in stats["region_distribution"]


def test_temporal_analyzer():
    ta = TemporalAnalyzer(bin_width_seconds=2.0)
    ta.record_detection(timestamp=1.0, class_name="C", track_id=1)
    ta.record_detection(timestamp=2.5, class_name="C", track_id=1)
    ta.record_detection(timestamp=3.0, class_name="D", track_id=2)
    
    rate = ta.detection_rate_over_time()
    assert len(rate["bin_edges"]) == 2
    assert rate["counts_per_bin"][0] == 2
    assert rate["counts_per_bin"][1] == 1
    
    class_r = ta.class_rate_over_time("C")
    assert class_r["counts_per_bin"] == [1, 1]
    
    unique_r = ta.unique_defects_over_time()
    assert unique_r["cumulative_unique"] == [1, 2]
    
    summ = ta.get_summary()
    assert summ["total_events"] == 3
    assert summ["class_counts"]["C"] == 2


def test_recurrence_detector():
    rd = RecurrenceDetector(min_unique_occurrences=2, time_window_seconds=10.0, alert_cooldown_seconds=0)
    
    rd.update_track(track_id=1, class_name="Crazing", region="Center", timestamp=1.0)
    rd.update_track(track_id=1, class_name="Crazing", region="Center", timestamp=5.0)
    
    patterns = rd.check_patterns(now=6.0)
    assert len(patterns) == 0
    
    rd.update_track(track_id=2, class_name="Crazing", region="Center", timestamp=6.0)
    patterns = rd.check_patterns(now=7.0)
    assert len(patterns) == 1
    
    assert patterns[0].class_name == "Crazing"
    assert patterns[0].region == "Center"
    assert patterns[0].unique_occurrences == 2
