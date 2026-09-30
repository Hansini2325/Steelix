from __future__ import annotations

import time
from typing import Any, Dict, List

from src.tracking.defect_tracker import DefectTracker
from src.utils.logger import get_logger

log = get_logger(__name__)


class TrackManager:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.tracker = DefectTracker()
        self.frame_count = 0
        self.start_ts = time.time()

    def process_frame_detections(
        self, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        self.frame_count += 1
        return self.tracker.update_tracks(
            detections,
            frame_idx=self.frame_count,
            timestamp=time.time(),
        )

    def get_all_tracks(self) -> List[Dict[str, Any]]:
        return self.tracker.get_track_summaries()

    def get_tracking_stats(self) -> Dict[str, Any]:
        tracks = self.get_all_tracks()
        unique = len(tracks)
        class_counts = {}
        for t in tracks:
            cn = t["class_name"]
            class_counts[cn] = class_counts.get(cn, 0) + 1

        top_class = None
        if class_counts:
            top_class = max(class_counts, key=class_counts.get)

        total_frames = self.frame_count
        duration = time.time() - self.start_ts

        return {
            "unique_tracked_defects": unique,
            "track_class_counts": class_counts,
            "top_tracked_class": top_class,
            "total_frames_tracked": total_frames,
            "tracking_duration_seconds": round(duration, 2),
        }
