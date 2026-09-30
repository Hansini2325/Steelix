from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

from src.utils.logger import get_logger

log = get_logger(__name__)


class DefectTrack:
    def __init__(self, track_id: int, class_id: int, class_name: str, display_name: str):
        self.track_id = track_id
        self.track_label = f"{display_name}-{track_id}"
        self.class_id = class_id
        self.class_name = class_name
        self.display_name = display_name
        self.hits = 0
        self.confidences: List[float] = []
        self.regions: List[str] = []
        self.first_seen_ts = time.time()
        self.last_seen_ts = self.first_seen_ts
        self.first_seen_frame = -1
        self.last_seen_frame = -1

    def update(self, conf: float, region: str, frame_idx: int, ts: float) -> None:
        self.hits += 1
        self.confidences.append(conf)
        self.regions.append(region)
        self.last_seen_ts = ts
        self.last_seen_frame = frame_idx
        if self.first_seen_frame == -1:
            self.first_seen_frame = frame_idx

    @property
    def avg_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)

    @property
    def most_common_region(self) -> str:
        if not self.regions:
            return "unknown"
        from collections import Counter
        return Counter(self.regions).most_common(1)[0][0]


class DefectTracker:
    def __init__(self):
        self.active_tracks: Dict[int, DefectTrack] = {}
        self.history: Dict[int, DefectTrack] = {}

    def update_tracks(
        self,
        frame_detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp: float,
    ) -> List[Dict[str, Any]]:

        for det in frame_detections:
            tid = det.get("track_id", -1)
            if tid <= 0:
                continue

            if tid not in self.active_tracks:
                dt = DefectTrack(
                    track_id=tid,
                    class_id=det["class_id"],
                    class_name=det["class_name"],
                    display_name=det.get("display_name", det["class_name"]),
                )
                self.active_tracks[tid] = dt
                self.history[tid] = dt
            else:
                dt = self.active_tracks[tid]

            dt.update(
                conf=det["confidence"],
                region=det.get("region", "Center"),
                frame_idx=frame_idx,
                ts=timestamp,
            )

            det["track_label"] = dt.track_label

        return frame_detections

    def get_track_summaries(self) -> List[Dict[str, Any]]:
        summaries = []
        for tid, dt in self.history.items():
            summaries.append({
                "track_id": dt.track_id,
                "track_label": dt.track_label,
                "class_id": dt.class_id,
                "class_name": dt.class_name,
                "display_name": dt.display_name,
                "avg_confidence": round(dt.avg_confidence, 4),
                "frame_count": dt.hits,
                "first_seen_ts": round(dt.first_seen_ts, 4),
                "last_seen_ts": round(dt.last_seen_ts, 4),
                "first_seen_frame": dt.first_seen_frame,
                "last_seen_frame": dt.last_seen_frame,
                "most_common_region": dt.most_common_region,
            })
        return summaries
