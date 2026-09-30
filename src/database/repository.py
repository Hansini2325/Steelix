from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.database.models import (
    SCHEMA_SQL,
    Detection,
    Inspection,
    ModelMetadata,
    Pattern,
    TimelineEvent,
    Track,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_DB_PATH = Path("storage/steelvision.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        self._init_db()
        log.info(f"Repository initialized: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        if self._connection is None or not self._is_connected():
            self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL;")
        return self._connection

    def _is_connected(self) -> bool:
        try:
            self._connection.execute("SELECT 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    def _init_db(self) -> None:
        try:
            conn = self._get_conn()
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            log.info("Database schema initialized.")
        except Exception as exc:
            log.error(f"Database initialization failed: {exc}")
            raise

    def generate_batch_id(self, date: Optional[str] = None) -> str:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        year, mmdd = date[:4], date[5:].replace("-", "")
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM inspections WHERE batch_id LIKE ?",
            (f"ST-{year}-{mmdd}-%",),
        ).fetchone()
        seq = (row["cnt"] if row else 0) + 1
        return f"ST-{year}-{mmdd}-{seq:03d}"

    def generate_inspection_id(self) -> str:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM inspections").fetchone()
        seq = (row["cnt"] if row else 0) + 1
        uid = uuid.uuid4().hex[:6].upper()
        return f"INS-{seq:05d}-{uid}"

    def create_inspection(self, inspection: Inspection) -> str:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO inspections (
                inspection_id, batch_id, batch_name,
                start_time, end_time, duration_seconds,
                input_source, operator_notes,
                inspection_status, status_reason,
                total_frame_detections, unique_tracked_defects,
                avg_confidence, low_conf_count,
                most_frequent_class, most_affected_region,
                pattern_count, most_recurring_class,
                highest_recurrence_region, longest_recurring_sequence,
                model_name, backend,
                conf_threshold, iou_threshold,
                class_counts_json, region_counts_json, heatmap_coords_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                inspection.inspection_id, inspection.batch_id, inspection.batch_name,
                inspection.start_time, inspection.end_time, inspection.duration_seconds,
                inspection.input_source, inspection.operator_notes,
                inspection.inspection_status, inspection.status_reason,
                inspection.total_frame_detections, inspection.unique_tracked_defects,
                inspection.avg_confidence, inspection.low_conf_count,
                inspection.most_frequent_class, inspection.most_affected_region,
                inspection.pattern_count, inspection.most_recurring_class,
                inspection.highest_recurrence_region, inspection.longest_recurring_sequence,
                inspection.model_name, inspection.backend,
                inspection.conf_threshold, inspection.iou_threshold,
                inspection.class_counts_json, inspection.region_counts_json,
                inspection.heatmap_coords_json,
            ),
        )
        conn.commit()
        log.info(f"Inspection persisted: {inspection.inspection_id}")
        return inspection.inspection_id

    def update_inspection(self, inspection: Inspection) -> None:
        conn = self._get_conn()
        conn.execute(
            """UPDATE inspections SET
                end_time=?, duration_seconds=?,
                inspection_status=?, status_reason=?,
                total_frame_detections=?, unique_tracked_defects=?,
                avg_confidence=?, low_conf_count=?,
                most_frequent_class=?, most_affected_region=?,
                pattern_count=?, most_recurring_class=?,
                highest_recurrence_region=?, longest_recurring_sequence=?,
                class_counts_json=?, region_counts_json=?,
                heatmap_coords_json=?
            WHERE inspection_id=?""",
            (
                inspection.end_time, inspection.duration_seconds,
                inspection.inspection_status, inspection.status_reason,
                inspection.total_frame_detections, inspection.unique_tracked_defects,
                inspection.avg_confidence, inspection.low_conf_count,
                inspection.most_frequent_class, inspection.most_affected_region,
                inspection.pattern_count, inspection.most_recurring_class,
                inspection.highest_recurrence_region, inspection.longest_recurring_sequence,
                inspection.class_counts_json, inspection.region_counts_json,
                inspection.heatmap_coords_json,
                inspection.inspection_id,
            ),
        )
        conn.commit()

    def get_inspection(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM inspections WHERE inspection_id = ?",
            (inspection_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_inspections(
        self,
        batch_id_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        query = "SELECT * FROM inspections WHERE 1=1"
        params: List[Any] = []

        if batch_id_filter:
            query += " AND batch_id LIKE ?"
            params.append(f"%{batch_id_filter}%")
        if status_filter:
            query += " AND inspection_status = ?"
            params.append(status_filter)
        if date_from:
            query += " AND start_time >= ?"
            params.append(date_from)
        if date_to:
            query += " AND start_time <= ?"
            params.append(date_to)

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def insert_detections(self, detections: List[Detection]) -> None:
        if not detections:
            return
        conn = self._get_conn()
        conn.executemany(
            """INSERT INTO detections (
                inspection_id, track_id, track_label, class_id,
                class_name, display_name, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                bbox_area_px2, frame_percentage, region,
                cx_norm, cy_norm, frame_idx, timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    d.inspection_id, d.track_id, d.track_label, d.class_id,
                    d.class_name, d.display_name, d.confidence,
                    d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2,
                    d.bbox_area_px2, d.frame_percentage, d.region,
                    d.cx_norm, d.cy_norm, d.frame_idx, d.timestamp,
                )
                for d in detections
            ],
        )
        conn.commit()

    def upsert_track(self, track: Track) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO tracks (
                inspection_id, track_id, track_label,
                class_id, class_name, display_name,
                avg_confidence, frame_count,
                first_seen_ts, last_seen_ts,
                first_seen_frame, last_seen_frame,
                most_common_region
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(inspection_id, track_id) DO UPDATE SET
                avg_confidence=excluded.avg_confidence,
                frame_count=excluded.frame_count,
                last_seen_ts=excluded.last_seen_ts,
                last_seen_frame=excluded.last_seen_frame,
                most_common_region=excluded.most_common_region""",
            (
                track.inspection_id, track.track_id, track.track_label,
                track.class_id, track.class_name, track.display_name,
                track.avg_confidence, track.frame_count,
                track.first_seen_ts, track.last_seen_ts,
                track.first_seen_frame, track.last_seen_frame,
                track.most_common_region,
            ),
        )
        conn.commit()

    def get_tracks(self, inspection_id: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM tracks WHERE inspection_id = ?",
            (inspection_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_pattern(self, pattern: Pattern) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO patterns (
                inspection_id, pattern_id, pattern_type,
                class_name, display_name, unique_occurrences,
                region, first_occurrence_ts, latest_occurrence_ts,
                duration_seconds, avg_interval_seconds,
                triggered_ts, description
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                pattern.inspection_id, pattern.pattern_id, pattern.pattern_type,
                pattern.class_name, pattern.display_name, pattern.unique_occurrences,
                pattern.region, pattern.first_occurrence_ts, pattern.latest_occurrence_ts,
                pattern.duration_seconds, pattern.avg_interval_seconds,
                pattern.triggered_ts, pattern.description,
            ),
        )
        conn.commit()

    def get_patterns(self, inspection_id: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM patterns WHERE inspection_id = ?",
            (inspection_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_timeline_event(self, event: TimelineEvent) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO timeline_events (inspection_id, event_type, timestamp, description, related_id)
               VALUES (?,?,?,?,?)""",
            (event.inspection_id, event.event_type, event.timestamp,
             event.description, event.related_id),
        )
        conn.commit()

    def get_timeline(self, inspection_id: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM timeline_events WHERE inspection_id = ? ORDER BY timestamp ASC",
            (inspection_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_model_metadata(self, meta: ModelMetadata) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO model_metadata (
                inspection_id, model_name, model_version,
                weights_path, backend, input_resolution,
                conf_threshold, iou_threshold,
                device, precision_mode
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                meta.inspection_id, meta.model_name, meta.model_version,
                meta.weights_path, meta.backend, meta.input_resolution,
                meta.conf_threshold, meta.iou_threshold,
                meta.device, meta.precision_mode,
            ),
        )
        conn.commit()

    def get_model_metadata(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM model_metadata WHERE inspection_id = ? LIMIT 1",
            (inspection_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_full_inspection(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        inspection = self.get_inspection(inspection_id)
        if not inspection:
            return None

        for json_key in ("class_counts_json", "region_counts_json", "heatmap_coords_json"):
            try:
                inspection[json_key.replace("_json", "")] = json.loads(inspection.get(json_key, "{}") or "{}")
            except Exception:
                inspection[json_key.replace("_json", "")] = {}

        inspection["tracks"] = self.get_tracks(inspection_id)
        inspection["patterns"] = self.get_patterns(inspection_id)
        inspection["timeline"] = self.get_timeline(inspection_id)
        inspection["model_metadata"] = self.get_model_metadata(inspection_id)

        return inspection

    def search_inspections(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM inspections
               WHERE batch_id LIKE ? OR input_source LIKE ?
               ORDER BY start_time DESC LIMIT ?""",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]
