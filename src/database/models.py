from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Inspection:
    inspection_id: str
    batch_id: str
    batch_name: Optional[str]
    start_time: str
    end_time: Optional[str]
    duration_seconds: float
    input_source: str
    operator_notes: Optional[str]
    inspection_status: str
    status_reason: Optional[str]
    total_frame_detections: int
    unique_tracked_defects: int
    avg_confidence: float
    low_conf_count: int
    most_frequent_class: Optional[str]
    most_affected_region: Optional[str]
    pattern_count: int
    most_recurring_class: Optional[str]
    highest_recurrence_region: Optional[str]
    longest_recurring_sequence: int
    model_name: str
    backend: str
    conf_threshold: float
    iou_threshold: float
    class_counts_json: str
    region_counts_json: str
    heatmap_coords_json: str
    db_id: Optional[int] = None


@dataclass
class Detection:
    inspection_id: str
    track_id: int
    track_label: str
    class_id: int
    class_name: str
    display_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    bbox_area_px2: float
    frame_percentage: float
    region: str
    cx_norm: float
    cy_norm: float
    frame_idx: int
    timestamp: float
    db_id: Optional[int] = None


@dataclass
class Track:
    inspection_id: str
    track_id: int
    track_label: str
    class_id: int
    class_name: str
    display_name: str
    avg_confidence: float
    frame_count: int
    first_seen_ts: float
    last_seen_ts: float
    first_seen_frame: int
    last_seen_frame: int
    most_common_region: str
    db_id: Optional[int] = None


@dataclass
class Pattern:
    inspection_id: str
    pattern_id: str
    pattern_type: str
    class_name: str
    display_name: str
    unique_occurrences: int
    region: Optional[str]
    first_occurrence_ts: float
    latest_occurrence_ts: float
    duration_seconds: float
    avg_interval_seconds: Optional[float]
    triggered_ts: float
    description: str
    db_id: Optional[int] = None


@dataclass
class TimelineEvent:
    inspection_id: str
    event_type: str
    timestamp: float
    description: str
    related_id: Optional[str] = None
    db_id: Optional[int] = None


@dataclass
class ModelMetadata:
    inspection_id: str
    model_name: str
    model_version: Optional[str]
    weights_path: str
    backend: str
    input_resolution: int
    conf_threshold: float
    iou_threshold: float
    device: str
    precision_mode: str
    db_id: Optional[int] = None


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS inspections (
    db_id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id           TEXT    NOT NULL UNIQUE,
    batch_id                TEXT    NOT NULL,
    batch_name              TEXT,
    start_time              TEXT    NOT NULL,
    end_time                TEXT,
    duration_seconds        REAL    DEFAULT 0,
    input_source            TEXT    NOT NULL,
    operator_notes          TEXT,
    inspection_status       TEXT    NOT NULL DEFAULT 'REVIEW',
    status_reason           TEXT,
    total_frame_detections  INTEGER DEFAULT 0,
    unique_tracked_defects  INTEGER DEFAULT 0,
    avg_confidence          REAL    DEFAULT 0,
    low_conf_count          INTEGER DEFAULT 0,
    most_frequent_class     TEXT,
    most_affected_region    TEXT,
    pattern_count           INTEGER DEFAULT 0,
    most_recurring_class    TEXT,
    highest_recurrence_region TEXT,
    longest_recurring_sequence INTEGER DEFAULT 0,
    model_name              TEXT,
    backend                 TEXT,
    conf_threshold          REAL,
    iou_threshold           REAL,
    class_counts_json       TEXT    DEFAULT '{}',
    region_counts_json      TEXT    DEFAULT '{}',
    heatmap_coords_json     TEXT    DEFAULT '[]',
    created_at              TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS detections (
    db_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id   TEXT    NOT NULL REFERENCES inspections(inspection_id),
    track_id        INTEGER NOT NULL,
    track_label     TEXT    NOT NULL,
    class_id        INTEGER NOT NULL,
    class_name      TEXT    NOT NULL,
    display_name    TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    bbox_x1         REAL,
    bbox_y1         REAL,
    bbox_x2         REAL,
    bbox_y2         REAL,
    bbox_area_px2   REAL    DEFAULT 0,
    frame_percentage REAL   DEFAULT 0,
    region          TEXT,
    cx_norm         REAL,
    cy_norm         REAL,
    frame_idx       INTEGER DEFAULT 0,
    timestamp       REAL    DEFAULT 0,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tracks (
    db_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id       TEXT    NOT NULL REFERENCES inspections(inspection_id),
    track_id            INTEGER NOT NULL,
    track_label         TEXT    NOT NULL,
    class_id            INTEGER NOT NULL,
    class_name          TEXT    NOT NULL,
    display_name        TEXT    NOT NULL,
    avg_confidence      REAL    DEFAULT 0,
    frame_count         INTEGER DEFAULT 0,
    first_seen_ts       REAL    DEFAULT 0,
    last_seen_ts        REAL    DEFAULT 0,
    first_seen_frame    INTEGER DEFAULT 0,
    last_seen_frame     INTEGER DEFAULT 0,
    most_common_region  TEXT,
    created_at          TEXT    DEFAULT (datetime('now')),
    UNIQUE (inspection_id, track_id)
);

CREATE TABLE IF NOT EXISTS patterns (
    db_id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id           TEXT    NOT NULL REFERENCES inspections(inspection_id),
    pattern_id              TEXT    NOT NULL,
    pattern_type            TEXT    NOT NULL,
    class_name              TEXT    NOT NULL,
    display_name            TEXT    NOT NULL,
    unique_occurrences      INTEGER DEFAULT 0,
    region                  TEXT,
    first_occurrence_ts     REAL    DEFAULT 0,
    latest_occurrence_ts    REAL    DEFAULT 0,
    duration_seconds        REAL    DEFAULT 0,
    avg_interval_seconds    REAL,
    triggered_ts            REAL    DEFAULT 0,
    description             TEXT,
    created_at              TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS timeline_events (
    db_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id   TEXT    NOT NULL REFERENCES inspections(inspection_id),
    event_type      TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    description     TEXT,
    related_id      TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_metadata (
    db_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id       TEXT    NOT NULL REFERENCES inspections(inspection_id),
    model_name          TEXT    NOT NULL,
    model_version       TEXT,
    weights_path        TEXT,
    backend             TEXT,
    input_resolution    INTEGER DEFAULT 640,
    conf_threshold      REAL,
    iou_threshold       REAL,
    device              TEXT,
    precision_mode      TEXT    DEFAULT 'FP32',
    created_at          TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_detections_inspection ON detections(inspection_id);
CREATE INDEX IF NOT EXISTS idx_tracks_inspection     ON tracks(inspection_id);
CREATE INDEX IF NOT EXISTS idx_patterns_inspection   ON patterns(inspection_id);
CREATE INDEX IF NOT EXISTS idx_timeline_inspection   ON timeline_events(inspection_id);
CREATE INDEX IF NOT EXISTS idx_inspections_batch     ON inspections(batch_id);
CREATE INDEX IF NOT EXISTS idx_inspections_status    ON inspections(inspection_status);
"""
