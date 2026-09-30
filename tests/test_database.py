from __future__ import annotations

import tempfile
from pathlib import Path

from src.database.models import Detection, Inspection, Track, Pattern
from src.database.repository import Repository


def test_database_repository():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        repo = Repository(db_path=db_path)
        
        ins = Inspection(
            inspection_id="INS-001",
            batch_id="BATCH-01",
            batch_name="Test Batch",
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            duration_seconds=60.0,
            input_source="test.jpg",
            operator_notes="",
            inspection_status="PASS",
            status_reason="",
            total_frame_detections=0,
            unique_tracked_defects=0,
            avg_confidence=0.0,
            low_conf_count=0,
            most_frequent_class="patches",
            most_affected_region="Center",
            pattern_count=0,
            most_recurring_class="",
            highest_recurrence_region="",
            longest_recurring_sequence=0,
            model_name="yolov8n",
            backend="pytorch",
            conf_threshold=0.3,
            iou_threshold=0.45,
            class_counts_json="{}",
            region_counts_json="{}",
            heatmap_coords_json="[]",
        )
        
        repo.create_inspection(ins)
        ins_data = repo.get_inspection("INS-001")
        assert ins_data is not None
        assert ins_data["inspection_status"] == "PASS"
        assert ins_data["batch_id"] == "BATCH-01"
        
        ins.inspection_status = "FLAGGED"
        repo.update_inspection(ins)
        ins_data2 = repo.get_inspection("INS-001")
        assert ins_data2["inspection_status"] == "FLAGGED"
        
        det = Detection(
            inspection_id="INS-001",
            track_id=1,
            track_label="patches-1",
            class_id=0,
            class_name="patches",
            display_name="Patches",
            confidence=0.9,
            bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10,
            bbox_area_px2=100.0,
            frame_percentage=1.0,
            region="Top-left",
            cx_norm=0.1, cy_norm=0.1,
            frame_idx=1,
            timestamp=0.1,
        )
        repo.insert_detections([det])
        
        tr = Track(
            inspection_id="INS-001",
            track_id=1,
            track_label="patches-1",
            class_id=0,
            class_name="patches",
            display_name="Patches",
            avg_confidence=0.9,
            frame_count=1,
            first_seen_ts=0.1,
            last_seen_ts=0.1,
            first_seen_frame=1,
            last_seen_frame=1,
            most_common_region="Top-left"
        )
        repo.upsert_track(tr)
        
        full_ins = repo.get_full_inspection("INS-001")
        assert len(full_ins["tracks"]) == 1
        assert full_ins["tracks"][0]["track_id"] == 1
        
        repo.close()
