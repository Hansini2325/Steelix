from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.passport.passport_manager import PassportManager
from src.passport.export_manager import export_passport_to_json, export_passport_to_csv
from src.passport.report_generator import ReportGenerator


def test_passport_lifecycle():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_passport.db"
        pm = PassportManager(db_path=db_path)
        
        ins_id = pm.start_inspection(
            input_source="test.mp4",
            model_name="yolov8n",
            backend="onnx",
            conf_threshold=0.35,
            iou_threshold=0.5,
            operator_notes="Test note",
        )
        
        assert ins_id.startswith("INS-")
        
        summary = {
            "duration_seconds": 10.5,
            "total_frame_detections": 15,
            "unique_tracked_defects": 3,
            "avg_confidence": 0.85,
            "low_conf_count": 0,
            "most_frequent_class": "inclusion",
            "most_affected_region": "Center",
            "class_counts": {"inclusion": 3},
            "region_counts": {"Center": 3},
        }
        
        pm.finalize_inspection(ins_id, summary)
        
        pp = pm.get_passport(ins_id)
        assert pp is not None
        assert pp["total_frame_detections"] == 15
        
        pm.close()
        
        json_path = Path(td) / "passport.json"
        export_passport_to_json(pp, json_path)
        assert json_path.exists()
        
        with open(json_path, "r") as f:
            js = json.load(f)
            assert js["inspection_id"] == ins_id
        
        rg = ReportGenerator()
        html_out = Path(td) / "report.html"
        rg.generate_html_report(pp, html_out)
        assert html_out.exists()
