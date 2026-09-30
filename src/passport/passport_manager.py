from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.database.models import Inspection, ModelMetadata
from src.database.repository import Repository
from src.utils.logger import get_logger

log = get_logger(__name__)


class PassportManager:
    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        if db_path:
            self.repo = Repository(db_path=Path(db_path))
        else:
            self.repo = Repository()

    def start_inspection(
        self,
        input_source: str,
        model_name: str,
        backend: str = "PyTorch",
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        batch_name: Optional[str] = None,
        operator_notes: Optional[str] = None,
    ) -> str:
        batch_id = self.repo.generate_batch_id()
        inspection_id = self.repo.generate_inspection_id()

        ins = Inspection(
            inspection_id=inspection_id,
            batch_id=batch_id,
            batch_name=batch_name,
            start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            end_time=None,
            duration_seconds=0.0,
            input_source=input_source,
            operator_notes=operator_notes,
            inspection_status="REVIEW",
            status_reason=None,
            total_frame_detections=0,
            unique_tracked_defects=0,
            avg_confidence=0.0,
            low_conf_count=0,
            most_frequent_class=None,
            most_affected_region=None,
            pattern_count=0,
            most_recurring_class=None,
            highest_recurrence_region=None,
            longest_recurring_sequence=0,
            model_name=model_name,
            backend=backend,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_counts_json="{}",
            region_counts_json="{}",
            heatmap_coords_json="[]",
        )
        self.repo.create_inspection(ins)

        meta = ModelMetadata(
            inspection_id=inspection_id,
            model_name=model_name,
            model_version=None,
            weights_path=model_name,
            backend=backend,
            input_resolution=640,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device="auto",
            precision_mode="FP32",
        )
        self.repo.insert_model_metadata(meta)

        return inspection_id

    def finalize_inspection(
        self,
        inspection_id: str,
        summary: Dict[str, Any],
    ) -> None:
        record = self.repo.get_inspection(inspection_id)
        if not record:
            log.error(f"Cannot finalize unknown inspection: {inspection_id}")
            return

        import json

        ins = Inspection(
            inspection_id=inspection_id,
            batch_id=record["batch_id"],
            batch_name=record["batch_name"],
            start_time=record["start_time"],
            end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            duration_seconds=summary.get("duration_seconds", 0.0),
            input_source=record["input_source"],
            operator_notes=record["operator_notes"],
            inspection_status="REVIEW",
            status_reason=None,
            total_frame_detections=summary.get("total_frame_detections", 0),
            unique_tracked_defects=summary.get("unique_tracked_defects", 0),
            avg_confidence=summary.get("avg_confidence", 0.0),
            low_conf_count=summary.get("low_conf_count", 0),
            most_frequent_class=summary.get("most_frequent_class"),
            most_affected_region=summary.get("most_affected_region"),
            pattern_count=summary.get("pattern_count", 0),
            most_recurring_class=summary.get("most_recurring_class"),
            highest_recurrence_region=summary.get("highest_recurrence_region"),
            longest_recurring_sequence=summary.get("longest_recurring_sequence", 0),
            model_name=record["model_name"],
            backend=record["backend"],
            conf_threshold=record["conf_threshold"],
            iou_threshold=record["iou_threshold"],
            class_counts_json=json.dumps(summary.get("class_counts", {})),
            region_counts_json=json.dumps(summary.get("region_counts", {})),
            heatmap_coords_json=json.dumps(summary.get("coords_sample", [])),
        )

        n_pass = summary.get("unique_tracked_defects", 0)
        flags = []
        if n_pass > 5:
            flags.append("High defect count")
        if summary.get("low_conf_count", 0) > 10:
            flags.append("Multiple low confidence detections")
        if summary.get("pattern_count", 0) > 0:
            flags.append("Recurring patterns found")

        if not flags and n_pass == 0:
            ins.inspection_status = "PASS"
        elif flags:
            ins.inspection_status = "FLAGGED"
            ins.status_reason = "; ".join(flags)
        else:
            ins.inspection_status = "REVIEW"

        self.repo.update_inspection(ins)
        log.info(f"Inspection finalized: {inspection_id} -> {ins.inspection_status}")

    def get_passport(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        return self.repo.get_full_inspection(inspection_id)

    def list_inspections(self, **kwargs) -> List[Dict[str, Any]]:
        return self.repo.list_inspections(**kwargs)

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self.repo.search_inspections(query)
