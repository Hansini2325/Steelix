from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.logger import get_logger

log = get_logger(__name__)


def export_passport_to_json(passport: Dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(passport, f, indent=2, default=str)
    log.info(f"Passport JSON exported -> {output_path}")
    return output_path


def export_passport_to_csv(passport: Dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = passport.get("tracks", [])
    if not tracks:
        log.warning("No tracks in passport to export to CSV.")
        return output_path

    if not isinstance(tracks, list):
        log.warning("Invalid tracks format.")
        return output_path

    if len(tracks) > 0 and isinstance(tracks[0], dict):
        keys = list(tracks[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for t in tracks:
                writer.writerow(t)

        log.info(f"Passport Tracks CSV exported -> {output_path}")

    return output_path
