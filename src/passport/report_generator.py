from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.utils.logger import get_logger

log = get_logger(__name__)


class ReportGenerator:
    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir

    def generate_html_report(self, passport: Dict[str, Any], output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ins_id = passport.get("inspection_id", "UNKNOWN")
        status = passport.get("inspection_status", "UNKNOWN")
        batch_id = passport.get("batch_id", "UNKNOWN")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<title>Inspection Report: {ins_id}</title>
<style>
  body {{ font-family: sans-serif; margin: 40px; background: #f9f9f9; color: #333; }}
  .header {{ border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-bottom: 20px; }}
  .status-PASS {{ color: green; font-weight: bold; }}
  .status-FLAGGED {{ color: red; font-weight: bold; }}
  .status-REVIEW {{ color: orange; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
  th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .card {{ background: #fff; padding: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Steel Defect Inspection Report</h1>
  <p><strong>Inspection ID:</strong> {ins_id}</p>
  <p><strong>Batch ID:</strong> {batch_id}</p>
  <p><strong>Status:</strong> <span class="status-{status}">{status}</span></p>
</div>

<div class="card">
  <h2>Summary</h2>
  <table>
    <tr><th>Start Time</th><td>{passport.get("start_time", "")}</td></tr>
    <tr><th>Duration (s)</th><td>{passport.get("duration_seconds", 0):.2f}</td></tr>
    <tr><th>Total Detections</th><td>{passport.get("total_frame_detections", 0)}</td></tr>
    <tr><th>Unique Defects</th><td>{passport.get("unique_tracked_defects", 0)}</td></tr>
    <tr><th>Avg Confidence</th><td>{passport.get("avg_confidence", 0):.3f}</td></tr>
    <tr><th>Reason</th><td>{passport.get("status_reason", "")}</td></tr>
  </table>
</div>
"""
        cc = passport.get("class_counts", {})
        if cc:
            html_content += """
<div class="card">
  <h2>Class Counts</h2>
  <table>
    <tr><th>Class</th><th>Count</th></tr>
"""
            for c_name, c_cnt in cc.items():
                html_content += f"    <tr><td>{c_name}</td><td>{c_cnt}</td></tr>\n"
            html_content += "  </table>\n</div>\n"

        html_content += """
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        log.info(f"HTML report saved -> {output_path}")
        return output_path

    def generate_pdf_report(self, passport: Dict[str, Any], output_path: Path) -> Path:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
        except ImportError:
            log.warning("ReportLab not installed, falling back to HTML report.")
            return self.generate_html_report(passport, output_path.with_suffix(".html"))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph(f"Steel Defect Inspection Report", styles['Title']))
        elements.append(Spacer(1, 12))

        ins_id = passport.get("inspection_id", "UNKNOWN")
        status = passport.get("inspection_status", "UNKNOWN")
        elements.append(Paragraph(f"<b>Inspection ID:</b> {ins_id}", styles['Normal']))
        elements.append(Paragraph(f"<b>Status:</b> {status}", styles['Normal']))
        elements.append(Paragraph(f"<b>Reason:</b> {passport.get('status_reason', '')}", styles['Normal']))
        elements.append(Spacer(1, 12))

        data = [
            ["Metric", "Value"],
            ["Start Time", passport.get("start_time", "")],
            ["Duration (s)", f"{passport.get('duration_seconds', 0):.2f}"],
            ["Total Detections", str(passport.get("total_frame_detections", 0))],
            ["Unique Defects", str(passport.get("unique_tracked_defects", 0))],
            ["Avg Conf", f"{passport.get('avg_confidence', 0):.3f}"],
            ["Top Class", str(passport.get("most_frequent_class", ""))],
        ]

        t = Table(data, colWidths=[200, 300])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(t)

        doc.build(elements)
        log.info(f"PDF report saved -> {output_path}")
        return output_path
