from __future__ import annotations
from pathlib import Path
import json
from .benchmarks import metric_records


def export_report(table, result: dict, path="outputs/verification_report.json"):
    Path(path).parent.mkdir(exist_ok=True)
    payload = {"data_mode": result["data_mode"], "regime": result["active_regime"], "metrics": metric_records(table)}
    Path(path).write_text(json.dumps(payload, indent=2))
    return path
