from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_dataframe(frame: pd.DataFrame, output_dir: str | Path, filename: str) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_json(data: dict[str, Any], output_dir: str | Path, filename: str) -> Path:
    output_dir = ensure_output_dir(output_dir)
    path = output_dir / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

