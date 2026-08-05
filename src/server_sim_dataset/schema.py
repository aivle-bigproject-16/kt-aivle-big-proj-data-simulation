from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CT_STEM = re.compile(r"^CT_[^_]+_[^_]+_(?P<battery>\d+)_(?P<axis>[xyz])_(?P<index>\d+)$", re.I)
RGB_STEM = re.compile(r"^RGB_[^_]+_[^_]+_(?P<battery>\d+)_(?P<index>\d+)$", re.I)


@dataclass(frozen=True)
class ParsedStem:
    modality: str
    battery_id: int
    axis: str
    original_index: int


def parse_stem(stem: str) -> ParsedStem:
    match = CT_STEM.match(stem)
    if match:
        return ParsedStem("CT", int(match["battery"]), match["axis"].lower(), int(match["index"]))
    match = RGB_STEM.match(stem)
    if match:
        return ParsedStem("RGB", int(match["battery"]), "", int(match["index"]))
    raise ValueError(f"Unsupported source stem: {stem}")


def points(value: Any) -> list[tuple[float, float]]:
    if value is None:
        return []
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        if len(value) % 2:
            raise ValueError("Flat polygon coordinate count must be even")
        result = [(float(value[i]), float(value[i + 1])) for i in range(0, len(value), 2)]
    elif isinstance(value, list):
        result = [(float(item[0]), float(item[1])) for item in value]
    else:
        raise ValueError("Polygon must be a list")
    if result and (len(result) < 3 or not all(math.isfinite(v) for p in result for v in p)):
        raise ValueError("Polygon must contain at least three finite points")
    return result


def defect_name(defect: dict[str, Any]) -> str:
    for key in ("class_name", "class", "label", "name", "defect_type", "type"):
        value = defect.get(key)
        if isinstance(value, str):
            return value
    return ""


def iter_defects(payload: dict[str, Any]) -> Iterable[tuple[str, list[tuple[float, float]]]]:
    raw = payload.get("defects") or []
    if isinstance(raw, dict):
        raw = raw.get("items") or raw.get("annotations") or [raw]
    if not isinstance(raw, list):
        raise ValueError("defects must be null, list, or object")
    for defect in raw:
        if not isinstance(defect, dict):
            continue
        polygon = points(defect.get("points") or defect.get("polygon") or defect.get("segmentation"))
        if polygon:
            yield defect_name(defect), polygon


def roi_bbox(payload: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    roi = (payload.get("data_info") or {}).get("roi")
    if roi is None:
        return 0, 0, width, height
    if not isinstance(roi, list) or len(roi) not in (2, 4):
        raise ValueError("data_info.roi must have 2 or 4 numeric elements")
    values = [int(round(float(value))) for value in roi]
    left, top, right, bottom = (0, 0, values[0], values[1]) if len(values) == 2 else tuple(values)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"ROI outside image: {values} vs {width}x{height}")
    return left, top, right, bottom


def split_name(path: Path) -> str:
    lowered = {part.lower() for part in path.parts}
    if "training" in lowered:
        return "training"
    if "validation" in lowered:
        return "validation"
    return "unknown"

