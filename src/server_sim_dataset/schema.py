from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CT_STEM = re.compile(r"^CT_[^_]+_[^_]+_(?P<battery>\d+)_(?P<axis>[xyz])_(?P<index>\d+)$", re.I)
RGB_STEM = re.compile(r"^RGB_[^_]+_[^_]+_(?P<battery>\d+)_(?P<index>\d+)$", re.I)


# 계획서 11.1 의 필수 컬럼 60 개에, 8.1 이 본문에서 요구하는 원본 추적 3 개와 7.5 의
# reserve 추적 7 개, 9.1 의 JPEG profile 1 개, 6.2 의 슬라이스 단위 seed 1 개를 더한
# 정본 목록이다. generation plan 과 dataset manifest 는 같은 스키마를 쓴다. plan 단계에
# 값이 정해지지 않는 컬럼은 빈 문자열로 두고 생성 단계에서 채운다.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id", "synthetic_id", "capture_set", "capture_group_id", "retry_of_sample_id",
    "modality", "original_battery_id", "output_battery_id", "product_status", "axis",
    "original_index", "source_sequence_order", "output_sequence_order", "source_split",
    "index_gap_before", "index_gap_size",
    "original_stem", "orig_image_relative_path", "orig_json_relative_path",
    "original_image_id", "original_image_file_name", "original_roi",
    "source_image_sha256", "source_json_sha256", "pixel_hash",
    "original_is_normal", "has_porosity", "has_damaged", "has_pollution",
    "original_defect_count", "output_defect_count", "class_instance_counts",
    "capture_quality", "failure_case", "failure_segment_id", "failure_artifact_mask_path",
    "generation_status", "exclusion_or_retry_reason",
    "base_augmentation_names", "normal_augmentation_parameters", "normal_augmentation_seed",
    "slice_seed", "normal_base_pixel_hash",
    "failure_window_start", "failure_window_end", "failure_method_order",
    "failure_augmentation_parameters", "augmentation_json_path", "augmentation_json_sha256",
    "reserve_candidates", "reserve_rank", "reserve_source_split",
    "reserve_original_battery_id", "reserve_original_index", "reserve_original_stem",
    "reserve_reason",
    "global_seed", "item_seed", "generator_version", "config_hash", "plan_sha256",
    "output_image_path", "output_json_path", "output_det_path", "output_seg_path",
    "jpeg_profile_id",
    "output_image_sha256", "output_json_sha256", "output_det_sha256", "output_seg_sha256",
    "quality_gate_passed", "quality_gate_metrics",
)

# 생성 단계에서만 값이 정해지는 컬럼이다. plan 에는 빈 문자열로 들어간다.
GENERATION_COLUMNS: frozenset[str] = frozenset({
    "pixel_hash", "output_defect_count", "class_instance_counts", "failure_artifact_mask_path",
    "generation_status", "exclusion_or_retry_reason", "normal_base_pixel_hash",
    "failure_method_order", "failure_augmentation_parameters", "augmentation_json_path",
    "augmentation_json_sha256", "reserve_rank", "reserve_source_split",
    "reserve_original_battery_id", "reserve_original_index", "reserve_original_stem",
    "reserve_reason", "generator_version", "plan_sha256",
    "output_image_path", "output_json_path", "output_det_path", "output_seg_path",
    "jpeg_profile_id", "output_image_sha256", "output_json_sha256", "output_det_sha256",
    "output_seg_sha256", "quality_gate_passed", "quality_gate_metrics",
})


def output_stem(capture_set: str, modality: str, output_battery_id: int, axis: str, index: int) -> str:
    """계획서 9.1 의 출력 stem 을 만든다.

    plan 과 generator 가 같은 규칙을 두 번 구현하면 어긋날 수 있으므로 한 곳에 둔다.
    """
    prefix = "initial" if capture_set == "initial_capture" else "recapture"
    middle = f"{axis}_" if modality == "CT" else ""
    return f"{prefix}_{modality}_{output_battery_id}_{middle}{index:06d}"


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

