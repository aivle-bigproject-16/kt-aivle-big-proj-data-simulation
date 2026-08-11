from __future__ import annotations

# 이 도구는 v1.2 의 cache CSV 스키마(image_stem 컬럼)와 "20 개 중 1 개 불량" 규칙을
# 전제한다. v1.3 은 cache v2 와 층화 배분을 쓰므로 둘 다 맞지 않는다. 계획서 11.2 의
# raw_extraction_feasibility.json 은 reports.feasibility_audit 가 만든다. 그쪽은 planner 의
# 층 배정과 배분 함수를 직접 불러 쓰므로 선정 로직과 어긋날 수 없다.

import argparse
import csv
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


CT_RE = re.compile(r"^CT_cell_(?P<form>[^_]+)_(?P<id>\d+)_(?P<axis>[xyz])_(?P<index>\d+)$")
RGB_RE = re.compile(r"^RGB_cell_(?P<form>[^_]+)_(?P<id>\d+)_(?P<index>\d+)$")
CT_NEEDS = {"x": 150, "y": 650, "z": 650}
RGB_NEED = 250


def _valid_defects(data: dict, modality: str) -> int:
    defects = data.get("defects") or []
    allowed = {"porosity"} if modality == "CT" else {"Damaged", "Pollution"}
    count = 0
    for defect in defects:
        if not isinstance(defect, dict):
            continue
        name = str(defect.get("name", ""))
        points = defect.get("points")
        if name.lower() in {value.lower() for value in allowed} and isinstance(points, list) and len(points) >= 6:
            count += 1
    return count


def _read_one(args: tuple[Path, dict[str, str]]) -> tuple[dict[str, str], dict]:
    raw_root, row = args
    path = raw_root / Path(row["raw_json_path"])
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return row, {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}

    modality = row["modality"]
    info = data.get("data_info") or {}
    image_info = data.get("image_info") or {}
    swelling = data.get("swelling") or {}
    errors: list[str] = []
    expected_type = "ct" if modality == "CT" else "rgb"
    if str(info.get("data_type", "")).lower() != expected_type:
        errors.append("data_type")
    if not isinstance(swelling.get("battery_outline"), list) or len(swelling.get("battery_outline") or []) < 6:
        errors.append("battery_outline")
    if Path(str(image_info.get("file_name", ""))).stem != row["image_stem"]:
        errors.append("image_file_name")
    defects = _valid_defects(data, modality)
    declared_normal = image_info.get("is_normal")
    if isinstance(declared_normal, bool) and declared_normal == (defects > 0):
        errors.append("is_normal_conflict")
    return row, {
        "errors": errors,
        "defect_count": defects,
        "battery_id_json": str(info.get("battery_ids", "")),
    }


def _window_exists(items: list[dict], size: int, want_defect: bool) -> bool:
    if len(items) < size:
        return False
    flags = [int(item["defect_count"] > 0) for item in items]
    current = sum(flags[:size])
    if (current > 0) == want_defect:
        return True
    for index in range(size, len(flags)):
        current += flags[index] - flags[index - size]
        if (current > 0) == want_defect:
            return True
    return False


def _normal_pool_has(items: list[dict], size: int) -> bool:
    return sum(item["defect_count"] == 0 for item in items) >= size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--scan-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    with args.scan_cache.open("r", encoding="utf-8-sig", newline="") as handle:
        cache_rows = [row for row in csv.DictReader(handle) if row["status"] == "valid"]

    # Training/Validation에 동일 key가 있으면 hash가 같은 복제는 하나만 유지하고,
    # 내용이 다른 충돌은 양쪽 모두 제외한다.
    keyed: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    malformed_stems: list[str] = []
    for row in cache_rows:
        match = CT_RE.match(row["image_stem"]) if row["modality"] == "CT" else RGB_RE.match(row["image_stem"])
        if not match:
            malformed_stems.append(row["image_stem"])
            continue
        group = match.groupdict()
        key = (row["modality"], int(group["id"]), group.get("axis", ""), int(group["index"]))
        row = dict(row)
        row.update({"parsed_id": group["id"], "axis": group.get("axis", ""), "parsed_index": group["index"]})
        keyed[key].append(row)

    selected: list[dict[str, str]] = []
    duplicate_same = duplicate_conflict = 0
    for rows in keyed.values():
        if len(rows) == 1:
            selected.append(rows[0])
        elif len({(row["image_sha256"], row["json_sha256"]) for row in rows}) == 1:
            duplicate_same += len(rows) - 1
            selected.append(sorted(rows, key=lambda row: row["raw_split"])[0])
        else:
            duplicate_conflict += len(rows)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        parsed = list(pool.map(_read_one, ((args.raw_root, row) for row in selected), chunksize=128))

    schema_errors: list[dict] = []
    ct: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    rgb: dict[int, list[dict]] = defaultdict(list)
    for row, result in parsed:
        errors = result.get("errors") or ([result["error"]] if "error" in result else [])
        if errors:
            schema_errors.append({"stem": row["image_stem"], "errors": errors})
            continue
        if str(int(result["battery_id_json"])) != str(int(row["parsed_id"])):
            schema_errors.append({"stem": row["image_stem"], "errors": ["battery_id_mismatch"]})
            continue
        defect_count = (
            int(row.get("porosity_component_count") or 0)
            if row["modality"] == "CT"
            else result["defect_count"]
        )
        item = {"index": int(row["parsed_index"]), "defect_count": defect_count, "stem": row["image_stem"]}
        if row["modality"] == "CT":
            ct[int(row["parsed_id"])][row["axis"]].append(item)
        else:
            rgb[int(row["parsed_id"])].append(item)

    for axes in ct.values():
        for items in axes.values():
            items.sort(key=lambda item: item["index"])
    for items in rgb.values():
        items.sort(key=lambda item: item["index"])

    ct_normal_contiguous: list[int] = []
    ct_normal: list[int] = []
    ct_defective: list[int] = []
    for battery_id, axes in ct.items():
        eligible = all(len(axes[axis]) >= need for axis, need in CT_NEEDS.items())
        if not eligible:
            continue
        normal_contiguous = all(_window_exists(axes[axis], need, False) for axis, need in CT_NEEDS.items())
        normal = all(_normal_pool_has(axes[axis], need) for axis, need in CT_NEEDS.items())
        # 각 축의 선택 window 중 적어도 한 축에 결함을 포함시킬 수 있어야 한다.
        defective = all(len(axes[axis]) >= need for axis, need in CT_NEEDS.items()) and any(
            _window_exists(axes[axis], need, True) for axis, need in CT_NEEDS.items()
        )
        if normal_contiguous:
            ct_normal_contiguous.append(battery_id)
        if normal:
            ct_normal.append(battery_id)
        if defective:
            ct_defective.append(battery_id)

    rgb_normal = sorted(battery_id for battery_id, items in rgb.items() if _window_exists(items, RGB_NEED, False))
    rgb_defective = sorted(battery_id for battery_id, items in rgb.items() if _window_exists(items, RGB_NEED, True))

    report = {
        "cache_valid_rows": len(cache_rows),
        "deduplicated_pairs": len(selected),
        "duplicate_same_removed": duplicate_same,
        "duplicate_conflict_rows_excluded": duplicate_conflict,
        "malformed_stem_count": len(malformed_stems),
        "schema_error_count": len(schema_errors),
        "schema_error_examples": schema_errors[:20],
        "CT": {
            "old_contiguous_normal_candidate_count": len(ct_normal_contiguous),
            "normal_candidate_count": len(ct_normal),
            "defective_candidate_count": len(ct_defective),
            "normal_candidate_ids": sorted(ct_normal),
            "defective_candidate_ids": sorted(ct_defective),
            "required_normal_ids": 19,
            "required_defective_ids": 1,
        },
        "RGB": {
            "normal_candidate_count": len(rgb_normal),
            "defective_candidate_count": len(rgb_defective),
            "normal_candidate_ids": rgb_normal,
            "defective_candidate_ids": rgb_defective,
            "required_normal_ids": 19,
            "required_defective_ids": 1,
        },
        "output_id_ranges_disjoint": set(range(1900000001, 1900000021)).isdisjoint(range(2900000001, 2900000021)),
    }
    report["plan_feasible"] = (
        len(ct_normal) >= 19
        and len(ct_defective) >= 1
        and len(rgb_normal) >= 19
        and len(rgb_defective) >= 1
        and not malformed_stems
        and not schema_errors
        and report["output_id_ranges_disjoint"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["plan_feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
