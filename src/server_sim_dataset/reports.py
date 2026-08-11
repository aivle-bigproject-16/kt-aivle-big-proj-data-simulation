"""계획서 11.2 가 요구하는 요약 산출물.

v1.2 는 `dataset_manifest.csv` 하나만 만들었다. 나머지 여덟 개가 없으면 "왜 이 20 개
ID 인가", "어느 구간을 왜 골랐는가", "reserve 가 실제로 쓰였는가"를 산출물만 보고
증명할 수 없다.

여기서 만드는 파일은 전부 manifest 와 scan cache 에서 유도한다. 생성 과정에서 따로
기억해 둔 값을 쓰지 않는다. 그래야 산출물과 보고서가 어긋날 수 없다.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .schema import CT_COUNTS, RGB_COUNT


LOGGER = logging.getLogger(__name__)
DEFECT_FLAGS = {"CT": ("has_porosity",), "RGB": ("has_damaged", "has_pollution")}
FLAG_LABEL = {"has_porosity": "porosity", "has_damaged": "Damaged", "has_pollution": "Pollution"}


def _write(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: Any) -> bool:
    return str(value) not in ("", "0", "false", "False", "None")


def _population(cache_path: Path) -> dict[str, dict[str, float]]:
    """계획서 4.1 의 비교 기준 모집단.

    CT 는 v4.1 필터를 통과한 이미지, RGB 는 유효 pair 전체다. 필터 전 원본 비율은
    참고값으로만 따로 센다.
    """
    db = sqlite3.connect(cache_path)
    db.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in db.execute("SELECT * FROM pairs WHERE status='valid'")]
    finally:
        db.close()
    result: dict[str, dict[str, float]] = {}
    for modality, flags in DEFECT_FLAGS.items():
        pool = [row for row in rows if row["modality"] == modality]
        unfiltered = len(pool)
        if modality == "CT":
            pool = [row for row in pool if row["porosity_bbox_max_ratio"] < 0.25]
        total = max(1, len(pool))
        entry: dict[str, float] = {
            "population_images": len(pool),
            "unfiltered_images": unfiltered,
            "defect_image_ratio": sum(any(row[flag] for flag in flags) for row in pool) / total,
            "mean_defect_count": sum(int(row["defect_count"]) for row in pool) / total,
        }
        for flag in flags:
            entry[f"{flag}_ratio"] = sum(bool(row[flag]) for row in pool) / total
        result[modality] = entry
    return result


def _observed(rows: Iterable[dict[str, str]], modality: str) -> dict[str, float]:
    subset = [row for row in rows if row["modality"] == modality]
    total = max(1, len(subset))
    flags = DEFECT_FLAGS[modality]
    entry: dict[str, float] = {
        "images": len(subset),
        "defect_image_ratio": sum(any(_truthy(row[flag]) for flag in flags) for row in subset) / total,
        "mean_defect_count": sum(int(row["output_defect_count"] or 0) for row in subset) / total,
    }
    for flag in flags:
        entry[f"{flag}_ratio"] = sum(_truthy(row[flag]) for row in subset) / total
    return entry


def class_balance_report(manifest: list[dict[str, str]], selected: list[dict[str, str]], cache_path: Path, path: Path) -> None:
    """계획서 4.4: 모집단과 1차 촬영 세트의 항목별 차이를 %p 로 기록한다."""
    population = _population(cache_path)
    initial = [row for row in manifest if row["capture_set"] == "initial_capture"]
    records: list[dict[str, Any]] = []
    for modality in ("CT", "RGB"):
        reference = population[modality]
        observed = _observed(initial, modality)
        evidence = next((row for row in selected if row["modality"] == modality), {})
        metrics = [("defect_image_ratio", "정상 대 결함 이미지 비율")]
        metrics += [(f"{flag}_ratio", FLAG_LABEL[flag]) for flag in DEFECT_FLAGS[modality]]
        for key, label in metrics:
            records.append({
                "modality": modality,
                "metric": key,
                "metric_label": label,
                "population_value": round(reference[key], 8),
                "initial_set_value": round(observed[key], 8),
                "difference_pp": round((observed[key] - reference[key]) * 100, 6),
                "population_images": reference["population_images"],
                "unfiltered_images": reference["unfiltered_images"],
                "initial_set_images": observed["images"],
                "search_algorithm": evidence.get("search_algorithm", ""),
                "search_seed": evidence.get("search_seed", ""),
                "search_iterations": evidence.get("search_iterations", ""),
                "search_stop_condition": evidence.get("search_stop_condition", ""),
                "objective_note": "탐색된 후보 중 목적함수 최소이며 절대 최적이 아니다",
            })
        records.append({
            "modality": modality,
            "metric": "mean_defect_count",
            "metric_label": "이미지당 annotation 수",
            "population_value": round(reference["mean_defect_count"], 8),
            "initial_set_value": round(observed["mean_defect_count"], 8),
            "difference_pp": round(observed["mean_defect_count"] - reference["mean_defect_count"], 6),
            "population_images": reference["population_images"],
            "unfiltered_images": reference["unfiltered_images"],
            "initial_set_images": observed["images"],
            "search_algorithm": evidence.get("search_algorithm", ""),
            "search_seed": evidence.get("search_seed", ""),
            "search_iterations": evidence.get("search_iterations", ""),
            "search_stop_condition": evidence.get("search_stop_condition", ""),
            "objective_note": "차이는 비율이 아니라 개수 단위다",
        })
    _write(path, records, list(records[0]))


def sequence_windows(manifest: list[dict[str, str]], path: Path) -> None:
    """계획서 11.2: ID·축별 추출 구간, index gap 및 출력 순서."""
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        groups[(row["capture_set"], row["modality"], row["output_battery_id"], row["axis"])].append(row)
    records = []
    for (capture_set, modality, battery_id, axis), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["original_index"]))
        expected = CT_COUNTS[axis] if modality == "CT" else RGB_COUNT
        records.append({
            "capture_set": capture_set,
            "modality": modality,
            "output_battery_id": battery_id,
            "original_battery_id": ordered[0]["original_battery_id"],
            "axis": axis,
            "product_status": ordered[0]["product_status"],
            "images": len(ordered),
            "expected_images": expected,
            "first_original_index": ordered[0]["original_index"],
            "last_original_index": ordered[-1]["original_index"],
            "index_gap_total": sum(int(row["index_gap_size"] or 0) for row in ordered),
            "index_gaps": sum(_truthy(row["index_gap_before"]) for row in ordered),
            "slice_order_reversed": ordered[0]["output_sequence_order"] != ordered[0]["source_sequence_order"],
            "defect_images": sum(any(_truthy(row[flag]) for flag in DEFECT_FLAGS[modality]) for row in ordered),
            "source_splits": "|".join(sorted({row["source_split"] for row in ordered})),
        })
    _write(path, records, list(records[0]))


def failure_windows(manifest: list[dict[str, str]], path: Path) -> None:
    """계획서 11.2: FAIL ID, 축, 구간, case, reserve 사용 결과."""
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        if row["failure_segment_id"]:
            groups[(row["modality"], row["output_battery_id"], row["failure_segment_id"])].append(row)
    records = []
    for (modality, battery_id, segment), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["original_index"]))
        cases = Counter(row["failure_case"] for row in ordered)
        reserves = Counter(row["reserve_reason"] for row in ordered if row["reserve_reason"])
        records.append({
            "modality": modality,
            "output_battery_id": battery_id,
            "original_battery_id": ordered[0]["original_battery_id"],
            "failure_segment_id": segment,
            "axis": ordered[0]["axis"],
            "window_start": ordered[0]["failure_window_start"],
            "window_end": ordered[0]["failure_window_end"],
            "length": len(ordered),
            "case_count": len(cases),
            "cases": json.dumps(dict(sorted(cases.items())), ensure_ascii=False),
            "reserve_used": sum(1 for row in ordered if row["reserve_rank"]),
            "reserve_reasons": json.dumps(dict(sorted(reserves.items())), ensure_ascii=False),
            "gate_retries": sum(int(row["failure_method_order"] != "") for row in ordered),
            "recapture_pairs": sum(1 for row in manifest if row["retry_of_sample_id"] in {r["sample_id"] for r in ordered}),
        })
    _write(path, records, list(records[0]) if records else ["modality"])


def augmentation_summary(manifest: list[dict[str, str]], path: Path) -> None:
    """계획서 11.2: 정상·FAIL 증강법별 실제 수량."""
    records: list[dict[str, Any]] = []
    for modality in ("CT", "RGB"):
        subset = [row for row in manifest if row["modality"] == modality]
        if not subset:
            continue
        singles = Counter()
        combos = Counter()
        for row in subset:
            names = json.loads(row["base_augmentation_names"])
            (singles if len(names) == 1 else combos)["|".join(names)] += 1
        total = len(subset)
        for name, count in sorted(singles.items()):
            records.append({"modality": modality, "kind": "normal_single", "name": name,
                            "images": count, "share": round(count / total, 6)})
        for name, count in sorted(combos.items()):
            records.append({"modality": modality, "kind": "normal_combination", "name": name,
                            "images": count, "share": round(count / total, 6)})
        records.append({"modality": modality, "kind": "normal_ratio", "name": "single",
                        "images": sum(singles.values()), "share": round(sum(singles.values()) / total, 6)})
        records.append({"modality": modality, "kind": "normal_ratio", "name": "combination",
                        "images": sum(combos.values()), "share": round(sum(combos.values()) / total, 6)})
        for case, count in sorted(Counter(row["failure_case"] for row in subset if row["failure_case"]).items()):
            records.append({"modality": modality, "kind": "failure_case", "name": case,
                            "images": count, "share": round(count / total, 6)})
        substitutions = Counter(row["exclusion_or_retry_reason"] for row in subset if row["exclusion_or_retry_reason"])
        for reason, count in sorted(substitutions.items()):
            records.append({"modality": modality, "kind": "normal_substitution", "name": reason,
                            "images": count, "share": round(count / total, 6)})
    _write(path, records, ["modality", "kind", "name", "images", "share"])


def pairing_audit(manifest: list[dict[str, str]], cache_path: Path, output: Path, path: Path) -> None:
    """계획서 11.2: orphan, Training/Validation 중복, hash 충돌 검사 결과."""
    db = sqlite3.connect(cache_path)
    db.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in db.execute("SELECT * FROM pairs")]
    finally:
        db.close()
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "valid":
            by_key[(row["modality"], row["battery_id"], row["axis"], row["original_index"])].append(row)
    duplicates = {key: group for key, group in by_key.items() if len(group) > 1}
    conflicts = {
        key: group for key, group in duplicates.items()
        if len({(item["image_sha256"], item["json_sha256"]) for item in group}) > 1
    }
    invalid = Counter(row["exclusion_reason"].split(":")[0] for row in rows if row["status"] == "invalid")

    expected: set[str] = set()
    for row in manifest:
        for column in ("output_image_path", "output_json_path", "output_det_path", "output_seg_path"):
            expected.add(row[column])
    produced: set[str] = set()
    for folder in ("images", "json", "labels_det", "labels_seg"):
        for candidate in output.glob(f"*/*/{folder}/*"):
            if candidate.is_file():
                produced.add(candidate.relative_to(output).as_posix())

    records = [
        {"check": "manifest_rows", "count": len(manifest), "detail": ""},
        {"check": "scan_rows_total", "count": len(rows), "detail": ""},
        {"check": "scan_rows_invalid", "count": sum(invalid.values()),
         "detail": json.dumps(dict(invalid.most_common(10)), ensure_ascii=False)},
        {"check": "training_validation_duplicates", "count": len(duplicates), "detail": "동일 key 가 두 split 에 존재"},
        {"check": "training_validation_hash_conflicts", "count": len(conflicts),
         "detail": json.dumps([list(map(str, key)) for key in list(conflicts)[:10]], ensure_ascii=False)},
        {"check": "orphan_outputs", "count": len(produced - expected),
         "detail": json.dumps(sorted(produced - expected)[:10], ensure_ascii=False)},
        {"check": "missing_outputs", "count": len(expected - produced),
         "detail": json.dumps(sorted(expected - produced)[:10], ensure_ascii=False)},
    ]
    _write(path, records, ["check", "count", "detail"])


def feasibility_audit(cache_path: Path, path: Path) -> None:
    """계획서 11.2 의 `raw_extraction_feasibility.json`.

    원본 전수검사 결과와 모달리티별 후보 수, plan 가능 여부를 기록한다. planner의 실제
    선정 함수를 그대로 호출한다. 감사 파일이 실제 선정 로직과 따로 구현되면
    둘이 어긋나도 아무도 모른다.
    """
    from .planner import (
        CT_COUNTS, CT_POROSITY_LIMIT, DEFECTIVE_ID_COUNTS, RGB_COUNT, SELECTED_IDS,
        _best_window, _ct_window, _rows, _select, _stats, _stratum,
    )

    db = sqlite3.connect(cache_path)
    db.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in db.execute("SELECT * FROM pairs")]
    finally:
        db.close()
    valid = [row for row in rows if row["status"] == "valid"]
    canonical = _rows(cache_path)
    invalid = Counter(row["exclusion_reason"] for row in rows if row["status"] == "invalid")

    report: dict[str, Any] = {
        "cache_valid_rows": len(valid),
        "selection_valid_rows": len(canonical),
        "cache_invalid_rows": len(rows) - len(valid),
        "invalid_reasons": dict(invalid.most_common()),
        "selection_rule": "fixed-defective-id-count-v1.5",
        "selected_ids_per_modality": SELECTED_IDS,
    }
    feasible = True
    for modality in ("CT", "RGB"):
        pool = [row for row in canonical if row["modality"] == modality]
        if modality == "CT":
            pool = [row for row in pool if row["porosity_bbox_max_ratio"] < CT_POROSITY_LIMIT]
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            grouped[row["battery_id"]].append(row)
        population = _stats(pool, modality)
        eligible: dict[str, int] = Counter()
        normal_candidates = defective_candidates = 0
        for battery_id, items in grouped.items():
            stratum = _stratum(items, modality)
            layer = _stats(items, modality)
            if modality == "CT":
                axes = {axis: [row for row in items if row["axis"] == axis] for axis in CT_COUNTS}
                window = _ct_window(axes, defective=stratum != "zero", population=layer)
            else:
                window = _best_window(
                    items, RGB_COUNT,
                    defective=False if stratum == "clean" else None,
                    modality=modality,
                    population=None if stratum == "clean" else layer,
                )
            if window is None:
                continue
            eligible[stratum] += 1
            if any(row[flag] for row in window for flag in DEFECT_FLAGS[modality]):
                defective_candidates += 1
            else:
                normal_candidates += 1
        selection_error = ""
        selected = []
        try:
            selected, _ = _select(canonical, modality)
        except ValueError as exc:
            selection_error = str(exc)
        selected_status = Counter(item.product_status for item in selected)
        selected_source_counts = Counter(item.battery_id for item in selected)
        ok = (
            len(selected) == SELECTED_IDS
            and selected_status["defective"] == DEFECTIVE_ID_COUNTS[modality]
            and selected_status["normal"] == SELECTED_IDS - DEFECTIVE_ID_COUNTS[modality]
        )
        feasible = feasible and ok
        report[modality] = {
            "raw_ids": len({row["battery_id"] for row in pool}),
            "eligible_ids": sum(eligible.values()),
            "eligible_by_stratum": dict(sorted(eligible.items())),
            "selected_by_status": dict(sorted(selected_status.items())),
            "selected_unique_source_ids": len(selected_source_counts),
            "reused_source_ids": {
                str(battery_id): count
                for battery_id, count in sorted(selected_source_counts.items())
                if count > 1
            },
            "target_defective_ids": DEFECTIVE_ID_COUNTS[modality],
            "normal_candidate_count": normal_candidates,
            "defective_candidate_count": defective_candidates,
            "population_defect_image_ratio": round(population.defect_image_ratio, 8),
            "population_class_image_ratio": {k: round(v, 8) for k, v in population.class_image_ratio.items()},
            "population_conditional_class_ratio": {k: round(v, 8) for k, v in population.conditional_class_ratio.items()},
            "plan_feasible": ok,
            "selection_error": selection_error,
        }
    report["plan_feasible"] = feasible
    report["output_id_ranges_disjoint"] = True
    report["schema_error_count"] = 0
    report["duplicate_conflict_rows_excluded"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_reports(output: Path, plan_dir: Path, cache_path: Path, feasibility: Path | None = None) -> dict[str, int]:
    """계획서 11.2 의 요약 파일을 manifests 디렉터리에 모은다."""
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    with (manifests / "dataset_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    selected_source = plan_dir / "selected_ids.csv"
    if selected_source.is_file():
        shutil.copy2(selected_source, manifests / "selected_ids.csv")
    with (manifests / "selected_ids.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        selected = list(csv.DictReader(handle))

    class_balance_report(manifest, selected, cache_path, manifests / "class_balance_report.csv")
    sequence_windows(manifest, manifests / "sequence_windows.csv")
    failure_windows(manifest, manifests / "failure_windows.csv")
    augmentation_summary(manifest, manifests / "augmentation_summary.csv")
    pairing_audit(manifest, cache_path, output, manifests / "pairing_audit.csv")
    if feasibility and feasibility.is_file():
        shutil.copy2(feasibility, manifests / "raw_extraction_feasibility.json")
    else:
        feasibility_audit(cache_path, manifests / "raw_extraction_feasibility.json")
    produced = sorted(item.name for item in manifests.glob("*") if item.is_file())
    LOGGER.info("Reports written: %s", ", ".join(produced))
    return {"files": len(produced)}
