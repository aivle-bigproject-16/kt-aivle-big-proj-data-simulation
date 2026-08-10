from __future__ import annotations

import csv
import json
import logging
import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .schema import CT_COUNTS, GENERATION_COLUMNS, MANIFEST_COLUMNS, RGB_COUNT, output_stem
from .util import atomic_json, config_hash, stable_seed


LOGGER = logging.getLogger(__name__)
GLOBAL_SEED = 20260723
CT_POROSITY_LIMIT = 0.25
NORMAL_AUGMENTATIONS = {
    "CT": ("brightness_contrast_gamma", "partial_histogram_blend", "normal_noise_poisson", "low_frequency_shading", "percentile_tone_curve", "weak_reconstruction_kernel", "synchronized_flip"),
    "RGB": ("safe_translate_rotate", "brightness_contrast_gamma", "rgb_channel_gain_tone", "low_frequency_lighting", "poisson_noise", "weak_reconstruction"),
}
FAILURE_CASES = {
    "CT": ("ct_cell_alignment_failure", "ct_acquisition_motion", "ct_insufficient_projection_sampling", "ct_low_signal_noise", "ct_beam_hardening_metal_streak"),
    "RGB": ("rgb_trigger_timing_failure", "rgb_uneven_lighting", "rgb_reflection_glare", "rgb_focus_failure", "rgb_underexposure", "rgb_overexposure", "rgb_surface_dust", "rgb_hair_contamination"),
}
DEFECT_FLAGS = {"CT": ("has_porosity",), "RGB": ("has_damaged", "has_pollution")}

SEARCH_ALGORITHM = "deterministic-sliding-window-v1.3"
SEARCH_STOP_CONDITION = "exhaustive over eligible windows of the fixed length"


def _rows(cache_path: Path) -> list[dict[str, Any]]:
    db = sqlite3.connect(cache_path)
    try:
        db.row_factory = sqlite3.Row
        raw = [dict(row) for row in db.execute("SELECT * FROM pairs WHERE status='valid'")]
    finally:
        db.close()
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        by_key[(row["modality"], row["battery_id"], row["axis"], row["original_index"])].append(row)
    result = []
    for key, candidates in by_key.items():
        hashes = {(row["image_sha256"], row["json_sha256"]) for row in candidates}
        if len(hashes) == 1:
            result.append(sorted(candidates, key=lambda row: row["source_split"])[0])
        else:
            LOGGER.warning("Excluded Training/Validation hash conflict: %s", key)
    return result


@dataclass(frozen=True)
class Stats:
    """계획서 4.4 의 1차·2차 기준을 계산하기 위한 집계값."""

    defect_image_ratio: float
    class_image_ratio: dict[str, float]
    mean_defect_count: float


def _stats(rows: list[dict[str, Any]], modality: str) -> Stats:
    total = max(1, len(rows))
    flags = DEFECT_FLAGS[modality]
    defective = sum(any(row[flag] for flag in flags) for row in rows)
    return Stats(
        defect_image_ratio=defective / total,
        class_image_ratio={flag: sum(bool(row[flag]) for row in rows) / total for flag in flags},
        mean_defect_count=sum(int(row["defect_count"]) for row in rows) / total,
    )


def _objective(window: list[dict[str, Any]], population: Stats, modality: str) -> tuple[float, float]:
    """계획서 4.4 의 1차·2차 목적값을 %p 단위로 돌려준다.

    1차는 정상 대 결함 이미지 비율의 절대 차이이고, 2차는 클래스별 이미지 비율 차이와
    이미지당 annotation 수 차이 중 큰 쪽이다. 두 값 모두 작을수록 비교 기준 모집단에
    가깝다.

    v1.2 는 후보 비율을 20 으로 나눈 뒤 모집단과 비교했다. 그 스케일에는 근거가 없고
    정렬을 사실상 무작위로 만들었다.
    """
    stats = _stats(window, modality)
    primary = abs(stats.defect_image_ratio - population.defect_image_ratio) * 100
    class_gap = max(
        abs(stats.class_image_ratio[flag] - population.class_image_ratio[flag]) * 100
        for flag in population.class_image_ratio
    )
    count_gap = abs(stats.mean_defect_count - population.mean_defect_count)
    if population.mean_defect_count > 0:
        count_gap = count_gap / population.mean_defect_count * 100
    return primary, max(class_gap, count_gap)


def _windows(rows: list[dict[str, Any]], length: int) -> Iterable[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: row["original_index"])
    for start in range(len(ordered) - length + 1):
        yield ordered[start:start + length]


def _gaps(window: list[dict[str, Any]]) -> int:
    return sum(max(0, b["original_index"] - a["original_index"] - 1) for a, b in zip(window, window[1:]))


def _best_window(
    rows: list[dict[str, Any]],
    length: int,
    *,
    defective: bool | None,
    modality: str,
    population: Stats | None = None,
) -> list[dict[str, Any]] | None:
    """고정 길이 연속 구간 중 하나를 결정론적으로 고른다.

    population 이 주어지면 계획서 4.5 의 4 항대로 목적함수를 먼저 최소화한다. 주어지지
    않으면 4.5 의 3 항대로 index gap 합계를 먼저 최소화하고 시작 index 로 동률을 깬다.
    """
    flags = DEFECT_FLAGS[modality]
    best: tuple[tuple[float, ...], list[dict[str, Any]]] | None = None
    for window in _windows(rows, length):
        has_defect = any(any(row[flag] for flag in flags) for row in window)
        if defective is not None and has_defect != defective:
            continue
        start = window[0]["original_index"]
        if population is None:
            key: tuple[float, ...] = (float(_gaps(window)), float(start))
        else:
            key = (*_objective(window, population, modality), float(_gaps(window)), float(start))
        if best is None or key < best[0]:
            best = (key, window)
    return None if best is None else best[1]


@dataclass
class Selection:
    battery_id: int
    product_status: str
    window: list[dict[str, Any]]
    primary_objective: float
    secondary_objective: float
    searched: int = 0
    rejection: str = ""
    reserve: list[tuple[str, list[int]]] = field(default_factory=list)


def _ct_window(
    axes: dict[str, list[dict[str, Any]]],
    *,
    defective: bool,
    population: Stats | None,
) -> list[dict[str, Any]] | None:
    parts: list[dict[str, Any]] = []
    for axis, count in CT_COUNTS.items():
        pool = axes[axis] if defective else [row for row in axes[axis] if not row["has_porosity"]]
        part = _best_window(
            pool,
            count,
            defective=True if defective else False,
            modality="CT",
            population=population if defective else None,
        )
        if part is None:
            return None
        parts.extend(part)
    return parts


def _select(rows: list[dict[str, Any]], modality: str) -> tuple[list[Selection], Stats]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["modality"] != modality:
            continue
        if modality == "CT" and row["porosity_bbox_max_ratio"] >= CT_POROSITY_LIMIT:
            continue
        grouped[row["battery_id"]].append(row)

    population = _stats([row for values in grouped.values() for row in values], modality)
    normal: list[Selection] = []
    defective: list[Selection] = []
    for battery_id, battery_rows in sorted(grouped.items()):
        if modality == "RGB":
            normal_window = _best_window(battery_rows, RGB_COUNT, defective=False, modality=modality)
            defect_window = _best_window(
                battery_rows, RGB_COUNT, defective=True, modality=modality, population=population
            )
        else:
            axes = {axis: [row for row in battery_rows if row["axis"] == axis] for axis in CT_COUNTS}
            normal_window = _ct_window(axes, defective=False, population=None)
            defect_window = _ct_window(axes, defective=True, population=population)
        if normal_window is not None:
            normal.append(Selection(battery_id, "normal", normal_window, 0.0, 0.0))
        if defect_window is not None:
            primary, secondary = _objective(defect_window, population, modality)
            defective.append(Selection(battery_id, "defective", defect_window, primary, secondary))

    # 계획서 4.5 의 2 항: 두 후보 집합에 동시에 속하는 ID 가 있으면 정상 후보 여유를
    # 보존하기 위해 제품불량 전용 후보를 먼저 쓴다.
    normal_ids = {item.battery_id for item in normal}
    exclusive = [item for item in defective if item.battery_id not in normal_ids]
    candidates = exclusive or defective
    candidates.sort(
        key=lambda item: (
            item.primary_objective,
            item.secondary_objective,
            stable_seed(GLOBAL_SEED, modality, item.battery_id),
        )
    )
    if not candidates:
        raise ValueError(
            f"{modality}: 계획서 4.5 의 제품불량 ID 후보가 없다. 반올림하지 않고 중단한다"
        )
    chosen_defect = candidates[0]
    chosen_normal = [item for item in normal if item.battery_id != chosen_defect.battery_id][:19]
    if len(chosen_normal) != 19:
        raise ValueError(
            f"{modality}: 계획서 4.5 는 제품정상 ID 19 개를 요구하는데 {len(chosen_normal)} 개뿐이다"
        )
    for item in chosen_normal:
        item.primary_objective, item.secondary_objective = _objective(item.window, population, modality)
    return [chosen_defect] + chosen_normal, population


def _normal_assignment(modality: str, slot: int, group_seed: int) -> tuple[list[str], dict[str, float]]:
    rng = random.Random(stable_seed(group_seed, "normal"))
    names = NORMAL_AUGMENTATIONS[modality]
    count = 2 if slot % 5 == 4 else 1
    selected = [names[(slot + offset) % len(names)] for offset in range(count)]
    params = {"brightness": rng.uniform(0.92, 1.08), "contrast": rng.uniform(0.92, 1.08), "gamma": rng.uniform(0.92, 1.08), "noise_sigma": rng.uniform(0.002, 0.008)}
    return selected, params


def _reserve_segments(
    window: list[dict[str, Any]],
    modality: str,
    axis: str,
    primary_slots: list[int],
    length: int,
) -> list[tuple[str, list[int]]]:
    """계획서 7.5 의 reserve 구간을 우선순위대로 미리 고정한다.

    1. 같은 ID·같은 축의 reserve 구간
    2. CT 는 같은 축이 모두 실패할 때 다른 축의 reserve 구간

    다른 적격 FAIL ID 로 넘어가는 4 순위는 두 ID 의 구간을 모두 알아야 하므로
    build_plan 에서 덧붙인다.
    """
    primary = set(primary_slots)
    same_axis = [index for index, row in enumerate(window) if modality != "CT" or row["axis"] == axis]
    alternates: list[tuple[str, list[int]]] = []
    for start in range(0, len(same_axis) - length + 1, length):
        slots = same_axis[start:start + length]
        if primary & set(slots):
            continue
        alternates.append(("same-axis", slots))
        if len(alternates) == 2:
            break
    if modality == "CT":
        for other in CT_COUNTS:
            if other == axis:
                continue
            pool = [index for index, row in enumerate(window) if row["axis"] == other]
            if len(pool) >= length:
                alternates.append(("other-axis", pool[:length]))
    return alternates


def _candidate_record(rank: int, reason: str, row: dict[str, Any]) -> dict[str, Any]:
    """reserve 후보 하나를 기록한다.

    생성 단계가 이 후보로 실제 교체하려면 경로와 원본 해시가 있어야 한다. 식별자만
    남기면 파일을 다시 찾아야 하고, 그러면 plan 에 고정한 의미가 없어진다.
    """
    return {
        "rank": rank,
        "reason": reason,
        "source_split": row["source_split"],
        "original_battery_id": row["battery_id"],
        "original_index": row["original_index"],
        "original_stem": row["original_stem"],
        "orig_image_relative_path": row["image_relative_path"],
        "orig_json_relative_path": row["json_relative_path"],
        "source_image_sha256": row["image_sha256"],
        "source_json_sha256": row["json_sha256"],
    }


def _sequence_metadata(
    window: list[dict[str, Any]], modality: str, reverse: bool
) -> dict[int, dict[str, int]]:
    """축 단위 순서와 index 누락을 계산한다.

    계획서 6.2 는 슬라이스 순서 역전이 있는 축의 실제 출력 순서를
    output_sequence_order 로 별도 기록하라고 규정한다. 13.4 는 숫자 index 의 누락을
    허용하되 기록하라고 규정한다.
    """
    by_axis: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(window):
        by_axis[row["axis"] if modality == "CT" else ""].append(position)
    metadata: dict[int, dict[str, int]] = {}
    for positions in by_axis.values():
        ordered = sorted(positions, key=lambda position: window[position]["original_index"])
        for order, position in enumerate(ordered):
            previous = window[ordered[order - 1]]["original_index"] if order else None
            current = window[position]["original_index"]
            gap = 0 if previous is None else max(0, current - previous - 1)
            metadata[position] = {
                "source_sequence_order": order,
                "output_sequence_order": len(ordered) - 1 - order if reverse else order,
                "index_gap_before": int(bool(gap)),
                "index_gap_size": gap,
            }
    return metadata


def _configuration(seed: int) -> dict[str, Any]:
    return {
        "global_seed": seed,
        "ct_counts": CT_COUNTS,
        "rgb_count": RGB_COUNT,
        "ct_porosity_limit": CT_POROSITY_LIMIT,
        "normal_augmentations": {key: list(value) for key, value in NORMAL_AUGMENTATIONS.items()},
        "failure_cases": {key: list(value) for key, value in FAILURE_CASES.items()},
        "search_algorithm": SEARCH_ALGORITHM,
    }


def build_plan(cache_path: Path, output_dir: Path, seed: int = GLOBAL_SEED) -> dict[str, Any]:
    rows = _rows(cache_path)
    selections: dict[str, list[Selection]] = {}
    populations: dict[str, Stats] = {}
    for modality in ("CT", "RGB"):
        selections[modality], populations[modality] = _select(rows, modality)

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "generation_plan.csv"
    selected_path = output_dir / "selected_ids.csv"
    settings = _configuration(seed)
    settings_hash = config_hash(settings)
    selected_records: list[dict[str, Any]] = []
    plan_records: list[dict[str, Any]] = []
    sample_counter = 0

    for modality in ("CT", "RGB"):
        base = 1_900_000_000 if modality == "CT" else 2_900_000_000
        chosen = selections[modality]
        population = populations[modality]
        fail_ids = {
            item.battery_id
            for item in sorted(chosen, key=lambda item: stable_seed(seed, modality, item.battery_id))[:2]
        }
        windows_by_id = {item.battery_id: item.window for item in chosen}
        fail_layout: dict[int, dict[str, Any]] = {}
        for item in chosen:
            if item.battery_id not in fail_ids:
                continue
            rng = random.Random(stable_seed(seed, modality, item.battery_id, "failure-window"))
            axis = rng.choice(list(CT_COUNTS)) if modality == "CT" else ""
            eligible = [
                index for index, row in enumerate(item.window)
                if modality != "CT" or row["axis"] == axis
            ]
            length = rng.randint(10, 25)
            start = rng.randint(0, len(eligible) - length)
            slots = eligible[start:start + length]
            k = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            cases = rng.sample(FAILURE_CASES[modality], k)
            fail_layout[item.battery_id] = {
                "axis": axis,
                "start": start,
                "length": length,
                "slots": slots,
                "cases": cases,
                "k": k,
                "reserve": _reserve_segments(item.window, modality, axis, slots, length),
            }

        for rank, item in enumerate(chosen, 1):
            output_battery_id = base + rank
            augmentation_slot = rank - 1
            probe_seed = stable_seed(seed, modality, item.battery_id, "normal-base")
            probe_names, _ = _normal_assignment(modality, augmentation_slot, probe_seed)
            # 계획서 6.2: 슬라이스 순서 역전 여부는 ID 단위로 한 번만 결정한다.
            reverse = modality == "CT" and "synchronized_flip" in probe_names and bool(probe_seed & 4)
            sequence = _sequence_metadata(item.window, modality, reverse)
            layout = fail_layout.get(item.battery_id)
            selected_records.append({
                "modality": modality,
                "rank": rank,
                "original_battery_id": item.battery_id,
                "output_battery_id": output_battery_id,
                "product_status": item.product_status,
                "fail_target": item.battery_id in fail_ids,
                "source_count": len(item.window),
                "search_algorithm": SEARCH_ALGORITHM,
                "search_seed": seed,
                "search_iterations": max(1, len(item.window)),
                "search_stop_condition": SEARCH_STOP_CONDITION,
                "primary_objective": round(item.primary_objective, 8),
                "secondary_objective": round(item.secondary_objective, 8),
                "population_defect_image_ratio": round(population.defect_image_ratio, 8),
                "population_mean_defect_count": round(population.mean_defect_count, 8),
                "slice_order_reversed": reverse,
                "rejection_reason": item.rejection,
            })

            initial_for_recapture: list[dict[str, Any]] = []
            for slot, row in enumerate(item.window):
                sample_counter += 1
                group_seed = stable_seed(seed, modality, item.battery_id, row["axis"], row["original_index"])
                normal_seed = probe_seed if modality == "CT" else group_seed
                assignment_slot = augmentation_slot if modality == "CT" else slot
                augmentations, parameters = _normal_assignment(modality, assignment_slot, normal_seed)
                failure_case = ""
                failure_segment = ""
                reserve_candidates: list[dict[str, Any]] = []
                if layout is not None and slot in layout["slots"]:
                    offset = layout["slots"].index(slot)
                    case_index = min(layout["k"] - 1, offset * layout["k"] // layout["length"])
                    failure_case = layout["cases"][case_index]
                    failure_segment = f"{modality}-{item.battery_id}-{layout['start']}-{layout['length']}"
                    for order, (reason, slots) in enumerate(layout["reserve"], 1):
                        reserve_candidates.append(
                            _candidate_record(order, reason, item.window[slots[offset]])
                        )
                    other = next(
                        (other_id for other_id in fail_ids if other_id != item.battery_id), None
                    )
                    if other is not None:
                        # 계획서 7.5: 다른 FAIL ID 로 넘어가도 이 ID 의 L 은 유지한다.
                        # 상대 ID 의 L 은 다를 수 있으므로 상대 구간이 아니라 상대 축의
                        # pool 에서 같은 길이만큼 가져온다.
                        other_window = windows_by_id[other]
                        other_axis = fail_layout[other]["axis"]
                        pool = [
                            index for index, candidate in enumerate(other_window)
                            if modality != "CT" or candidate["axis"] == other_axis
                        ]
                        if len(pool) >= layout["length"]:
                            reserve_candidates.append(
                                _candidate_record(
                                    len(reserve_candidates) + 1,
                                    "other-fail-id",
                                    other_window[pool[offset]],
                                )
                            )
                record = {
                    "sample_id": f"S{sample_counter:08d}",
                    "synthetic_id": output_stem("initial_capture", modality, output_battery_id, row["axis"], row["original_index"]),
                    "capture_set": "initial_capture",
                    "capture_group_id": f"G{stable_seed(modality, item.battery_id, row['axis'], row['original_index']):016x}",
                    "retry_of_sample_id": "",
                    "modality": modality,
                    "original_battery_id": item.battery_id,
                    "output_battery_id": output_battery_id,
                    "product_status": item.product_status,
                    "axis": row["axis"],
                    "original_index": row["original_index"],
                    "source_sequence_order": sequence[slot]["source_sequence_order"],
                    "output_sequence_order": sequence[slot]["output_sequence_order"],
                    "source_split": row["source_split"],
                    "index_gap_before": sequence[slot]["index_gap_before"],
                    "index_gap_size": sequence[slot]["index_gap_size"],
                    "original_stem": row["original_stem"],
                    "orig_image_relative_path": row["image_relative_path"],
                    "orig_json_relative_path": row["json_relative_path"],
                    "original_image_id": row["original_image_id"],
                    "original_image_file_name": Path(row["image_relative_path"]).name,
                    "original_roi": row["original_roi_json"],
                    "source_image_sha256": row["image_sha256"],
                    "source_json_sha256": row["json_sha256"],
                    "original_is_normal": row["original_is_normal"],
                    "has_porosity": row["has_porosity"],
                    "has_damaged": row["has_damaged"],
                    "has_pollution": row["has_pollution"],
                    "original_defect_count": row["defect_count"],
                    "capture_quality": "FAIL" if failure_case else "PASS",
                    "failure_case": failure_case,
                    "failure_segment_id": failure_segment,
                    "base_augmentation_names": json.dumps(augmentations),
                    "normal_augmentation_parameters": json.dumps(parameters, sort_keys=True),
                    "normal_augmentation_seed": normal_seed,
                    "slice_seed": stable_seed(normal_seed, row["axis"], row["original_index"], "|".join(augmentations)),
                    "failure_window_start": layout["start"] if failure_case else "",
                    "failure_window_end": layout["start"] + layout["length"] - 1 if failure_case else "",
                    "reserve_candidates": json.dumps(reserve_candidates, ensure_ascii=False) if reserve_candidates else "",
                    "global_seed": seed,
                    "item_seed": stable_seed(group_seed, failure_case or "PASS"),
                    "config_hash": settings_hash,
                }
                plan_records.append(record)
                initial_for_recapture.append(record)

            if item.battery_id in fail_ids:
                for initial in initial_for_recapture:
                    sample_counter += 1
                    retry = dict(initial)
                    retry.update({
                        "sample_id": f"S{sample_counter:08d}",
                        "synthetic_id": output_stem("recapture", modality, output_battery_id, retry["axis"], retry["original_index"]),
                        "retry_of_sample_id": initial["sample_id"],
                        "capture_set": "recapture",
                        "capture_quality": "PASS",
                        "failure_case": "",
                        "failure_segment_id": "",
                        "failure_window_start": "",
                        "failure_window_end": "",
                        "reserve_candidates": "",
                    })
                    plan_records.append(retry)

    expected = {"initial_capture": 34000, "recapture": 3400}
    actual = {name: sum(row["capture_set"] == name for row in plan_records) for name in expected}
    if actual != expected:
        raise AssertionError(f"Plan quantity mismatch: {actual} != {expected}")
    ct_ids = {row["output_battery_id"] for row in selected_records if row["modality"] == "CT"}
    rgb_ids = {row["output_battery_id"] for row in selected_records if row["modality"] == "RGB"}
    if ct_ids & rgb_ids:
        raise AssertionError("CT/RGB output battery IDs overlap")

    for record in plan_records:
        for column in MANIFEST_COLUMNS:
            record.setdefault(column, "")
        for column in GENERATION_COLUMNS:
            record[column] = ""

    with plan_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(plan_records)
    with selected_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_records[0]))
        writer.writeheader()
        writer.writerows(selected_records)

    summary = {
        "initial_capture": 34000,
        "recapture": 3400,
        "total": 37400,
        "CT_selected_ids": 20,
        "RGB_selected_ids": 20,
        "seed": seed,
        "config_hash": settings_hash,
        "planner_version": __version__,
        "label_source": "extracted-json-only",
    }
    atomic_json(output_dir / "plan_summary.json", summary)
    LOGGER.info("Generation plan complete: %s", plan_path)
    return summary
