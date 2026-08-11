from __future__ import annotations

import csv
import json
import logging
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .schema import (
    CT_AXIS_COORDINATES,
    CT_COUNTS,
    GENERATION_COLUMNS,
    MANIFEST_COLUMNS,
    RGB_COUNT,
    ct_axis_transform,
    output_stem,
)
from .util import atomic_json, config_hash, stable_seed


LOGGER = logging.getLogger(__name__)
GLOBAL_SEED = 20260723
CT_POROSITY_LIMIT = 0.25
SELECTED_IDS = 20
DEFECTIVE_ID_COUNTS = {"CT": 1, "RGB": 2}
NORMAL_AUGMENTATIONS = {
    "CT": ("brightness_contrast_gamma", "partial_histogram_blend", "normal_noise_poisson", "low_frequency_shading", "percentile_tone_curve", "weak_reconstruction_kernel", "synchronized_flip"),
    "RGB": ("safe_translate_rotate", "brightness_contrast_gamma", "rgb_channel_gain_tone", "low_frequency_lighting", "poisson_noise", "weak_reconstruction"),
}
FAILURE_CASES = {
    "CT": ("ct_cell_alignment_failure", "ct_acquisition_motion", "ct_insufficient_projection_sampling", "ct_low_signal_noise", "ct_beam_hardening_metal_streak"),
    "RGB": ("rgb_trigger_timing_failure", "rgb_uneven_lighting", "rgb_reflection_glare", "rgb_focus_failure", "rgb_underexposure", "rgb_overexposure", "rgb_surface_dust", "rgb_hair_contamination"),
}
DEFECT_FLAGS = {"CT": ("has_porosity",), "RGB": ("has_damaged", "has_pollution")}

# 계획서 v1.5 §4.3: RGB는 정확히 두 제품불량 ID를 선택하고, 그 ID의 결함 이미지 안에서
# 클래스 구성 목표에 가장 가까운 구간 조합을 찾는다.
# Product status is fixed by ID count.  This objective applies only to the
# defect-image classes inside the two selected defective RGB IDs.
# 계획서 7.3 의 FAIL 구간 길이 L.
FAIL_LENGTH_RANGE = (10, 25)

SEARCH_ALGORITHM = "deterministic-fixed-id-v1.5"
SEARCH_STOP_CONDITION = "deterministic label-carrier selection and contiguous-window validation"


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
    """계획서 4.4 의 기준을 계산하기 위한 집계값.

    conditional_class_ratio 는 결함 이미지만을 분모로 한 클래스 구성이다. RGB 는 결함이
    배터리 단위라 불량 ID 의 사진은 사실상 전부 결함을 담는다. 그래서 세트 전체의 결함
    이미지 비율은 20 개 중 몇 개를 불량으로 뽑느냐로 정해지고 원본 비율과 비교할 수 없다.
    비교 가능한 것은 결함 이미지 안에서의 클래스 구성이다.
    """

    defect_image_ratio: float
    class_image_ratio: dict[str, float]
    conditional_class_ratio: dict[str, float]
    mean_defect_count: float
    conditional_defect_count: float


def _stats(rows: list[dict[str, Any]], modality: str) -> Stats:
    total = max(1, len(rows))
    flags = DEFECT_FLAGS[modality]
    defective = sum(any(row[flag] for flag in flags) for row in rows)
    defect_total = sum(int(row["defect_count"]) for row in rows)
    denominator = max(1, defective)
    return Stats(
        defect_image_ratio=defective / total,
        class_image_ratio={flag: sum(bool(row[flag]) for row in rows) / total for flag in flags},
        conditional_class_ratio={flag: sum(bool(row[flag]) for row in rows) / denominator for flag in flags},
        mean_defect_count=defect_total / total,
        conditional_defect_count=defect_total / denominator,
    )


def _objective(window: list[dict[str, Any]], population: Stats, modality: str) -> tuple[float, float, float]:
    """계획서 4.4 의 목적값을 %p 단위로 돌려준다. 작을수록 모집단에 가깝다.

    1차는 정상 대 결함 이미지 비율의 절대 차이다. 2차는 계획서 본문의 순서를 그대로
    따라 클래스별 이미지 비율 차이를 먼저 보고, 그 다음에 이미지당 annotation 수 차이를
    본다. 두 값을 하나로 합치면 안 된다. 실제 원본에서 annotation 수 항이 클래스 비율 항을
    가려서, `Damaged` 가 하나도 없는 ID 가 `Damaged` 를 가진 동률 후보를 이기는 일이
    생겼다.

    v1.2 는 후보 비율을 20 으로 나눈 뒤 모집단과 비교했다. 그 스케일에는 근거가 없고
    정렬을 사실상 무작위로 만들었다.
    """
    stats = _stats(window, modality)
    if modality == "RGB":
        primary = max(
            abs(stats.conditional_class_ratio[flag] - population.conditional_class_ratio[flag]) * 100
            for flag in population.conditional_class_ratio
        )
        count_gap = abs(stats.conditional_defect_count - population.conditional_defect_count)
        if population.conditional_defect_count > 0:
            count_gap = count_gap / population.conditional_defect_count * 100
        # 3 순위는 불량 ID 가 얼마나 온전히 불량인지다. 원본에서 불량 배터리는 거의 모든
        # 사진에 결함이 보이므로 결함 비율이 1 에 가까운 구간을 선호한다.
        return primary, count_gap, (1.0 - stats.defect_image_ratio) * 100
    primary = abs(stats.defect_image_ratio - population.defect_image_ratio) * 100
    class_gap = max(
        abs(stats.class_image_ratio[flag] - population.class_image_ratio[flag]) * 100
        for flag in population.class_image_ratio
    )
    count_gap = abs(stats.mean_defect_count - population.mean_defect_count)
    if population.mean_defect_count > 0:
        count_gap = count_gap / population.mean_defect_count * 100
    return primary, class_gap, count_gap


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

    후보마다 구간을 다시 순회하면 구간 길이에 비례해 느려진다. RGB 는 적격 ID 가 700 개를
    넘고 구간 길이가 250 이라 그 방식으로는 plan 생성만 수십 분이 걸린다. 누적합을 미리
    만들어 후보 하나를 상수 시간에 평가한다.
    """
    ordered = sorted(rows, key=lambda row: row["original_index"])
    total = len(ordered)
    if total < length:
        return None
    flags = DEFECT_FLAGS[modality]
    defect_images = [0] * (total + 1)
    per_flag = {flag: [0] * (total + 1) for flag in flags}
    defect_counts = [0] * (total + 1)
    for position, row in enumerate(ordered):
        defect_images[position + 1] = defect_images[position] + (1 if any(row[flag] for flag in flags) else 0)
        for flag in flags:
            per_flag[flag][position + 1] = per_flag[flag][position] + (1 if row[flag] else 0)
        defect_counts[position + 1] = defect_counts[position] + int(row["defect_count"])

    best: tuple[tuple[float, ...], int] | None = None
    for start in range(total - length + 1):
        end = start + length
        defects_here = defect_images[end] - defect_images[start]
        if defective is not None and (defects_here > 0) != defective:
            continue
        first = ordered[start]["original_index"]
        # 연속 구간의 gap 합계는 양 끝 index 차이에서 구간 길이를 빼면 나온다.
        gaps = ordered[end - 1]["original_index"] - first - (length - 1)
        if population is None:
            key: tuple[float, ...] = (float(gaps), float(first))
        else:
            if modality == "RGB":
                denominator = max(1, defects_here)
                primary = max(
                    abs((per_flag[flag][end] - per_flag[flag][start]) / denominator - population.conditional_class_ratio[flag]) * 100
                    for flag in flags
                )
                count_gap = abs((defect_counts[end] - defect_counts[start]) / denominator - population.conditional_defect_count)
                if population.conditional_defect_count > 0:
                    count_gap = count_gap / population.conditional_defect_count * 100
                key = (primary, count_gap, (1.0 - defects_here / length) * 100, float(gaps), float(first))
            else:
                primary = abs(defects_here / length - population.defect_image_ratio) * 100
                class_gap = max(
                    abs((per_flag[flag][end] - per_flag[flag][start]) / length - population.class_image_ratio[flag]) * 100
                    for flag in flags
                )
                count_gap = abs((defect_counts[end] - defect_counts[start]) / length - population.mean_defect_count)
                if population.mean_defect_count > 0:
                    count_gap = count_gap / population.mean_defect_count * 100
                key = (primary, class_gap, count_gap, float(gaps), float(first))
        if best is None or key < best[0]:
            best = (key, start)
    return None if best is None else ordered[best[1]:best[1] + length]


@dataclass
class Selection:
    battery_id: int
    product_status: str
    window: list[dict[str, Any]]
    primary_objective: float
    secondary_objective: float
    count_objective: float = 0.0
    stratum: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    searched: int = 0
    rejection: str = ""
    reserve: list[tuple[str, list[int]]] = field(default_factory=list)
    source_reuse_ordinal: int = 1
    defect_role: str = ""

    @property
    def instance_key(self) -> tuple[int, int]:
        """Identity of one output-producing use of a source battery."""
        return self.battery_id, self.source_reuse_ordinal


def _ct_window(
    axes: dict[str, list[dict[str, Any]]],
    *,
    defective: bool,
    population: Stats | None,
) -> list[dict[str, Any]] | None:
    """CT 한 ID 의 x/y/z 구간을 고른다.

    제품정상 후보는 계획서 4.5 의 3 항대로 porosity 를 제거한 pool 안에서 index gap 이
    가장 작은 연속 구간을 고른다.

    제품불량 후보는 축마다 결함 구간을 요구하지 않는다. 계획서 4.5 의 판정 기준은
    "선택된 x/y/z 구간에 porosity annotation 이 하나 이상"이며 축별 조건이 아니다. 실제
    원본에서 porosity 는 한 축에 몰려 있어서, 축마다 결함을 요구하면 적격 ID 가 0 개가
    된다. 대신 축별 구간은 4.5 의 4 항대로 목적함수로 고르고, 그렇게 고른 결과에 porosity
    가 하나도 없을 때만 결함 구간이 있는 축 하나를 바꾼다.
    """
    parts: list[dict[str, Any]] = []
    for axis, count in CT_COUNTS.items():
        if defective:
            part = _best_window(axes[axis], count, defective=None, modality="CT", population=population)
        else:
            pool = [row for row in axes[axis] if not row["has_porosity"]]
            part = _best_window(pool, count, defective=False, modality="CT", population=None)
        if part is None:
            return None
        parts.extend(part)
    if not defective:
        return parts
    if any(row["has_porosity"] for row in parts):
        return parts
    for axis, count in CT_COUNTS.items():
        replacement = _best_window(axes[axis], count, defective=True, modality="CT", population=population)
        if replacement is not None:
            return [row for row in parts if row["axis"] != axis] + replacement
    return None


# 계획서 4.5 의 층 경계. CT 는 전처리 v4.1 의 positive_rate_bin 을 그대로 쓴다. 같은
# 원본을 두 파이프라인이 서로 다르게 나누면 비교가 불가능해지기 때문이다.
CT_POSITIVE_RATE_BINS = (
    ("zero", 0.0, 1e-9),
    ("very_low", 1e-9, 0.05),
    ("low_mid", 0.05, 0.30),
    ("mid_high", 0.30, 0.70),
    ("very_high", 0.70, 1.01),
)
DEFECT_FREE_STRATA = {"CT": "zero", "RGB": "clean"}


def _max_defect_run(item: "Selection", modality: str) -> int:
    """구간 안에서 결함 프레임이 가장 많은 축의 결함 장수.

    FAIL 구간은 한 축 안에서 연속으로 잡히므로, 교차 칸을 채우려면 그 축에 결함이
    충분히 있어야 한다.
    """
    flags = DEFECT_FLAGS[modality]
    counts: Counter = Counter()
    for row in item.window:
        if any(row[flag] for flag in flags):
            counts[row["axis"] if modality == "CT" else ""] += 1
    return max(counts.values(), default=0)


def _stratum(rows: list[dict[str, Any]], modality: str) -> str:
    """ID 하나를 원본 통계로 층에 배정한다.

    CT 는 porosity 양성률 구간, RGB 는 보유 클래스 조합이다. RGB 의 결함은 배터리 단위라
    양성률이 0 아니면 1 에 가까워서 양성률 구간이 의미가 없다.
    """
    if modality == "CT":
        rate = sum(1 for row in rows if row["has_porosity"]) / max(1, len(rows))
        for name, low, high in CT_POSITIVE_RATE_BINS:
            if low <= rate < high:
                return name
        return CT_POSITIVE_RATE_BINS[-1][0]
    damaged = any(row["has_damaged"] for row in rows)
    pollution = any(row["has_pollution"] for row in rows)
    if damaged and pollution:
        return "both"
    if damaged:
        return "damaged_only"
    if pollution:
        return "pollution_only"
    return "clean"


def _first_ordered_labeled_window(
    items: list[dict[str, Any]], flag: str, length: int = RGB_COUNT
) -> list[dict[str, Any]] | None:
    """Return the earliest ordered window carrying the requested label."""
    ordered = sorted(items, key=lambda row: row["original_index"])
    for start in range(len(ordered) - length + 1):
        window = ordered[start:start + length]
        if any(row[flag] for row in window):
            return window
    return None


def _select_rgb(
    grouped: dict[int, list[dict[str, Any]]], population: Stats, seed: int
) -> list[Selection]:
    """Select distinct Pollution and Damaged carrier IDs without ratio optimization."""
    role_candidates: dict[str, list[tuple[int, list[dict[str, Any]]]]] = {}
    for role, flag in (("pollution", "has_pollution"), ("damaged", "has_damaged")):
        candidates = []
        for battery_id, items in sorted(grouped.items()):
            window = _first_ordered_labeled_window(items, flag)
            if window is not None:
                candidates.append((battery_id, window))
        role_candidates[role] = candidates

    pairs = [
        (pollution, damaged)
        for pollution in role_candidates["pollution"]
        for damaged in role_candidates["damaged"]
        if pollution[0] != damaged[0]
    ]
    if not pairs:
        raise ValueError("RGB: distinct Pollution and Damaged carrier IDs are required")
    pollution, damaged = min(
        pairs,
        key=lambda pair: stable_seed(seed, "RGB", "label-carriers", pair[0][0], pair[1][0]),
    )

    selections: list[Selection] = []
    used: set[int] = set()
    for role, (battery_id, window) in (("pollution", pollution), ("damaged", damaged)):
        primary, secondary, counted = _objective(window, population, "RGB")
        selections.append(
            Selection(
                battery_id, "defective", window, primary, secondary, counted,
                stratum=_stratum(grouped[battery_id], "RGB"), defect_role=role,
            )
        )
        used.add(battery_id)

    normal_needed = SELECTED_IDS - len(selections)
    normal: list[Selection] = []
    for battery_id, items in sorted(grouped.items()):
        if battery_id in used:
            continue
        window = _best_window(items, RGB_COUNT, defective=False, modality="RGB", population=None)
        if window is None:
            continue
        normal.append(Selection(battery_id, "normal", window, 0.0, 0.0, 0.0, stratum="clean"))
    # 계획서 4.5 의 6 항: 동일 목적값에서는 전역 seed 로 결정론적으로 선택한다.
    normal.sort(key=lambda item: stable_seed(seed, "RGB", "normal", item.battery_id))
    if len(normal) < normal_needed:
        raise ValueError(
            f"RGB: 무결함 ID 가 {len(normal)} 개뿐이라 {normal_needed} 개를 채울 수 없다. 반올림하지 않고 중단한다"
        )
    selections.extend(normal[:normal_needed])
    selections.sort(key=lambda item: (item.product_status != "defective", item.stratum, item.battery_id))
    LOGGER.info(
        "RGB label-carrier selection: Pollution ID %d, Damaged ID %d",
        pollution[0], damaged[0],
    )
    return selections


def _ordered_distinct_windows(
    items: list[dict[str, Any]], length: int, limit: int
) -> list[list[dict[str, Any]]]:
    """Choose distinct, spread-out windows without requiring adjacent indexes.

    Missing raw indexes are valid and are recorded later as sequence gaps.  A
    reused source may share some rows when it does not contain enough data for
    disjoint windows, but the complete selected index tuple is never repeated.
    """
    ordered = sorted(items, key=lambda row: row["original_index"])
    maximum_start = len(ordered) - length
    if maximum_start < 0 or limit < 1:
        return []
    available = list(range(maximum_start + 1))
    starts: list[int] = []
    while available and len(starts) < limit:
        start = max(
            available,
            key=lambda candidate: (
                min((abs(candidate - chosen) for chosen in starts), default=maximum_start + 1),
                -candidate,
            ),
        )
        starts.append(start)
        available.remove(start)
    return [ordered[start:start + length] for start in starts]


def _select(
    rows: list[dict[str, Any]], modality: str, seed: int = GLOBAL_SEED
) -> tuple[list[Selection], Stats]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["modality"] != modality:
            continue
        if modality == "CT" and row["porosity_bbox_max_ratio"] >= CT_POROSITY_LIMIT:
            continue
        grouped[row["battery_id"]].append(row)

    population = _stats([row for values in grouped.values() for row in values], modality)
    free = DEFECT_FREE_STRATA[modality]

    if modality == "RGB":
        return _select_rgb(grouped, population, seed), population

    # 층별 모집단을 따로 잡는다. 층화를 해 놓고 목적함수만 전체 모집단과 비교하면 각 층이
    # 전체 평균을 흉내내다가 층 사이 희석이 생긴다.
    layers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    strata_of: dict[int, str] = {}
    for battery_id, battery_rows in grouped.items():
        stratum = _stratum(battery_rows, modality)
        strata_of[battery_id] = stratum
        layers[stratum].extend(battery_rows)
    layer_population = {name: _stats(items, modality) for name, items in layers.items()}
    candidates: dict[str, list[Selection]] = defaultdict(list)
    for battery_id, battery_rows in sorted(grouped.items()):
        stratum = strata_of[battery_id]
        reference = layer_population[stratum]
        axes = {axis: [row for row in battery_rows if row["axis"] == axis] for axis in CT_COUNTS}
        window = _ct_window(axes, defective=stratum != free, population=reference)
        if window is None:
            continue
        primary, secondary, counted = _objective(window, reference, modality)
        product_status = "defective" if any(
            row[flag] for row in window for flag in DEFECT_FLAGS[modality]
        ) else "normal"
        candidates[stratum].append(
            Selection(battery_id, product_status, window, primary, secondary, counted, stratum=stratum)
        )

    if not candidates:
        raise ValueError(f"{modality}: 적격 ID 가 없다")
    normal = list(candidates.get(free, []))
    defective = [
        item for stratum, items in candidates.items() if stratum != free for item in items
    ]
    sort_key = lambda item: (
        item.primary_objective,
        item.secondary_objective,
        item.count_objective,
        stable_seed(seed, modality, item.battery_id),
    )
    normal.sort(key=sort_key)
    defective.sort(key=sort_key)
    defective_needed = DEFECTIVE_ID_COUNTS[modality]
    normal_needed = SELECTED_IDS - defective_needed
    if len(defective) < defective_needed or not normal:
        raise ValueError(
            f"{modality}: fixed ID ratio requires {defective_needed} defective and "
            f"{normal_needed} normal output IDs (normal sources may be reused); eligible counts are "
            f"{len(defective)} defective and {len(normal)} normal"
        )
    windows_by_source: dict[int, dict[str, list[list[dict[str, Any]]]]] = {}
    for source in normal:
        source_rows = grouped[source.battery_id]
        windows_by_source[source.battery_id] = {
            axis: _ordered_distinct_windows(
                [
                    row for row in source_rows
                    if row["axis"] == axis and not row["has_porosity"]
                ],
                length,
                normal_needed,
            )
            for axis, length in CT_COUNTS.items()
        }

    reuse_counts: Counter[int] = Counter()
    repeated_normal: list[Selection] = []
    while len(repeated_normal) < normal_needed:
        progress = False
        for source in normal:
            reuse_ordinal = reuse_counts[source.battery_id] + 1
            axis_windows = windows_by_source[source.battery_id]
            if any(len(axis_windows[axis]) < reuse_ordinal for axis in CT_COUNTS):
                continue
            reused_window = [
                row
                for axis in CT_COUNTS
                for row in axis_windows[axis][reuse_ordinal - 1]
            ]
            reuse_counts[source.battery_id] = reuse_ordinal
            repeated_normal.append(
                replace(source, window=reused_window, source_reuse_ordinal=reuse_ordinal)
            )
            progress = True
            if len(repeated_normal) == normal_needed:
                break
        if not progress:
            total_capacity = sum(
                min(len(axis_windows[axis]) for axis in CT_COUNTS)
                for axis_windows in windows_by_source.values()
            )
            raise ValueError(
                f"CT: {normal_needed} normal output IDs require distinct "
                f"ordered windows, but eligible source capacity is {total_capacity}"
            )
    chosen = defective[:defective_needed] + repeated_normal
    # 불량 ID 를 먼저 배치해 출력 ID 번호가 제품 상태와 무관하게 흩어지지 않도록 한다.
    chosen.sort(key=lambda item: (
        item.product_status != "defective", item.stratum,
        item.battery_id, item.source_reuse_ordinal,
    ))
    LOGGER.info(
        "%s fixed-status selection: %s (defective %d/%d)",
        modality,
        Counter(item.stratum for item in chosen),
        sum(1 for item in chosen if item.product_status == "defective"),
        len(chosen),
    )
    return chosen, population


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
    window: list[dict[str, Any]], modality: str, reversed_axes: set[str]
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
    for axis, positions in by_axis.items():
        ordered = sorted(positions, key=lambda position: window[position]["original_index"])
        for order, position in enumerate(ordered):
            previous = window[ordered[order - 1]]["original_index"] if order else None
            current = window[position]["original_index"]
            gap = 0 if previous is None else max(0, current - previous - 1)
            metadata[position] = {
                "source_sequence_order": order,
                "output_sequence_order": (
                    len(ordered) - 1 - order if axis in reversed_axes else order
                ),
                "index_gap_before": int(bool(gap)),
                "index_gap_size": gap,
            }
    return metadata


def _configuration(seed: int) -> dict[str, Any]:
    """결과를 바꾸는 설정 전체. 계획서 13.3 의 config_hash 재료다.

    ID 개수, 클래스 구성, CT 좌표계와 FAIL 구간 등 출력 의미를 바꾸는 값은 모두 해시에
    포함되어야 한다.
    """
    return {
        "global_seed": seed,
        "ct_counts": CT_COUNTS,
        "ct_axis_coordinates": CT_AXIS_COORDINATES,
        "rgb_count": RGB_COUNT,
        "selected_ids": SELECTED_IDS,
        "ct_porosity_limit": CT_POROSITY_LIMIT,
        "ct_positive_rate_bins": [[name, low, high] for name, low, high in CT_POSITIVE_RATE_BINS],
        "defective_id_counts": DEFECTIVE_ID_COUNTS,
        "ct_normal_source_reuse": (
            "capacity-aware-round-robin-with-ordered-distinct-windows-and-output-instance-seed"
        ),
        "rgb_defective_id_roles": ["pollution", "damaged"],
        "fail_length_range": list(FAIL_LENGTH_RANGE),
        "normal_augmentations": {key: list(value) for key, value in NORMAL_AUGMENTATIONS.items()},
        "failure_cases": {key: list(value) for key, value in FAILURE_CASES.items()},
        "search_algorithm": SEARCH_ALGORITHM,
    }


def build_plan(cache_path: Path, output_dir: Path, seed: int = GLOBAL_SEED) -> dict[str, Any]:
    rows = _rows(cache_path)
    selections: dict[str, list[Selection]] = {}
    populations: dict[str, Stats] = {}
    for modality in ("CT", "RGB"):
        selections[modality], populations[modality] = _select(rows, modality, seed)

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
        source_reuse_totals = Counter(item.battery_id for item in chosen)
        # 계획서 v1.5 §7: FAIL 대상 2개를 제품 상태별로 하나씩 고른다. 두 개를 한
        # 무더기에서 뽑으면 촬영실패와 제품불량이 겹치는 칸이 비어 버린다. v1.3 산출물이
        # 실제로 그랬다.
        fail_ids: set[tuple[int, int]] = set()
        for status in ("defective", "normal"):
            pool = [item for item in chosen if item.product_status == status]
            if status == "defective":
                # 결함이 한두 장뿐인 ID 를 FAIL 대상으로 잡으면 교차 칸이 한 장짜리가
                # 된다. FAIL 구간 최대 길이만큼 결함이 이어지는 축을 가진 ID 로 좁힌다.
                dense = [item for item in pool if _max_defect_run(item, modality) >= FAIL_LENGTH_RANGE[1]]
                pool = dense or pool
            if pool:
                target = min(pool, key=lambda item: stable_seed(
                    seed, modality, "fail-target", *item.instance_key
                ))
                fail_ids.add(target.instance_key)
        for item in sorted(chosen, key=lambda item: stable_seed(
            seed, modality, "fail-fill", *item.instance_key
        )):
            if len(fail_ids) >= 2:
                break
            fail_ids.add(item.instance_key)
        windows_by_id = {item.instance_key: item.window for item in chosen}
        fail_layout: dict[tuple[int, int], dict[str, Any]] = {}
        for item in chosen:
            item_key = item.instance_key
            if item_key not in fail_ids:
                continue
            rng = random.Random(stable_seed(
                seed, modality, *item.instance_key, "failure-window"
            ))
            flags = DEFECT_FLAGS[modality]
            defect_at = [any(row[flag] for flag in flags) for row in item.window]
            defective_target = item.product_status == "defective"
            if modality == "CT":
                axes = list(CT_COUNTS)
                if defective_target:
                    # 계획서 v1.5 §7: 불량 ID의 FAIL 구간은 결함이 있는 축에서 고른다.
                    with_defect = [
                        candidate for candidate in axes
                        if any(defect_at[index] for index, row in enumerate(item.window) if row["axis"] == candidate)
                    ]
                    axes = with_defect or axes
                axis = rng.choice(axes)
            else:
                axis = ""
            eligible = [
                index for index, row in enumerate(item.window)
                if modality != "CT" or row["axis"] == axis
            ]
            length = rng.randint(*FAIL_LENGTH_RANGE)
            starts = list(range(0, len(eligible) - length + 1))
            if defective_target:
                # 촬영실패와 제품불량이 함께 나타나는 이미지를 충분히 만든다. 한 장만
                # 겹치면 그 조합을 학습에도 평가에도 쓸 수 없다.
                def overlap(position: int) -> int:
                    return sum(defect_at[index] for index in eligible[position:position + length])
                rich = [s for s in starts if overlap(s) >= length // 2]
                starts = rich or [s for s in starts if overlap(s)] or starts
            start = rng.choice(starts)
            slots = eligible[start:start + length]
            k = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            cases = rng.sample(FAILURE_CASES[modality], k)
            fail_layout[item_key] = {
                "axis": axis,
                "start": start,
                "length": length,
                "slots": slots,
                "cases": cases,
                "k": k,
                "reserve": _reserve_segments(item.window, modality, axis, slots, length),
            }

        for rank, item in enumerate(chosen, 1):
            item_key = item.instance_key
            output_battery_id = base + rank
            augmentation_slot = rank - 1
            probe_seed = stable_seed(
                seed, modality, *item.instance_key,
                output_battery_id, "normal-base",
            )
            probe_names, _ = _normal_assignment(modality, augmentation_slot, probe_seed)
            # 계획서 6.2: 슬라이스 순서 역전 여부는 ID 단위로 한 번만 결정한다.
            reversed_axes = {
                axis for axis in CT_COUNTS
                if modality == "CT"
                and "synchronized_flip" in probe_names
                and ct_axis_transform(probe_seed, axis).reverse_slices
            }
            sequence = _sequence_metadata(item.window, modality, reversed_axes)
            layout = fail_layout.get(item_key)
            selected_records.append({
                "modality": modality,
                "rank": rank,
                "original_battery_id": item.battery_id,
                "output_battery_id": output_battery_id,
                "product_status": item.product_status,
                "defect_role": item.defect_role,
                "stratum": item.stratum,
                "fail_target": item_key in fail_ids,
                "source_count": len(item.window),
                "source_reuse_ordinal": item.source_reuse_ordinal,
                "source_reuse_total": source_reuse_totals[item.battery_id],
                "search_algorithm": SEARCH_ALGORITHM,
                "search_seed": seed,
                "search_iterations": max(1, len(item.window)),
                "search_stop_condition": SEARCH_STOP_CONDITION,
                "target_defective_ids": item.evidence.get(
                    "target_defective_ids", DEFECTIVE_ID_COUNTS[modality]
                ),
                "primary_objective": round(item.primary_objective, 8),
                "secondary_objective": round(item.secondary_objective, 8),
                "annotation_count_objective": round(item.count_objective, 8),
                "population_defect_image_ratio": round(population.defect_image_ratio, 8),
                "population_mean_defect_count": round(population.mean_defect_count, 8),
                "slice_order_reversed": bool(reversed_axes),
                "slice_order_reversed_by_axis": json.dumps(
                    {axis: axis in reversed_axes for axis in CT_COUNTS}, sort_keys=True
                ) if modality == "CT" else False,
                "rejection_reason": item.rejection,
            })

            initial_for_recapture: list[dict[str, Any]] = []
            for slot, row in enumerate(item.window):
                sample_counter += 1
                group_seed = stable_seed(
                    seed, modality, output_battery_id, *item.instance_key,
                    row["axis"], row["original_index"],
                )
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
                    failure_segment = (
                        f"{modality}-{output_battery_id}-{layout['start']}-{layout['length']}"
                    )
                    for order, (reason, slots) in enumerate(layout["reserve"], 1):
                        reserve_candidates.append(
                            _candidate_record(order, reason, item.window[slots[offset]])
                        )
                    other = next(
                        (other_id for other_id in fail_ids if other_id != item_key), None
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
                    "capture_group_id": f"G{stable_seed(modality, output_battery_id, row['axis'], row['original_index']):016x}",
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

            if item_key in fail_ids:
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
