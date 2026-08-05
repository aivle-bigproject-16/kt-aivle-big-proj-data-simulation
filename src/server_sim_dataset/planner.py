from __future__ import annotations

import csv
import json
import logging
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .util import atomic_json, stable_seed


LOGGER = logging.getLogger(__name__)
GLOBAL_SEED = 20260723
CT_COUNTS = {"x": 150, "y": 650, "z": 650}
NORMAL_AUGMENTATIONS = {
    "CT": ("brightness_contrast_gamma", "partial_histogram_blend", "normal_noise_poisson", "low_frequency_shading", "percentile_tone_curve", "weak_reconstruction_kernel", "synchronized_flip"),
    "RGB": ("safe_translate_rotate", "brightness_contrast_gamma", "rgb_channel_gain_tone", "low_frequency_lighting", "poisson_noise", "weak_reconstruction"),
}
FAILURE_CASES = {
    "CT": ("ct_cell_alignment_failure", "ct_acquisition_motion", "ct_insufficient_projection_sampling", "ct_low_signal_noise", "ct_beam_hardening_metal_streak"),
    "RGB": ("rgb_trigger_timing_failure", "rgb_uneven_lighting", "rgb_reflection_glare", "rgb_focus_failure", "rgb_underexposure", "rgb_overexposure", "rgb_surface_dust", "rgb_hair_contamination"),
}


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


def _windows(rows: list[dict[str, Any]], length: int) -> Iterable[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: row["original_index"])
    for start in range(len(ordered) - length + 1):
        yield ordered[start:start + length]


def _best_window(rows: list[dict[str, Any]], length: int, defective: bool) -> list[dict[str, Any]] | None:
    candidates = []
    for window in _windows(rows, length):
        defect_count = sum(bool(r["has_porosity"] or r["has_damaged"] or r["has_pollution"]) for r in window)
        if (defect_count > 0) != defective:
            continue
        gaps = sum(max(0, b["original_index"] - a["original_index"] - 1) for a, b in zip(window, window[1:]))
        candidates.append((gaps, window[0]["original_index"], window))
    return min(candidates, default=(0, 0, None), key=lambda item: item[:2])[2]


def _select(rows: list[dict[str, Any]], modality: str) -> list[tuple[int, str, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["modality"] == modality and (modality != "CT" or row["porosity_bbox_max_ratio"] < 0.25):
            grouped[row["battery_id"]].append(row)
    normal: list[tuple[int, list[dict[str, Any]]]] = []
    defective: list[tuple[int, list[dict[str, Any]]]] = []
    for battery_id, battery_rows in sorted(grouped.items()):
        if modality == "RGB":
            normal_window = _best_window(battery_rows, 250, False)
            defect_window = _best_window(battery_rows, 250, True)
        else:
            axes = {axis: [r for r in battery_rows if r["axis"] == axis] for axis in CT_COUNTS}
            normal_parts = []
            for axis, count in CT_COUNTS.items():
                pool = [r for r in axes[axis] if not r["has_porosity"]]
                part = _best_window(pool, count, False)
                if part is None: normal_parts = []; break
                normal_parts.extend(part)
            normal_window = normal_parts or None
            defect_parts = []
            for axis, count in CT_COUNTS.items():
                part = _best_window(axes[axis], count, False) or next(_windows(axes[axis], count), None)
                if part is None: defect_parts = []; break
                defect_parts.extend(part)
            if defect_parts and not any(r["has_porosity"] for r in defect_parts):
                for axis, count in CT_COUNTS.items():
                    replacement = _best_window(axes[axis], count, True)
                    if replacement is not None:
                        defect_parts = [r for r in defect_parts if r["axis"] != axis] + replacement
                        break
            defect_window = defect_parts if defect_parts and any(r["has_porosity"] for r in defect_parts) else None
        if normal_window is not None:
            normal.append((battery_id, normal_window))
        if defect_window is not None:
            defective.append((battery_id, defect_window))
    population = [row for values in grouped.values() for row in values]
    population_ratio = sum(bool(r["has_porosity"] or r["has_damaged"] or r["has_pollution"]) for r in population) / max(1, len(population))
    defect_only = [candidate for candidate in defective if candidate[0] not in {item[0] for item in normal}]
    defect_candidates = defect_only or defective
    defect_candidates.sort(key=lambda candidate: (
        abs(sum(bool(r["has_porosity"] or r["has_damaged"] or r["has_pollution"]) for r in candidate[1]) / len(candidate[1]) / 20 - population_ratio),
        stable_seed(GLOBAL_SEED, modality, candidate[0]),
    ))
    defect_choice = defect_candidates[:1]
    if not defect_choice:
        raise ValueError(f"{modality}: no eligible defective ID")
    normal_choice = [candidate for candidate in normal if candidate[0] != defect_choice[0][0]][:19]
    if len(normal_choice) != 19:
        raise ValueError(f"{modality}: requires 19 normal IDs, found {len(normal_choice)}")
    return [(defect_choice[0][0], "defective", defect_choice[0][1])] + [(bid, "normal", window) for bid, window in normal_choice]


def _normal_assignment(modality: str, slot: int, group_seed: int) -> tuple[list[str], dict[str, float]]:
    rng = random.Random(stable_seed(group_seed, "normal"))
    names = NORMAL_AUGMENTATIONS[modality]
    count = 2 if slot % 5 == 4 else 1
    selected = [names[(slot + offset) % len(names)] for offset in range(count)]
    params = {"brightness": rng.uniform(0.92, 1.08), "contrast": rng.uniform(0.92, 1.08), "gamma": rng.uniform(0.92, 1.08), "noise_sigma": rng.uniform(0.002, 0.008)}
    return selected, params


def build_plan(cache_path: Path, output_dir: Path, seed: int = GLOBAL_SEED) -> dict[str, int]:
    rows = _rows(cache_path)
    selections = {modality: _select(rows, modality) for modality in ("CT", "RGB")}
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "generation_plan.csv"
    selected_path = output_dir / "selected_ids.csv"
    selected_records = []
    plan_records: list[dict[str, Any]] = []
    sample_counter = 0
    for modality in ("CT", "RGB"):
        base = 1_900_000_000 if modality == "CT" else 2_900_000_000
        chosen = selections[modality]
        fail_ids = {item[0] for item in sorted(chosen, key=lambda x: stable_seed(seed, modality, x[0]))[:2]}
        for rank, (battery_id, product_status, window) in enumerate(chosen, 1):
            output_battery_id = base + rank
            selected_records.append({"modality": modality, "rank": rank, "original_battery_id": battery_id, "output_battery_id": output_battery_id, "product_status": product_status, "fail_target": battery_id in fail_ids, "source_count": len(window)})
            fail_slots: dict[int, tuple[str, str]] = {}
            if battery_id in fail_ids:
                rng = random.Random(stable_seed(seed, modality, battery_id, "failure-window"))
                eligible_indices = list(range(len(window)))
                if modality == "CT":
                    axis = rng.choice(list(CT_COUNTS))
                    eligible_indices = [i for i, row in enumerate(window) if row["axis"] == axis]
                length = rng.randint(10, 25)
                start = rng.randint(0, len(eligible_indices) - length)
                slots = eligible_indices[start:start + length]
                k = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
                cases = rng.sample(FAILURE_CASES[modality], k)
                for offset, position in enumerate(slots):
                    case_index = min(k - 1, offset * k // length)
                    fail_slots[position] = (f"{modality}-{battery_id}-{start}-{length}", cases[case_index])
            initial_for_recapture = []
            for slot, row in enumerate(window):
                sample_counter += 1
                group_seed = stable_seed(seed, modality, battery_id, row["axis"], row["original_index"])
                normal_seed = stable_seed(seed, modality, battery_id, "normal-base") if modality == "CT" else group_seed
                assignment_slot = rank - 1 if modality == "CT" else slot
                augmentations, parameters = _normal_assignment(modality, assignment_slot, normal_seed)
                failure_segment, failure_case = fail_slots.get(slot, ("", ""))
                record = {"sample_id": f"S{sample_counter:08d}", "capture_group_id": f"G{stable_seed(modality,battery_id,row['axis'],row['original_index']):016x}", "retry_of_sample_id": "", "capture_set": "initial_capture", "modality": modality, "original_battery_id": battery_id, "output_battery_id": output_battery_id, "product_status": product_status, "axis": row["axis"], "original_index": row["original_index"], "source_split": row["source_split"], "original_stem": row["original_stem"], "orig_image_relative_path": row["image_relative_path"], "orig_json_relative_path": row["json_relative_path"], "source_image_sha256": row["image_sha256"], "source_json_sha256": row["json_sha256"], "capture_quality": "FAIL" if failure_case else "PASS", "failure_case": failure_case, "failure_segment_id": failure_segment, "base_augmentation_names": json.dumps(augmentations), "normal_augmentation_parameters": json.dumps(parameters, sort_keys=True), "normal_augmentation_seed": normal_seed, "slice_seed": stable_seed(normal_seed,row["axis"],row["original_index"],"|".join(augmentations)), "item_seed": stable_seed(group_seed, failure_case or "PASS"), "global_seed": seed}
                plan_records.append(record); initial_for_recapture.append(record)
            if battery_id in fail_ids:
                for initial in initial_for_recapture:
                    sample_counter += 1
                    retry = dict(initial); retry.update({"sample_id": f"S{sample_counter:08d}", "retry_of_sample_id": initial["sample_id"], "capture_set": "recapture", "capture_quality": "PASS", "failure_case": "", "failure_segment_id": ""})
                    plan_records.append(retry)
    expected = {"initial_capture": 34000, "recapture": 3400}
    actual = {name: sum(row["capture_set"] == name for row in plan_records) for name in expected}
    if actual != expected:
        raise AssertionError(f"Plan quantity mismatch: {actual} != {expected}")
    if set(r["output_battery_id"] for r in selected_records if r["modality"] == "CT") & set(r["output_battery_id"] for r in selected_records if r["modality"] == "RGB"):
        raise AssertionError("CT/RGB output battery IDs overlap")
    for path, records in ((plan_path, plan_records), (selected_path, selected_records)):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = {"initial_capture": 34000, "recapture": 3400, "total": 37400, "CT_selected_ids": 20, "RGB_selected_ids": 20, "seed": seed, "label_source": "extracted-json-only"}
    atomic_json(output_dir / "plan_summary.json", summary)
    LOGGER.info("Generation plan complete: %s", plan_path)
    return summary
