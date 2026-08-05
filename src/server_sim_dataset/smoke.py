from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .cache import ensure_cache
from .generator import generate, verify
from .planner import build_plan
from .schema import iter_defects, parse_stem
from .util import atomic_json, safe_relative, sha256_file, stable_seed


LOGGER = logging.getLogger(__name__)


def _quick_json_candidates(raw_root: Path) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    """Read only enough extracted JSON files to cover CT/RGB normal and defective routes."""
    selected: dict[str, list[tuple[Path, dict[str, Any]]]] = {"CT": [], "RGB": []}
    states: dict[str, set[bool]] = {"CT": set(), "RGB": set()}
    label_roots = [
        path for path in raw_root.glob("**/02.라벨링데이터") if path.is_dir()
    ]
    if not label_roots:
        raise FileNotFoundError("No extracted 02.라벨링데이터 directory found under raw root")
    for label_root in sorted(label_roots):
        for directory, _, filenames in os.walk(label_root):
            for filename in filenames:
                if not filename.lower().endswith(".json"):
                    continue
                path = Path(directory, filename)
                try:
                    parsed = parse_stem(path.stem)
                    payload = json.loads(path.read_text(encoding="utf-8-sig"))
                    recognized = {
                        name.lower() for name, _ in iter_defects(payload)
                    } & ({"porosity"} if parsed.modality == "CT" else {"damaged", "pollution"})
                    defective = bool(recognized)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                if defective not in states[parsed.modality]:
                    selected[parsed.modality].append((path, payload))
                    states[parsed.modality].add(defective)
                if all(len(states[modality]) == 2 for modality in ("CT", "RGB")):
                    return selected
    missing = [modality for modality in ("CT", "RGB") if len(states[modality]) < 2]
    raise ValueError(f"Quick smoke could not find both normal and defective JSON for: {missing}")


def _find_images(raw_root: Path, stems: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for split in ("Training", "Validation"):
        source_root = raw_root / "3.개방데이터" / "1.데이터" / split / "01.원천데이터"
        if not source_root.is_dir():
            continue
        for directory, _, filenames in os.walk(source_root):
            for filename in filenames:
                path = Path(directory, filename)
                if path.stem in stems and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    found.setdefault(path.stem, path)
            if stems <= found.keys():
                return found
    missing = sorted(stems - found.keys())
    raise FileNotFoundError(f"Quick smoke paired images not found: {missing}")


def write_quick_smoke_plan(raw_root: Path, smoke_plan: Path, per_group: int) -> dict[str, int]:
    candidates = _quick_json_candidates(raw_root)
    stems = {path.stem for values in candidates.values() for path, _ in values}
    images = _find_images(raw_root, stems)
    rows: list[dict[str, Any]] = []
    counter = 0
    cases = {"CT": "ct_low_signal_noise", "RGB": "rgb_underexposure"}
    for modality in ("CT", "RGB"):
        sources = candidates[modality]
        for route_index, (capture_set, quality) in enumerate((
            ("initial_capture", "PASS"),
            ("initial_capture", "FAIL"),
            ("recapture", "PASS"),
        )):
            for position in range(per_group):
                # Initial PASS uses the normal source. Initial FAIL and its recapture
                # use the same defective source, but live in different capture folders.
                source_index = 0 if route_index == 0 else min(1, len(sources) - 1)
                json_path, payload = sources[source_index]
                image_path = images[json_path.stem]
                parsed = parse_stem(json_path.stem)
                counter += 1
                seed = stable_seed("quick-smoke", modality, capture_set, quality, position)
                rows.append({
                    "sample_id": f"Q{counter:05d}",
                    "capture_group_id": f"QG{modality}{position:03d}",
                    "retry_of_sample_id": "" if capture_set == "initial_capture" else f"quick-{modality}-{position}",
                    "capture_set": capture_set,
                    "modality": modality,
                    "original_battery_id": parsed.battery_id,
                    "output_battery_id": (1_900_000_001 if modality == "CT" else 2_900_000_001) + position,
                    "product_status": "defective" if any(True for _ in iter_defects(payload)) else "normal",
                    "axis": parsed.axis,
                    "original_index": parsed.original_index,
                    "source_split": "training" if "Training" in json_path.parts else "validation",
                    "original_stem": json_path.stem,
                    "orig_image_relative_path": safe_relative(raw_root, image_path),
                    "orig_json_relative_path": safe_relative(raw_root, json_path),
                    "source_image_sha256": sha256_file(image_path),
                    "source_json_sha256": sha256_file(json_path),
                    "capture_quality": quality,
                    "failure_case": cases[modality] if quality == "FAIL" else "",
                    "failure_segment_id": f"quick-{modality}" if quality == "FAIL" else "",
                    "base_augmentation_names": json.dumps(["brightness_contrast_gamma"]),
                    "normal_augmentation_parameters": json.dumps({"brightness": 1.02, "contrast": 1.01, "gamma": 0.99, "noise_sigma": 0.003}),
                    "normal_augmentation_seed": seed,
                    "slice_seed": stable_seed(seed, parsed.axis, parsed.original_index),
                    "item_seed": stable_seed(seed, "failure"),
                    "global_seed": 20260723,
                })
    smoke_plan.parent.mkdir(parents=True, exist_ok=True)
    with smoke_plan.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{row['modality']}_{row['capture_set']}_{row['capture_quality']}"] += 1
    counts["total"] = len(rows)
    return dict(counts)


def select_smoke_rows(rows: list[dict[str, str]], per_group: int) -> list[dict[str, str]]:
    """Select every important generation route instead of the first N plan rows."""
    if per_group < 1:
        raise ValueError("per_group must be at least 1")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        capture_set = row["capture_set"]
        quality = row["capture_quality"]
        if capture_set == "recapture":
            quality = "PASS"
        grouped[(row["modality"], capture_set, quality)].append(row)
    required = [
        (modality, capture_set, quality)
        for modality in ("CT", "RGB")
        for capture_set, quality in (
            ("initial_capture", "PASS"),
            ("initial_capture", "FAIL"),
            ("recapture", "PASS"),
        )
    ]
    missing = ["/".join(key) for key in required if not grouped.get(key)]
    if missing:
        raise ValueError(f"Generation plan lacks smoke-test routes: {missing}")
    selected: list[dict[str, str]] = []
    for key in required:
        # Spread picks across the available route, which avoids testing one ID only.
        candidates = grouped[key]
        count = min(per_group, len(candidates))
        positions = sorted({round(index * (len(candidates) - 1) / max(1, count - 1)) for index in range(count)})
        selected.extend(candidates[position] for position in positions)
    return selected


def write_smoke_plan(source_plan: Path, smoke_plan: Path, per_group: int) -> dict[str, int]:
    with source_plan.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_smoke_rows(rows, per_group)
    smoke_plan.parent.mkdir(parents=True, exist_ok=True)
    with smoke_plan.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    counts: dict[str, int] = defaultdict(int)
    for row in selected:
        counts[f"{row['modality']}_{row['capture_set']}_{row['capture_quality']}"] += 1
    counts["total"] = len(selected)
    return dict(counts)


def run_smoke_test(
    raw_root: Path,
    cache_path: Path,
    plan_dir: Path,
    output: Path,
    engine_root: Path,
    per_group: int = 2,
    refresh_cache: bool = False,
    full_scan: bool = False,
) -> dict[str, Any]:
    """Run cache, planning, stratified generation, and integrity checks end to end."""
    started = time.perf_counter()
    summary_path = output / "smoke_test_summary.json"
    try:
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Smoke output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        smoke_plan = output / "smoke_generation_plan.csv"
        if full_scan:
            _, cache_reused = ensure_cache(raw_root, cache_path, refresh_cache)
            full_plan = plan_dir / "generation_plan.csv"
            plan_reused = full_plan.is_file()
            if not plan_reused:
                build_plan(cache_path, plan_dir)
            route_counts = write_smoke_plan(full_plan, smoke_plan, per_group)
            mode = "full-plan"
        else:
            cache_reused = False
            plan_reused = False
            route_counts = write_quick_smoke_plan(raw_root, smoke_plan, per_group)
            mode = "quick-no-full-scan"
        LOGGER.info("Smoke plan ready: %d samples across six required routes", route_counts["total"])
        generation = generate(raw_root, smoke_plan, output / "dataset", engine_root)
        verification = verify(output / "dataset")
        result: dict[str, Any] = {
            "status": "passed",
            "mode": mode,
            "cache_reused": cache_reused,
            "plan_reused": plan_reused,
            "label_source": "extracted-json-only",
            "route_counts": route_counts,
            "generation": generation,
            "verification": verification,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "output": str(output.resolve()),
        }
        atomic_json(summary_path, result)
        LOGGER.info("Smoke test PASSED: %s", summary_path)
        return result
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        atomic_json(summary_path, failure)
        LOGGER.exception("Smoke test FAILED; details written to %s", summary_path)
        raise
