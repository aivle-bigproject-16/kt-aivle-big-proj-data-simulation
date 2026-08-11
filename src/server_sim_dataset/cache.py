from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

from PIL import Image

from .schema import iter_defects, parse_stem, points, roi_bbox, split_name
from .util import safe_relative, sha256_file


LOGGER = logging.getLogger(__name__)
# v2 에서 original_image_id, original_roi_json, defect_count 를 추가했다. 계획서 8.1 이
# manifest 에 원본 식별자와 원본 ROI 원본값을 보존하라고 규정하는데 v1 스키마에는 두
# 값이 없어 생성 단계에서 원본 JSON 을 다시 열어야 했다.
CACHE_VERSION = 2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ARCHIVE_SUFFIXES = {".tar", ".tgz", ".gz", ".zip", ".7z", ".rar"}


def _files(root: Path, suffixes: set[str]) -> Iterator[Path]:
    for directory, names, filenames in os.walk(root):
        names[:] = [name for name in names if Path(name).suffix.lower() not in ARCHIVE_SUFFIXES]
        for filename in filenames:
            path = Path(directory, filename)
            if path.suffix.lower() in suffixes:
                yield path


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE pairs (
 id INTEGER PRIMARY KEY, status TEXT NOT NULL, exclusion_reason TEXT NOT NULL,
 source_split TEXT NOT NULL, modality TEXT NOT NULL, battery_id INTEGER NOT NULL,
 axis TEXT NOT NULL, original_index INTEGER NOT NULL, original_stem TEXT NOT NULL,
 image_relative_path TEXT NOT NULL, json_relative_path TEXT NOT NULL,
 image_sha256 TEXT NOT NULL, json_sha256 TEXT NOT NULL, width INTEGER NOT NULL,
 height INTEGER NOT NULL, roi_json TEXT NOT NULL, outline_json TEXT NOT NULL,
 defects_json TEXT NOT NULL, original_is_normal INTEGER NOT NULL,
 porosity_bbox_max_ratio REAL NOT NULL, has_porosity INTEGER NOT NULL,
 has_damaged INTEGER NOT NULL, has_pollution INTEGER NOT NULL,
 original_image_id INTEGER NOT NULL, original_roi_json TEXT NOT NULL,
 defect_count INTEGER NOT NULL
);
CREATE INDEX pairs_key ON pairs(modality,battery_id,axis,original_index);
CREATE INDEX pairs_status ON pairs(status);
"""


def cache_is_usable(cache_path: Path, raw_root: Path) -> bool:
    if not cache_path.is_file():
        return False
    try:
        db = _database(cache_path)
        try:
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
        finally:
            db.close()
        return metadata.get("cache_version") == str(CACHE_VERSION) and Path(metadata.get("raw_root", "")) == raw_root.resolve()
    except sqlite3.Error:
        return False


def _scan_pair(task: tuple[str, str, str]) -> tuple[bool, list[object]]:
    """JSON 하나와 짝 이미지를 읽어 pairs 행 하나를 만든다.

    프로세스 풀에서 실행하므로 모듈 최상위에 두고 인자와 반환값을 모두 picklable 하게
    유지한다. 원본 이미지 27 만 장의 SHA-256 이 스캔 시간을 지배하므로 이 함수를
    병렬로 돌리는 것이 전체 소요를 결정한다.
    """
    raw_root, json_text, image_text = Path(task[0]), Path(task[1]), task[2]
    try:
        parsed = parse_stem(json_text.stem)
        if not image_text:
            raise ValueError("same-split paired image not found")
        image_path = Path(image_text)
        payload = json.loads(json_text.read_text(encoding="utf-8-sig"))
        info = payload.get("image_info") or {}
        data = payload.get("data_info") or {}
        if int(data.get("battery_ids")) != parsed.battery_id:
            raise ValueError("battery ID mismatch")
        if Path(str(info.get("file_name", ""))).stem != json_text.stem:
            raise ValueError("image_info.file_name mismatch")
        with Image.open(image_path) as image:
            width, height = image.size
        roi = roi_bbox(payload, width, height) if parsed.modality == "CT" else (0, 0, width, height)
        outline = points((payload.get("swelling") or {}).get("battery_outline"))
        if len(outline) < 3:
            raise ValueError("missing battery outline")
        defects = [{"name": name, "points": polygon} for name, polygon in iter_defects(payload)]
        names = {item["name"] for item in defects}
        ratios = []
        roi_width, roi_height = roi[2] - roi[0], roi[3] - roi[1]
        for item in defects:
            if item["name"].lower() == "porosity":
                xs = [p[0] for p in item["points"]]; ys = [p[1] for p in item["points"]]
                ratios.append(max((max(xs)-min(xs))/roi_width, (max(ys)-min(ys))/roi_height))
        declared_normal = bool(info.get("is_normal"))
        recognized = any(name.lower() in {"porosity", "damaged", "pollution"} for name in names)
        if declared_normal == recognized:
            raise ValueError("image_info.is_normal conflicts with defects")
        try:
            original_image_id = int(info.get("id"))
        except (TypeError, ValueError):
            original_image_id = -1
        lowered = {name.lower() for name in names}
        return True, ["valid", "", split_name(json_text), parsed.modality, parsed.battery_id, parsed.axis,
            parsed.original_index, json_text.stem, safe_relative(raw_root, image_path), safe_relative(raw_root, json_text),
            sha256_file(image_path), sha256_file(json_text), width, height, json.dumps(roi), json.dumps(outline),
            json.dumps(defects), int(declared_normal), max(ratios, default=0.0), int("porosity" in lowered),
            int("damaged" in lowered), int("pollution" in lowered),
            original_image_id, json.dumps(data.get("roi")), len(defects)]
    except Exception as exc:
        try:
            parsed = parse_stem(json_text.stem)
            modality, battery, axis, original_index = parsed.modality, parsed.battery_id, parsed.axis, parsed.original_index
        except ValueError:
            modality, battery, axis, original_index = "", -1, "", -1
        return False, ["invalid", str(exc), split_name(json_text), modality, battery, axis, original_index,
            json_text.stem, "", safe_relative(raw_root, json_text), "", sha256_file(json_text), 0, 0,
            "[]", "[]", "[]", 1, 0.0, 0, 0, 0, -1, "null", 0]


def _tasks(raw_root: Path, image_index: dict[str, list[Path]]) -> Iterator[tuple[str, str, str]]:
    for json_path in _files(raw_root, {".json"}):
        json_split = split_name(json_path)
        candidates = [path for path in image_index.get(json_path.stem, []) if split_name(path) == json_split]
        yield str(raw_root), str(json_path), str(candidates[0]) if len(candidates) == 1 else ""


def build_cache(raw_root: Path, cache_path: Path, workers: int = 1) -> dict[str, int]:
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    image_index: dict[str, list[Path]] = defaultdict(list)
    LOGGER.info("JSON-only scan: indexing source images")
    for index, image_path in enumerate(_files(raw_root, IMAGE_SUFFIXES), 1):
        image_index[image_path.stem].append(image_path)
        if index % 25000 == 0:
            LOGGER.info("Image index: %s files", f"{index:,}")
    db = _database(temporary)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", [
        ("cache_version", str(CACHE_VERSION)), ("raw_root", str(raw_root)), ("label_source", "extracted-json-only")
    ])
    counts = defaultdict(int)
    insert = "INSERT INTO pairs(status,exclusion_reason,source_split,modality,battery_id,axis,original_index,original_stem,image_relative_path,json_relative_path,image_sha256,json_sha256,width,height,roi_json,outline_json,defects_json,original_is_normal,porosity_bbox_max_ratio,has_porosity,has_damaged,has_pollution,original_image_id,original_roi_json,defect_count) VALUES(" + ",".join("?" * 25) + ")"
    LOGGER.info("JSON-only scan: parsing extracted JSON with %d worker(s)", workers)
    stream = _tasks(raw_root, image_index)
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers)
        # map 은 입력 순서를 유지하므로 병렬로 돌려도 삽입 순서가 결정론적이다.
        results = pool.map(_scan_pair, stream, chunksize=64)
    else:
        pool = None
        results = map(_scan_pair, stream)
    try:
        for index, (ok, values) in enumerate(results, 1):
            counts["json_seen"] += 1
            counts["valid" if ok else "invalid"] += 1
            db.execute(insert, values)
            if index % 1000 == 0:
                db.commit()
            if index % 25000 == 0:
                LOGGER.info(
                    "JSON parsed: %s | valid %s | invalid %s",
                    f"{index:,}",
                    f"{counts['valid']:,}",
                    f"{counts['invalid']:,}",
                )
    finally:
        if pool is not None:
            pool.shutdown()
    db.commit(); db.close()
    temporary.replace(cache_path)
    LOGGER.info("Scan cache complete: %s", cache_path)
    return dict(counts)


def ensure_cache(raw_root: Path, cache_path: Path, refresh: bool = False, workers: int = 1) -> tuple[Path, bool]:
    if not refresh and cache_is_usable(cache_path, raw_root):
        LOGGER.info("Reusing scan cache; filesystem scan skipped: %s", cache_path)
        return cache_path, True
    build_cache(raw_root, cache_path, workers=workers)
    return cache_path, False


def source_hashes(cache_path: Path) -> dict[str, tuple[str, str]]:
    """원본 stem 별 (image_sha256, json_sha256) 를 돌려준다.

    생성 단계가 행마다 원본을 다시 해싱하면 4000x4000 JPEG 37,400 장의 I/O 가 전체
    시간을 지배한다. 캐시가 스캔 시점에 이미 계산해 둔 값이 있으므로 그것과 대조한다.
    원본 파일 자체를 다시 해싱하는 것은 계획서 13.3 의 엄격 검사가 필요할 때만 한다.
    """
    db = _database(cache_path)
    try:
        rows = db.execute(
            "SELECT original_stem,image_sha256,json_sha256 FROM pairs WHERE status='valid'"
        )
        return {row["original_stem"]: (row["image_sha256"], row["json_sha256"]) for row in rows}
    finally:
        db.close()


def export_cache_csv(cache_path: Path, output: Path) -> None:
    db = _database(cache_path)
    try:
      with output.open("w", encoding="utf-8-sig", newline="") as handle:
        rows = db.execute("SELECT * FROM pairs ORDER BY id")
        writer = csv.writer(handle); writer.writerow([column[0] for column in rows.description]); writer.writerows(rows)
    finally:
        db.close()
