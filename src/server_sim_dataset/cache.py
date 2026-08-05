from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from PIL import Image

from .schema import iter_defects, parse_stem, points, roi_bbox, split_name
from .util import safe_relative, sha256_file


LOGGER = logging.getLogger(__name__)
CACHE_VERSION = 1
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
 has_damaged INTEGER NOT NULL, has_pollution INTEGER NOT NULL
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


def build_cache(raw_root: Path, cache_path: Path) -> dict[str, int]:
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
    insert = "INSERT INTO pairs(status,exclusion_reason,source_split,modality,battery_id,axis,original_index,original_stem,image_relative_path,json_relative_path,image_sha256,json_sha256,width,height,roi_json,outline_json,defects_json,original_is_normal,porosity_bbox_max_ratio,has_porosity,has_damaged,has_pollution) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    LOGGER.info("JSON-only scan: parsing extracted JSON files")
    for index, json_path in enumerate(_files(raw_root, {".json"}), 1):
        counts["json_seen"] += 1
        values: list[object]
        try:
            parsed = parse_stem(json_path.stem)
            json_split = split_name(json_path)
            candidates = [path for path in image_index.get(json_path.stem, []) if split_name(path) == json_split]
            if len(candidates) != 1:
                raise ValueError(f"same-split paired image count={len(candidates)}")
            image_path = candidates[0]
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
            info = payload.get("image_info") or {}
            data = payload.get("data_info") or {}
            if int(data.get("battery_ids")) != parsed.battery_id:
                raise ValueError("battery ID mismatch")
            if Path(str(info.get("file_name", ""))).stem != json_path.stem:
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
            values = ["valid", "", split_name(json_path), parsed.modality, parsed.battery_id, parsed.axis,
                parsed.original_index, json_path.stem, safe_relative(raw_root,image_path), safe_relative(raw_root,json_path),
                sha256_file(image_path), sha256_file(json_path), width, height, json.dumps(roi), json.dumps(outline),
                json.dumps(defects), int(declared_normal), max(ratios, default=0.0), int("porosity" in {n.lower() for n in names}),
                int("damaged" in {n.lower() for n in names}), int("pollution" in {n.lower() for n in names})]
            counts["valid"] += 1
        except Exception as exc:
            counts["invalid"] += 1
            try:
                parsed = parse_stem(json_path.stem)
                modality,battery,axis,original_index=parsed.modality,parsed.battery_id,parsed.axis,parsed.original_index
            except ValueError:
                modality,battery,axis,original_index="",-1,"",-1
            values = ["invalid", str(exc), split_name(json_path), modality,battery,axis,original_index,json_path.stem,
                "",safe_relative(raw_root,json_path),"",sha256_file(json_path),0,0,"[]","[]","[]",1,0.0,0,0,0]
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
    db.commit(); db.close()
    temporary.replace(cache_path)
    LOGGER.info("Scan cache complete: %s", cache_path)
    return dict(counts)


def ensure_cache(raw_root: Path, cache_path: Path, refresh: bool = False) -> tuple[Path, bool]:
    if not refresh and cache_is_usable(cache_path, raw_root):
        LOGGER.info("Reusing scan cache; filesystem scan skipped: %s", cache_path)
        return cache_path, True
    build_cache(raw_root, cache_path)
    return cache_path, False


def export_cache_csv(cache_path: Path, output: Path) -> None:
    db = _database(cache_path)
    try:
      with output.open("w", encoding="utf-8-sig", newline="") as handle:
        rows = db.execute("SELECT * FROM pairs ORDER BY id")
        writer = csv.writer(handle); writer.writerow([column[0] for column in rows.description]); writer.writerows(rows)
    finally:
        db.close()
