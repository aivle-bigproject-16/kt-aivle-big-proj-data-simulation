from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from server_sim_dataset import __version__
from server_sim_dataset.cache import ensure_cache
from server_sim_dataset.generator import generate, package_outputs, verify
from server_sim_dataset.schema import MANIFEST_COLUMNS, parse_stem, roi_bbox
from server_sim_dataset.util import sha256_file, stable_seed


def _source(root: Path, stem: str = "CT_cell_pouch_101_x_001") -> tuple[Path, Path]:
    image_path = root / "Training" / "01.원천데이터" / f"{stem}.jpg"
    json_path = root / "Training" / "02.라벨링데이터" / f"{stem}.json"
    image_path.parent.mkdir(parents=True); json_path.parent.mkdir(parents=True)
    Image.new("L", (16, 12), 120).save(image_path)
    payload = {
        "data_info": {"data_type": "ct", "battery_ids": 101, "roi": [8, 12]},
        "swelling": {"swelling": False, "battery_outline": [1, 1, 7, 1, 7, 10, 1, 10]},
        "defects": None,
        "image_info": {"width": 16, "height": 12, "file_name": image_path.name, "id": 1, "is_normal": True},
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return image_path, json_path


class PipelineTests(unittest.TestCase):
    def test_logging_does_not_use_unsupported_comma_printf_format(self) -> None:
        source_root = Path(__file__).parents[1] / "src" / "server_sim_dataset"
        offenders = []
        for path in source_root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "%,d" in text or "%,i" in text:
                offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_parse_and_roi(self) -> None:
        parsed = parse_stem("CT_cell_pouch_101_z_009")
        self.assertEqual((parsed.modality, parsed.battery_id, parsed.axis, parsed.original_index), ("CT", 101, "z", 9))
        self.assertEqual(roi_bbox({"data_info": {"roi": [100, 200]}}, 400, 300), (0, 0, 100, 200))

    def test_json_only_cache_is_reused_and_ignores_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory); _source(tmp_path)
            (tmp_path / "labels.tar").write_bytes(b"not opened")
            cache = tmp_path / "cache" / "scan.sqlite"
            _, reused = ensure_cache(tmp_path, cache)
            self.assertFalse(reused)
            db = sqlite3.connect(cache)
            try:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM pairs WHERE status='valid'").fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT value FROM metadata WHERE key='label_source'").fetchone()[0], "extracted-json-only")
            finally:
                db.close()
            _, reused = ensure_cache(tmp_path, cache)
            self.assertTrue(reused)

    def test_generate_ct_roi_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory); image_path, json_path = _source(tmp_path / "raw")
            plan = tmp_path / "plan.csv"
            # plan 과 manifest 는 같은 72 컬럼 스키마를 쓴다. 생성 단계 소관 컬럼은 비운다.
            row = {column: "" for column in MANIFEST_COLUMNS}
            row.update({
                "sample_id": "S00000001",
                "synthetic_id": "initial_CT_1900000001_x_000001",
                "capture_group_id": "G1", "retry_of_sample_id": "", "capture_set": "initial_capture",
                "modality": "CT", "original_battery_id": 101, "output_battery_id": 1900000001,
                "product_status": "normal", "axis": "x", "original_index": 1,
                "source_sequence_order": 0, "output_sequence_order": 0,
                "index_gap_before": 0, "index_gap_size": 0,
                "source_split": "training", "original_stem": image_path.stem,
                "orig_image_relative_path": image_path.relative_to(tmp_path / "raw").as_posix(),
                "orig_json_relative_path": json_path.relative_to(tmp_path / "raw").as_posix(),
                "original_image_id": 1, "original_image_file_name": image_path.name,
                "original_roi": json.dumps([8, 12]),
                "source_image_sha256": sha256_file(image_path), "source_json_sha256": sha256_file(json_path),
                "original_is_normal": 1, "has_porosity": 0, "has_damaged": 0, "has_pollution": 0,
                "original_defect_count": 0,
                "capture_quality": "PASS", "failure_case": "", "failure_segment_id": "",
                "base_augmentation_names": json.dumps(["brightness_contrast_gamma"]),
                "normal_augmentation_parameters": json.dumps({"brightness": 1.0, "contrast": 1.0, "gamma": 1.0, "noise_sigma": 0.003}),
                "normal_augmentation_seed": stable_seed("normal"),
                "slice_seed": stable_seed("slice"),
                "item_seed": stable_seed("item"), "global_seed": 20260723,
                "config_hash": "test",
            })
            with plan.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
                writer.writeheader(); writer.writerow(row)
            output = tmp_path / "output"; summary = generate(tmp_path / "raw", plan, output)
            self.assertEqual(summary["succeeded"], 1)
            with (output / "manifests" / "dataset_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                produced = list(csv.DictReader(handle))
            self.assertEqual(set(produced[0]), set(MANIFEST_COLUMNS))
            self.assertEqual(produced[0]["generator_version"], __version__)
            self.assertTrue(produced[0]["plan_sha256"])
            self.assertIn(produced[0]["jpeg_profile_id"], {"source-qtable", "common-q95-s444"})
            output_image = next((output / "initial_capture" / "CT" / "images").glob("*.jpg"))
            with Image.open(output_image) as generated:
                self.assertEqual(generated.size, (8, 12))
            output_json = json.loads(next((output / "initial_capture" / "CT" / "json").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(output_json["data_info"]["roi"], [0, 0, 8, 12])
            self.assertEqual(output_json["data_info"]["battery_ids"], 1900000001)
            self.assertEqual(verify(output), {"samples": 1, "errors": 0, "orphans": 0})
            self.assertEqual(package_outputs(output)["zip_files"], 17)

    def test_stable_seed_is_reproducible(self) -> None:
        self.assertEqual(stable_seed("a", 1), stable_seed("a", 1))
        self.assertNotEqual(stable_seed("a", 1), stable_seed("a", 2))


if __name__ == "__main__":
    unittest.main()
