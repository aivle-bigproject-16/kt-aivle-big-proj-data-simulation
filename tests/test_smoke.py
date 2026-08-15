from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server_sim_dataset.smoke import select_smoke_rows, write_quick_smoke_plan


class SmokeSelectionTests(unittest.TestCase):
    def test_selects_all_twelve_product_status_and_capture_routes(self) -> None:
        rows = []
        for modality in ("CT", "RGB"):
            for product_status in ("normal", "defective"):
                for capture_set, qualities in (
                    ("initial_capture", ("PASS", "FAIL")),
                    ("recapture", ("PASS",)),
                ):
                    for quality in qualities:
                        for index in range(4):
                            rows.append({
                                "modality": modality,
                                "product_status": product_status,
                                "capture_set": capture_set,
                                "capture_quality": quality,
                                "sample_id": f"{modality}-{product_status}-{capture_set}-{quality}-{index}",
                            })
        selected = select_smoke_rows(rows, per_group=2)
        self.assertEqual(len(selected), 24)
        routes = {
            (row["modality"], row["product_status"], row["capture_set"], row["capture_quality"])
            for row in selected
        }
        self.assertEqual(len(routes), 12)

    def test_rejects_plan_without_fail_route(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks smoke-test routes"):
            select_smoke_rows([
                {
                    "modality": "CT", "product_status": "normal",
                    "capture_set": "initial_capture", "capture_quality": "PASS",
                }
            ], per_group=1)

    def test_quick_plan_builds_twelve_routes_and_links_recaptures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = {"CT": [], "RGB": []}
            images = {}
            for modality, stems in {
                "CT": ("CT_cell_pouch_101_x_000", "CT_cell_pouch_102_x_000"),
                "RGB": ("RGB_cell_cylindrical_0001_001", "RGB_cell_cylindrical_0002_001"),
            }.items():
                normal_path = root / f"{stems[0]}.json"
                defective_path = root / f"{stems[1]}.json"
                normal_payload = {"defects": []}
                defect_name = "porosity" if modality == "CT" else "Damaged"
                defective_payload = {"defects": [{"name": defect_name, "points": [0, 0, 1, 0, 1, 1]}]}
                candidates[modality] = [(normal_path, normal_payload), (defective_path, defective_payload)]
                for path in (normal_path, defective_path):
                    image_path = root / f"{path.stem}.jpg"
                    image_path.write_bytes(b"smoke")
                    path.write_text("{}", encoding="utf-8")
                    images[path.stem] = image_path

            plan = root / "smoke.csv"
            with patch("server_sim_dataset.smoke._quick_json_candidates", return_value=candidates), patch(
                "server_sim_dataset.smoke._find_images", return_value=images
            ):
                counts = write_quick_smoke_plan(root, plan, per_group=1)

            with plan.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(counts["total"], 12)
            self.assertEqual(len(rows), 12)
            sample_ids = {row["sample_id"] for row in rows}
            for row in rows:
                if row["capture_set"] == "recapture":
                    self.assertIn(row["retry_of_sample_id"], sample_ids)
            routes = {
                (row["modality"], row["product_status"], row["capture_set"], row["capture_quality"])
                for row in rows
            }
            self.assertEqual(len(routes), 12)

    def test_quick_plan_rejects_zero_per_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            write_quick_smoke_plan(Path("unused"), Path("unused.csv"), per_group=0)


if __name__ == "__main__":
    unittest.main()

