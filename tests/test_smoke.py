from __future__ import annotations

import unittest

from server_sim_dataset.smoke import select_smoke_rows


class SmokeSelectionTests(unittest.TestCase):
    def test_selects_all_six_routes(self) -> None:
        rows = []
        for modality in ("CT", "RGB"):
            for capture_set, qualities in (
                ("initial_capture", ("PASS", "FAIL")),
                ("recapture", ("PASS",)),
            ):
                for quality in qualities:
                    for index in range(4):
                        rows.append({
                            "modality": modality,
                            "capture_set": capture_set,
                            "capture_quality": quality,
                            "sample_id": f"{modality}-{capture_set}-{quality}-{index}",
                        })
        selected = select_smoke_rows(rows, per_group=2)
        self.assertEqual(len(selected), 12)
        routes = {(row["modality"], row["capture_set"], row["capture_quality"]) for row in selected}
        self.assertEqual(len(routes), 6)

    def test_rejects_plan_without_fail_route(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks smoke-test routes"):
            select_smoke_rows([
                {"modality": "CT", "capture_set": "initial_capture", "capture_quality": "PASS"}
            ], per_group=1)


if __name__ == "__main__":
    unittest.main()

