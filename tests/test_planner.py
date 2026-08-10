"""planner 회귀 테스트.

수정계획서 v1.3 의 S2 단계에서 구현보다 먼저 작성했다. v1.3 구현이 끝나기 전에는
일부가 실패하는 것이 정상이며, 실패 목록이 곧 구현의 완료 기준이다.

각 테스트에는 대응하는 결함 ID 와 계획서 조항을 적어 둔다.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from server_sim_dataset import planner
from server_sim_dataset.schema import MANIFEST_COLUMNS

from _synthetic import CT_DEFECT_ID, RGB_DEFECT_ID, build_synthetic_cache


# 정본 목록은 schema.MANIFEST_COLUMNS 다. 여기서는 목록이 통째로 사라지거나 줄어드는
# 사고만 잡는다. 어떤 컬럼이 실제로 채워지는지는 아래 ManifestSchemaTests 가 본다.
EXPECTED_COLUMN_COUNT = 72


class PlannerFixture(unittest.TestCase):
    """합성 cache 로 plan 을 한 번만 만들어 여러 테스트가 공유한다."""

    plan_rows: list[dict[str, str]]
    selected_rows: list[dict[str, str]]
    summary: dict[str, object]
    plan_dir: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name)
        cache = build_synthetic_cache(root / "cache" / "scan.sqlite")
        cls.plan_dir = root / "plan"
        cls.summary = planner.build_plan(cache, cls.plan_dir)
        cls.plan_rows = cls._read(cls.plan_dir / "generation_plan.csv")
        cls.selected_rows = cls._read(cls.plan_dir / "selected_ids.csv")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    @staticmethod
    def _read(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))


class QuantityTests(PlannerFixture):
    def test_plan_quantities_match_section_13_1(self) -> None:
        counts = Counter(row["capture_set"] for row in self.plan_rows)
        self.assertEqual(counts["initial_capture"], 34000)
        self.assertEqual(counts["recapture"], 3400)
        self.assertEqual(len(self.plan_rows), 37400)

    def test_output_battery_id_ranges_are_disjoint(self) -> None:
        ct = {int(row["output_battery_id"]) for row in self.plan_rows if row["modality"] == "CT"}
        rgb = {int(row["output_battery_id"]) for row in self.plan_rows if row["modality"] == "RGB"}
        self.assertEqual(ct, set(range(1_900_000_001, 1_900_000_021)))
        self.assertEqual(rgb, set(range(2_900_000_001, 2_900_000_021)))
        self.assertEqual(ct & rgb, set())

    def test_stratified_allocation_follows_the_eligible_distribution(self) -> None:
        """계획서 4.5(개정): 선정 20 개를 원본 ID 의 층 분포에 비례해 배분한다.

        v1.2 와 v1.3 초안은 20 개 중 정확히 1 개만 제품불량으로 고정했다. 그 규칙에서는
        세트의 클래스 비율이 불량 ID 하나로 정해져 모집단과 비교할 수 없다. 실제 원본에서
        RGB 결함은 배터리 단위라 ID 의 67.5% 가 불량인데, 5% 규칙은 결함 이미지 비율을
        0.05 로 묶어 버렸다.

        이 fixture 는 적격 ID 가 정확히 20 개이므로 배분은 전부를 뽑는 것과 같다.
        """
        expected = {
            "CT": {"zero": 19, "low_mid": 1},
            "RGB": {"clean": 19, "both": 1},
        }
        for modality, strata in expected.items():
            ids = [row for row in self.selected_rows if row["modality"] == modality]
            self.assertEqual(len(ids), 20, modality)
            self.assertEqual(Counter(row["stratum"] for row in ids), Counter(strata), modality)
            defective = sum(1 for row in ids if row["product_status"] == "defective")
            self.assertEqual(defective, 1, f"{modality}: 이 fixture 는 결함 ID 가 하나뿐이다")

    def test_allocation_is_proportional_and_respects_supply(self) -> None:
        """배분은 최대잉여법이고, 후보가 모자란 층의 몫은 다른 층으로 넘어간다."""
        available = {"a": 50, "b": 30, "c": 20}
        self.assertEqual(planner._allocate(available, 10, available), {"a": 5, "b": 3, "c": 2})
        scarce = {"a": 50, "b": 30, "c": 20}
        supply = {"a": 50, "b": 1, "c": 20}
        result = planner._allocate(scarce, 10, supply)
        self.assertEqual(result["b"], 1)
        self.assertEqual(sum(result.values()), 10)
        for name, count in result.items():
            self.assertLessEqual(count, supply[name])


class SelectionTests(PlannerFixture):
    def test_ct_defective_window_is_chosen_for_defect_ratio(self) -> None:
        """F-07, 계획서 4.5 의 4 항.

        결함 ID 의 구간은 결함 비율 기준으로 골라야 한다. v1.2 는 무결함 구간을 먼저 찾고
        porosity 가 하나도 없을 때만 한 축을 사후 교체했기 때문에, 결함이 한 축에만 남고
        1,450 장 중 650 장에만 porosity 가 있었다.

        계획서 4.5 의 판정 기준 자체는 선택 전체에 porosity 가 하나 이상이면 되고 축별
        조건이 아니다. 이 fixture 는 세 축 모두에 결함 구간이 존재하도록 만들었으므로,
        목적함수로 고르면 세 축 모두 결함을 포함하는 구간이 선택된다.
        """
        defective = [
            row for row in self.plan_rows
            if row["modality"] == "CT"
            and row["product_status"] == "defective"
            and row["capture_set"] == "initial_capture"
        ]
        self.assertEqual(len(defective), 1450)
        self.assertEqual(int(defective[0]["original_battery_id"]), CT_DEFECT_ID)
        self.assertIn(
            "has_porosity",
            defective[0],
            "결함 분포를 검증하려면 plan 이 has_porosity 를 옮겨 실어야 한다",
        )
        axes_with_defect = {row["axis"] for row in defective if row["has_porosity"] == "1"}
        self.assertEqual(
            axes_with_defect,
            {"x", "y", "z"},
            "결함 ID 의 x/y/z 구간이 모두 결함 비율 기준으로 선택되어야 한다",
        )

    def test_rgb_defective_window_is_the_defect_capable_id(self) -> None:
        defective = [
            row for row in self.selected_rows
            if row["modality"] == "RGB" and row["product_status"] == "defective"
        ]
        self.assertEqual(len(defective), 1)
        self.assertEqual(int(defective[0]["original_battery_id"]), RGB_DEFECT_ID)

    def test_objective_function_is_exposed_without_arbitrary_scale(self) -> None:
        """F-06, 계획서 4.4.

        v1.2 의 정렬 키는 후보 구간의 결함 비율을 20 으로 나눈 뒤 모집단 비율과
        비교했다. 이 스케일에는 근거가 없고 목적함수를 무력화한다. 목적값은 별도
        함수로 노출되어야 검증과 보고가 가능하다.
        """
        objective = getattr(planner, "_objective", None)
        self.assertIsNotNone(objective, "계획서 4.4 의 목적함수가 _objective 로 노출되어야 한다")
        source = Path(planner.__file__).read_text(encoding="utf-8")
        self.assertNotIn(
            "/ 20 - population_ratio",
            source,
            "근거 없는 /20 스케일이 남아 있다",
        )

    def test_selected_ids_records_search_evidence(self) -> None:
        """계획서 4.4: 알고리즘, seed, 반복 횟수, 종료 조건, 목적값을 기록한다."""
        required = {
            "search_algorithm", "search_seed", "search_iterations",
            "search_stop_condition", "primary_objective", "secondary_objective",
        }
        missing = required - set(self.selected_rows[0])
        self.assertEqual(missing, set(), f"selected_ids.csv 에 선정 근거 컬럼이 없다: {sorted(missing)}")


class AugmentationAssignmentTests(PlannerFixture):
    def test_single_and_double_ratio_is_80_20(self) -> None:
        """계획서 6.1: 단일 정상 증강 80%, 2 개 조합 20%."""
        for modality in ("CT", "RGB"):
            rows = [
                row for row in self.plan_rows
                if row["modality"] == modality and row["capture_set"] == "initial_capture"
            ]
            doubles = sum(len(json.loads(row["base_augmentation_names"])) == 2 for row in rows)
            self.assertAlmostEqual(doubles / len(rows), 0.20, places=2, msg=modality)

    def test_ct_assigns_one_augmentation_block_per_id(self) -> None:
        """계획서 6.2: CT 정상 증강은 battery_id 전체를 하나의 block 으로 적용한다."""
        by_id = defaultdict(set)
        for row in self.plan_rows:
            if row["modality"] == "CT":
                by_id[row["output_battery_id"]].add(row["base_augmentation_names"])
        for battery_id, names in by_id.items():
            self.assertEqual(len(names), 1, f"CT {battery_id} 에 증강 종류가 여러 개 배정되었다")


class ReserveTests(PlannerFixture):
    def test_fail_ids_have_ranked_reserve_candidates(self) -> None:
        """F-02, 계획서 7.5.

        reserve 목록은 generation plan 에 미리 고정되어야 한다. 생성 시점에 즉석에서
        고르면 재현성이 없다.
        """
        fail_rows = [row for row in self.plan_rows if row["failure_case"]]
        self.assertTrue(fail_rows, "FAIL 행이 없다")
        for column in ("reserve_candidates", "reserve_rank", "reserve_original_stem", "reserve_reason"):
            self.assertIn(column, fail_rows[0], f"plan 에 {column} 이 없다")
        for row in fail_rows:
            candidates = json.loads(row["reserve_candidates"] or "[]")
            self.assertTrue(candidates, f"{row['sample_id']} 에 reserve 후보가 없다")
            self.assertEqual(
                [item["rank"] for item in candidates],
                list(range(1, len(candidates) + 1)),
                "reserve 후보의 우선순위가 연속된 정수가 아니다",
            )
            for item in candidates:
                self.assertTrue(item["original_stem"])
                self.assertTrue(item["reason"])

    def test_reserve_does_not_reuse_the_primary_failure_slots(self) -> None:
        """계획서 7.5: reserve 는 주 FAIL 구간과 겹치지 않는 대체 구간이어야 한다."""
        primary = {
            row["original_stem"] for row in self.plan_rows
            if row["failure_case"] and row["capture_set"] == "initial_capture"
        }
        for row in self.plan_rows:
            for item in json.loads(row["reserve_candidates"] or "[]"):
                if item["reason"] == "same-axis":
                    self.assertNotIn(item["original_stem"], primary)

    def test_fail_window_length_is_between_10_and_25(self) -> None:
        """계획서 7.3: FAIL 구간 길이 L 은 10~25 이다."""
        lengths = Counter()
        for row in self.plan_rows:
            if row["failure_segment_id"]:
                lengths[(row["modality"], row["original_battery_id"], row["failure_segment_id"])] += 1
        self.assertTrue(lengths)
        for key, length in lengths.items():
            self.assertGreaterEqual(length, 10, key)
            self.assertLessEqual(length, 25, key)


class ManifestSchemaTests(PlannerFixture):
    def test_schema_declares_the_full_column_set(self) -> None:
        """수정계획서 v1.3 의 5 장이 정의한 72 개다."""
        self.assertEqual(len(MANIFEST_COLUMNS), EXPECTED_COLUMN_COUNT)
        self.assertEqual(len(set(MANIFEST_COLUMNS)), EXPECTED_COLUMN_COUNT, "중복 컬럼이 있다")

    def test_plan_header_is_exactly_the_schema(self) -> None:
        self.assertEqual(list(self.plan_rows[0]), list(MANIFEST_COLUMNS))

    def test_plan_carries_every_column_the_manifest_needs(self) -> None:
        """F-04, 계획서 11.1.

        manifest 는 plan 행에 생성 결과를 덧붙여 만든다. 따라서 생성 단계에서만 정해지는
        컬럼을 뺀 나머지는 plan 시점에 이미 있어야 한다.
        """
        produced_at_generation = {
            "generation_status", "output_image_path", "output_json_path", "output_det_path",
            "output_seg_path", "output_image_sha256", "output_json_sha256", "output_det_sha256",
            "output_seg_sha256", "augmentation_json_path", "augmentation_json_sha256",
            "output_defect_count", "class_instance_counts", "failure_method_order",
            "failure_augmentation_parameters", "failure_artifact_mask_path", "pixel_hash",
            "normal_base_pixel_hash", "quality_gate_passed", "quality_gate_metrics",
            "exclusion_or_retry_reason", "generator_version", "jpeg_profile_id",
        }
        expected = set(MANIFEST_COLUMNS) - produced_at_generation
        missing = expected - set(self.plan_rows[0])
        self.assertEqual(missing, set(), f"plan 에 없는 컬럼 {len(missing)} 개: {sorted(missing)}")

    def test_index_gap_is_recorded(self) -> None:
        """계획서 13.4: 숫자 index 의 누락은 허용하고 기록한다."""
        self.assertIn("index_gap_before", self.plan_rows[0])
        self.assertIn("index_gap_size", self.plan_rows[0])


if __name__ == "__main__":
    unittest.main()
