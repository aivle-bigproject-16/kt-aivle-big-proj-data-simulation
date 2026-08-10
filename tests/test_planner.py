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

from _synthetic import CT_DEFECT_ID, RGB_DEFECT_ID, build_synthetic_cache


# 계획서 11.1 의 60 개 컬럼에, 8.1 이 본문에서 요구하는 원본 추적 3 개와 7.5 의
# reserve 6 개, 9.1 의 JPEG profile 1 개, 6.2 의 slice 단위 seed 1 개를 더한 목록이다.
# 수정계획서 v1.3 의 5 장이 정본이며, 구현이 끝나면 schema.MANIFEST_COLUMNS 로 옮긴다.
MANIFEST_COLUMNS = (
    # 식별
    "sample_id", "synthetic_id", "capture_set", "capture_group_id", "retry_of_sample_id",
    "modality", "original_battery_id", "output_battery_id", "product_status", "axis",
    # 원본 순서
    "original_index", "source_sequence_order", "output_sequence_order", "source_split",
    # index 누락
    "index_gap_before", "index_gap_size",
    # 원본 추적
    "original_stem", "orig_image_relative_path", "orig_json_relative_path",
    "original_image_id", "original_image_file_name", "original_roi",
    # 원본 무결성
    "source_image_sha256", "source_json_sha256", "pixel_hash",
    # 제품 결함
    "original_is_normal", "has_porosity", "has_damaged", "has_pollution",
    # 객체 수
    "original_defect_count", "output_defect_count", "class_instance_counts",
    # 촬영 품질
    "capture_quality", "failure_case", "failure_segment_id", "failure_artifact_mask_path",
    # 생성 상태
    "generation_status", "exclusion_or_retry_reason",
    # 정상 증강
    "base_augmentation_names", "normal_augmentation_parameters", "normal_augmentation_seed",
    "slice_seed", "normal_base_pixel_hash",
    # FAIL 증강
    "failure_window_start", "failure_window_end", "failure_method_order",
    "failure_augmentation_parameters", "augmentation_json_path", "augmentation_json_sha256",
    # reserve
    "reserve_source_split", "reserve_original_battery_id", "reserve_original_index",
    "reserve_original_stem", "reserve_rank", "reserve_reason",
    # 재현성
    "global_seed", "item_seed", "generator_version", "config_hash", "plan_sha256",
    # 출력
    "output_image_path", "output_json_path", "output_det_path", "output_seg_path",
    "jpeg_profile_id",
    # 출력 무결성
    "output_image_sha256", "output_json_sha256", "output_det_sha256", "output_seg_sha256",
    # 품질 검증
    "quality_gate_passed", "quality_gate_metrics",
)


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

    def test_exactly_one_defective_id_per_modality(self) -> None:
        """계획서 4.5: ID 기준 제품불량 비율은 정확히 1/20 이다."""
        for modality in ("CT", "RGB"):
            statuses = Counter(
                row["product_status"] for row in self.selected_rows if row["modality"] == modality
            )
            self.assertEqual(statuses["defective"], 1, modality)
            self.assertEqual(statuses["normal"], 19, modality)


class SelectionTests(PlannerFixture):
    def test_ct_defective_window_covers_every_axis(self) -> None:
        """F-07, 계획서 4.5 의 4 항.

        결함 ID 의 구간은 처음부터 결함 비율 기준으로 골라야 한다. 무결함 구간을 먼저
        찾고 한 축만 사후 교체하면 결함이 한 축에 몰린다. v1.2 산출물에서 CT 결함 ID
        1,450 장 중 650 장에만, 그것도 한 축에만 porosity 가 있었던 것이 이 결함이다.
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
