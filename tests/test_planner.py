"""계획서 v1.5 planner 회귀 테스트."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from server_sim_dataset import planner
from server_sim_dataset.schema import MANIFEST_COLUMNS, ct_axis_transform

from _synthetic import (
    CT_DEFECT_ID,
    CT_NORMAL_IDS,
    build_synthetic_cache,
)


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
    def test_v15_configuration_records_id_targets_and_ct_coordinates(self) -> None:
        settings = planner._configuration(planner.GLOBAL_SEED)
        self.assertEqual(settings["defective_id_counts"], {"CT": 1, "RGB": 2})
        self.assertEqual(
            settings["ct_axis_coordinates"],
            {"x": ("X", "Y", "Z"), "y": ("Y", "X", "Z"), "z": ("Z", "X", "Y")},
        )

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

    def test_ct_uses_one_defective_and_nineteen_normal_ids(self) -> None:
        """계획서 v1.5 §4.1: CT 제품 상태를 ID 개수로 고정한다."""
        ids = [row for row in self.selected_rows if row["modality"] == "CT"]
        self.assertEqual(len(ids), 20)
        self.assertEqual(Counter(row["stratum"] for row in ids), Counter({"zero": 19, "low_mid": 1}))
        self.assertEqual(sum(1 for row in ids if row["product_status"] == "defective"), 1)

    def test_rgb_selects_one_pollution_carrier_and_one_damaged_carrier(self) -> None:
        defective = [
            row for row in self.selected_rows
            if row["modality"] == "RGB" and row["product_status"] == "defective"
        ]
        self.assertEqual(len(defective), 2)
        self.assertEqual({row["defect_role"] for row in defective}, {"pollution", "damaged"})
        self.assertEqual(len({row["original_battery_id"] for row in defective}), 2)
        for selected in defective:
            flag = "has_pollution" if selected["defect_role"] == "pollution" else "has_damaged"
            rows = [
                row for row in self.plan_rows
                if row["capture_set"] == "initial_capture"
                and row["output_battery_id"] == selected["output_battery_id"]
            ]
            self.assertTrue(any(row[flag] == "1" for row in rows))

    def test_rgb_uses_two_defective_output_ids(self) -> None:
        counts = Counter(
            (row["modality"], row["product_status"])
            for row in self.selected_rows
        )
        self.assertEqual(counts[("CT", "defective")], 1)
        self.assertEqual(counts[("CT", "normal")], 19)
        self.assertEqual(counts[("RGB", "defective")], 2)
        self.assertEqual(counts[("RGB", "normal")], 18)

    def test_capture_quality_is_stratified_across_product_status(self) -> None:
        """계획서 v1.5 §7: FAIL 대상을 제품 상태별로 하나씩 고른다.

        v1.3 산출물에는 촬영실패이면서 제품불량인 이미지가 한 장도 없었다. 두 축이
        독립이라고 규정해 놓고 교차 칸이 비면 그 조합을 학습에도 평가에도 쓸 수 없다.
        """
        for modality, flags in (("CT", ["has_porosity"]), ("RGB", ["has_damaged", "has_pollution"])):
            table = Counter(
                (row["capture_quality"], any(row[flag] == "1" for flag in flags))
                for row in self.plan_rows if row["modality"] == modality
            )
            for quality in ("PASS", "FAIL"):
                for defective in (True, False):
                    self.assertGreater(
                        table[(quality, defective)], 0,
                        f"{modality} {quality}/{'불량' if defective else '정상'} 칸이 비었다",
                    )

class SelectionTests(PlannerFixture):
    def test_ct_reuses_normal_sources_to_fill_nineteen_output_ids(self) -> None:
        normal = [
            row for row in self.selected_rows
            if row["modality"] == "CT" and row["product_status"] == "normal"
        ]
        self.assertEqual(len(normal), 19)
        self.assertEqual(len({row["original_battery_id"] for row in normal}), len(CT_NORMAL_IDS))
        self.assertEqual(len({row["output_battery_id"] for row in normal}), 19)
        self.assertTrue(any(int(row["source_reuse_total"]) > 1 for row in normal))

        initial = [
            row for row in self.plan_rows
            if row["modality"] == "CT"
            and row["product_status"] == "normal"
            and row["capture_set"] == "initial_capture"
        ]
        seeds_by_output = {
            row["output_battery_id"]: row["normal_augmentation_seed"] for row in initial
        }
        groups_by_output = defaultdict(set)
        signatures_by_source = defaultdict(set)
        for row in initial:
            groups_by_output[row["output_battery_id"]].add(row["capture_group_id"])
            signatures_by_source[row["original_battery_id"]].add((
                row["base_augmentation_names"],
                row["normal_augmentation_parameters"],
            ))
        self.assertEqual(len(set(seeds_by_output.values())), 19)
        self.assertEqual(sum(len(groups) for groups in groups_by_output.values()), len(initial))
        for source_id, signatures in signatures_by_source.items():
            output_count = len({
                row["output_battery_id"] for row in initial
                if row["original_battery_id"] == source_id
            })
            self.assertEqual(len(signatures), output_count)

    def test_reused_ct_source_uses_distinct_ordered_index_windows_with_gaps(self) -> None:
        initial = [
            row for row in self.plan_rows
            if row["modality"] == "CT"
            and row["product_status"] == "normal"
            and row["capture_set"] == "initial_capture"
        ]
        by_source = defaultdict(lambda: defaultdict(dict))
        for row in initial:
            axes = by_source[row["original_battery_id"]][row["output_battery_id"]]
            axes.setdefault(row["axis"], []).append(int(row["original_index"]))
        for outputs in by_source.values():
            if len(outputs) < 2:
                continue
            signatures = defaultdict(set)
            for axes in outputs.values():
                for axis, indexes in axes.items():
                    ordered = sorted(indexes)
                    self.assertTrue(all(right > left for left, right in zip(ordered, ordered[1:])))
                    signatures[axis].add(tuple(ordered))
            for axis, windows in signatures.items():
                self.assertEqual(len(windows), len(outputs), axis)

    def test_ct_index_gaps_match_the_raw_index_difference(self) -> None:
        groups = defaultdict(list)
        for row in self.plan_rows:
            if row["modality"] == "CT" and row["capture_set"] == "initial_capture":
                groups[(row["output_battery_id"], row["axis"])].append(row)
        observed_gap = False
        for key, rows in groups.items():
            ordered = sorted(rows, key=lambda row: int(row["source_sequence_order"]))
            previous = None
            for row in ordered:
                current = int(row["original_index"])
                gap_size = 0 if previous is None else max(0, current - previous - 1)
                self.assertEqual(int(row["index_gap_before"]), int(bool(gap_size)), key)
                self.assertEqual(int(row["index_gap_size"]), gap_size, key)
                observed_gap = observed_gap or bool(gap_size)
                previous = current
        self.assertTrue(observed_gap)

    def test_ordered_windows_accept_consecutive_and_gapped_indexes(self) -> None:
        helper = planner._ordered_distinct_windows
        consecutive = [{"original_index": index} for index in range(6)]
        mixed = [{"original_index": index} for index in (0, 1, 7, 9, 10, 19)]
        self.assertEqual(
            [[row["original_index"] for row in window] for window in helper(consecutive, 3, 2)],
            [[0, 1, 2], [3, 4, 5]],
        )
        self.assertEqual(
            [[row["original_index"] for row in window] for window in helper(mixed, 3, 2)],
            [[0, 1, 7], [9, 10, 19]],
        )

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

    def test_ct_slice_order_comes_from_the_same_3d_transform_as_plane_flips(self) -> None:
        rows = [
            row for row in self.plan_rows
            if row["modality"] == "CT"
            and row["capture_set"] == "initial_capture"
            and "synchronized_flip" in json.loads(row["base_augmentation_names"])
        ]
        self.assertTrue(rows)
        ids = {row["output_battery_id"] for row in rows}
        for battery_id in ids:
            for axis in ("x", "y", "z"):
                axis_rows = sorted(
                    (
                        row for row in rows
                        if row["output_battery_id"] == battery_id and row["axis"] == axis
                    ),
                    key=lambda row: int(row["source_sequence_order"]),
                )
                seed = int(axis_rows[0]["normal_augmentation_seed"])
                actual_reversed = (
                    int(axis_rows[0]["output_sequence_order"])
                    > int(axis_rows[-1]["output_sequence_order"])
                )
                self.assertEqual(actual_reversed, ct_axis_transform(seed, axis).reverse_slices)


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
        """v1.5가 유지하는 72개 기본 manifest 컬럼 계약이다."""
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
