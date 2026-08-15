"""FAIL 경로 회귀 테스트.

실제 v2.0 엔진 대신 stub 을 주입한다. 여기서 검증하려는 것은 엔진의 화질이 아니라
계획서 7.5 의 재시도·reserve 순서와 7.7 의 추적성, 8.1 의 크기 동기화이기 때문이다.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

from PIL import Image

from server_sim_dataset import generator
from server_sim_dataset.schema import MANIFEST_COLUMNS
from server_sim_dataset.util import sha256_file, stable_seed


class Identity:
    def apply_point(self, x: float, y: float) -> tuple[float, float]:
        return x, y


@dataclass
class StubResult:
    image: Image.Image
    transform: Identity = field(default_factory=Identity)
    records: list[dict[str, Any]] = field(default_factory=list)
    object_mask: Image.Image | None = None


class StubEngine:
    """앞의 fail_first 번 호출만 게이트에서 떨어지는 엔진."""

    __name__ = "stub_engine"

    def __init__(self, fail_first: int = 0, scale: float = 1.0, reject_cases: set[str] | None = None) -> None:
        self.fail_first = fail_first
        self.scale = scale
        self.reject_cases = reject_cases or set()
        self.calls = 0
        self.seeds: list[int] = []

    def apply_failure_case(self, image, modality, failure_case, seed, object_mask, defect_mask=None):
        self.calls += 1
        self.seeds.append(seed)
        if self.calls <= self.fail_first or failure_case in self.reject_cases:
            raise ValueError("quality gate rejected the attempt")
        size = (max(2, int(image.width * self.scale)), max(2, int(image.height * self.scale)))
        # 실제 엔진처럼 픽셀을 바꾼다. 그대로 돌려주면 정상 증강 결과와 구분되지 않는다.
        darkened = image.resize(size).point(lambda value: max(0, value - 20))
        return StubResult(
            image=darkened,
            records=[
                {"order": 0, "type": failure_case, "severity": 0.8, "parameters": {"gain": 1.2}},
                {"order": 1, "type": "noise", "severity": 0.5, "parameters": {"sigma": 0.01}},
            ],
        )


def _write_source(root: Path, stem: str, battery_id: int) -> tuple[Path, Path]:
    image_path = root / "Training" / "01.원천데이터" / f"{stem}.jpg"
    json_path = root / "Training" / "02.라벨링데이터" / f"{stem}.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 12), 120).save(image_path)
    payload = {
        "data_info": {"data_type": "ct", "battery_ids": battery_id, "roi": [8, 12]},
        "swelling": {"swelling": False, "battery_outline": [1, 1, 7, 1, 7, 10, 1, 10]},
        "defects": None,
        "image_info": {"width": 16, "height": 12, "file_name": image_path.name, "id": 7, "is_normal": True},
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return image_path, json_path


def _source_fields(root: Path, image_path: Path, json_path: Path) -> dict[str, Any]:
    return {
        "orig_image_relative_path": image_path.relative_to(root).as_posix(),
        "orig_json_relative_path": json_path.relative_to(root).as_posix(),
        "source_image_sha256": sha256_file(image_path),
        "source_json_sha256": sha256_file(json_path),
    }


def _fail_row(root: Path, primary: dict[str, Any], reserve: dict[str, Any] | None) -> dict[str, Any]:
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update({
        "sample_id": "S00000001",
        "synthetic_id": "initial_CT_1900000001_x_000001",
        "capture_group_id": "G1", "capture_set": "initial_capture", "modality": "CT",
        "original_battery_id": 101, "output_battery_id": 1900000001,
        "product_status": "normal", "axis": "x", "original_index": 1,
        "source_sequence_order": 0, "output_sequence_order": 0,
        "index_gap_before": 0, "index_gap_size": 0, "source_split": "training",
        "original_stem": "CT_cell_pouch_101_x_000001",
        "original_image_id": 7, "original_image_file_name": "CT_cell_pouch_101_x_000001.jpg",
        "original_roi": json.dumps([8, 12]),
        "original_is_normal": 1, "original_defect_count": 0,
        "capture_quality": "FAIL", "failure_case": "ct_low_signal_noise",
        "failure_segment_id": "CT-101-0-10",
        "failure_window_start": 0, "failure_window_end": 9,
        "base_augmentation_names": json.dumps(["brightness_contrast_gamma"]),
        "normal_augmentation_parameters": json.dumps({"brightness": 1.0, "contrast": 1.0, "gamma": 1.0, "noise_sigma": 0.003}),
        "normal_augmentation_seed": stable_seed("normal"), "slice_seed": stable_seed("slice"),
        "item_seed": stable_seed("item"), "global_seed": 20260723, "config_hash": "test",
    })
    row.update(primary)
    if reserve is not None:
        row["reserve_candidates"] = json.dumps([{
            "rank": 1, "reason": "same-axis", "source_split": "training",
            "original_battery_id": 101, "original_index": 400,
            "original_stem": "CT_cell_pouch_101_x_000400",
            **reserve,
        }])
    return row


class FailurePathTests(unittest.TestCase):
    def _run(self, engine: StubEngine, *, with_reserve: bool) -> tuple[Path, list[dict[str, str]]]:
        root = Path(self._temporary.name)
        raw = root / "raw"
        primary_image, primary_json = _write_source(raw, "CT_cell_pouch_101_x_000001", 101)
        reserve_fields = None
        if with_reserve:
            reserve_image, reserve_json = _write_source(raw, "CT_cell_pouch_101_x_000400", 101)
            reserve_fields = _source_fields(raw, reserve_image, reserve_json)
        row = _fail_row(raw, _source_fields(raw, primary_image, primary_json), reserve_fields)
        plan = root / "plan.csv"
        with plan.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
            writer.writeheader(); writer.writerow(row)
        output = root / "output"
        # _engine_for 는 프로세스마다 엔진을 한 번만 import 하므로, 테스트가 한 프로세스
        # 안에서 여러 stub 을 쓰려면 그 캐시를 함께 되돌려야 한다.
        with mock.patch.object(generator, "_failure_engine", return_value=engine), \
                mock.patch.object(generator, "_WORKER_ENGINE", None):
            generator.generate(raw, plan, output)
        with (output / "manifests" / "dataset_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            return output, list(csv.DictReader(handle))

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)

    def test_primary_source_is_used_when_the_gate_passes(self) -> None:
        engine = StubEngine()
        _, manifest = self._run(engine, with_reserve=True)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(manifest[0]["reserve_rank"], "")
        self.assertEqual(manifest[0]["quality_gate_passed"], "true")

    def test_reserve_is_used_only_after_eight_fixed_attempts(self) -> None:
        """계획서 7.5: 8 회 모두 실패한 경우에만 다음 reserve 소스로 이동한다."""
        engine = StubEngine(fail_first=8)
        _, manifest = self._run(engine, with_reserve=True)
        self.assertEqual(engine.calls, 9, "주 구간에서 정확히 8 회를 쓴 뒤 reserve 로 넘어가야 한다")
        self.assertEqual(len(set(engine.seeds[:8])), 8, "재시도 seed 가 서로 달라야 한다")
        self.assertEqual(manifest[0]["reserve_rank"], "1")
        self.assertEqual(manifest[0]["reserve_reason"], "same-axis")
        self.assertEqual(manifest[0]["reserve_original_stem"], "CT_cell_pouch_101_x_000400")

    def test_exhausting_every_reserve_stops_generation(self) -> None:
        engine = StubEngine(fail_first=100)
        with self.assertRaisesRegex(RuntimeError, "exhausted every reserve"):
            self._run(engine, with_reserve=True)

    def test_exhausted_case_falls_back_deterministically(self) -> None:
        engine = StubEngine(reject_cases={"ct_low_signal_noise"})
        _, manifest = self._run(engine, with_reserve=True)
        self.assertNotEqual(manifest[0]["failure_case"], "ct_low_signal_noise")
        self.assertTrue(manifest[0]["exclusion_or_retry_reason"].startswith("failure_case_fallback:"))
        self.assertEqual(manifest[0]["quality_gate_passed"], "true")

    def test_json_size_follows_the_engine_output(self) -> None:
        """B-03, 계획서 8.1: width/height 가 실제 이미지와 다르면 검증 실패다."""
        engine = StubEngine(scale=0.5)
        output, manifest = self._run(engine, with_reserve=False)
        produced = json.loads((output / manifest[0]["output_json_path"]).read_text(encoding="utf-8"))
        with Image.open(output / manifest[0]["output_image_path"]) as image:
            actual = image.size
        self.assertEqual((produced["image_info"]["width"], produced["image_info"]["height"]), actual)
        self.assertEqual(produced["data_info"]["roi"], [0, 0, actual[0], actual[1]])

    def test_augmentation_json_records_severity_and_transform_order(self) -> None:
        """계획서 7.7: failure_case, severity, 변환 순서, 실측값, seed, 출력 SHA-256."""
        engine = StubEngine()
        output, manifest = self._run(engine, with_reserve=False)
        record = json.loads((output / manifest[0]["augmentation_json_path"]).read_text(encoding="utf-8"))
        self.assertEqual(record["failure_case"], "ct_low_signal_noise")
        self.assertAlmostEqual(record["severity"], 0.8)
        self.assertEqual([item["type"] for item in record["transforms"]], ["ct_low_signal_noise", "noise"])
        self.assertTrue(record["output_sha256"])
        measurements = record["automatic_checks"]["measurements"]
        for key in ("mean_luminance_delta", "std_ratio", "defect_area_retention"):
            self.assertIn(key, measurements)
        self.assertEqual(
            json.loads(manifest[0]["failure_method_order"]), ["ct_low_signal_noise", "noise"]
        )
        self.assertTrue(json.loads(manifest[0]["quality_gate_metrics"]))

    def test_normal_base_pixel_hash_is_recorded_for_the_recapture_check(self) -> None:
        """계획서 13.5: 초기 촬영과 재촬영 pair 의 정상 기본 결과를 pixel hash 로 비교한다."""
        engine = StubEngine()
        _, manifest = self._run(engine, with_reserve=False)
        self.assertTrue(manifest[0]["normal_base_pixel_hash"])
        self.assertNotEqual(manifest[0]["normal_base_pixel_hash"], manifest[0]["pixel_hash"])


if __name__ == "__main__":
    unittest.main()
