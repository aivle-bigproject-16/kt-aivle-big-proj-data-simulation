"""생성계획서 본문에 특정 문구가 남아 있는지만 확인하는 도구.

이름과 달리 구현이 그 문구를 지키는지는 검사하지 않는다. 계획서를 개정하다가 합의된
조항이 통째로 사라지는 사고를 막는 용도다. 구현이 계획서를 지키는지는 tests/ 의 회귀
테스트가 본다. v1.2 에서 이 도구가 전부 통과하는 동안 결함 20 건이 그대로 있었다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_TEXT = {
    "ct_threshold_25": "porosity_bbox_max_ratio >= 0.25",
    "json_only_labels": "풀린 `.json`만 라벨 소스로 사용하며 TAR·ZIP 등 압축파일은 열거나 후보로 등록하지 않는다",
    "ct_format": "CT 입력은 4000×4000 grayscale JPEG",
    "rgb_format": "RGB 입력·출력은 1920×1080 RGB PNG",
    "ct_roi_crop": "최종 CT도 해당 ROI 크기의",
    "ct_roi_coordinates": "[0, 0, output_width, output_height]",
    "outline_path": "swelling.battery_outline",
    "defect_path": "defects[].points",
    "json_battery": "data_info.battery_ids",
    "json_file_name": "image_info.file_name",
    "normal_pool": "normal_pair_pool",
    "deterministic_normal": "실행할 때마다 새로 무작위 추첨하지",
    "slice_seed": "slice_seed=stable_seed(id_seed, axis, original_index, augmentation_name)",
    "ct_output_ids": "1900000001~1900000100",
    "rgb_output_ids": "2900000001~2900000100",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    args = parser.parse_args()

    plan = args.plan.read_text(encoding="utf-8-sig")
    audit = json.loads(args.feasibility.read_text(encoding="utf-8-sig"))
    checks = {name: value in plan for name, value in REQUIRED_TEXT.items()}
    checks.update(
        {
            "raw_plan_feasible": audit.get("plan_feasible") is True,
            "raw_schema_errors_zero": audit.get("schema_error_count") == 0,
            "raw_hash_conflicts_zero": audit.get("duplicate_conflict_rows_excluded") == 0,
            "ct_normal_candidates": audit.get("CT", {}).get("normal_candidate_count", 0) >= 19,
            "ct_defective_candidates": audit.get("CT", {}).get("defective_candidate_count", 0) >= 1,
            "rgb_normal_candidates": audit.get("RGB", {}).get("normal_candidate_count", 0) >= 19,
            "rgb_defective_candidates": audit.get("RGB", {}).get("defective_candidate_count", 0) >= 1,
            "output_ids_disjoint": audit.get("output_id_ranges_disjoint") is True,
        }
    )
    failed = [name for name, passed in checks.items() if not passed]
    result = {"passed": not failed, "checks": checks, "failed": failed}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
