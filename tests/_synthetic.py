"""planner 테스트를 위한 합성 scan cache 생성기.

실제 원본은 이미지 수십만 장이라 테스트에서 스캔할 수 없다. planner 는 cache 의
``pairs`` 테이블만 읽으므로, 같은 스키마의 SQLite 를 직접 만들어 넣는다.

여기서 만드는 모집단은 계획서 4.2 와 4.3 의 적격 조건을 최소한으로 넘기도록
구성한다. CT 는 ID 당 x 150 장, y 650 장, z 650 장이 필요하고 RGB 는 ID 당 250 장이
필요하므로, 각 축과 ID 에 그보다 조금 많은 pair 를 준다.

결함 ID 후보는 CT 101 번과 RGB 901 번 하나씩이며, 두 ID 모두 무결함 구간과 결함
구간을 동시에 가진다. 이렇게 두는 이유는 계획서 4.5 가 결함 ID 의 구간을 결함 비율
기준으로 고르라고 규정하는데, 무결함 구간이 함께 존재할 때만 그 규정을 어기는
구현을 잡아낼 수 있기 때문이다.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from server_sim_dataset.cache import SCHEMA


CT_DEFECT_ID = 101
# Realistic shortage: nine eligible normal sources must fill 19 output products.
CT_NORMAL_IDS = tuple(range(102, 111))

# 여러 라벨 조합을 만들어 서로 다른 Pollution/Damaged 담당 ID를 고를 수 있게 한다.
RGB_POLLUTION_ID = 901
RGB_MIXED_ID = 902
RGB_SPARE_ID = 903
RGB_DEFECT_IDS = (RGB_POLLUTION_ID, RGB_MIXED_ID, RGB_SPARE_ID)
RGB_DEFECT_ID = RGB_POLLUTION_ID
RGB_NORMAL_IDS = tuple(range(904, 922))

# 결함 ID 는 무결함 구간도 확보할 수 있을 만큼 pool 을 넉넉히 준다.
CT_DEFECT_POOL = {"x": 400, "y": 900, "z": 900}
CT_NORMAL_POOL = {"x": 450, "y": 1950, "z": 1950}
RGB_DEFECT_POOL = 300
RGB_NORMAL_POOL = 300

# 결함은 각 pool 의 앞쪽 구간에만 배치한다. 뒤쪽에는 무결함 구간이 남는다.
CT_DEFECT_SPAN = 120
RGB_DEFECT_SPAN = 120

OUTLINE = [10.0, 10.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0]


def _row(
    modality: str,
    battery_id: int,
    axis: str,
    index: int,
    *,
    porosity: bool = False,
    damaged: bool = False,
    pollution: bool = False,
) -> tuple[Any, ...]:
    stem = (
        f"CT_cell_pouch_{battery_id}_{axis}_{index:06d}"
        if modality == "CT"
        else f"RGB_cell_cylinder_{battery_id}_{index:06d}"
    )
    defects: list[dict[str, Any]] = []
    if porosity:
        defects.append({"name": "porosity", "points": [[20.0, 20.0], [40.0, 20.0], [40.0, 40.0]]})
    if damaged:
        defects.append({"name": "Damaged", "points": [[20.0, 20.0], [40.0, 20.0], [40.0, 40.0]]})
    if pollution:
        defects.append({"name": "Pollution", "points": [[50.0, 50.0], [70.0, 50.0], [70.0, 70.0]]})
    has_defect = bool(defects)
    return (
        "valid",
        "",
        "training",
        modality,
        battery_id,
        axis,
        index,
        stem,
        f"raw/{stem}.jpg",
        f"raw/{stem}.json",
        f"image-{stem}",
        f"json-{stem}",
        100,
        100,
        json.dumps([0, 0, 100, 100]),
        json.dumps(OUTLINE),
        json.dumps(defects),
        int(not has_defect),
        0.1 if porosity else 0.0,
        int(porosity),
        int(damaged),
        int(pollution),
        abs(hash(stem)) % 1_000_000,
        json.dumps([100, 100]) if modality == "CT" else "null",
        len(defects),
    )


def _ct_rows() -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for axis, count in CT_DEFECT_POOL.items():
        for index in range(count):
            rows.append(_row("CT", CT_DEFECT_ID, axis, index, porosity=index < CT_DEFECT_SPAN))
    for battery_id in CT_NORMAL_IDS:
        for axis, count in CT_NORMAL_POOL.items():
            for index in range(count):
                rows.append(_row("CT", battery_id, axis, index))
    return rows


def _rgb_rows() -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for index in range(RGB_DEFECT_POOL):
        # 5 프레임 중 4 개는 Pollution 단독, 1 개는 동시 결함.
        rows.append(_row("RGB", RGB_POLLUTION_ID, "", index,
                         pollution=True, damaged=index % 5 == 4))
    for index in range(RGB_DEFECT_POOL):
        # 짝수는 동시 결함, 홀수는 Damaged 단독.
        rows.append(_row("RGB", RGB_MIXED_ID, "", index,
                         damaged=True, pollution=index % 2 == 0))
    for index in range(RGB_DEFECT_POOL):
        rows.append(_row("RGB", RGB_SPARE_ID, "", index, pollution=True))
    for battery_id in RGB_NORMAL_IDS:
        for index in range(RGB_NORMAL_POOL):
            rows.append(_row("RGB", battery_id, "", index))
    return rows


def build_synthetic_cache(path: Path) -> Path:
    """계획서 적격 조건을 만족하는 최소 규모의 합성 cache 를 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)",
            [("cache_version", "1"), ("raw_root", str(path.parent)), ("label_source", "extracted-json-only")],
        )
        connection.executemany(
            "INSERT INTO pairs("
            "status,exclusion_reason,source_split,modality,battery_id,axis,original_index,original_stem,"
            "image_relative_path,json_relative_path,image_sha256,json_sha256,width,height,roi_json,"
            "outline_json,defects_json,original_is_normal,porosity_bbox_max_ratio,has_porosity,"
            "has_damaged,has_pollution,original_image_id,original_roi_json,defect_count) "
            "VALUES(" + ",".join("?" * 25) + ")",
            _ct_rows() + _rgb_rows(),
        )
        connection.commit()
    finally:
        connection.close()
    return path
