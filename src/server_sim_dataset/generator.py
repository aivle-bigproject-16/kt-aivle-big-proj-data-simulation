from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import math
import platform
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, JpegImagePlugin

from . import __version__
from .planner import FAILURE_CASES, INITIAL_QUANTITIES, RECAPTURE_QUANTITIES
from .schema import (
    MANIFEST_COLUMNS,
    ct_axis_transform,
    iter_defects,
    output_stem,
    points,
    roi_bbox,
    sequence_length,
)
from .util import atomic_json, sha256_file, stable_seed


LOGGER = logging.getLogger(__name__)
CLASS_IDS = {"CT": {"porosity": 0}, "RGB": {"damaged": 0, "pollution": 1}}

# 계획서 6.2 의 연속 강도 envelope 제어점 수. 650 슬라이스 축에서도 인접 프레임 사이
# 변화가 전체 폭의 5% 안에 들어오도록 4 점으로 둔다.
_ENVELOPE_CONTROL_POINTS = 4
_SHADING_LIMIT = 0.08
# 계획서 6.3 의 이동·회전 고정 범위와 frame padding 조건.
_MAX_SHIFT = 5.0
_MAX_ROTATION = 5.0
_FRAME_PADDING = 0.01
_AFFINE_ATTEMPTS = 4
# 계획서 9.1: 원본 JPEG 조건을 재현할 수 없을 때 쓰는 승인된 공통 profile.
_COMMON_JPEG_PROFILE = "common-q95-s444"
_SOURCE_JPEG_PROFILE = "source-qtable"
# 계획서 7.5 의 고정 재시도 횟수. 이 횟수를 모두 쓴 뒤에만 reserve 로 넘어간다.
_FAILURE_ATTEMPTS = 8


def _library_versions() -> dict[str, str]:
    """계획서 7.7 이 요구하는 실행 환경 기록."""
    versions: dict[str, str] = {}
    for name in ("PIL", "numpy", "shapely", "psutil"):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:
            versions[name] = "unavailable"
    return versions


def _engine_provenance(engine: Any, engine_root: Path | None) -> dict[str, str]:
    """계획서 13.3 이 요구하는 v2.0 설정과 augment.py 의 SHA-256 기록."""
    if engine is None:
        return {"version": "not-loaded", "augment_sha256": ""}
    module_file = getattr(engine, "__file__", "")
    package = sys.modules.get(engine.__name__.split(".")[0])
    return {
        "version": str(getattr(package, "__version__", "unknown")),
        "augment_path": module_file,
        "augment_sha256": sha256_file(Path(module_file)) if module_file and Path(module_file).is_file() else "",
        "engine_root": str(engine_root) if engine_root else "",
    }


def _gamma(image: Image.Image, value: float) -> Image.Image:
    table = [round(255 * ((i / 255) ** value)) for i in range(256)]
    return image.point(table if image.mode == "L" else table * 3)


@dataclass(frozen=True)
class PlaneAffine:
    """중심 기준 회전과 평행이동으로 이루어진 평면 변환.

    이미지와 polygon 이 같은 변환을 받아야 하므로 정변환을 여기에 정의하고, 이미지에는
    그 역변환 계수를 넘긴다. PIL 의 transform 은 출력 좌표를 입력 좌표로 되돌리는
    역매핑을 받기 때문이다.
    """

    dx: float
    dy: float
    radians: float
    center: tuple[float, float]

    def apply_point(self, x: float, y: float) -> tuple[float, float]:
        cos, sin = math.cos(self.radians), math.sin(self.radians)
        cx, cy = self.center
        u, v = x - cx, y - cy
        return cos * u - sin * v + cx + self.dx, sin * u + cos * v + cy + self.dy

    def inverse_coefficients(self) -> tuple[float, float, float, float, float, float]:
        cos, sin = math.cos(self.radians), math.sin(self.radians)
        cx, cy = self.center
        tx, ty = cx + self.dx, cy + self.dy
        return (cos, sin, -cos * tx - sin * ty + cx, -sin, cos, sin * tx - cos * ty + cy)


@dataclass
class NormalResult:
    """정상 증강의 결과와 그 근거.

    계획서 6.3 은 이동·회전이 취소되거나 대체되었을 때 그 사유를 manifest 에 기록하라고
    규정한다. 이미지만 돌려주면 무엇이 실제로 적용되었는지 알 수 없다.
    """

    image: Image.Image
    flip_horizontal: bool = False
    flip_vertical: bool = False
    affine: PlaneAffine | None = None
    applied: list[str] = field(default_factory=list)
    retry_reason: str = ""


def _envelope(id_seed: int, axis: str, length: int, low: float, high: float) -> list[float]:
    """ID·축 단위로 한 번 생성하는 저주파 강도 곡선.

    계획서 6.2 는 음영·노이즈처럼 슬라이스별로 변해야 하는 효과를 인접 프레임 사이에서
    연속적으로 변화시키라고 규정한다. 슬라이스마다 독립 난수를 뽑으면 인접 프레임의
    강도가 불연속으로 튄다.
    """
    if length < 1:
        raise ValueError("envelope length must be positive")
    rng = np.random.Generator(np.random.PCG64(stable_seed(id_seed, axis, "envelope")))
    control = rng.uniform(low, high, size=_ENVELOPE_CONTROL_POINTS)
    positions = np.linspace(0.0, 1.0, num=_ENVELOPE_CONTROL_POINTS)
    return [float(value) for value in np.interp(np.linspace(0.0, 1.0, num=length), positions, control)]


def _fits(affine: PlaneAffine, geometry: list[list[tuple[float, float]]], width: int, height: int) -> bool:
    """변환 후에도 outline 과 모든 defect polygon 이 frame padding 안에 남는지 본다."""
    pad_x, pad_y = max(1.0, width * _FRAME_PADDING), max(1.0, height * _FRAME_PADDING)
    for polygon in geometry:
        for x, y in polygon:
            moved_x, moved_y = affine.apply_point(x, y)
            if not (pad_x <= moved_x <= width - 1 - pad_x and pad_y <= moved_y <= height - 1 - pad_y):
                return False
    return True


def _safe_affine(
    image: Image.Image,
    rng: np.random.Generator,
    geometry: list[list[tuple[float, float]]] | None,
) -> tuple[PlaneAffine | None, str]:
    """계획서 6.3 의 이동·회전 품질 게이트.

    고정 범위를 바로 적용하지 않고, outline 과 모든 defect polygon 으로 허용 범위를 먼저
    확인한다. 통과하지 못하면 같은 증강법의 강도를 낮춰 재시도하고, 그래도 실패하면
    이동·회전을 취소한다.
    """
    width, height = image.size
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    for attempt in range(_AFFINE_ATTEMPTS):
        scale = 0.5 ** attempt
        candidate = PlaneAffine(
            dx=float(rng.uniform(-_MAX_SHIFT, _MAX_SHIFT)) * scale,
            dy=float(rng.uniform(-_MAX_SHIFT, _MAX_SHIFT)) * scale,
            radians=math.radians(float(rng.uniform(-_MAX_ROTATION, _MAX_ROTATION)) * scale),
            center=center,
        )
        if geometry is None or _fits(candidate, geometry, width, height):
            return candidate, "" if attempt == 0 else f"reduced-strength-attempt-{attempt}"
    return None, "padding-gate-failed-replaced-with-optical"


def _normal(
    image: Image.Image,
    names: list[str],
    params: dict[str, float],
    slice_seed: int,
    id_seed: int,
    *,
    sequence: tuple[int, int] = (0, 1),
    axis: str = "",
    geometry: list[list[tuple[float, float]]] | None = None,
) -> NormalResult:
    """정상 증강을 적용한다.

    seed 를 둘로 나눈다. 슬라이스마다 달라져야 하는 효과는 slice_seed 를 쓰고, ID 전체에
    동기 적용되어야 하는 효과는 id_seed 를 쓴다. v1.2 는 seed 하나만 받아 반전까지
    슬라이스마다 새로 뽑았고, 그 결과 CT 4,350 장의 볼륨 방향이 프레임마다 뒤바뀌었다.
    """
    rng = np.random.Generator(np.random.PCG64(slice_seed))
    result = NormalResult(image=image)
    position, length = sequence
    position = min(max(position, 0), max(0, length - 1))
    for name in names:
        out = result.image
        if name == "brightness_contrast_gamma":
            out = _gamma(ImageEnhance.Contrast(ImageEnhance.Brightness(out).enhance(params["brightness"])).enhance(params["contrast"]), params["gamma"])
        elif name in {"normal_noise_poisson", "poisson_noise"}:
            base = params["noise_sigma"]
            sigma = _envelope(id_seed, f"{axis}|noise", length, base * 0.75, base * 1.25)[position]
            array = np.asarray(out).astype(np.float32) / 255
            noisy = rng.poisson(np.clip(array, 0, 1) * 180) / 180 + rng.normal(0, sigma, array.shape)
            out = Image.fromarray(np.uint8(np.clip(noisy, 0, 1) * 255), mode=out.mode)
        elif name in {"low_frequency_shading", "low_frequency_lighting"}:
            amplitude = _envelope(id_seed, f"{axis}|shading", length, -_SHADING_LIMIT, _SHADING_LIMIT)[position]
            width, height = out.size
            field_x = np.linspace(-1, 1, width)
            profile = 1 + amplitude * field_x
            array = np.asarray(out).astype(np.float32) * (profile[None, :] if out.mode == "L" else profile[None, :, None])
            out = Image.fromarray(np.uint8(np.clip(array, 0, 255)), mode=out.mode)
        elif name in {"weak_reconstruction_kernel", "weak_reconstruction"}:
            out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.25, 0.65))))
        elif name in {"partial_histogram_blend", "percentile_tone_curve", "rgb_channel_gain_tone"}:
            equalized = ImageOps.equalize(out)
            out = Image.blend(out, equalized, float(rng.uniform(0.05, 0.15)))
        elif name == "synchronized_flip":
            # One ID-level 3-D reflection is projected onto this orthogonal view.
            transform = ct_axis_transform(id_seed, axis)
            result.flip_horizontal = transform.flip_horizontal
            result.flip_vertical = transform.flip_vertical
            if result.flip_horizontal:
                out = ImageOps.mirror(out)
            if result.flip_vertical:
                out = ImageOps.flip(out)
        elif name == "safe_translate_rotate":
            affine, reason = _safe_affine(out, rng, geometry)
            result.retry_reason = reason
            if affine is None:
                # 계획서 6.3 의 3 항: 적용 가능한 광학 증강으로 대체한다.
                out = ImageEnhance.Color(out).enhance(float(rng.uniform(0.96, 1.04)))
                result.applied.append("rgb_channel_gain_tone")
                result.image = out
                continue
            result.affine = affine
            out = out.transform(
                out.size,
                Image.AFFINE,
                affine.inverse_coefficients(),
                resample=Image.BICUBIC,
                fillcolor=out.getpixel((0, 0)),
            )
        result.applied.append(name)
        result.image = out
    if not result.applied:
        raise ValueError("계획서 6.3 은 모든 PASS 이미지에 최소 1 개 정상 증강을 요구한다")
    return result


def _transform_polygon(polygon: list[tuple[float,float]], width: int, height: int, *, offset=(0,0), flip_x=False, flip_y=False, affine=None) -> list[tuple[float,float]]:
    result=[]
    for x,y in polygon:
        x-=offset[0]; y-=offset[1]
        # PIL 의 mirror·flip 은 픽셀을 W-1-x, H-1-y 로 옮긴다. W-x 로 옮기면 반전된 모든
        # 슬라이스에 1 픽셀 계통 오차가 남는다.
        if flip_x: x=width-1-x
        if flip_y: y=height-1-y
        if affine is not None: x,y=affine.apply_point(x,y)
        x=min(max(x,0.0),float(width)); y=min(max(y,0.0),float(height)); result.append((x,y))
    if len({(round(x,4),round(y,4)) for x,y in result}) < 3: raise ValueError("polygon vanished after transform")
    return result


def _replace_points(container: Any, polygon: list[tuple[float,float]]) -> Any:
    if isinstance(container, list) and (not container or isinstance(container[0], (int,float))):
        return [coordinate for point in polygon for coordinate in point]
    return [[x,y] for x,y in polygon]


def _update_annotations(payload: dict[str,Any], width:int, height:int, *, offset=(0,0), flip_x=False, flip_y=False, affine=None) -> list[tuple[str,list[tuple[float,float]]]]:
    swelling=payload.get("swelling") or {}; original=swelling.get("battery_outline")
    if original:
        polygon=_transform_polygon(points(original),width,height,offset=offset,flip_x=flip_x,flip_y=flip_y,affine=affine)
        swelling["battery_outline"]=_replace_points(original,polygon); payload["swelling"]=swelling
    defects=payload.get("defects") or []
    iterable=defects if isinstance(defects,list) else defects.get("items",[])
    for defect in iterable:
        for key in ("points","polygon","segmentation"):
            if key in defect and defect[key]:
                polygon=_transform_polygon(points(defect[key]),width,height,offset=offset,flip_x=flip_x,flip_y=flip_y,affine=affine)
                defect[key]=_replace_points(defect[key],polygon); break
    return list(iter_defects(payload))


def _shift(polygon: list[tuple[float,float]], offset: tuple[int,int]) -> list[tuple[float,float]]:
    return [(x-offset[0], y-offset[1]) for x,y in polygon]


def _save_jpeg(image: Image.Image, path: Path, quantization: Any, subsampling: int) -> str:
    """계획서 9.1 대로 저장하고 실제로 쓴 profile ID 를 돌려준다.

    원본 조건을 재현할 수 없으면 승인된 공통 profile 로 내려가되, 어느 쪽을 썼는지
    manifest 에 남겨야 한다. 조용히 내려가면 산출물마다 압축 조건이 달라진 사실을
    사후에 알 수 없다.
    """
    if quantization:
        try:
            image.save(path, format="JPEG", qtables=quantization, subsampling=subsampling)
            return _SOURCE_JPEG_PROFILE
        except (OSError, ValueError) as exc:
            LOGGER.warning("Source qtable rejected for %s (%s); using common profile", path.name, exc)
    image.save(path, format="JPEG", quality=95, subsampling=0)
    return _COMMON_JPEG_PROFILE


def _mask(size: tuple[int,int], polygons: list[list[tuple[float,float]]]) -> Image.Image:
    result=Image.new("L",size,0); draw=ImageDraw.Draw(result)
    for polygon in polygons: draw.polygon(polygon,fill=255)
    return result


def _failure_engine(engine_root: Path | None):
    if engine_root:
        source=engine_root/"src"
        if not source.is_dir(): raise FileNotFoundError(f"v2.0 engine src not found: {source}")
        sys.path.insert(0,str(source))
    try: return importlib.import_module("quality_fail_augment.augment")
    except ImportError as exc: raise RuntimeError("FAIL samples require quality-fail-augment v2.0; pass --engine-root") from exc


def _labels(path_det:Path,path_seg:Path,defects:list[tuple[str,list[tuple[float,float]]]],modality:str,width:int,height:int)->None:
    det=[]; seg=[]
    for name,polygon in defects:
        class_id=CLASS_IDS[modality].get(name.lower())
        if class_id is None: continue
        xs=[p[0] for p in polygon]; ys=[p[1] for p in polygon]
        left,right,top,bottom=min(xs),max(xs),min(ys),max(ys)
        det.append(f"{class_id} {(left+right)/(2*width):.8f} {(top+bottom)/(2*height):.8f} {(right-left)/width:.8f} {(bottom-top)/height:.8f}")
        seg.append(str(class_id)+" "+" ".join(f"{value:.8f}" for p in polygon for value in (p[0]/width,p[1]/height)))
    path_det.write_text("\n".join(det)+("\n" if det else ""),encoding="utf-8")
    path_seg.write_text("\n".join(seg)+("\n" if seg else ""),encoding="utf-8")


@dataclass
class Rendered:
    """한 소스를 읽어 정상 증강까지 끝낸 상태."""

    image: Image.Image
    payload: dict[str,Any]
    defects: list[tuple[str,list[tuple[float,float]]]]
    normal: NormalResult
    outline: list[tuple[float,float]]
    quantization: Any = None
    subsampling: int = -1


def _pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _polygon_area(polygon: list[tuple[float,float]]) -> float:
    total = 0.0
    for (x1,y1),(x2,y2) in zip(polygon, polygon[1:] + polygon[:1]):
        total += x1*y2 - x2*y1
    return abs(total) / 2


def _measurements(before: Image.Image, after: Image.Image, defects_before: list, defects_after: list) -> dict[str,float]:
    """계획서 7.7 의 automatic_checks.measurements.

    엔진은 severity 와 변환 파라미터만 돌려주므로 적용 전후의 실측값은 여기서 잰다.
    """
    first = np.asarray(before.convert("L"), dtype=np.float32)
    second = np.asarray(after.convert("L"), dtype=np.float32)
    area_before = sum(_polygon_area(polygon) for _,polygon in defects_before)
    area_after = sum(_polygon_area(polygon) for _,polygon in defects_after)
    return {
        "mean_luminance_delta": round(float(second.mean() - first.mean()), 6),
        "std_ratio": round(float(second.std() / first.std()) if first.std() else 0.0, 6),
        "defect_area_retention": round(area_after / area_before, 6) if area_before else 1.0,
    }


def _render(raw_root: Path, row: dict[str,str], source: dict[str,Any], *, strict_source_hash: bool) -> Rendered:
    """소스 하나를 읽어 ROI crop 과 정상 증강까지 적용한다.

    reserve 로 교체하면 소스 이미지 자체가 바뀌므로 이 경로를 다시 타야 한다. 주 구간과
    reserve 가 같은 코드를 쓰지 않으면 두 경로의 증강 결과가 어긋난다.
    """
    modality = row["modality"]
    image_path = (raw_root / source["orig_image_relative_path"]).resolve()
    json_path = (raw_root / source["orig_json_relative_path"]).resolve()
    if raw_root not in image_path.parents or raw_root not in json_path.parents:
        raise ValueError("Plan path escapes raw root")
    if strict_source_hash and (
        sha256_file(image_path) != source["source_image_sha256"]
        or sha256_file(json_path) != source["source_json_sha256"]
    ):
        raise ValueError(f"Source hash mismatch: {row['sample_id']}")
    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    quantization = None
    subsampling = -1
    with Image.open(image_path) as handle:
        if modality == "CT" and handle.format == "JPEG":
            # 계획서 9.1: 원본 JPEG 의 quantization table 과 subsampling 을 재사용한다.
            quantization = getattr(handle, "quantization", None)
            try: subsampling = JpegImagePlugin.get_sampling(handle)
            except Exception: subsampling = -1
        image = handle.convert("L" if modality == "CT" else "RGB")
    offset = (0,0)
    if modality == "CT":
        roi = roi_bbox(payload, *image.size); offset = (roi[0], roi[1]); image = image.crop(roi)
    names = json.loads(row["base_augmentation_names"])
    params = json.loads(row["normal_augmentation_parameters"])
    # 계획서 6.3 의 이동·회전 게이트는 출력 좌표계의 outline 과 defect polygon 으로 판정한다.
    geometry = [
        _shift(polygon, offset) for polygon in
        [points((payload.get("swelling") or {}).get("battery_outline"))] + [p for _,p in iter_defects(payload)]
        if polygon
    ]
    normal = _normal(
        image, names, params,
        int(row.get("slice_seed") or row["normal_augmentation_seed"]),
        int(row["normal_augmentation_seed"]),
        sequence=(int(row.get("source_sequence_order") or 0), sequence_length(modality, row["axis"])),
        axis=row["axis"], geometry=geometry or None,
    )
    width, height = normal.image.size
    defects = _update_annotations(
        payload, width, height, offset=offset,
        flip_x=normal.flip_horizontal, flip_y=normal.flip_vertical, affine=normal.affine,
    )
    outline = points((payload.get("swelling") or {}).get("battery_outline"))
    return Rendered(normal.image, payload, defects, normal, outline, quantization, subsampling)


def _apply_failure(engine: Any, rendered: Rendered, row: dict[str,str]) -> dict[str,Any] | None:
    """계획서 7.5 의 고정 재시도. 8 회 모두 실패하면 None 을 돌려 reserve 로 넘긴다.

    증강 종류나 강도 범위를 임의로 완화해 통과시키지 않는다.
    """
    defect_polygons = [polygon for _,polygon in rendered.defects]
    before = rendered.image
    last_error: Exception | None = None
    for attempt in range(_FAILURE_ATTEMPTS):
        try:
            result = engine.apply_failure_case(
                before, row["modality"], row["failure_case"],
                stable_seed(row["item_seed"], "attempt", attempt),
                _mask(before.size, [rendered.outline]),
                _mask(before.size, defect_polygons) if defect_polygons else None,
            )
        except (ValueError, RuntimeError) as exc:
            last_error = exc
            continue
        # 엔진이 크기를 바꾸는 case 가 있으므로 적용 후 실제 크기를 다시 읽는다.
        width, height = result.image.size
        defects = _update_annotations(rendered.payload, width, height, affine=result.transform)
        return {
            "attempt": attempt,
            "result": result,
            "defects": defects,
            "width": width,
            "height": height,
            "measurements": _measurements(before, result.image, rendered.defects, defects),
        }
    LOGGER.warning(
        "FAIL gate exhausted %d attempts for %s: %s", _FAILURE_ATTEMPTS, row["sample_id"], last_error
    )
    return None


def _completed(output: Path) -> dict[str, dict[str, str]]:
    """이미 만들어져 검증까지 통과한 행을 돌려준다.

    v1.2 의 --resume 은 빈 디렉터리 검사만 껐고 실제로는 전량을 다시 만들면서 manifest 를
    덮어썼다. 이름과 동작이 어긋나 위험했다. 여기서는 출력 4 종이 모두 존재하고 해시가
    일치하는 행만 재사용한다.
    """
    manifest_path = output / "manifests" / "dataset_manifest.csv"
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    usable: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("generation_status") != "success":
            continue
        for column, digest in (
            ("output_image_path", "output_image_sha256"), ("output_json_path", "output_json_sha256"),
            ("output_det_path", "output_det_sha256"), ("output_seg_path", "output_seg_sha256"),
        ):
            path = output / row[column]
            if not path.is_file() or sha256_file(path) != row[digest]:
                break
        else:
            usable[row["sample_id"]] = row
    return usable


_WORKER_ENGINE: Any = None


def _engine_for(engine_root: Path | None) -> Any:
    """프로세스마다 엔진을 한 번만 import 한다."""
    global _WORKER_ENGINE
    if _WORKER_ENGINE is None:
        _WORKER_ENGINE = _failure_engine(engine_root)
    return _WORKER_ENGINE


def _generate_one(task: tuple[Any, ...]) -> tuple[dict[str, Any], int, int]:
    """plan 행 하나를 산출물로 만든다.

    프로세스 풀에서 실행하므로 모듈 최상위에 둔다. 행끼리 상태를 공유하지 않고 seed 가
    전부 plan 에 박혀 있으므로 병렬로 돌려도 결과는 같다.
    """
    raw_root, output, row, plan_hash, engine_root, strict_source_hash = task
    modality = row["modality"]
    rendered = _render(raw_root, row, row, strict_source_hash=strict_source_hash)
    normal = rendered.normal
    payload = rendered.payload
    defects = rendered.defects
    image = rendered.image
    width, height = image.size
    normal_base_pixel_hash = _pixel_hash(image)
    augmentation = {
        "normal_augmentations": json.loads(row["base_augmentation_names"]),
        "applied_augmentations": normal.applied,
        "normal_parameters": json.loads(row["normal_augmentation_parameters"]),
        "seed": int(row["normal_augmentation_seed"]),
        "slice_seed": int(row.get("slice_seed") or row["normal_augmentation_seed"]),
        "flip": {
            # Keep x/y aliases for existing sidecar consumers; these are image
            # coordinates, not global CT X/Y coordinates.
            "x": normal.flip_horizontal,
            "y": normal.flip_vertical,
            "horizontal": normal.flip_horizontal,
            "vertical": normal.flip_vertical,
        },
        "retry_reason": normal.retry_reason,
        "failure_case": "",
        "automatic_checks": {"passed": True, "measurements": {}},
    }
    used_reserve: dict[str, Any] | None = None
    artifact_mask_path = ""
    reserve_hit = 0
    retries = 0
    if row["failure_case"]:
        engine = _engine_for(engine_root)
        planned_failure_case = row["failure_case"]
        fallback_cases = sorted(
            (case for case in FAILURE_CASES[modality] if case != planned_failure_case),
            key=lambda case: stable_seed(row["item_seed"], "failure-case-fallback", case),
        )
        applied_failure_case = planned_failure_case
        outcome = None
        primary_rendered = rendered
        reserve_candidates = json.loads(row.get("reserve_candidates") or "[]")
        for failure_case in (planned_failure_case, *fallback_cases):
            attempt_row = dict(row, failure_case=failure_case)
            for candidate in (None, *reserve_candidates):
                rendered = (
                    primary_rendered if candidate is None
                    else _render(raw_root, row, candidate, strict_source_hash=strict_source_hash)
                )
                outcome = _apply_failure(engine, rendered, attempt_row)
                if outcome is None:
                    retries += _FAILURE_ATTEMPTS
                    continue
                retries += outcome["attempt"]
                applied_failure_case = failure_case
                if candidate is not None:
                    used_reserve = candidate
                    reserve_hit = 1
                    LOGGER.info(
                        "Reserve rank %s used for %s (%s)",
                        candidate["rank"], row["sample_id"], candidate["reason"],
                    )
                if failure_case != planned_failure_case:
                    LOGGER.warning(
                        "Failure case fallback for %s: %s -> %s",
                        row["sample_id"], planned_failure_case, failure_case,
                    )
                payload = rendered.payload
                normal = rendered.normal
                augmentation.update({
                    "applied_augmentations": normal.applied,
                    "retry_reason": normal.retry_reason,
                    "flip": {
                        "x": normal.flip_horizontal,
                        "y": normal.flip_vertical,
                        "horizontal": normal.flip_horizontal,
                        "vertical": normal.flip_vertical,
                    },
                })
                normal_base_pixel_hash = _pixel_hash(rendered.image)
                break
            if outcome is not None:
                break
        if outcome is None:
            raise RuntimeError(f"FAIL quality gate exhausted every reserve and fallback case for {row['sample_id']}")
        result = outcome["result"]
        defects = outcome["defects"]
        image = result.image
        width, height = outcome["width"], outcome["height"]
        augmentation.update({
            "failure_case": applied_failure_case,
            "planned_failure_case": planned_failure_case,
            "applied_failure_case": applied_failure_case,
            "failure_attempt": outcome["attempt"],
            "transforms": result.records,
            "severity": max((record.get("severity", 0.0) for record in result.records), default=0.0),
            "automatic_checks": {"passed": True, "measurements": outcome["measurements"]},
        })
        if getattr(result, "object_mask", None) is not None:
            mask_dir = output / row["capture_set"] / modality / "failure_masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_file = mask_dir / ((row.get("synthetic_id") or row["sample_id"]) + ".mask.png")
            result.object_mask.save(mask_file, format="PNG")
            artifact_mask_path = mask_file.relative_to(output).as_posix()

    stem = row.get("synthetic_id") or output_stem(
        row["capture_set"], modality, int(row["output_battery_id"]), row["axis"], int(row["original_index"])
    )
    base = output / row["capture_set"] / modality
    for folder in ("images", "json", "labels_det", "labels_seg", "augmentation_json"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    image_path = base / "images" / (stem + (".jpg" if modality == "CT" else ".png"))
    json_path = base / "json" / (stem + ".json")
    det_path = base / "labels_det" / (stem + ".txt")
    seg_path = base / "labels_seg" / (stem + ".txt")
    jpeg_profile = ""
    if modality == "CT":
        jpeg_profile = _save_jpeg(image, image_path, rendered.quantization, rendered.subsampling)
    else:
        image.save(image_path, format="PNG")
    info = payload.setdefault("image_info", {})
    data = payload.setdefault("data_info", {})
    data["battery_ids"] = int(row["output_battery_id"])
    info.update({
        "id": stable_seed(row["sample_id"]), "file_name": image_path.name,
        "width": width, "height": height, "is_normal": not bool(defects),
    })
    if modality == "CT":
        data["roi"] = [0, 0, width, height]
    atomic_json(json_path, payload)
    _labels(det_path, seg_path, defects, modality, width, height)
    output_image_sha = sha256_file(image_path)
    augmentation["output_sha256"] = output_image_sha
    aug_path = base / "augmentation_json" / (stem + ".augmentation.json")
    atomic_json(aug_path, augmentation)
    checks = augmentation["automatic_checks"]
    transforms = augmentation.get("transforms", [])
    result_row = dict(row)
    retry_reasons = [reason for reason in (normal.retry_reason,) if reason]
    if row["failure_case"] and applied_failure_case != planned_failure_case:
        retry_reasons.append(f"failure_case_fallback:{planned_failure_case}->{applied_failure_case}")
    result_row.update({
        "generation_status": "success",
        "jpeg_profile_id": jpeg_profile,
        "exclusion_or_retry_reason": ";".join(retry_reasons),
        "failure_case": applied_failure_case if row["failure_case"] else "",
        "generator_version": __version__,
        "plan_sha256": plan_hash,
        "pixel_hash": _pixel_hash(image),
        "normal_base_pixel_hash": normal_base_pixel_hash,
        "output_defect_count": len(defects),
        "class_instance_counts": json.dumps(Counter(name for name, _ in defects), sort_keys=True),
        "failure_artifact_mask_path": artifact_mask_path,
        "failure_method_order": json.dumps([record.get("type", "") for record in transforms]),
        "failure_augmentation_parameters": json.dumps(transforms, ensure_ascii=False),
        "quality_gate_passed": str(bool(checks["passed"])).lower(),
        "quality_gate_metrics": json.dumps(checks["measurements"], sort_keys=True),
        "reserve_rank": used_reserve["rank"] if used_reserve else "",
        "reserve_reason": used_reserve["reason"] if used_reserve else "",
        "reserve_source_split": used_reserve["source_split"] if used_reserve else "",
        "reserve_original_battery_id": used_reserve["original_battery_id"] if used_reserve else "",
        "reserve_original_index": used_reserve["original_index"] if used_reserve else "",
        "reserve_original_stem": used_reserve["original_stem"] if used_reserve else "",
        "output_image_path": image_path.relative_to(output).as_posix(),
        "output_json_path": json_path.relative_to(output).as_posix(),
        "output_det_path": det_path.relative_to(output).as_posix(),
        "output_seg_path": seg_path.relative_to(output).as_posix(),
        "output_image_sha256": output_image_sha,
        "output_json_sha256": sha256_file(json_path),
        "output_det_sha256": sha256_file(det_path),
        "output_seg_sha256": sha256_file(seg_path),
        "augmentation_json_path": aug_path.relative_to(output).as_posix(),
        "augmentation_json_sha256": sha256_file(aug_path),
    })
    return result_row, reserve_hit, retries


def generate(
    raw_root: Path,
    plan_path: Path,
    output: Path,
    engine_root: Path | None = None,
    limit: int | None = None,
    resume: bool = False,
    strict_source_hash: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(f"Output directory is not empty: {output}")
    with plan_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[:limit]
    plan_hash = sha256_file(plan_path)
    done: dict[str, dict[str, str]] = {}
    if resume:
        done = _completed(output)
        if done:
            LOGGER.info("Resuming: %s rows already verified", f"{len(done):,}")
    pending = [row for row in rows if row["sample_id"] not in done]
    if any(row["failure_case"] for row in pending):
        # 계획서 13.3 은 실행에 쓴 augment.py 의 SHA-256 을 남기라고 규정한다. 워커만
        # 엔진을 import 하면 부모가 그 값을 알 수 없으므로 여기서 한 번 확인한다.
        _engine_for(engine_root)
    tasks = [
        (raw_root, output, row, plan_hash, engine_root, strict_source_hash)
        for row in pending
    ]
    produced: list[dict[str, Any]] = []
    reserve_used = 0
    quality_gate_retries = 0
    if workers > 1 and tasks:
        pool = ProcessPoolExecutor(max_workers=workers)
        # map 은 입력 순서를 유지하므로 병렬로 돌려도 manifest 순서가 결정론적이다.
        results = pool.map(_generate_one, tasks, chunksize=8)
    else:
        pool = None
        results = map(_generate_one, tasks)
    started = time.perf_counter()
    try:
        for index, (result_row, reserve_hit, retries) in enumerate(results, 1):
            produced.append(result_row)
            reserve_used += reserve_hit
            quality_gate_retries += retries
            if index % 200 == 0 or index == len(tasks):
                rate = index / max(1e-9, time.perf_counter() - started)
                LOGGER.info(
                    "Generation %s/%s (%.2f%%) | %.1f img/s | eta %.1f min | reserve %d | retries %d",
                    f"{index:,}", f"{len(tasks):,}", 100 * index / len(tasks),
                    rate, (len(tasks) - index) / rate / 60, reserve_used, quality_gate_retries,
                )
    finally:
        if pool is not None:
            pool.shutdown()
    manifest = [done[row["sample_id"]] if row["sample_id"] in done else None for row in rows]
    iterator = iter(produced)
    manifest = [item if item is not None else next(iterator) for item in manifest]
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if manifest:
        with (manifests / "dataset_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
            writer.writeheader()
            writer.writerows(manifest)
    summary = {
        "generator_version": __version__,
        "planned": len(rows),
        "succeeded": len(manifest),
        "regenerated": len(produced),
        "reused": len(rows) - len(produced),
        "reserve_used": reserve_used,
        "quality_gate_retries": quality_gate_retries,
        "workers": workers,
        "plan_sha256": plan_hash,
        "config_hash": rows[0].get("config_hash", "") if rows else "",
        "label_source": "extracted-json-only",
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": _library_versions(),
        "failure_engine": _engine_provenance(_WORKER_ENGINE, engine_root),
    }
    atomic_json(output / "generation_summary.json", summary)
    atomic_json(manifests / "generation_summary.json", summary)
    return summary


EXPECTED_QUANTITIES = {
    **{("initial_capture", modality): count for modality, count in INITIAL_QUANTITIES.items()},
    **{("recapture", modality): count for modality, count in RECAPTURE_QUANTITIES.items()},
}


def verify(output:Path,plan_path:Path|None=None,expect_full:bool=False)->dict[str,int]:
    """계획서 13.1 과 13.5 의 검증.

    v1.2 의 verify 는 manifest 에 적힌 행만 순회했다. 생성이 중도에 끊겨 manifest 가
    짧아진 산출물도 그대로 통과한다. 계획 대비 수량과 orphan 파일, 출력 JSON 과 실제
    이미지의 일치까지 본다.
    """
    manifest_path=output/"manifests"/"dataset_manifest.csv"
    with manifest_path.open("r",encoding="utf-8-sig",newline="") as handle:
        rows=list(csv.DictReader(handle))
    errors:list[str]=[]
    expected_paths:set[str]=set()
    for row in rows:
        for column,hash_column in (("output_image_path","output_image_sha256"),("output_json_path","output_json_sha256"),("output_det_path","output_det_sha256"),("output_seg_path","output_seg_sha256")):
            path=output/row[column]
            expected_paths.add(row[column])
            if not path.is_file() or sha256_file(path)!=row[hash_column]: errors.append(f"{row['sample_id']}:{column}")
        payload=json.loads((output/row["output_json_path"]).read_text(encoding="utf-8-sig")) if (output/row["output_json_path"]).is_file() else {}
        info=payload.get("image_info") or {}
        image_file=output/row["output_image_path"]
        if image_file.is_file():
            with Image.open(image_file) as handle_image:
                actual=handle_image.size
            if (info.get("width"),info.get("height"))!=actual:
                errors.append(f"{row['sample_id']}:json-size {info.get('width')}x{info.get('height')} != {actual[0]}x{actual[1]}")
            if info.get("file_name")!=image_file.name:
                errors.append(f"{row['sample_id']}:json-file-name")
            if row["modality"]=="CT" and (payload.get("data_info") or {}).get("roi")!=[0,0,actual[0],actual[1]]:
                errors.append(f"{row['sample_id']}:json-roi")
    if plan_path is not None:
        with plan_path.open("r",encoding="utf-8-sig",newline="") as handle:
            planned=sum(1 for _ in csv.DictReader(handle))
        if planned!=len(rows): errors.append(f"plan rows {planned} != manifest rows {len(rows)}")
    if expect_full:
        counts=Counter((row["capture_set"],row["modality"]) for row in rows)
        for key,expected in EXPECTED_QUANTITIES.items():
            if counts.get(key,0)!=expected:
                errors.append(f"quantity {key}: {counts.get(key,0)} != {expected}")
    produced:set[str]=set()
    for folder in ("images","json","labels_det","labels_seg"):
        for candidate in output.glob(f"*/*/{folder}/*"):
            if candidate.is_file(): produced.add(candidate.relative_to(output).as_posix())
    orphans=sorted(produced-expected_paths)
    if orphans: errors.append(f"orphan outputs {len(orphans)}: {orphans[:3]}")
    if errors: raise ValueError(f"Verification failed ({len(errors)}): {errors[:5]}")
    return {"samples":len(rows),"errors":0,"orphans":0}


def package_outputs(
    output: Path,
    plan_dir: Path | None = None,
    cache_path: Path | None = None,
    feasibility: Path | None = None,
    expect_full: bool = False,
) -> dict[str, int]:
    """계획서 10 의 17 개 ZIP 을 만든다.

    압축 전에 요약 파일을 먼저 생성하고, 압축 후에는 각 아카이브를 다시 열어 파일 수와
    stem 집합을 대조한다. 계획서 10 이 "ZIP 생성 후 각 ZIP 내부에서도 pair 와 파일 수를
    재검증한다"고 규정하는데 v1.2 에는 이 단계가 없었다.
    """
    plan_path = plan_dir / "generation_plan.csv" if plan_dir else None
    verify(output, plan_path if plan_path and plan_path.is_file() else None, expect_full)
    if plan_dir is not None and cache_path is not None:
        from .reports import build_reports
        build_reports(output, plan_dir, cache_path, feasibility)

    expected: dict[str, set[str]] = {}
    zip_count = 0
    for capture_set, prefix in (("initial_capture", "initial"), ("recapture", "recapture")):
        for modality in ("CT", "RGB"):
            for folder in ("images", "json", "labels_det", "labels_seg"):
                source = output / capture_set / modality / folder
                archive = output / f"{prefix}_{modality}_{folder}.zip"
                names: set[str] = set()
                # 이미지는 이미 JPEG/PNG 로 압축되어 있다. deflate 를 다시 걸면 용량은
                # 사실상 그대로인데 32GB 를 재압축하느라 시간만 든다.
                compression = zipfile.ZIP_STORED if folder == "images" else zipfile.ZIP_DEFLATED
                with zipfile.ZipFile(archive, "w", compression=compression, compresslevel=None if compression == zipfile.ZIP_STORED else 6) as handle:
                    for path in sorted(source.glob("*")):
                        if path.is_file():
                            handle.write(path, path.name)
                            names.add(path.stem)
                expected[archive.name] = names
                zip_count += 1

    with zipfile.ZipFile(output / "manifests.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted((output / "manifests").glob("*")):
            if path.is_file():
                handle.write(path, path.name)
        # 계획서 7.7 의 추적성 산출물. v1.2 는 만들고도 어느 ZIP 에도 넣지 않아
        # manifest 가 참조하는 파일이 배포본에 없었다.
        for path in sorted(output.glob("*/*/augmentation_json/*.json")):
            handle.write(path, f"augmentation_json/{path.name}")
        for path in sorted(output.glob("*/*/failure_masks/*.png")):
            handle.write(path, f"failure_masks/{path.name}")
    zip_count += 1

    mismatches = []
    for capture_set, prefix in (("initial_capture", "initial"), ("recapture", "recapture")):
        for modality in ("CT", "RGB"):
            stems = [expected[f"{prefix}_{modality}_{folder}.zip"] for folder in ("images", "json", "labels_det", "labels_seg")]
            if len({frozenset(item) for item in stems}) != 1:
                mismatches.append(f"{prefix}_{modality}: image/json/det/seg stem sets differ")
    for name, names in expected.items():
        with zipfile.ZipFile(output / name) as handle:
            if len(handle.namelist()) != len(names):
                mismatches.append(f"{name}: archive holds {len(handle.namelist())} of {len(names)}")
    if mismatches:
        raise ValueError(f"ZIP verification failed: {mismatches[:5]}")
    return {"zip_files": zip_count, "archives_verified": len(expected) + 1}
