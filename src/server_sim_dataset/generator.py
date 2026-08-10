from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import math
import platform
import sys
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, JpegImagePlugin

from . import __version__
from .schema import iter_defects, output_stem, points, roi_bbox, sequence_length
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
    flip_x: bool = False
    flip_y: bool = False
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
            # 계획서 6.2: 반전 여부와 방향은 ID 별로 한 번만 결정한다.
            result.flip_x = bool(id_seed & 1)
            result.flip_y = bool(id_seed & 2)
            if result.flip_x:
                out = ImageOps.mirror(out)
            if result.flip_y:
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


def generate(raw_root:Path,plan_path:Path,output:Path,engine_root:Path|None=None,limit:int|None=None,resume:bool=False)->dict[str,Any]:
    raw_root=raw_root.resolve(); output=output.resolve()
    if output.exists() and any(output.iterdir()) and not resume: raise ValueError(f"Output directory is not empty: {output}")
    with plan_path.open("r",encoding="utf-8-sig",newline="") as handle:
        rows=list(csv.DictReader(handle))
    if limit: rows=rows[:limit]
    plan_hash=sha256_file(plan_path)
    engine=None; manifest=[]; failures=0
    for index,row in enumerate(rows,1):
        modality=row["modality"]; source_image=(raw_root/row["orig_image_relative_path"]).resolve(); source_json=(raw_root/row["orig_json_relative_path"]).resolve()
        if raw_root not in source_image.parents or raw_root not in source_json.parents: raise ValueError("Plan path escapes raw root")
        if sha256_file(source_image)!=row["source_image_sha256"] or sha256_file(source_json)!=row["source_json_sha256"]: raise ValueError(f"Source hash mismatch: {row['sample_id']}")
        payload=json.loads(source_json.read_text(encoding="utf-8-sig"))
        quantization=None; subsampling=-1
        with Image.open(source_image) as source:
            if modality=="CT" and source.format=="JPEG":
                # 계획서 9.1: 원본 JPEG 의 quantization table 과 subsampling 을 재사용한다.
                quantization=getattr(source,"quantization",None)
                try: subsampling=JpegImagePlugin.get_sampling(source)
                except Exception: subsampling=-1
            image=source.convert("L" if modality=="CT" else "RGB")
        offset=(0,0)
        if modality=="CT":
            roi=roi_bbox(payload,*image.size); offset=(roi[0],roi[1]); image=image.crop(roi)
        names=json.loads(row["base_augmentation_names"]); params=json.loads(row["normal_augmentation_parameters"])
        slice_seed=int(row.get("slice_seed") or row["normal_augmentation_seed"])
        id_seed=int(row["normal_augmentation_seed"])
        # 계획서 6.3 의 이동·회전 게이트는 출력 좌표계의 outline 과 defect polygon 으로 판정한다.
        geometry=[_shift(polygon,offset) for polygon in
            [points((payload.get("swelling") or {}).get("battery_outline"))]+[p for _,p in iter_defects(payload)] if polygon]
        normal=_normal(image,names,params,slice_seed,id_seed,
            sequence=(int(row.get("source_sequence_order") or 0),sequence_length(modality,row["axis"])),
            axis=row["axis"],geometry=geometry or None)
        image=normal.image; width,height=image.size
        defects=_update_annotations(payload,width,height,offset=offset,flip_x=normal.flip_x,flip_y=normal.flip_y,affine=normal.affine)
        augmentation={"normal_augmentations":names,"applied_augmentations":normal.applied,
            "normal_parameters":params,"seed":id_seed,"slice_seed":slice_seed,
            "flip":{"x":normal.flip_x,"y":normal.flip_y},"retry_reason":normal.retry_reason,
            "failure_case":"","automatic_checks":{"passed":True}}
        if row["failure_case"]:
            if engine is None: engine=_failure_engine(engine_root)
            outline=points((payload.get("swelling") or {}).get("battery_outline")); defect_polygons=[polygon for _,polygon in defects]
            last_error=None
            for attempt in range(8):
                try:
                    attempt_seed=stable_seed(row["item_seed"],"attempt",attempt)
                    result=engine.apply_failure_case(image,modality,row["failure_case"],attempt_seed,_mask(image.size,[outline]),_mask(image.size,defect_polygons) if defect_polygons else None)
                    augmentation["failure_attempt"]=attempt
                    break
                except (ValueError,RuntimeError) as exc:
                    last_error=exc
            else:
                raise RuntimeError(f"FAIL quality gate exhausted 8 fixed-range attempts for {row['sample_id']}: {last_error}")
            image=result.image; defects=_update_annotations(payload,width,height,affine=result.transform)
            augmentation.update({"failure_case":row["failure_case"],"transforms":result.records})
        stem=row.get("synthetic_id") or output_stem(row["capture_set"],modality,int(row["output_battery_id"]),row["axis"],int(row["original_index"]))
        base=output/row["capture_set"]/modality
        for folder in ("images","json","labels_det","labels_seg","augmentation_json"): (base/folder).mkdir(parents=True,exist_ok=True)
        image_path=base/"images"/(stem+(".jpg" if modality=="CT" else ".png")); json_path=base/"json"/(stem+".json"); det_path=base/"labels_det"/(stem+".txt"); seg_path=base/"labels_seg"/(stem+".txt")
        jpeg_profile=""
        if modality=="CT":
            jpeg_profile=_save_jpeg(image,image_path,quantization,subsampling)
        else: image.save(image_path,format="PNG")
        info=payload.setdefault("image_info",{}); data=payload.setdefault("data_info",{}); data["battery_ids"]=int(row["output_battery_id"]); info.update({"id":stable_seed(row["sample_id"]),"file_name":image_path.name,"width":width,"height":height,"is_normal":not bool(defects)})
        if modality=="CT": data["roi"]=[0,0,width,height]
        atomic_json(json_path,payload); _labels(det_path,seg_path,defects,modality,width,height)
        augmentation["output_sha256"]=sha256_file(image_path); aug_path=base/"augmentation_json"/(stem+".augmentation.json"); atomic_json(aug_path,augmentation)
        result_row=dict(row); result_row.update({"generation_status":"success",
            "jpeg_profile_id":jpeg_profile,"exclusion_or_retry_reason":normal.retry_reason,
            "generator_version":__version__,"plan_sha256":plan_hash,
            "output_image_path":image_path.relative_to(output).as_posix(),"output_json_path":json_path.relative_to(output).as_posix(),"output_det_path":det_path.relative_to(output).as_posix(),"output_seg_path":seg_path.relative_to(output).as_posix(),"output_image_sha256":sha256_file(image_path),"output_json_sha256":sha256_file(json_path),"output_det_sha256":sha256_file(det_path),"output_seg_sha256":sha256_file(seg_path),"augmentation_json_path":aug_path.relative_to(output).as_posix(),"augmentation_json_sha256":sha256_file(aug_path)})
        manifest.append(result_row)
        if index%25==0 or index==len(rows):
            LOGGER.info(
                "Generation %s/%s (%.2f%%) | failed %d",
                f"{index:,}",
                f"{len(rows):,}",
                100*index/len(rows),
                failures,
            )
    manifests=output/"manifests"; manifests.mkdir(parents=True,exist_ok=True)
    if manifest:
        with (manifests/"dataset_manifest.csv").open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    summary={"generator_version":__version__,"planned":len(rows),"succeeded":len(manifest),"failed":failures,"python":sys.version,"platform":platform.platform(),"plan_sha256":sha256_file(plan_path),"label_source":"extracted-json-only"}; atomic_json(output/"generation_summary.json",summary); return summary


def verify(output:Path)->dict[str,int]:
    manifest_path=output/"manifests"/"dataset_manifest.csv"
    with manifest_path.open("r",encoding="utf-8-sig",newline="") as handle:
        rows=list(csv.DictReader(handle))
    errors=[]
    for row in rows:
        for column,hash_column in (("output_image_path","output_image_sha256"),("output_json_path","output_json_sha256"),("output_det_path","output_det_sha256"),("output_seg_path","output_seg_sha256")):
            path=output/row[column]
            if not path.is_file() or sha256_file(path)!=row[hash_column]: errors.append(f"{row['sample_id']}:{column}")
    if errors: raise ValueError(f"Verification failed ({len(errors)}): {errors[:5]}")
    return {"samples":len(rows),"errors":0}


def package_outputs(output: Path) -> dict[str, int]:
    verify(output)
    zip_count = 0
    mapping = {"images": "images", "json": "json", "labels_det": "labels_det", "labels_seg": "labels_seg"}
    for capture_set, prefix in (("initial_capture", "initial"), ("recapture", "recapture")):
        for modality in ("CT", "RGB"):
            for folder, suffix in mapping.items():
                source = output / capture_set / modality / folder
                archive = output / f"{prefix}_{modality}_{suffix}.zip"
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
                    for path in sorted(source.glob("*")):
                        if path.is_file():
                            handle.write(path, path.name)
                zip_count += 1
    with zipfile.ZipFile(output / "manifests.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted((output / "manifests").glob("*")):
            if path.is_file():
                handle.write(path, path.name)
    zip_count += 1
    return {"zip_files": zip_count}
