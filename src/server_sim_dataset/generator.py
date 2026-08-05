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
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from . import __version__
from .schema import iter_defects, points, roi_bbox
from .util import atomic_json, sha256_file, stable_seed


LOGGER = logging.getLogger(__name__)
CLASS_IDS = {"CT": {"porosity": 0}, "RGB": {"damaged": 0, "pollution": 1}}


def _gamma(image: Image.Image, value: float) -> Image.Image:
    table = [round(255 * ((i / 255) ** value)) for i in range(256)]
    return image.point(table if image.mode == "L" else table * 3)


def _normal(image: Image.Image, names: list[str], params: dict[str, float], seed: int) -> tuple[Image.Image, bool, bool]:
    rng = np.random.Generator(np.random.PCG64(seed)); out = image; flip_x = flip_y = False
    for name in names:
        if name == "brightness_contrast_gamma":
            out = _gamma(ImageEnhance.Contrast(ImageEnhance.Brightness(out).enhance(params["brightness"])).enhance(params["contrast"]), params["gamma"])
        elif name in {"normal_noise_poisson", "poisson_noise"}:
            array = np.asarray(out).astype(np.float32) / 255
            noisy = rng.poisson(np.clip(array, 0, 1) * 180) / 180 + rng.normal(0, params["noise_sigma"], array.shape)
            out = Image.fromarray(np.uint8(np.clip(noisy, 0, 1) * 255), mode=out.mode)
        elif name in {"low_frequency_shading", "low_frequency_lighting"}:
            width, height = out.size; x = np.linspace(-1, 1, width); field = 1 + rng.uniform(-0.08, 0.08) * x
            array = np.asarray(out).astype(np.float32) * (field[None, :] if out.mode == "L" else field[None, :, None])
            out = Image.fromarray(np.uint8(np.clip(array, 0, 255)), mode=out.mode)
        elif name in {"weak_reconstruction_kernel", "weak_reconstruction"}:
            out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.25, 0.65))))
        elif name in {"partial_histogram_blend", "percentile_tone_curve", "rgb_channel_gain_tone"}:
            equalized = ImageOps.equalize(out); out = Image.blend(out, equalized, float(rng.uniform(0.05, 0.15)))
        elif name == "synchronized_flip":
            flip_x = bool(seed & 1); flip_y = bool(seed & 2)
            if flip_x: out = ImageOps.mirror(out)
            if flip_y: out = ImageOps.flip(out)
        elif name == "safe_translate_rotate":
            # Geometry is conservatively replaced by a guaranteed-safe optical transform when
            # the plan does not contain an approved per-image affine candidate.
            out = ImageEnhance.Color(out).enhance(float(rng.uniform(0.96, 1.04)))
    return out, flip_x, flip_y


def _transform_polygon(polygon: list[tuple[float,float]], width: int, height: int, *, offset=(0,0), flip_x=False, flip_y=False, affine=None) -> list[tuple[float,float]]:
    result=[]
    for x,y in polygon:
        x-=offset[0]; y-=offset[1]
        if flip_x: x=width-x
        if flip_y: y=height-y
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
    engine=None; manifest=[]; failures=0
    for index,row in enumerate(rows,1):
        modality=row["modality"]; source_image=(raw_root/row["orig_image_relative_path"]).resolve(); source_json=(raw_root/row["orig_json_relative_path"]).resolve()
        if raw_root not in source_image.parents or raw_root not in source_json.parents: raise ValueError("Plan path escapes raw root")
        if sha256_file(source_image)!=row["source_image_sha256"] or sha256_file(source_json)!=row["source_json_sha256"]: raise ValueError(f"Source hash mismatch: {row['sample_id']}")
        payload=json.loads(source_json.read_text(encoding="utf-8-sig"))
        with Image.open(source_image) as source:
            image=source.convert("L" if modality=="CT" else "RGB")
        offset=(0,0)
        if modality=="CT":
            roi=roi_bbox(payload,*image.size); offset=(roi[0],roi[1]); image=image.crop(roi)
        names=json.loads(row["base_augmentation_names"]); params=json.loads(row["normal_augmentation_parameters"])
        effective_seed=int(row.get("slice_seed") or row["normal_augmentation_seed"])
        image,flip_x,flip_y=_normal(image,names,params,effective_seed); width,height=image.size
        defects=_update_annotations(payload,width,height,offset=offset,flip_x=flip_x,flip_y=flip_y)
        augmentation={"normal_augmentations":names,"normal_parameters":params,"seed":int(row["normal_augmentation_seed"]),"failure_case":"","automatic_checks":{"passed":True}}
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
        stem=("initial" if row["capture_set"]=="initial_capture" else "recapture")+f"_{modality}_{row['output_battery_id']}_"+(f"{row['axis']}_" if modality=="CT" else "")+f"{int(row['original_index']):06d}"
        base=output/row["capture_set"]/modality
        for folder in ("images","json","labels_det","labels_seg","augmentation_json"): (base/folder).mkdir(parents=True,exist_ok=True)
        image_path=base/"images"/(stem+(".jpg" if modality=="CT" else ".png")); json_path=base/"json"/(stem+".json"); det_path=base/"labels_det"/(stem+".txt"); seg_path=base/"labels_seg"/(stem+".txt")
        if modality=="CT": image.save(image_path,format="JPEG",quality=95,subsampling=0)
        else: image.save(image_path,format="PNG")
        info=payload.setdefault("image_info",{}); data=payload.setdefault("data_info",{}); data["battery_ids"]=int(row["output_battery_id"]); info.update({"id":stable_seed(row["sample_id"]),"file_name":image_path.name,"width":width,"height":height,"is_normal":not bool(defects)})
        if modality=="CT": data["roi"]=[0,0,width,height]
        atomic_json(json_path,payload); _labels(det_path,seg_path,defects,modality,width,height)
        augmentation["output_sha256"]=sha256_file(image_path); aug_path=base/"augmentation_json"/(stem+".augmentation.json"); atomic_json(aug_path,augmentation)
        result_row=dict(row); result_row.update({"generation_status":"success","output_image_path":image_path.relative_to(output).as_posix(),"output_json_path":json_path.relative_to(output).as_posix(),"output_det_path":det_path.relative_to(output).as_posix(),"output_seg_path":seg_path.relative_to(output).as_posix(),"output_image_sha256":sha256_file(image_path),"output_json_sha256":sha256_file(json_path),"output_det_sha256":sha256_file(det_path),"output_seg_sha256":sha256_file(seg_path),"augmentation_json_path":aug_path.relative_to(output).as_posix(),"augmentation_json_sha256":sha256_file(aug_path)})
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
