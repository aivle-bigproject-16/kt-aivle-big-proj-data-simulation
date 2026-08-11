from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .cache import ensure_cache, export_cache_csv
from .generator import generate, package_outputs, verify
from .planner import build_plan
from .smoke import run_smoke_test


def parser() -> argparse.ArgumentParser:
    root=argparse.ArgumentParser(prog="server-sim-dataset")
    root.add_argument("--log-level",default="INFO",choices=("DEBUG","INFO","WARNING","ERROR"))
    root.add_argument("--workers",type=int,default=1,help="원본 스캔과 생성에 쓸 프로세스 수")
    commands=root.add_subparsers(dest="command",required=True)
    scan=commands.add_parser("scan",help="scan extracted JSON and build reusable SQLite cache")
    scan.add_argument("--raw-root",type=Path,required=True); scan.add_argument("--cache",type=Path,required=True); scan.add_argument("--refresh-cache",action="store_true"); scan.add_argument("--export-csv",type=Path)
    plan=commands.add_parser("plan",help="create deterministic 37,400-row generation plan")
    plan.add_argument("--raw-root",type=Path,required=True); plan.add_argument("--cache",type=Path,required=True); plan.add_argument("--output",type=Path,required=True); plan.add_argument("--refresh-cache",action="store_true")
    gen=commands.add_parser("generate",help="generate images, JSON, YOLO labels and manifest")
    gen.add_argument("--raw-root",type=Path,required=True); gen.add_argument("--plan",type=Path,required=True); gen.add_argument("--output",type=Path,required=True); gen.add_argument("--engine-root",type=Path); gen.add_argument("--limit",type=int); gen.add_argument("--resume",action="store_true")
    gen.add_argument("--strict-source-hash",action="store_true",help="행마다 원본 파일을 다시 해싱해 plan 과 대조한다")
    check=commands.add_parser("verify",help="verify output files and hashes")
    check.add_argument("--output",type=Path,required=True); check.add_argument("--plan",type=Path)
    check.add_argument("--expect-full",action="store_true",help="계획서 13.1 의 고정 수량까지 대조한다")
    package=commands.add_parser("package",help="verify output and create the 17 planned ZIP archives")
    package.add_argument("--output",type=Path,required=True); package.add_argument("--plan-dir",type=Path)
    package.add_argument("--cache",type=Path); package.add_argument("--feasibility",type=Path)
    package.add_argument("--expect-full",action="store_true")
    smoke=commands.add_parser("smoke",help="run stratified CT/RGB PASS/FAIL/recapture smoke test")
    smoke.add_argument("--raw-root",type=Path,required=True); smoke.add_argument("--cache",type=Path,required=True)
    smoke.add_argument("--plan-dir",type=Path,required=True); smoke.add_argument("--output",type=Path,required=True)
    smoke.add_argument("--engine-root",type=Path,required=True); smoke.add_argument("--per-group",type=int,default=2)
    smoke.add_argument("--refresh-cache",action="store_true")
    smoke.add_argument("--full-scan",action="store_true",help="also build/reuse the complete cache and 37,400-row plan")
    return root


def main(argv:list[str]|None=None)->None:
    args=parser().parse_args(argv); logging.basicConfig(level=getattr(logging,args.log_level),format="%(asctime)s | %(levelname)s | %(message)s")
    if args.command=="scan":
        cache,reused=ensure_cache(args.raw_root,args.cache,args.refresh_cache,args.workers)
        if args.export_csv: export_cache_csv(cache,args.export_csv)
        result={"cache":str(cache.resolve()),"scan_skipped":reused,"label_source":"extracted-json-only"}
    elif args.command=="plan":
        cache,reused=ensure_cache(args.raw_root,args.cache,args.refresh_cache,args.workers); result=build_plan(cache,args.output); result["scan_skipped"]=reused
    elif args.command=="generate": result=generate(args.raw_root,args.plan,args.output,args.engine_root,args.limit,args.resume,args.strict_source_hash,args.workers)
    elif args.command=="verify": result=verify(args.output,args.plan,args.expect_full)
    elif args.command=="package": result=package_outputs(args.output,args.plan_dir,args.cache,args.feasibility,args.expect_full)
    else: result=run_smoke_test(args.raw_root,args.cache,args.plan_dir,args.output,args.engine_root,args.per_group,args.refresh_cache,args.full_scan)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
