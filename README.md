# CT/RGB 서버 시뮬레이션 데이터셋 파이프라인 v1.5

실행 정본은 `시뮬레이션_데이터셋_생성_계획서_v1.5(2026-08-11).md`다. 이전 v1.2 생성
계획서와 v1.3 수정계획서는 변경 이력과 근거 확인용으로 보존한다.

v1.5는 CT 파일명의 x/y/z를 공통 3-D 좌표계에 매핑해 이미지·폴리곤 반전과 슬라이스
순서를 하나의 ID 단위 변환에서 파생한다. 제품 상태는 이미지 수가 아니라 ID 수로 고정하며,
CT는 불량 1/20·정상 19/20, RGB는 불량 2/20·정상 18/20이다.

라벨은 원본 폴더에 풀려 있는 `.json`만 읽는다. TAR·ZIP 등 압축파일은 열거나 fallback으로
사용하지 않는다.

## 요구 사항

- Python 3.10 이상
- `quality-fail-augment` v2.0 (FAIL 샘플 생성에 필요)
- 의존성은 `requirements.lock`으로 고정한다. 엔진과 같은 프로세스에서 동작하므로 버전이
  어긋나면 리샘플링 구현 차이만으로 출력 해시가 달라진다.

## 설치

엔진 저장소를 이 저장소의 형제 디렉터리에 두는 것을 전제로 한다.

```
<workspace>/
  kt-aivle-big-proj-data-simulation/   <- 이 저장소
  kt-aivle-big-proj-data-augmentation/ <- quality-fail-augment v2.0
```

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock "pytest>=8"
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -e ..\kt-aivle-big-proj-data-augmentation
```

엔진을 editable로 설치하면 `--engine-root`로 `sys.path`를 조작하지 않아도 된다.

## 환경 변수

| 변수 | 용도 |
| --- | --- |
| `RAW_ROOT` | 원본 루트. 예: `C:\...\103.배터리 불량 이미지 데이터` |
| `QUALITY_FAIL_ENGINE_ROOT` | 엔진 저장소 경로. 테스트에서 통합 테스트를 켜는 데 쓴다 |

아래 예제는 `$raw`, `$py`를 미리 정의한 것으로 본다.

```powershell
$py  = ".\.venv\Scripts\python.exe"
$raw = "C:\Users\<user>\...\103.배터리 불량 이미지 데이터"
```

## 1. 원본 스캔과 캐시 생성

```powershell
& $py -X utf8 -m server_sim_dataset.cli --workers 6 scan `
  --raw-root $raw `
  --cache ".\cache\raw_scan_v2.sqlite"
```

- `--workers`는 프로세스 수다. 스캔 시간은 원본 이미지의 SHA-256이 지배하므로 여기서
  가장 크게 줄어든다.
- 같은 명령을 다시 실행하면 `scan_skipped: true`가 출력되고 원본 스캔을 생략한다.
- 캐시 스키마는 v2다. v1 캐시는 `original_image_id`와 원본 ROI 값을 갖고 있지 않으므로
  재사용되지 않고 자동으로 다시 만들어진다.

## 2. 187,000장 generation plan 생성

```powershell
& $py -X utf8 -m server_sim_dataset.cli plan `
  --raw-root $raw `
  --cache ".\cache\raw_scan_v2.sqlite" `
  --output ".\work\plan"
```

계획은 CT/RGB 각각 100개 출력 ID를 만든다. ID당 이미지 수는 기존과 같아 1차 CT
145,000장, RGB 25,000장과 재촬영 CT 14,500장, RGB 2,500장으로 구성된다. 제품불량
비율은 CT 5%, RGB 10%, 촬영 FAIL 대상 ID 비율은 각 모달리티 10%로 v1.5와 같다.
generation plan과 dataset manifest는 같은 72개 컬럼 스키마를 쓴다. 생성 단계에서만
정해지는 컬럼은 plan에 빈 문자열로 들어간다.

## 3. 전체 생성 전 smoke test

CT/RGB 각각에서 `initial PASS`, `initial FAIL`, `recapture PASS`를 골라 생성한다. plan의
앞부분만 만드는 `--limit`과 달리 촬영실패 엔진과 재촬영 경로까지 지나간다.

```powershell
& $py -X utf8 -m server_sim_dataset.cli smoke `
  --raw-root $raw `
  --cache ".\cache\raw_scan_v2.sqlite" `
  --plan-dir ".\work\plan" `
  --output ".\work\smoke" `
  --engine-root "..\kt-aivle-big-proj-data-augmentation" `
  --per-group 2 --full-scan
```

성공하면 `work\smoke\smoke_test_summary.json`의 `status`가 `passed`가 된다. 실패해도 같은
JSON에 오류 종류와 메시지가 남는다.

## 4. 생성

```powershell
& $py -X utf8 -m server_sim_dataset.cli --workers 6 generate `
  --raw-root $raw `
  --plan ".\work\plan\generation_plan.csv" `
  --output ".\work\output" `
  --engine-root "..\kt-aivle-big-proj-data-augmentation"
```

- 기본은 캐시에 기록된 원본 해시를 신뢰한다. 원본이 바뀌었을 가능성을 확인하려면
  `--strict-source-hash`를 준다. 행마다 원본을 다시 해싱하므로 느리다.
- 중단된 작업은 `--resume`으로 이어서 한다. 출력 4종이 모두 존재하고 해시가 일치하는
  행만 재사용하고 나머지만 다시 만든다.

## 5. 결과 검증

```powershell
& $py -X utf8 -m server_sim_dataset.cli verify `
  --output ".\work\output" `
  --plan ".\work\plan\generation_plan.csv" --expect-full
```

모든 이미지·JSON·detection TXT·segmentation TXT의 존재와 SHA-256, 계획 대비 수량, orphan
파일, 출력 JSON과 실제 이미지의 크기·파일명·ROI 일치를 검사한다.

## 6. 계획서의 17개 ZIP 생성

```powershell
& $py -X utf8 -m server_sim_dataset.cli package `
  --output ".\work\output" `
  --plan-dir ".\work\plan" `
  --cache ".\cache\raw_scan_v2.sqlite" `
  --feasibility ".\raw_extraction_feasibility.json" `
  --expect-full
```

검증을 통과한 경우에만 ZIP을 만든다. `--plan-dir`과 `--cache`를 주면 계획서 11.2의 요약
파일을 먼저 생성해 `manifests.zip`에 함께 담는다. 압축 후에는 각 아카이브를 다시 열어
파일 수와 stem 집합을 대조한다.

`manifests.zip`에 들어가는 것은 다음과 같다.

- `dataset_manifest.csv`, `selected_ids.csv`, `class_balance_report.csv`
- `sequence_windows.csv`, `failure_windows.csv`, `augmentation_summary.csv`
- `pairing_audit.csv`, `generation_summary.json`, `raw_extraction_feasibility.json`
- `augmentation_json/` 전체와 `failure_masks/`

## 7. 배포

산출물은 아래 경로에만 올린다. 내 드라이브 아래의 팀 공유 폴더다.

```
gdrive:AIVLE_BigProject/data_simulation/<산출물폴더>
```

버전별 하위 폴더를 쓰고 기존 버전은 덮어쓰지 않는다. 예: `server-simulation-v1.4-zips`,
`server-simulation-v1.5-zips`.

```powershell
rclone lsd gdrive:                                  # 최상위 폴더명을 눈으로 확인한다
rclone copy ".\work\output" `
  "gdrive:AIVLE_BigProject/data_simulation/server-simulation-v1.5-zips" `
  --include "*.zip" --max-depth 1 --transfers 4 -P
rclone check ".\work\output" `
  "gdrive:AIVLE_BigProject/data_simulation/server-simulation-v1.5-zips" `
  --include "*.zip" --max-depth 1 --one-way
```

Google Drive는 폴더 이름의 **대소문자를 구분한다**. `AIVLE_Bigproject`처럼 한 글자만
달라도 `rclone copy`는 오류를 내지 않고 같은 이름의 다른 폴더를 새로 만든다. 팀은 그
폴더를 보지 못한다. 경로를 직접 타이핑하지 말고 `rclone lsd gdrive:`의 출력을 그대로
복사해서 쓰고, 업로드 뒤에는 `rclone check`와 `rclone lsd gdrive:`를 다시 확인한다.

## 테스트

```powershell
$env:QUALITY_FAIL_ENGINE_ROOT = "..\kt-aivle-big-proj-data-augmentation"
& $py -X utf8 -m pytest -q
```

한글 assertion 메시지가 cp949 콘솔에서 깨지므로 `-X utf8`을 붙인다.
