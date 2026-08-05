# CT/RGB 서버 시뮬레이션 데이터셋 파이프라인 v1.2

현재 계획서 `시뮬레이션_데이터셋_생성_계획서_v1.2(2026-08-05).md`를 구현한 독립 코드다.
라벨은 원본 폴더에 풀려 있는 `.json`만 읽는다. TAR·ZIP 등 압축파일은 열거나 fallback으로
사용하지 않는다.

## 설치

```powershell
$repo = "C:\Users\User\Documents\Codex\2026-07-31\dl-r"
$python = "C:\Users\User\Documents\Codex\rgb-augmentation-venv\Scripts\python.exe"
Set-Location -LiteralPath $repo
& $python -m pip install -e ".[test]"
```

## 1. 최초 스캔과 캐시 생성

```powershell
& $python -m server_sim_dataset.cli scan `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --cache ".\cache\raw_scan_v1.sqlite" `
  --export-csv ".\cache\raw_scan_v1.csv"
```

같은 명령을 다시 실행하면 `scan_skipped: true`가 출력되고 원본 스캔을 생략한다. 원본이
변경되어 캐시를 다시 만들 때만 `--refresh-cache`를 추가한다.

## 2. 37,400장 generation plan 생성

```powershell
& $python -m server_sim_dataset.cli plan `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --cache ".\cache\raw_scan_v1.sqlite" `
  --output "E:\server-simulation-v1.2-plan"
```

캐시가 있으면 이 단계도 스캔을 생략한다. 계획은 1차 CT 29,000장, RGB 5,000장과 재촬영
CT 2,900장, RGB 500장으로 구성된다.

## 3. 생성

촬영실패 효과는 검증된 `quality-fail-augment v2.0` 엔진을 호출한다.

```powershell
& $python -m server_sim_dataset.cli generate `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --plan "E:\server-simulation-v1.2-plan\generation_plan.csv" `
  --output "E:\server-simulation-v1.2-output" `
  --engine-root "C:\Users\User\Documents\Codex\2026-07-22\aivle-bigproject-16-kt-aivle-big\work\kt-aivle-big-proj-data-augmentation-v1.9-severe"
```

기본 동작은 원본 image/JSON SHA-256을 generation plan과 대조하므로 원본이 바뀐 경우 즉시
중단한다.

## 3.1 전체 생성 전 smoke test

아래 명령은 CT/RGB 각각에서 `initial PASS`, `initial FAIL`, `recapture PASS`를 2장씩 골라
총 12장을 생성한다. 따라서 단순히 plan의 앞부분만 생성하는 `--limit`보다 촬영실패 엔진과
재촬영 경로까지 확실하게 검사한다.

```powershell
& $python -m server_sim_dataset.cli smoke `
  --raw-root "E:\103.배터리 불량 이미지 데이터" `
  --cache ".\cache\raw_scan_v1.sqlite" `
  --plan-dir "E:\server-simulation-v1.2-plan" `
  --output "E:\server-simulation-v1.2-smoke" `
  --engine-root "C:\Users\User\Documents\Codex\2026-07-22\aivle-bigproject-16-kt-aivle-big\work\kt-aivle-big-proj-data-augmentation-v1.9-severe" `
  --per-group 2
```

- cache가 있으면 원본 스캔을 생략한다.
- generation plan이 있으면 plan 생성도 생략한다.
- smoke 출력 폴더는 없거나 비어 있어야 한다.
- 성공하면 `E:\server-simulation-v1.2-smoke\smoke_test_summary.json`의 `status`가
  `passed`가 된다.
- 실패해도 같은 JSON에 오류 종류와 메시지가 기록된다.

동일한 명령을 PowerShell 래퍼로 실행할 수도 있다.

```powershell
Set-Location -LiteralPath "C:\Users\User\Documents\Codex\2026-07-31\dl-r"
.\run_smoke_test.ps1
```

래퍼는 저장소의 `src`를 `PYTHONPATH`에 자동 등록하므로 `pip install -e .`를 먼저 하지
않았어도 실행할 수 있다.

출력 폴더 이름을 바꿀 때는 다음과 같이 실행한다.

```powershell
.\run_smoke_test.ps1 -Output "E:\server-simulation-v1.2-smoke-2"
```

## 4. 결과 검증

```powershell
& $python -m server_sim_dataset.cli verify `
  --output "E:\server-simulation-v1.2-output"
```

모든 이미지·JSON·detection TXT·segmentation TXT의 존재 여부와 SHA-256을 manifest 기준으로
검증한다.

## 5. 계획서의 17개 ZIP 생성

```powershell
& $python -m server_sim_dataset.cli package `
  --output "E:\server-simulation-v1.2-output"
```

출력 검증을 먼저 통과한 경우에만 initial/recapture의 CT/RGB 이미지·JSON·detection·
segmentation ZIP 16개와 `manifests.zip`을 생성한다.
