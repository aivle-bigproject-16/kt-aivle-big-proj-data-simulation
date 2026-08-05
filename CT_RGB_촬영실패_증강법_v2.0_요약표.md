# CT·RGB 촬영실패 증강법 v2.0 요약표

이 문서는 `quality-fail-augment v2.0`에 적용된 **촬영·획득 실패 증강법만** 정리한다.
제품 제조 불량을 새로 합성하는 증강법은 포함하지 않는다. 한 이미지에는 failure case 하나를
적용하되, 해당 case를 표현하기 위한 내부 증강 단계는 여러 개일 수 있다.

## CT 촬영실패 증강법

| case ID | 촬영 실패 케이스 | 실제 원인 | 표현할 CT 이상 | 내부 증강 단계 |
| --- | --- | --- | --- | --- |
| `ct_cell_alignment_failure` | 셀 위치·정렬 설정 실패 | 배터리 또는 검사 지그의 위치 이탈, 중심 정렬 오류, 촬영 영역 설정 오류 | 배터리 일부가 촬영 영역 밖으로 벗어나 외곽과 내부 구조 일부가 잘림 | `alignment_edge_crop` |
| `ct_acquisition_motion` | CT 획득 중 배터리 움직임 | 지그 고정력 부족, 진동, 회전축 흔들림, 스캔 도중 위치 변화 | 구조 경계의 방향성 흐림, 동일 구조가 이동 방향으로 이중으로 보이는 ghosting | `directional_motion_blur` → `double_edge_ghosting` |
| `ct_insufficient_projection_sampling` | 투영 영상 수집 부족 | projection view 누락, 회전 구간 미수집, 제한각 촬영, 조기 획득 종료 | 구조 주변 streak, 방향성 aliasing, 경계·세부 구조의 재구성 손실 | `radon_projection_drop` → `filtered_back_projection` |
| `ct_low_signal_noise` | X-ray 신호·광자량 부족 | 관전류·노출시간 부족, 투과 광자 부족, 검출기 read noise 영향 증가 | 전체 신호 감소, 입자성 Poisson noise 증가, 대비와 미세 구조 식별력 저하 | `signal_to_transmission` → `poisson_sampling` → `read_noise` → `low_contrast_attenuation` |
| `ct_beam_hardening_metal_streak` | 고밀도 물질 투과·보정 실패 | 금속 또는 고밀도 부품에서 저에너지 광자 흡수, photon starvation, 보정 불충분 | 고밀도 영역 주변 cupping·명암 왜곡과 방사형 밝고 어두운 streak | `dense_material_mask` → `cupping_field` → `metal_anchored_streaks` |

## RGB 촬영실패 증강법

| case ID | 촬영 실패 케이스 | 실제 원인 | 표현할 RGB 이상 | 내부 증강 단계 |
| --- | --- | --- | --- | --- |
| `rgb_trigger_timing_failure` | 촬영 트리거 타이밍 실패 | 센서 감지 시점 오류, 카메라 트리거 지연·선행, 컨베이어와 촬영 시점 불일치 | 배터리 일부가 프레임 밖으로 잘리고, 이동 중 촬영된 경우 진행 방향 흐림 발생 | `timing_edge_crop` → 선택적 `conveyor_motion_blur` |
| `rgb_uneven_lighting` | 조명 점등 실패·불균일 조명 | LED 광량 편차, 일부 LED 미점등, 조명 위치 불량, 광축과 배터리 위치 불일치 | 배터리 한쪽은 과도하게 밝고 반대쪽은 어두운 밝기 기울기, 국부적인 암부 | `lighting_gradient` → 선택적 `led_dead_zone` |
| `rgb_reflection_glare` | 반사광 억제 실패 | 조명 입사각·카메라 각도 오류, 편광 억제 부족, 금속·필름 표면의 정반사 | 배터리 표면의 길고 밝은 반사 core, 주변 bloom, 일부 결함·표면 정보 가림 | `surface_aware_specular_reflection` → `highlight_bloom` |
| `rgb_focus_failure` | 카메라 초점 설정 실패 | 초점 거리 설정 오류, autofocus 실패, 렌즈–배터리 거리 변화, 진동 | 배터리 외곽과 표면 결함 경계가 전반적으로 흐려지고 미세 정보가 감소 | `defocus_blur` |
| `rgb_underexposure` | 노출 부족 | 노출시간 부족, 조리개·게인 설정 오류, 조명 광량 부족 | 전체 신호와 밝기 감소, 암부 정보 손실, shot noise와 sensor read noise 증가 | `linear_exposure_reduction` → `signal_dependent_shot_noise` → `sensor_read_noise` |
| `rgb_overexposure` | 노출 과다 | 노출시간·센서 gain 과다, 조명 광량 과다, 조리개 설정 오류 | 배터리 표면의 넓은 포화, 색·질감·결함 경계 손실, 선택적인 highlight bloom | `overexposure` → 선택적 `highlight_bloom` |
| `rgb_surface_dust` | 렌즈·보호유리 먼지 오염 | 렌즈 또는 카메라 보호창에 부착된 먼지·입자성 오염물 | 촬영 위치에 고정된 흐린 원형·타원형 그림자와 주변 halo | `lens_dust_shadow` |
| `rgb_hair_contamination` | 렌즈·보호유리 섬유 오염 | 렌즈 또는 보호창에 부착된 머리카락·실·섬유성 오염물 | 화면을 가로지르는 가늘고 굽은 반투명 선형 그림자와 초점 이탈 halo | `lens_fiber_shadow` |

## 적용 해석

- CT와 RGB의 촬영실패 label은 제품 결함 label과 독립적이다.
- `porosity`, `Damaged`, `Pollution` 등 원본 제품 결함 annotation은 유지한다.
- 먼지·섬유·glare 같은 촬영 아티팩트는 제품 결함 annotation으로 추가하지 않는다.
- `선택적` 단계는 해당 failure case 안에서 seed와 적용 확률에 따라 함께 적용될 수 있는
  보조 효과이며 별도의 두 번째 failure case가 아니다.
