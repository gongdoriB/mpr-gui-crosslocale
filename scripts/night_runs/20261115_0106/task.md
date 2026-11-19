# 앱 군집 부트스트랩을 4개 모델로 넓히기 (밤 작업)

## 목표
문항이 같은 앱 화면에서 여러 개 나오므로, 문항 단위 부트스트랩 구간이 너무 좁을 수 있다. 이미 Qwen3-VL-8B 실험 A에 대해 앱 군집 부트스트랩을 했다. 같은 방법을 다른 모델과 REL에도 적용해, 부호와 0 포함 여부 판정이 바뀌는지 표로 남긴다. 결과를 해석하거나 원고 문장을 쓰지 않는다. 수치와 판정 변화만 기록한다.

## 입력 (읽기만)
- 기존 방법과 코드: ${PROJECT_ROOT}/robustness_free_20261114/README_KO.md (3절), task1_task3_expA.py, common_expA.py, outputs/task3_app_clusters.csv
- 실험 A 행 단위 결과:
  - 8B: ${PROJECT_ROOT}/expA_results_20261114/ (analysis_qwen3, analysis_internvl35 폴더와 그 입력을 task1_task3_expA.py가 읽는 방식 그대로 찾을 것)
  - 4B: ${PROJECT_ROOT}/analysis_newmodels_20261114/{qwen3_vl_4b,internvl35_4b}/expA_analyze/glmm_rows.csv (행 단위), 원자료 ${PROJECT_ROOT}/model_results_20261114/<key>/unpacked/
- REL 대비 정의와 의존 문항 층: ${PROJECT_ROOT}/analysis_newmodels_20261114/run_frozen_analysis.py 와 각 모델 results.json (rel 항목). 이 스크립트의 함수를 import해서 같은 quartet 표를 만들 것. 스크립트 자체는 수정 금지.

## 할 일
1. 실험 A: 4개 모델(qwen3 8B, internvl35 8B, qwen3_vl_4b, internvl35_4b) × 대비 3개(T−O, R−T, R−O) × 언어 범위 3개(all4, noTH_screen, noTH_both). 문항 단위 CI와 앱 군집 CI를 나란히. 앱 식별 규칙과 seed 20261113, B=10000은 README 3절 그대로.
2. REL: 4개 모델 × 대비 4개(R−O, C−O, F−C, F−R) × 같은 3개 범위. seed 42, B=10000. 문항 단위 값은 각 모델 results.json의 값과 소수 첫째 자리까지 같은지 먼저 확인하고, 다르면 blocked.md에 적고 그 모델 REL은 건너뛴다.
3. 판정 변화 열: 0을 포함하는지 여부가 문항 단위와 앱 군집 사이에 달라지는 행을 표시.
4. 먼저 실험 A qwen3 8B로 기존 outputs/task3_cluster_bootstrap_expA_qwen3.csv 값을 재현해 코드를 확인한 뒤 나머지를 돌린다. 재현이 안 되면 멈추고 blocked.md에 적는다.
5. 모델마다 끝날 때 CSV를 바로 저장한다(중간에 멈춰도 남도록).

## 산출물 (출력 폴더 안에만)
- app_cluster_expA.csv, app_cluster_rel.csv (열: model, exp, scope, contrast, estimate, item_ci_lo, item_ci_hi, app_ci_lo, app_ci_hi, width_ratio, zero_call_changed)
- app_cluster_KO.md: 표 요약과 판정이 바뀐 행 목록, 코드 경로, seed, B
- 사용한 스크립트 파일(.py)
- REPORT.md (한국어)

## 완료 기준
두 CSV에 4개 모델 모든 조합이 있거나, 빠진 조합의 이유가 blocked.md에 있다.

## 제약
- 원본 파일 수정·이동·삭제 금지. 출력 폴더 밖에 쓰지 않는다.
- 데이터 분모를 표에 적는다(쌍 수, 문항 수, 앱 수).
- "확실", "증명" 같은 단정 금지. 앱이 독립 군집이라는 가정은 확인하지 않은 것으로 적는다.
