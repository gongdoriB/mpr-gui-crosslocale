# 모델 확장 큐 실행 안내: Qwen3-VL 2B/4B, InternVL3.5 1B/2B/4B, UI-TARS-1.5-7B

작성 2026-11-14, 같은 날 리뷰 수정 반영. 서버에서는 아무것도 실행하지 않았다(미검증). 로컬에서는 6개 설정 전부에 `--dry-run`과 `--mock-model`(REL 전체 18,432회, 실험 A 전체 5,688회까지)을 돌렸고 큐 스크립트의 흐름을 가짜 명령으로 시험했다. 실제 모델 로드와 GPU 추론은 한 번도 해보지 않았다.

수정 전 파일은 `superseded/`에 `_v1`로 남겼다.

## 돌리는 것

큐 전체 순서: 시작 점검(묶음 파일, sha256, `timeout`, GPU 이름) → `OVERNIGHT_STATUS`의 FINISHED 대기(상한 있음) → GPU 비어 있음 확인 → 네트워크 속도 확인 → 저장소 점검(`mpr-gui-crosslocale`, `mpr-ivl`) → 묶음 풀기 → Qwen3-VL-8B 배치 1 점검 → 8B 캐시 삭제 → 모델들.

모델마다 같은 순서다: 내려받기, smoke와 게이트, REL O/R/C/F 전체(366문항, 고유 추론 18,432회), 실험 A O/T/R 전체(1,896쌍 x 3 = 5,688회), 결과 묶기, 캐시 정리. 배치 1, 탐욕 생성, max_new_tokens 2, english_fixed 시스템 프롬프트, 라벨 점수 on.

| 순서 | 키 | 모델 | 저장소 | 고정 revision | 디스크 |
| --- | --- | --- | --- | --- | --- |
| 1 | qwen3_vl_2b | Qwen/Qwen3-VL-2B-Instruct | mpr-gui-crosslocale | 89644892e4d85e24eaac8bacfd4f463576704203 | 4.27 GB |
| 2 | qwen3_vl_4b | Qwen/Qwen3-VL-4B-Instruct | mpr-gui-crosslocale | ebb281ec70b05090aa6165b016eac8ec08e71b17 | 8.89 GB |
| 3 | internvl35_1b | OpenGVLab/InternVL3_5-1B | mpr-ivl | 2f71cf52542334823e48a46ffba0e2bc9add3446 | 2.14 GB |
| 4 | internvl35_2b | OpenGVLab/InternVL3_5-2B | mpr-ivl | 7d7bd7bcc35b77b391883ebd7f9686cfa0e25cc0 | 4.71 GB |
| 5 | internvl35_4b | OpenGVLab/InternVL3_5-4B | mpr-ivl | 481f6e32467eab4e922ccd7fd6cf420441a62331 | 9.48 GB |
| 6 | uitars_15_7b | ByteDance-Seed/UI-TARS-1.5-7B | mpr-gui-crosslocale | 683d002dd99d8f95104d31e70391a39348857f4e | 33.18 GB |

revision과 크기는 2026-11-14에 HF API(읽기 전용)로 읽은 값이다. UI-TARS는 가중치가 fp32로 저장되어 7B인데도 33 GB다(bfloat16으로 불러온다).

## 설정에서 같은 점과 다른 점

- 설정 파일은 각 8B 설정에서 model_id와 revision만 바꿨다(`addon/01_code/configs/models/`). Qwen3 쪽은 `qwen3_vl_8b_4090_channels_last.yaml`, InternVL 쪽은 `internvl3_5_8b_4090.yaml` 기준이다. 저장소에 원래 있던 `qwen3_vl_4b*.yaml`은 max_new_tokens 8, revision 없음 등 설정이 달라서 쓰지 않았고, 새 파일 이름은 겹치지 않는다.
- 이미지 픽셀 한도는 8B와 같다. Qwen3: min 262,144, max 2,097,152. InternVL: 448 타일, max_num 8, 썸네일 사용(최대 2,304 토큰).
- UI-TARS는 Qwen2.5-VL 구조(`Qwen2_5_VLForConditionalGeneration`)다. 8B 설정이 없는 계열이라 기존 `qwen2_5_vl_7b.yaml`과 같은 픽셀 한도(262,144 ~ 2,097,152)를 썼다. Qwen2.5-VL은 토큰 하나가 28x28 픽셀이라 같은 한도에서도 이미지당 최대 약 2,674토큰으로, Qwen3(32x32, 2,048토큰)보다 많다. 설정 파일에 있던 `vision_token_limit: 2048`은 실제 한도와 맞지 않아 뺐다(코드에서는 메타데이터로만 쓰임).
- UI-TARS가 기존 어댑터로 로드되는지: 모델의 config.json과 가중치 이름 729개가 Qwen2.5-VL-7B-Instruct(기존 설정이 쓰는 모델)와 완전히 같다. 로컬에서 meta 장치로 `Qwen2_5_VLForConditionalGeneration`을 만들어 파라미터 이름 729개가 나오는 것까지 확인했다. 실제 가중치 로드와 추론은 서버에서만 확인된다.
- 8B 이전 실행은 REL 배치 4(Qwen3), 실험 A 배치 3(Qwen3)이었고 이번 모델은 모두 배치 1이다. 크기 간 비교에서 배치로 인한 수치 차이가 조금 섞일 수 있다.
- 실험 A는 이전 실행 계약이 없는 모델용으로 `run_rel_T_generic.py`를 따로 만들었다(원본 `expA/run_rel_T.py`의 복사본, 원본 폴더는 수정하지 않았다). 이전 실행과의 일치 게이트(anchor)는 없다. 대신 run_contract.json에 설정 sha256, 코드 해시, 패키지 버전을 남긴다. 시스템 프롬프트는 같은 저장소의 8B 계약과 같은지 확인한다.
- 게이트(`check_generic_gate.py`): 모든 행 성공, 배치 크기 1, 파싱 실패율 50% 초과면 실패. 파싱 실패 수는 어느 경우든 기록한다. smoke가 실패한 모델은 전체 실행을 건너뛰고 다음 모델로 넘어간다. 11월 14일에 아래 규칙을 더했다. 앞의 셋은 어기면 실패다.
  - 파싱된 생성 라벨이 전부 같으면 실패(파싱된 행이 20개 이상일 때 검사).
  - A–D 확률 질량(`label_logprobs`의 exp 합)의 중앙값이 0.5 미만이면 실패. 필드가 없으면 검사하지 않고 "not evaluated"로 기록한다.
  - 토큰 수가 모델 계열 범위를 벗어나면 실패. Qwen3-VL은 시각 토큰 2,048 이하(2,097,152 / 32²), UI-TARS는 2,674 이하(2,097,152 / 28²), InternVL3.5는 1,792~2,304(8B 6×6 실행에서 나온 값이 정확히 이 둘이다). 어댑터가 `visual_token_count`를 기록하지 않으면 보고만 한다. 저장소 코드를 확인해 보니 Qwen3와 UI-TARS 어댑터는 `prompt_token_count`만 기록하고 `visual_token_count`는 기록하지 않는다(8B Qwen3 실행 결과에서도 79,020행 전부 비어 있었다). 그래서 이 두 계열은 프롬프트 토큰 수로 대신 본다. Qwen3는 800~2,700(8B 실행에서 1,711~2,515), UI-TARS는 800~3,300(2,674 + 글 부분 여유, 이 상한은 실측이 아니라 계산이다). InternVL 어댑터는 `visual_token_count`를 기록한다.
  - smoke 정확도는 기록하고 0.30 미만이면 경고만 한다(실패 아님). 우연 수준은 0.25다.
  - smoke 속도의 중앙값으로 전체 단계 예상 시간을 계산해 `gate_*.json`의 `timing.eta_full_seconds`에 남긴다. 이 값이 그 단계의 timeout보다 길면(이미 끝낸 만큼은 빼고) 그 단계를 시작하지 않고 `STATUS`와 `QUEUE_STATUS`에 `STOPPED: ... ETA` 줄과 `rel=STOPPED_ETA` 또는 `expa=STOPPED_ETA`를 남긴다. 다른 단계는 계속 돈다. 일부러 돌리려면 `IGNORE_ETA=1`.
- 단계 timeout 표기를 `1h30m` 같은 두 단위에서 `90m`처럼 분 하나로 바꿨다. GNU `timeout`은 `1h30m`을 받지 않는다(`invalid time interval`로 바로 끝난다). 이전 버전의 실험 A 1.5시간, REL 4.5시간·5.5시간 단계가 이 때문에 시작도 못 했을 것이다. 로컬 가짜 `timeout`을 GNU처럼 그런 값을 거부하게 고쳐서 확인했다.

## 실행 방법

1. 이전 8B 작업이 도는 동안은 아무것도 하지 않는다. 큐 스크립트가 `/workspace/OVERNIGHT_STATUS`에 FINISHED가 찍히길 기다린다. 먼저 올려두는 것은 괜찮다. 시작 점검은 묶음 파일·sha256·`timeout`·GPU 이름뿐이라 대기 전에 끝난다. 저장소(`mpr-gui-crosslocale`, `mpr-ivl`)와 venv 점검은 대기가 끝난 뒤에 한다(이전 작업이 그것들을 만드는 중일 수 있어서).
2. 로컬의 두 파일을 서버 `/workspace/`에 올린다(Jupyter 업로드 또는 scp).
   - `model_expansion_sizes_uitars_20261114.tar.gz`
   - `queue_sizes_uitars.sh`
3. 서버 터미널에서 확인한다.
   ```bash
   cd /workspace && ls model_expansion_sizes_uitars_20261114.tar.gz queue_sizes_uitars.sh
   sha256sum model_expansion_sizes_uitars_20261114.tar.gz   # ADDON_SHA256.txt의 값과 같아야 한다
   ```
4. tmux 안에서 시작한다.
   ```bash
   tmux new -s queue        # 이미 있으면 tmux attach -t queue
   bash /workspace/queue_sizes_uitars.sh
   ```
   일부 모델만: `ONLY="qwen3_vl_2b internvl35_1b" bash /workspace/queue_sizes_uitars.sh` (순서는 위 표 순서 그대로 쓴다.)
   이전 작업이 끝났는데 FINISHED 줄이 없을 때만: `SKIP_WAIT=1 bash ...`. 이때도 GPU 메모리 사용이 1,500 MiB 아래가 될 때까지(최대 30분) 확인하고 아니면 멈춘다.
   네트워크가 20 MB/s 미만이면 멈춘다. 그래도 하려면 `FORCE_SLOW_NET=1`.
   같은 큐를 두 번 띄우면 두 번째는 "already running"으로 끝난다(`flock`).
5. 진행 확인: `cat /workspace/QUEUE_STATUS`. 모델별 상세는 `/workspace/queue_logs/<키>_<단계>.log`와 각 저장소의 `followup_rel_4lang/results/sizes_<키>/STATUS`.
6. 끝나면 `/workspace/QUEUE_DONE` 파일이 생긴다. 내용은 `OK`(큐를 끝까지 돌았다. 모델별 결과는 `QUEUE_STATUS` 마지막 줄의 ok/NOT_COMPLETE) 또는 `STOPPED rc=<번호>`(STOP으로 끝났다. 이유는 `QUEUE_STATUS`)다. Ctrl-C나 kill로 끊으면 만들지 않는다. 다음 큐(MiniCPM)는 둘 중 어느 쪽이든 시작한다. `/workspace/queue_results/`와 `QUEUE_STATUS`를 내려받는다. 이름에 `_PARTIAL`이 붙은 tar는 일부 단계가 실패했다는 뜻이다. 같은 명령을 다시 치면 끝난 단계는 건너뛰고 이어서 돈다.
7. 인스턴스는 스크립트가 끄지 않는다. 결과를 받은 뒤 Destroy는 직접 한다(MiniCPM 큐가 이어서 돌 예정이면 그 큐가 끝난 뒤).

FINISHED 대기는 상한이 있다. `OVERNIGHT_MAX_H`(기본 14시간)가 지나도 FINISHED가 없거나, `run_all_overnight.sh` 프로세스가 10분(`OVERNIGHT_GRACE_MIN`) 넘게 없는데 FINISHED도 없으면(이전 작업이 스스로 STOP한 경우) STOP하고 `QUEUE_DONE`에 `STOPPED rc=1`을 쓴다. 터미널 한 줄로 두 스크립트를 이어 실행하면 그 줄이 `run_all_overnight.sh`를 포함해 프로세스가 계속 있는 것으로 보일 수 있다. 그 경우 상한(14시간)이 안전망이다.

대기가 끝난 뒤 큐가 하는 일:
1. GPU 메모리 사용 1,500 MiB 미만 확인(최대 30분).
2. 네트워크 확인: huggingface.co와 download.pytorch.org에서 각각 15초 동안 범위 요청으로 받아 MB/s를 `QUEUE_STATUS`에 적는다(`/workspace/queue_results/net_check_sizes.txt`에도). 이 큐는 torch를 새로 받지 않아서 huggingface.co 값만 판정에 쓰고, 20 MB/s 미만이거나 연결이 안 되면 멈춘다. 테스트 URL이 HTTP 4xx를 돌려주면 속도를 모르는 것이라 경고만 하고 계속한다.
3. 저장소 점검. 한쪽 저장소가 없으면 그 계열 모델만 건너뛰고 다른 쪽은 돈다. 둘 다 없으면 STOP.
4. addon을 저장소에 풀어 sha256 확인.
5. Qwen3-VL-8B 배치 1 점검(아래). 8B 캐시를 지우기 전에, 오버나이트가 `qwen3=ok`를 보고했을 때만 한다.
6. 8B 캐시 삭제는 `OVERNIGHT_STATUS`에 `FINISHED qwen3=ok internvl35=ok`가 있을 때만 한다(`mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct`, `mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B`). 하나라도 FAILED면 캐시를 그대로 두고 디스크 여유 점검(키별 최소 여유)에 맡긴다. 8B 결과 파일, venv, 이전 results 폴더는 건드리지 않는다.

### Qwen3-VL-8B 배치 1 점검

Qwen3-VL-8B의 이전 REL 실행은 배치 4(실험 A는 배치 3)였고 이번 큐의 모델은 전부 배치 1이다. 배치 크기가 답을 얼마나 바꾸는지 보려고, 8B 캐시를 지우기 전에 같은 8B를 배치 1로 작은 부분집합에 돌려 이전 배치 4 답과 맞춰 본다. 가볍게 해 둔 확인이고 실패해도 큐는 멈추지 않는다.
- 대상: 실험 A 주 분석의 앞 28문항(`parallel_id` 정렬 순서, 문항당 12쌍 × O/T/R 3조건) = 추론 1,008회(로컬 mock 실행으로 1,008회 확인). 이 중 O와 R 672행은 이전 REL 실행(`rstem_ropts_inputs/reference_qwen3_20261107_OR.jsonl`, 배치 4)의 답과 비교할 수 있다. T 행은 이전 답이 없어 비교하지 않는다.
- 실행: `followup_rel_4lang/model_expansion/batch1_check_qwen3_8b.sh`가 `run_rel_T_generic.py`에 8B 설정(`qwen3_vl_8b_4090_channels_last.yaml`, sha256 확인)을 주고, `batch1_compare.py`가 라벨 일치율을 낸다. 네트워크 없이(`HF_HUB_OFFLINE=1`) 이미 받아 둔 8B 캐시를 쓴다. 시간 제한 1시간. 추정 15~25분(0.7~0.8 s/추론 × 1,008 + 모델 로드, 미검증).
- 결과: `/workspace/queue_results/batch1_check_qwen3_8b/batch1_vs_batch4.json`(일치 행 수, 일치율, 조건별, gold 확률 차이 평균·최대, 불일치 예시 25개)와 `QUEUE_STATUS`의 한 줄. 합격/불합격 기준은 두지 않았다. 일치율이 얼마면 문제인지는 이 값을 보고 정할 일이다.
- 이미 결과 파일이 있으면 다시 하지 않는다.

### 캐시 정리 규칙

모델 하나를 끝까지 마치면(smoke, REL, 실험 A 모두 ok) 그 모델 캐시를 지운다. 실패하거나 시간 초과로 끝나면 캐시를 남겨 두어 같은 명령으로 다시 돌릴 때 다시 내려받지 않게 한다(`hf download`는 받던 파일을 이어 받는다. 내려받기 실패나 시간 초과 때도 부분 파일을 지우지 않는다). 단, 남기면 뒤에 오는 모델의 최소 여유(위 표의 "최소 여유")를 채우지 못하게 될 때는 지운다. 기준은 남은 모델 중 가장 큰 최소 여유이고, 마지막 모델이면 MiniCPM 큐가 요구하는 40 GB(`HANDOFF_FREE_GB`)다. `KEEP_FAILED_CACHE=0`이면 실패해도 항상 지운다.

내려받기 같은 네트워크 단계는 실패하면 최대 3번(30초, 60초 간격) 다시 시도한다. 이전 서버에서 이미지 받기 중 SSL EOF, `git clone` 중 gnutls 핸드셰이크 오류가 났기 때문이다.

## 시간 추정 (모델당 추론 24,120회 = REL 18,432 + 실험 A 5,688)

기준은 이 GPU에서 8B 실측 0.73~0.78 s/추론이다(Qwen3 7,584회 99분, InternVL 93분). 작은 모델의 속도는 측정한 적이 없고 아래 값은 모두 추정이다(미검증). 이미지 전처리와 약 2,000토큰 prefill, 라벨 점수 계산 같은 고정 비용이 있어 파라미터 수에 비례해 줄지 않을 수 있다.

| 모델 | 가정 s/추론 | 추론 시간 | 단계별 timeout (smoke / REL / 실험 A) |
| --- | --- | --- | --- |
| Qwen3-VL-2B | 0.30~0.45 | 2.0~3.0 h | 45분 / 4시간 / 1.5시간 |
| Qwen3-VL-4B | 0.45~0.60 | 3.0~4.0 h | 45분 / 5시간 / 2시간 |
| InternVL3.5-1B | 0.25~0.40 | 1.7~2.7 h | 45분 / 4시간 / 1.5시간 |
| InternVL3.5-2B | 0.30~0.50 | 2.0~3.3 h | 45분 / 4.5시간 / 1.75시간 |
| InternVL3.5-4B | 0.45~0.65 | 3.0~4.4 h | 45분 / 5.5시간 / 2시간 |
| UI-TARS-1.5-7B | 0.70~0.90 | 4.7~6.0 h | 1시간 / 8시간 / 3시간 |

합계 약 17~24시간에 모델 내려받기와 로딩(모델당 10~30분, UI-TARS는 33 GB라 더 걸릴 수 있음)이 더해져 19~26시간 정도다. 시간당 요금을 곱한 비용은 요금을 몰라 계산하지 않았다(확인 필요). 전부 돌릴지 일부만 돌릴지는 시간과 비용을 보고 정할 일이라 `ONLY`로 고를 수 있게 해 두었다. 이 파일은 그 결정을 담지 않는다.

## 디스크 추정 (93 GB)

현재 작업 후 사용 약 60 GB로 가정(확인 필요). 8B 캐시 삭제로 약 34.6 GB가 비어(Qwen3-VL-8B 17.55 GB, InternVL3.5-8B 17.07 GB) 사용량은 약 26 GB, 여유는 약 67 GB가 된다. 끝까지 마친 모델은 캐시를 지우고, 실패하거나 시간 초과인 모델은 디스크가 허용하면 남긴다(위 캐시 정리 규칙). 오버나이트가 FAILED로 끝나면 8B 캐시도 남으므로 그만큼 여유가 줄어 UI-TARS가 디스크 부족으로 건너뛰어질 수 있다. 가장 큰 UI-TARS는 33.2 GB라 사용량이 약 60 GB까지 올라간다. 다운로드 전에 여유 공간을 확인하고(키별 최소 여유: 1B 12, 2B 14, 4B 20, UI-TARS 52 GB) 모자라면 그 모델을 건너뛴다. 결과 tar는 모델당 수십~수백 MB로 추정(미검증).

## 위험

| 위험 | 크기 | 대처 |
| --- | --- | --- |
| 시간과 비용 | 19~26 h 추정, 요금 확인 필요 | `ONLY`로 일부만, 도중에 tmux에서 Ctrl-C로 멈추고 같은 명령으로 재개 |
| 8B 캐시 삭제 | 8B를 다시 돌리려면 약 34.6 GB를 다시 내려받아야 함 | 8B 결과 tar는 큐 시작 전에 내려받아 둔다 |
| 실제 로드 실패 | 모델 하나가 smoke에서 멈춤. 다음 모델로 계속 진행 | `queue_logs/<키>_smoke.log`를 보내준다. 설정을 임의로 고치지 않는다 |
| UI-TARS가 라벨만 출력하지 않음 | GUI 에이전트용 모델이라 파싱 실패가 많을 수 있음 | 실패율 50% 초과면 게이트가 멈추고 smoke 결과는 tar에 남는다 |
| InternVL 원격 코드 | 1B/2B/4B는 8B와 같은 원격 코드 구조이나 실제 로드는 미검증 | smoke에서 확인됨 |
| 큐가 중간에 STOP | `QUEUE_DONE`에 `STOPPED rc=1`이 써지고 뒤의 MiniCPM 큐는 그것을 신호로 시작한다(MiniCPM 큐는 `run_all_overnight.sh`가 아직 돌고 있으면 그것이 끝날 때까지 더 기다린다) | `QUEUE_STATUS`의 STOP 사유를 보고 고친 뒤 다시 실행 |
| 이전 작업이 스스로 STOP | FINISHED가 안 찍힘 | 10분 뒤 이 큐도 STOP(위). 이전 작업 로그를 먼저 본다 |
| 8B 배치 1 점검이 GPU를 쓰는 15~25분 | 점검 중에 다른 GPU 작업이 있으면 메모리 부족 | 앞의 GPU 비어 있음 확인 뒤에만 시작한다. 실패해도 큐는 계속한다 |
| 실패한 모델의 캐시가 디스크를 차지 | 최대 33 GB(UI-TARS) | 위 캐시 규칙. 남은 모델의 최소 여유를 못 채우면 지운다 |
| timeout이 걸린 단계 | 그 단계는 부분 결과만 남음(`_PARTIAL`) | 같은 명령으로 재개하면 이어서 돈다 |
| GPU 동시 사용 | 이전 작업이 남아 있으면 메모리 부족 | FINISHED 확인 후 GPU 프로세스가 사라질 때까지 최대 30분 기다리고, 안 사라지면 멈춘다 |

## 로컬에서 한 검사 (모두 통과)

- `bash -n queue_sizes_uitars.sh`, `bash -n addon/followup_rel_4lang/model_expansion/run_model.sh`
- 6개 설정 모두 `run_inference.py --dry-run`(고유 추론 18,432 확인), `run_rel_T_generic.py --dry-run`, 두 러너의 `--mock-model` smoke와 전체 실행, 게이트 통과. 게이트는 파싱 실패 61%, 행 누락, 배치 크기 3, 실패 행을 만든 가짜 결과에서 실패로 판정하는 것도 확인했다.
- 가짜 `/workspace`와 가짜 `hf`/`nvidia-smi`/`timeout`/`df`로 큐 전체를 돌려 확인한 것: 다운로드 실패 모델 건너뛰기, 시간 초과 단계의 `_PARTIAL` 처리, 디스크 부족 건너뛰기, 이미 끝난 모델 건너뛰기, QUEUE_DONE 생성.
- 11월 14일 수정 뒤 다시 한 검사(`local_tests/`, 결과 파일은 같은 폴더):
  - `python -I local_tests/test_sz_gate_rules.py <check_generic_gate.py> <임시폴더>`: 새 게이트 규칙마다 통과/실패 경계 값을 만든 가짜 출력으로 시험(라벨 전부 같음, A–D 질량 0.2·0.5, 필드 없음, 시각/프롬프트 토큰 범위 경계 2048/2049, 2674/2675, InternVL 1792/2304/1536/2560, 정확도 경고, ETA 계산). 25개 모두 통과(`sz_gate_rules_result.txt`).
  - `bash local_tests/run_sz_queue_tests.sh all`: 가짜 `/workspace`에 가짜 `hf`/`nvidia-smi`/`curl`/`timeout`/`df`/`flock`과 모의 추론으로 큐 전체를 돌린다. 확인한 것: 오버나이트 ok/ok에서 배치 1 점검(1,008회, 비교 672행)과 8B 캐시 삭제, qwen3=ok internvl35=FAILED에서 점검은 하고 캐시는 유지, qwen3=FAILED에서 점검 건너뜀, 프로세스 없음 STOP(grace), 프로세스가 있어도 상한에서 STOP, 한쪽 저장소 없음(그 계열만 건너뜀, 대기 중에 저장소가 생겨도 인식), GPU 1,500 MiB 이상 STOP(`SKIP_WAIT=1`에서도), 두 번째 실행 거부, SIGTERM이면 `QUEUE_DONE` 없음, 네트워크 느림/무응답 STOP과 `FORCE_SLOW_NET=1`, 내려받기 재시도, 실패한 내려받기·시간 초과의 캐시 유지, 재실행 때 재다운로드 없음, 디스크 부족 시 캐시 삭제, ETA 중단. 63개 모두 통과(`sz_queue_tests_result.txt`).
  - `bash local_tests/run_sz_tests.sh`: 6개 모델의 `--dry-run`, 모의 smoke, 모의 REL 18,432회, 모의 실험 A 5,688회와 게이트(60단계 모두 exit 0, `sz_test_summary.txt`).
  - 위 모든 시험은 가짜 `timeout`이 GNU처럼 `1h30m` 같은 값을 거부하게 한 뒤의 결과다.
- 묶음 sha256(11월 14일 수정판): `model_expansion_sizes_uitars_20261114.tar.gz` = `79367d4108b2ba4b4d8049286158753c68e14a65291936feaacedf85497ffa7b`(`ADDON_SHA256.txt`, `queue_sizes_uitars.sh`의 `ADDON_SHA`와 같다). 이전 판은 `15ef9b21…8763`이고 `superseded/`에 있다. `queue_sizes_uitars.sh`는 `SHA256SUMS`에 있다. 서버에서는 `sha256sum -c`로 확인한다.
- 못 한 것: 실제 모델 로드와 생성, 실제 GPU 시간, 서버의 `timeout`/`hf`/`df -BG` 동작, FINISHED 대기 루프를 실제 서버 파일로 확인하는 것. Mac에서 InternVL3.5-1B CPU 로드 시험은 로컬 파이썬 환경에 PIL이 없고 transformers 5.x라 서버(4.57.6)와 달라 의미가 작아 건너뛰었다.
