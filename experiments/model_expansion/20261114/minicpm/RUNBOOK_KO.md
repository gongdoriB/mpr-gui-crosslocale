# MiniCPM-V-4.5 실행 안내 (RTX 4090 한 장, 큐의 마지막 작업)

작성: 2026-11-14, 같은 날 리뷰 수정 반영(수정 전 파일은 `superseded/`에 `_v1`로 남겼다). 서버에서는 아무것도 실행하지 않았다. 로컬에서 만든 것과 검증한 것은 6번에, 검증하지 못한 것은 7번에 적었다(미검증 표시).

## 1. 무엇을 돌리나

| 항목 | 내용 |
| --- | --- |
| 모델 | `openbmb/MiniCPM-V-4_5`, 커밋 `14a2004908140699317e2109e90573c1e6323c36` (Hugging Face API로 확인, 2026-11-14) |
| 크기 | 파라미터 8,695,895,280개(bf16), safetensors 17,391,790,560바이트(17.4GB). 언어모델 Qwen3-8B + SigLIP. bf16로 24GB에 들어간다(가중치 17.4GB). 8B가 안 들어갈 이유가 없어 MiniCPM-V-4(4B)는 쓰지 않는다. |
| 코드 | 팀원 저장소 `followup-rel-4lang-inference` 브랜치 커밋 `6664400`에 어댑터 패치 1개를 얹는다. 폴더는 `/workspace/mpr-minicpm`, 가상환경은 그 안의 `.venv`. `mpr-gui-crosslocale`, `mpr-ivl`은 건드리지 않는다. |
| 패키지 | 다른 Qwen3 실행과 같은 고정: torch 2.9.1+cu130, transformers 4.57.6, accelerate 1.15.0, huggingface-hub 0.36.2, tokenizers 0.22.2, pillow 13.0.0, numpy 2.5.4 (`01_code/requirements-qwen3-4090.txt`) |
| 실험 1 | REL O/R/C/F, `rel_all_366` 전체. `followup_rel_4lang/run_inference.py --model-config ... --batch-size 1 --resume`. 실제 추론 18,432회(조건 행 23,424개). 11월 7일·9일 실행과 같은 호출이다. |
| 실험 2 | 실험 A O/T/R, 주 분석 1,896쌍 × 3 = 5,688회. `expA/run_rel_T.py --model generic`. 이전 실행이 없는 모델용 모드를 추가했다(앵커 비교 없음). |
| 공통 | english_fixed 시스템 프롬프트, `mpr_label_only_v1` 사용자 프롬프트, 그리디 `max_new_tokens=2`, 배치 1, A–D 첫 토큰 점수 |

## 2. 어댑터 설계 (`01_code/src/mpr_crosslocale/models/minicpmv.py`)

- 프롬프트: 모델의 chat 템플릿에 `system`=english_fixed 문장, `user`=`(<image>./</image>)\n` + `mpr_label_only_v1` 문구를 넣는다. `enable_thinking=False`라서 템플릿이 어시스턴트 시작에 빈 `<think>\n\n</think>\n\n`를 붙인다. 모델 자체의 `chat()`은 샘플링/빔서치가 기본이라 쓰지 않고, 같은 전처리 뒤 `generate()`를 그리디로 부른다.
- 이미지: 모델 자체 슬라이싱(`max_slice_nums=9`, 슬라이스당 448px 안팎)을 그대로 쓴다. 슬라이스 하나가 64토큰이라 시각 토큰은 최대 640개다. `rel_all_366`의 이미지 1,464장을 실제 전처리기로 세어 보니 1,444장은 576토큰(9조각), 20장은 640토큰(10조각)이다. 원본은 중앙값 약 2.6MP이고 96%가 2.1MP를 넘는다. Qwen3(`max_pixels` 2,097,152, 최대 약 2,048토큰), InternVL3.5(8타일, 최대 2,304토큰)와 비교하면 입력 화소는 비슷한 크기로 줄어들지만 토큰 수는 3분의 1 이하다. 모델이 압축 방식으로 설계된 결과이고, 9조각은 모델이 학습된 최대치라 올리지 않았다. 논문에는 "시각 토큰 예산이 다르다"고 적어야 한다.
- 점수: 첫 생성 위치의 로짓에서 A, B, C, D 토큰(각각 한 토큰)의 log-softmax를 읽는다. InternVL 어댑터와 같은 `next_token_single_label`이다. 한 가지 다른 점은 로짓을 float32로 올린 뒤 log-softmax를 한다는 것이다.
- 생성: `max_new_tokens=2`, `do_sample=false`, `num_beams=1`. 출력 길이/파싱은 다른 모델과 같다.
- 기록 필드: `visual_token_count`, `num_patches_list`(조각 수), `prompt_token_count`.
- 팀원 코드 변경은 최소로 했다(패치 4개 파일): 어댑터 새 파일, 설정 새 파일, `run_cross_locale._build_model_adapter`에 `minicpmv` 분기, `run_rel_followup`의 모델 계열 허용 목록에 `minicpmv`와 `code_identity`에 어댑터 파일 추가.
- 어텐션: 언어모델 SDPA(Qwen 실행과 같음). SigLIP 비전 탑은 모델 코드가 SDPA를 지원하지 않아 자동으로 eager가 된다. flash-attn은 설치하지 않는다.

## 3. 올릴 파일과 실행

`upload/`의 세 파일을 서버 `/workspace/`에 올린다: `minicpm_addon_20261114.tar.gz`, `queue_minicpm.sh`, `SHA256SUMS`.

```bash
cd /workspace && sha256sum -c SHA256SUMS
tmux new -s minicpm          # 이미 다른 큐가 tmux에 있으면 새 창(Ctrl-b c)
bash queue_minicpm.sh --plan # 아무것도 바꾸지 않고 단계만 출력
bash queue_minicpm.sh        # /workspace/QUEUE_DONE이 생길 때까지 기다린 뒤 시작
```

`queue_minicpm.sh`가 하는 일(순서대로):

1. `/workspace/QUEUE_DONE` 대기(60초 간격, 최대 48시간, `MAX_WAIT_H`로 바꾼다). 파일이 있기만 하면 시작한다. 내용은 `OK`(앞 큐가 끝까지 돌았다)든 `STOPPED rc=1`(앞 큐가 STOP했다)든 상관없다. 앞 큐가 일찍 STOP하면 `QUEUE_DONE`이 먼저 생기는데 이때 `run_all_overnight.sh`가 아직 GPU를 쓰고 있을 수 있어서, 그 프로세스가 있는 동안은 더 기다린다. 그다음 GPU 메모리 사용이 1.5GB 아래가 될 때까지 최대 30분 더 본다. 기본 대기 상한을 72시간에서 48시간으로 줄였다. 앞 큐가 끝날 때 `QUEUE_DONE`을 반드시 쓰게 바꿨으므로 이 상한은 안전망이다(오버나이트 최대 약 12시간 + 앞 큐 19~26시간 추정).
1b. 네트워크 확인: huggingface.co(MiniCPM 모델 파일 한 조각)와 download.pytorch.org에서 각각 15초 동안 범위 요청으로 받아 MB/s를 `QUEUE_STATUS`에 적는다(`queue_results/minicpm/net_check.txt`에도). huggingface.co가 20 MB/s 미만이거나 응답이 없으면 아무것도 받지 않고 멈춘다(torch를 새로 받아야 할 때, 즉 venv에 torch가 없을 때는 download.pytorch.org도 기준에 넣는다). 그래도 하려면 `FORCE_SLOW_NET=1 bash queue_minicpm.sh`. 테스트 URL이 HTTP 4xx를 돌려주면 속도를 모르는 것이라 경고만 하고 계속한다. 이 확인은 앞 큐가 끝난 뒤(GPU 대기 다음)에 한다. 큐 시작 시점의 속도는 몇 시간 뒤의 속도가 아니기 때문이다.
2. 점검: 묶음 sha256, Python 3.12, GPU가 RTX 4090, `/workspace` 빈 공간 40GB 이상. 하나라도 어긋나면 이유를 `QUEUE_STATUS`에 쓰고 멈춘다.
3. 설정(제한 90분): 저장소 clone(최대 5번), 커밋 `6664400` 고정, 묶음 풀기와 해시 검사, `git am`, venv와 패키지(`pip install` 각 3번), 데이터 묶음 `unpack` / `fetch-images`(최대 6번) / `verify`.
4. 모델(제한 120분): `hf download`를 위 커밋으로(최대 3번, 30초·60초 간격, 실패하거나 시간이 지나도 받은 파일은 캐시에 남고 이어 받는다), 파일 크기 검사, 토크나이저·전처리기 로드와 A–D 단일 토큰 확인. 제한 시간을 75분에서 120분으로 늘렸다(재시도가 들어가서).
5. smoke(각 제한 40분): REL 80회, 실험 A 108회. 통과 기준은 모든 행 성공, 파싱 실패 50% 이하, 그리고 아래 점검 규칙이다. 하나라도 실패하면 본 실행으로 가지 않는다. smoke 정확도와 예상 시간은 `QUEUE_STATUS`에 한 줄로 남는다.
6. 본 실행: REL 18,432회(제한 6시간), 실험 A 5,688회(제한 3시간). REL이 실패해도 실험 A는 시도한다. 단계마다 시작 전에 smoke 중앙값 × 남은 추론 수로 예상 시간을 계산해 제한 시간보다 길면 그 단계를 시작하지 않고 `STOP step ...` 줄을 남긴다(`MINICPM_DONE`에는 `full_rel_eta` 같은 이름으로 들어간다). `IGNORE_ETA=1`이면 그냥 시작한다.
7. 결과를 `/workspace/queue_results/minicpm/minicpm_results_YYYYMMDD.tar.gz`(+ `.sha256`)로 묶고, 캐시를 정리하고, `QUEUE_STATUS`에 한 줄 붙이고, `/workspace/MINICPM_DONE`을 만든다(내용이 `OK` 또는 `FAILED steps: ...`). 단계 실패나 점검 실패여도 이 마무리는 항상 한다. Ctrl-C로 끊으면 `MINICPM_DONE`을 만들지 않는다(다시 실행하면 이어서 한다). 캐시 규칙: pip와 torch 캐시는 항상 지운다. 모델 캐시(`.cache/huggingface`)는 전부 성공했으면 지우고, 단계가 실패하거나 시간 초과였으면 남겨서 다시 실행할 때 다시 받지 않게 한다. 다만 남은 디스크 여유가 `NEXT_NEEDS_GB`(기본 52, 뒤 큐 6×6의 UI-TARS 내려받기 기준)보다 작으면 지운다. `KEEP_MODEL_CACHE=1`이면 성공해도 지우지 않는다.

점검 규칙(`check_minicpm_gates.py`, `check_T_gate.py generic`). 앞의 셋은 어기면 게이트 실패다.
- 파싱된 생성 라벨이 전부 같으면 실패(파싱된 행 20개 이상일 때 검사).
- A–D 확률 질량(`label_logprobs`의 exp 합) 중앙값이 0.5 미만이면 실패(필드가 없으면 "not evaluated"로 기록).
- `visual_token_count`가 576~640을 벗어나면 실패. 이 범위는 REL 이미지 1,464장을 실제 전처리기로 센 값이다(576토큰 1,444장, 640토큰 20장). MiniCPM 경로에서 `visual_token_count`가 기록되는지 확인했다. 어댑터 패치가 `visual_token_count`, `num_patches_list`, `prompt_token_count`를 출력에 담고(패치 293~329행), REL 실행기와 실험 A 실행기가 모두 이 필드를 예측 행에 그대로 옮긴다. 임의 가중치 소형 모델로 실제 REL 이미지를 돌린 결과에서 `visual_token_count` 576, `prompt_token_count` 705, `num_patches_list` [9]가 기록된 것을 다시 확인했다. 실험 A 실행기를 소형 모델로 돌려 본 것은 아니고 코드 경로가 같다는 것까지만 확인했다. 필드가 비어 있으면 실패 대신 보고만 한다. 범위를 바꾸려면 `GATE_VISUAL_RANGE=lo,hi`.
- smoke 정확도는 기록하고 0.30 미만이면 경고만 한다(우연 수준 0.25). 추론당 중앙값과 전체 예상 시간(`eta_full_seconds`)도 게이트 JSON에 남는다.

같은 명령을 다시 치면 끝난 설정과 예측은 건너뛰고 이어서 한다. 인스턴스는 자동으로 끄지 않는다.

진행 확인:

```bash
tail -n 20 /workspace/QUEUE_STATUS
tail -n 3 /workspace/mpr-minicpm/followup_rel_4lang/results/minicpm_rel/logs/full.log
cat /workspace/mpr-minicpm/followup_rel_4lang/results/expA_minicpm/STATUS
```

## 4. 시간·디스크 추정 (미검증)

근거는 같은 4090의 InternVL3.5 실측이다. REL 18,432회 평균 0.727초(시각 토큰 평균 2,180) = 3.72시간. MiniCPM은 시각 토큰이 약 576으로 훨씬 짧아 추론당 0.3~0.6초로 본다. 실제 속도는 smoke 게이트 JSON의 `eta_hours_full_rel_at_median`에 남는다.

| 단계 | 추정 |
| --- | --- |
| 설정(패키지 + 이미지 1.3GB) | 20~40분 |
| 모델 내려받기 17.4GB | 10~25분 |
| smoke 2개 + 모델 로드 | 10~20분 |
| REL 18,432회 | 1.5~3.1시간 |
| 실험 A 5,688회 | 0.5~1.0시간 |
| 합계 | 약 3~5.5시간 (큐 대기 시간 제외) |

디스크 최대치는 모델 17.4GB + venv 약 8GB(추정) + 이미지 1.3GB + 결과 1GB 안팎으로 30GB 안팎이다. 점검 기준은 40GB. 끝난 뒤에는 모델 캐시를 지워 venv와 저장소(약 10GB)가 남는다.

## 5. 위험과 대처

| 위험 | 크기 | 대처 |
| --- | --- | --- |
| 큐 앞 작업이 `QUEUE_DONE`을 안 만든다 | 대기 | 앞 큐(`queue_sizes_uitars.sh`)는 이제 STOP해도 `QUEUE_DONE`을 쓴다(Ctrl-C/kill 제외). 48시간 뒤 자동 중단. `QUEUE_STATUS`로 확인하고, 앞 작업이 끝났다면 직접 `touch /workspace/QUEUE_DONE` |
| 앞 작업이 아직 GPU를 쓰는 중에 시작 | OOM으로 둘 다 죽을 수 있음 | `QUEUE_DONE` 뒤에도 GPU 사용 1.5GB 미만을 확인한다 |
| 24GB OOM | 가중치 17.4GB에 활성화 수 GB. 최대 사용량은 실측하지 못함(미검증) | 배치 1이라 가능성은 낮다. 나면 연속 3회 실패에서 멈추고 `FAILED`로 기록된다. 이때는 알려주고 판단을 받는다 |
| 원격 코드와 transformers 4.57.6 불일치 | 모델 로드 단계에서 실패 | 로컬에서 같은 버전·같은 원격 코드로 작은 임의 가중치 모델을 돌려 통과시켰다. 실제 가중치로는 미검증. 실패하면 모델 단계에서 멈춘다 |
| 디스크 부족 | 내려받기 실패 | 시작 전 40GB 점검. 실패하면 다른 폴더를 지우지 말고 알려준다 |
| 모델 코드를 Hub에서 받음 | `trust_remote_code` | 커밋을 고정했고 코드를 읽어 확인했다(원격 .py 파일 7개). 같은 커밋이면 내용이 바뀌지 않는다 |
| 네트워크 중단 | 설정/내려받기 실패 | 각 네트워크 단계는 3~6번 다시 시도한다. 그래도 실패하면 같은 명령을 다시 치면 이어서 한다(받던 모델 파일은 캐시에 남는다) |
| 네트워크가 느림 | 이전 서버가 이랬다 | 큐 시작 시 20 MB/s 점검이 멈춘다. 다른 인스턴스로 옮기거나 `FORCE_SLOW_NET=1` |
| 새 점검 규칙이 smoke를 막음 | 라벨이 전부 같음, A–D 질량 낮음, 토큰 수 범위 밖 | `QUEUE_STATUS`의 `smoke failed` 줄과 `gate_smoke.json`의 `problems`를 본다. 모델 이상일 수도 범위 가정이 틀렸을 수도 있다. 임의로 풀지 말고 알려준다 |
| 큐 스크립트가 잘못 작동 | 다른 작업에 영향 | `/workspace/mpr-minicpm`과 `queue_results/minicpm` 밖은 건드리지 않는다. `QUEUE_STATUS`에는 한 줄씩 추가만 한다 |
| 모델 라이선스 | 공개 연구용 사용 조건이 있을 수 있음 | 확인 필요 (MiniCPM 모델 라이선스 문서를 읽지 않았다) |

## 6. 로컬에서 한 검증

환경: Mac(arm64, 18GB), Python 3.12, torch 2.9.1, transformers 4.57.6(서버와 같은 버전). 실제 MiniCPM-V-4.5 원격 코드와 토크나이저·전처리기를 그대로 쓰고 가중치만 임의로 만든 약 8천만 파라미터 모델을 썼다. 출력은 의미 없는 글자이고 배관 검증용이다.

- 어댑터 직접 실행(float32, CPU, 실제 REL 이미지): 입력 만들기, 슬라이싱, `generate`, 첫 토큰 로짓, A–D 토큰 id(32, 33, 34, 35) 정상. 렌더된 프롬프트 모양 확인.
- REL 실행기 전체 경로(`run_inference.py`와 같은 `run_rel_followup.main`, CUDA 확인과 장치만 패치, bf16): 예측 파일·`results.jsonl`·`summary.json` 정상 생성.
- `--dry-run`, `--mock-model`: REL 전체(18,432회 계획, 가짜 모델 전체), 실험 A 전체(5,688회 계획, 가짜 모델 3문항과 전체)와 두 게이트 스크립트 통과.
- 서버 설정 재현: 원본 저장소를 새로 clone해 `6664400`으로 맞추고 묶음 풀기, 해시 검사, `git am`(성공), `data_bundle unpack/verify`까지 확인.
- 큐 스크립트: `bash -n`, `--plan`, 가짜 `nvidia-smi`/`flock`/`timeout`/`curl`/`hf`/`df`를 붙인 흐름 시험(`local_tests/run_minicpm_queue_tests.sh`). 무거운 단계(설정, 모델, smoke, 전체)만 `local_tests/make_stub_queue.py`가 가짜 함수로 바꾸고 나머지(대기, GPU 확인, 네트워크 점검, 재시도, ETA 중단, 캐시 규칙, 정리, `MINICPM_DONE`)는 실제 스크립트다.
- 점검 규칙 단위 시험: `python3 -I local_tests/test_minicpm_gate_rules.py runners <임시폴더>` (결과 `local_tests/gate_rules_result.txt`).
- 수정판 시험 결과: 점검 규칙 25개(`local_tests/gate_rules_result.txt`), 큐 흐름 46개(`local_tests/minicpm_queue_tests_result.txt`: 내용이 `STOPPED rc=1`인 `QUEUE_DONE`으로 시작, 대기와 상한, `run_all_overnight.sh`가 도는 동안 대기, 네트워크 STOP/강행/404 경고, 내려받기 재시도, 실패·시간 초과 때 캐시 유지와 디스크 부족 때 삭제, ETA 중단과 `IGNORE_ETA=1`, SIGTERM, 중복 실행 거부) 모두 통과.
- 묶음 sha256(11월 14일 수정판): `minicpm_addon_20261114.tar.gz` = `e0dab45c5f0dafc5292b1ebcd8596463bf5bc53bd116a036bf84597558f8c909`(`queue_minicpm.sh`의 `ADDON_SHA`와 같다), `queue_minicpm.sh`는 `upload/SHA256SUMS`에 있다. 이전 판은 `32b272f4…cebdf3`이고 `superseded/`에 있다.
- 재현용 스크립트는 `tests/`에 있다(`make_tiny.py`, `t_adapter.py`, `t_rel_tiny.py`, `t_budget.py`).

## 7. 검증하지 못한 것 (미검증)

- 실제 17.4GB 가중치 로드, GPU 메모리 최대치, 실제 속도, 실제 출력 품질(파싱 실패율).
- `pip install -r requirements-qwen3-4090.txt`가 서버(CUDA 13 템플릿)에서 새 venv에 그대로 설치되는지(다른 실행에서 같은 파일을 썼다).
- `queue_minicpm.sh`의 설정·모델·실행 단계를 진짜 서버에서 돌려 본 적이 없다(무거운 단계는 비우고 흐름만 시험). 재시도 줄(`retry 5 clone_repo`, `retry 3 pip install`, `retry 6 fetch-images`, `retry 3 hf download`)은 스크립트 문자열 검사와 가짜 `hf`로만 확인했다.
- 네트워크 점검에 쓰는 MiniCPM 모델 파일 URL(`resolve/<커밋>/model-00001-of-00004.safetensors`)이 실제로 있는지 확인하지 않았다. 없으면 HTTP 404로 보고 경고만 하고 계속한다.
- 점검 규칙의 MiniCPM 시각 토큰 범위(576~640)는 REL 이미지에서만 센 값이다. 실험 A의 이미지는 같은 이미지 집합이라 같다고 보지만 실제 실행 전에는 확인하지 못했다.
- 실험 A의 T 조건과 앵커 비교는 이 모델에 이전 실행이 없어 하지 않는다. 다른 모델과의 비교는 같은 18,432/5,688 입력을 쓴 결과끼리 한다.
- 결과 해석(MiniCPM의 정확도, 시각 토큰 차이의 영향)은 하지 않았다.

## 8. 받을 결과

`/workspace/queue_results/minicpm/minicpm_results_YYYYMMDD.tar.gz` 하나. 안에: `minicpm_rel/`(smoke, full, 게이트 JSON, 로그), `expA_minicpm/`(같은 구성), `requirements.freeze.txt`, `git_*.txt`, `nvidia_smi.txt`, `STATUS`, 큐 로그. 다 받은 뒤 인스턴스 정리는 사용자가 정한다.
