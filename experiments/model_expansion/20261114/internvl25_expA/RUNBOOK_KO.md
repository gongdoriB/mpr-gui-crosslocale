# InternVL2.5-8B 실험 A 실행 안내 (RTX 4090 한 장, 큐의 맨 끝)

작성: 2026-11-14. 서버에서는 아무것도 실행하지 않았고, 올리지도 않았다. 로컬에서 확인한 것은 6번에, 확인하지 못한 것은 7번에 적었다(미검증, 확인 필요 표시).

## 1. 무엇을 돌리나

| 항목 | 내용 |
| --- | --- |
| 모델 | `OpenGVLab/InternVL2_5-8B`, 커밋 `e9e4c0dc1db56bfab10458671519b7fa3dd29463` (HF API로 확인, 2026-11-14. 이 저장소의 `main`이 지금도 이 커밋이고 마지막 수정은 2025-03-25다. 2026-10-05 REL 실행도 같은 커밋이다.) |
| 크기 | 파라미터 8,075,365,376개(bf16). safetensors 4조각 합계 16,150,800,432바이트(16.2GB). 저장소 전체는 예시 이미지·로그를 합쳐 16.2GB에 몇 MB가 더 붙는다. |
| 언어모델 | InternLM2.5-7B-chat. `config.json`의 `llm_config.architectures`는 `InternLM2ForCausalLM`, `_name_or_path`는 `internlm/internlm2_5-7b-chat`, 모델 카드 표도 같다. 비전은 InternViT-300M-448px-V2_5. Qwen 계열이 아닌 언어모델이다. |
| 라이선스 | 모델 카드에 MIT, 언어모델은 Apache-2.0으로 적혀 있다(모델 카드 기준, 별도 라이선스 문서는 읽지 않음). |
| 실험 | 실험 A O/T/R. 1,896쌍 × 3조건 = 5,688회, 배치 1. 주 분석 제외 항목 `rel::5/forest_9.jpg::zh::ru`는 T 입력 파일의 main 분할에 들어 있다. 이 쌍도 돌리고(5,688에 포함), 제외는 분석 단계에서 한다. |
| 프롬프트 | english_fixed 시스템 프롬프트, `mpr_label_only_v1` 사용자 프롬프트, 그리디 `max_new_tokens=2`, A–D 첫 토큰 점수. 다른 모델의 실험 A와 같다. |
| 모델 설정 | 2026-10-05 REL 실행과 같은 `internvl2_5_8b.yaml`(바이트 단위로 같다). eager 어텐션(flash-attn 설치 없음), `max_num` 7 + 썸네일, 최대 2,048 시각 토큰, 448px 타일당 256토큰. |
| 코드 | 서버의 `/workspace/mpr-ivl`(커밋 `6664400` + InternVL3.5 패치 0001–0005)을 그대로 쓴다. 패치 0001이 `internvl2_5` 계열을 같은 어댑터 경로로 받고, 시스템 프롬프트를 명시하면 모델의 기본(중국어) 시스템 메시지를 대체하게 한다. 새 패치는 만들지 않았다. |
| 추가분 | 묶음 `internvl25_expA_addon_20261114.tar.gz` 하나. `followup_rel_4lang/expA_ivl25/`라는 새 폴더만 만든다: 실행 스크립트, `run_rel_T_generic.py`(sizes 묶음의 파일에서 허용 계열에 `internvl2_5` 하나만 추가), `check_generic_gate.py`(InternVL2.5 점검 규칙 추가), 설정 yaml, 이전 REL 실행 계약 JSON(시스템 프롬프트 대조용). |

기존 파일은 수정하지 않는다. `mpr-ivl`에는 새 폴더가 하나 생기고 결과 폴더와 모델 캐시가 생긴다.

## 2. 올릴 파일과 실행

`internvl25_expA/`의 세 파일을 서버 `/workspace/`에 올린다: `queue_internvl25_expA.sh`, `internvl25_expA_addon_20261114.tar.gz`, `SHA256SUMS`.

```bash
cd /workspace && sha256sum -c SHA256SUMS
tmux new-window -n ivl25      # 다른 큐가 도는 tmux 안이면 새 창. 새 세션이면 tmux new -s ivl25
bash queue_internvl25_expA.sh --plan   # 아무것도 바꾸지 않고 단계만 출력
bash queue_internvl25_expA.sh          # /workspace/QUEUE6_DONE이 생길 때까지 잠만 자고 기다린다
```

대기 중에는 60초마다 파일 하나를 확인할 뿐이라 앞 큐에 영향이 없다. `queue_6x6.sh`가 시작할 때 낡은 `QUEUE6_DONE`을 지우므로, 6x6 큐를 한 번도 시작하지 않은 상태에서 낡은 `QUEUE6_DONE`이 남아 있으면 이 스크립트가 바로 시작한다. 확인하고 올린다.

`queue_internvl25_expA.sh`가 하는 일(순서대로):

1. `/workspace/QUEUE6_DONE` 대기(최대 168시간, `MAX_WAIT_H`로 바꾼다). 내용은 상관없다. 파일이 생긴 뒤에도 `run_all_overnight.sh`, `queue_6x6.sh`, `queue_minicpm.sh`, `queue_sizes_uitars.sh` 중 하나라도 돌고 있으면 더 기다리고, 그다음 GPU 메모리 사용이 1.5GB 아래가 될 때까지 최대 30분 본다.
2. 네트워크 확인: huggingface.co(이 모델의 첫 조각)와 download.pytorch.org에서 15초씩 범위 요청으로 받아 MB/s를 `QUEUE7_STATUS`에 적는다. huggingface.co가 20 MB/s 미만이거나 응답이 없으면 멈춘다(`FORCE_SLOW_NET=1`로 강행). download.pytorch.org는 기록만 한다(torch를 새로 받지 않기 때문).
3. 점검: 묶음 sha256, GPU가 RTX 4090, `/workspace` 빈 공간 30GB 이상(이미 받은 캐시가 있으면 생략), `/workspace/mpr-ivl`에 `.git`과 `.venv`가 있음. 없으면 새로 만들지 않고 멈춘다.
4. 설정(제한 45분): 저장소가 `6664400`의 후손이고 패치 표지(`INTERNVL_FAMILIES`, `system_message if system_prompt is None`)가 있는지 확인, 묶음 풀기와 해시 검사, 묶음 안 yaml이 저장소의 `internvl2_5_8b.yaml`과 같은지 `cmp`, venv가 torch 2.9.1+cu130·transformers 4.57.6·timm·einops·sentencepiece를 갖췄는지 확인, 실험 A 입력 해시 검사, `data_bundle.py verify`(이미지가 없으면 내려받기). venv가 맞지 않을 때만 저장소의 제약 파일로 `pip install`을 한다.
5. 모델(제한 120분): `hf download`를 위 커밋으로(최대 3번, 받다 만 파일은 캐시에 남아 이어 받는다). safetensors 4조각 합계가 16,150,800,432바이트인지, `llm_config.architectures`가 `InternLM2ForCausalLM`인지, 어댑터가 쓰는 토크나이저 로더가 로드되고 A, B, C, D가 각각 한 토큰(id 290, 309, 289, 302, 이전 실행 기록과 같아야 함)인지 확인한다.
6. smoke(제한 40분): 3문항 × 12쌍 × 3조건 = 108회와 게이트. 게이트가 실패하면 본 실행으로 가지 않는다. 정확도·추론당 중앙값·예상 전체 시간이 `QUEUE7_STATUS`에 한 줄로 남는다.
7. 본 실행(제한 3시간): 5,688회. 시작 전에 smoke 중앙값 × 5,688로 예상 시간을 계산해 3시간보다 길면 시작하지 않고 `STOP step full_expa ...`를 남긴다(`IGNORE_ETA=1`이면 그냥 시작). 끝나면 5,688행이 모두 성공이고 게이트 규칙을 통과하는지 확인한다.
8. 결과를 `/workspace/queue_results/internvl25_expA/internvl25_expA_results_YYYYMMDD.tar.gz`(+ `.sha256`)로 묶고, 성공했으면 이 모델의 캐시 폴더(`mpr-ivl/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B`)만 지운다. 실패하거나 시간이 초과되면 캐시를 남겨 다시 실행할 때 다시 받지 않게 한다. 마지막으로 `/workspace/QUEUE7_DONE`에 `OK` 또는 `FAILED steps: ...`를 쓴다. Ctrl-C/kill로 끊으면 `QUEUE7_DONE`을 만들지 않는다(같은 명령을 다시 치면 이어서 한다).

게이트 규칙(`check_generic_gate.py expa`, 어기면 실패): 모든 행 성공, 배치 크기 1, 파싱 실패 50% 이하, 파싱된 라벨이 전부 같으면 실패(20행 이상일 때), A–D 확률 질량 중앙값 0.5 미만이면 실패, 시각 토큰 수가 768~1792를 벗어나면 실패, 모든 `rendered_prompt`에 english_fixed 문장이 있고 기본 중국어 시스템 메시지("书生")가 없어야 한다. 시각 토큰 범위는 2026-10-05 REL 실행에서 실제로 나온 값(768이 13,984행, 1,792가 4,448행)이다. smoke 정확도가 0.30 미만이면 경고만 한다.

진행 확인:

```bash
tail -n 20 /workspace/QUEUE7_STATUS
tail -n 3 /workspace/mpr-ivl/followup_rel_4lang/results/expA_internvl25_8b/logs/expa_full.log
cat /workspace/mpr-ivl/followup_rel_4lang/results/expA_internvl25_8b/STATUS
```

받을 결과: `queue_results/internvl25_expA/` 아래 tar.gz 하나(`expA_internvl25_8b/`의 smoke·full 예측과 게이트 JSON, 로그, `meta/`의 패키지 목록·커밋·nvidia-smi, 큐 로그). 인스턴스는 자동으로 끄지 않는다.

## 3. 시간·디스크 추정 (미검증)

근거는 둘이다. 같은 4090의 InternVL3.5-8B 실험 A는 5,688회에 약 70분(REL 실측 0.727초/회, 시각 토큰 평균 약 2,180). InternVL2.5-8B는 시각 토큰이 평균 약 1,015(768이 76%, 1,792가 24%)라 같은 4090에서 회당 더 짧을 가능성이 크다. 2026-10-05 REL 실행(B200 MIG 2g.45gb 조각, 18,432회)의 회당 중앙값은 0.27초, 평균 0.36초였다. 4090에서의 값은 측정한 적이 없으므로 회당 0.3~0.6초로 짐작한다.

| 단계 | 추정 |
| --- | --- |
| 모델 내려받기 16.2GB | 3~14분(20~100 MB/s) |
| 설정 + 이미지 확인 | 2~10분 |
| smoke 108회 + 모델 로드 | 5~10분 |
| 실험 A 5,688회 | 30~60분 (3시간 안에 못 끝날 예상이면 시작하지 않는다) |
| 합계 | 약 1~1.5시간 (큐 대기 제외) |

디스크는 모델 16.2GB가 최대 증가분이고 결과는 수십 MB다. 점검 기준은 30GB(16.2 × 1.3 + 8). 성공하면 모델 캐시를 지워 16.2GB가 돌아온다. 앞 큐들이 이미 `mpr-ivl`을 만들어 둔 상태가 전제다.

## 4. 위험과 대처

| 위험 | 크기 | 대처 |
| --- | --- | --- |
| 6x6 큐가 중간에 STOP해 `QUEUE6_DONE`이 안 생김 | 이 스크립트가 최대 168시간 잠만 자고 기다림. 인스턴스 요금은 앞 큐가 멈춘 시점부터 이 대기만큼 낭비될 수 있음 | `queue_6x6.sh`는 끝까지 갔을 때만 `QUEUE6_DONE`을 만든다(조기 STOP에는 안 만듦). 30분마다 `QUEUE7_STATUS`에 "queue_6x6.sh running / NOT running"과 `QUEUE6_STATUS`의 마지막 줄을 적는다. 앞 큐가 죽었고 GPU가 비었으면 `touch /workspace/QUEUE6_DONE`으로 시작시킨다 |
| 앞 큐가 GPU를 쓰는 중에 시작 | 24GB에 두 모델이 안 들어가 둘 다 죽을 수 있음 | 다른 큐/overnight 프로세스가 없고 GPU 사용 1.5GB 미만일 때만 진행 |
| 24GB OOM | 가중치 16.2GB + 시각 토큰 최대 1,792 + eager 어텐션. 24GB 카드에서의 최대 사용량은 측정하지 못함. 이전 REL 실행은 45GB 조각이라 24GB 적합 여부를 보여 주지 못한다. 같은 크기의 InternVL3.5-8B(시각 토큰 최대 2,304)가 4090에서 돌아간 것이 간접 근거다 | 배치 1. 연속 3회 실패하면 멈추고 `FAILED`로 기록. 이때는 임의로 설정을 바꾸지 말고 로그를 보내 판단을 받는다 |
| 원격 코드와 패키지 불일치 | 모델 로드 단계에서 실패 | 6번 참고. 이전 실행과 같은 torch/transformers 조합이고 로컬에서 같은 원격 코드로 작은 임의 가중치를 돌려 통과시켰다. 실제 가중치·4090으로는 미검증. 실패하면 모델/smoke 단계에서 멈추고 `QUEUE7_DONE`에 `FAILED steps`가 적힌다 |
| 모델 코드를 Hub에서 받아 실행(`trust_remote_code`) | 외부 코드 실행 | 커밋을 고정했다. 해당 커밋의 .py 8개를 로컬에 받아 `os.system`, `subprocess`, 네트워크, `eval/exec` 호출이 없는지 검색했다(없음). 줄 단위로 읽어 감사한 것은 아니다 |
| 공유 저장소 변경 | `mpr-ivl`에 새 폴더 `followup_rel_4lang/expA_ivl25/`, 결과 폴더, 모델 캐시가 생김 | 기존 파일은 덮어쓰지 않는다. venv는 점검이 실패할 때만 제약 파일 기준으로 `pip install`한다(이 시점에는 다른 큐가 모두 끝나 있다). 지우는 것은 이 모델의 캐시 폴더 하나뿐 |
| 네트워크가 느리거나 끊김 | 16.2GB 내려받기 실패 | 20 MB/s 점검, 내려받기 3번 재시도(30초·60초 간격), 받던 조각은 이어 받는다. 그래도 실패하면 같은 명령을 다시 친다 |
| 디스크 부족 | 내려받기 실패 | 시작 전 30GB 점검. 다른 폴더를 지우지 말고 알려준다 |
| 점검 규칙이 smoke를 막음 | 라벨이 전부 같음, A–D 질량 낮음, 시각 토큰 범위 밖, 시스템 프롬프트가 안 바뀜 | `QUEUE7_STATUS`의 `smoke gate` 줄과 `expa_smoke/gate_expa_smoke.json`의 `problems`를 본다. 범위나 가정이 틀렸을 수도 있으니 임의로 풀지 말고 알려준다 |
| 시간 초과 | 본 실행 3시간 제한 | 예상이 제한보다 길면 시작하지 않는다. 시작한 뒤에 넘으면 `TIMEOUT`으로 기록하고 캐시와 부분 예측을 남긴다. 같은 명령을 다시 치면 이어서 한다 |
| 비용 | 앞 큐가 끝난 뒤 1~1.5시간 더 4090 임대 | 인스턴스를 끄는 판단은 사람이 한다. 결과를 모두 받은 뒤 Destroy(Stop은 디스크 요금이 남는다) |

## 5. 설계에서 정한 것

- 실험 A만 돌린다. REL O/R/C/F는 이 모델의 2026-10-05 결과(18,432회, 파싱 실패 0)가 이미 있다.
- 저장소는 `mpr-ivl`을 쓴다. 이 저장소에는 `internvl2_5` 어댑터 경로와 시스템 프롬프트 대체 패치가 들어 있고, venv에 timm·einops·sentencepiece가 있다. 별도 저장소를 새로 만들면 clone과 설치로 시간만 쓴다.
- 모델 설정은 2026-10-05 실행의 설정을 그대로 쓴다(`max_num` 7, 2,048 시각 토큰 상한). InternVL3.5-8B 실행(`max_num` 8, 2,304)과 타일 예산이 다르다는 점은 논문에서 밝혀야 한다.
- `--reference-contract`로 2026-10-05 REL 실행 계약을 주었다. 시스템 프롬프트와 템플릿이 그 계약과 같지 않으면 멈춘다. 모델 설정 차이를 로컬 `--dry-run`으로 비교했더니 차이가 없었다(`plan.json`의 `config_keys_that_differ: []`).
- 아이디어(실행하지 않음): O/R는 2026-10-05 REL 결과를 재사용하고 T 1,896회만 돌리면 약 3분의 1 시간이다. 다만 하드웨어(B200 MIG 조각 대 4090)와 패키지 버전이 달라 같은 입력에서도 일부 답이 뒤집힐 수 있고, 그러면 O/T/R 쌍비교에 하드웨어 잡음이 섞인다. 그래서 세 조건을 한 번에 돌리는 쪽으로 만들었다.

## 6. 로컬에서 한 검증

환경: Mac(arm64), Python 3.12, torch 2.9.1, transformers 4.57.6(서버와 같은 버전), timm 1.0.31, sentencepiece 0.2.0. 결과 파일은 `local_tests/`에 있다.

- 토크나이저와 프롬프트(`check_tokenizer_prompt.py`, 결과 `tokenizer_prompt_result.txt`): 해당 커밋의 `tokenizer.model`·`tokenization_internlm2.py`·`conversation.py`·`config.json`을 받아(모델 가중치는 받지 않음) 어댑터의 `_encode_question`을 돌렸다. A, B, C, D는 각각 한 토큰이고 id가 290, 309, 289, 302로, 2026-10-05 서버 실행의 `label_token_ids`와 같다. english_fixed를 주면 렌더된 프롬프트의 시스템 메시지가 영어 문장으로 대체되고 "书生" 기본 메시지는 사라진다. `None`을 주면 기본 중국어 메시지가 그대로 남는 것도 확인했다. 렌더된 형태는 서버 실행의 `rendered_prompt` 기록과 같다.
- 원격 코드 로드(`tiny_remote_code_test.py`, 결과 `tiny_remote_code_result.txt`): 같은 커밋의 원격 코드 파일로, 모델 크기만 줄인 임의 가중치(약 1,200만 파라미터) 모델을 만들어 `AutoModel.from_pretrained(trust_remote_code=True)`, 어댑터의 `ensure_generation_mixin`, eager 어텐션, `generate_one`(첫 토큰 A–D 점수)까지 CPU float32로 끝까지 돌렸다. 타일 3개 = 시각 토큰 768, 프롬프트 858토큰으로 서버 기록의 최소값과 같다. 출력은 의미 없는 글자다(배관 확인용).
- 실행 스크립트 모의 실행: 복제한 `mpr-ivl` 상태(커밋 `6664400` + 패치 5개, 데이터·이미지 포함)에서 `run_expA_ivl25.sh smoke`와 `full`을 `--mock-model`로 끝까지 돌렸다. 계획이 5,688회·1,896쌍, `config_keys_that_differ` 없음, smoke 108행, 전체 5,688행 게이트 통과. ETA 중단(종료 코드 3)과 `IGNORE_ETA=1`도 확인했다.
- 게이트 규칙 단위 시험(`test_gate_rules.py`, 결과 `gate_rules_result.txt`): 정상(768, 1792)은 통과, 시각 토큰 범위 밖, 기본 중국어 시스템 메시지, 라벨 전부 동일, A–D 질량 낮음은 각각 실패 7개 모두 기대대로.
- 큐 스크립트(`run_ivl25_queue_tests.sh`, 결과 `queue_tests_result.txt`): `--plan`, `bash -n`, 가짜 `nvidia-smi`/`flock`/`timeout`/`curl`/`hf`/`df`를 붙인 흐름 시험. "real" 시험은 실제 `setup()`, 실제 `model()`(진짜 소형 파일 + 실제 크기의 빈 safetensors로 만든 가짜 캐시, 오프라인), 실제 실행 스크립트(모의 모델), 실제 게이트를 큐 전체로 돌린다. "stub" 시험은 무거운 단계만 가짜로 바꿔 대기, 다른 큐 대기, GPU 대기, 네트워크 점검, 재시도, 시간 초과, ETA 종료 코드, 캐시 규칙, 중복 실행 거부, SIGTERM을 확인한다. 잘못된 저장소·설정·torch 버전·묶음 해시·디스크·safetensors 크기에서 멈추는 음성 시험도 있다. 최종 판 기준 62개 확인이 모두 통과했다(`queue_tests_result.txt`). 시험에서는 pip 색인 주소를 막아 두어 어디에도 접속하지 않는다.
- 이전 실행에서 가져온 사실(서버 `server_sync_4090_20261114/mirror/.../results/full/internvl2_5_8b/`의 `run_contract.json`·`predictions.jsonl.gz`): 2026-10-05 REL 18,432회, 성공 23,424행 전부, 파싱 실패 0, torch 2.9.1+cu130·transformers 4.57.6·accelerate 1.15.0·timm 1.0.30·sentencepiece 0.2.0, Python 3.11.11, GPU는 B200 MIG 2g.45gb. 이 기록이 6번의 호환성 근거다.

묶음 sha256: `internvl25_expA_addon_20261114.tar.gz` = `32031bdf30e153b369f6e85e32f03848747f1afd546386fe8457e40f2e7cffda`(`queue_internvl25_expA.sh`의 `ADDON_SHA`와 같다), 큐 스크립트는 `SHA256SUMS`에 있다. 묶음을 다시 만들면(`build_addon.sh`) 해시가 바뀌므로 `SHA256SUMS`와 스크립트가 함께 갱신된다.

## 7. 검증하지 못한 것 (미검증)

- 실제 16.2GB 가중치 로드, 4090(24GB)에서의 최대 GPU 메모리, 실제 속도, 실제 출력 품질. 3번의 시간 추정은 짐작이다.
- torch 2.9.1 / transformers 4.57.6에서 이 모델의 원격 코드가 동작하는지: 같은 버전 조합의 서버 실행 기록(2026-10-05)과 로컬 임의 가중치 시험이라는 두 근거가 있다. 다만 이번 서버 venv는 Python 3.12, timm 1.0.31(이전은 Python 3.11.11, timm 1.0.30)이고 GPU도 다르다. 이 조합에서 실제 가중치로 돌려 본 적은 없다. 모델 로드 단계에서 실패하면 큐는 거기서 멈추고 로그에 이유가 남는다. 이 경우 대안은 이전 실행의 패키지 구성(Python 3.11, timm 1.0.30)을 맞춘 별도 venv이고, 그 가능성은 확인 필요다. 기존 venv는 건드리지 않는 쪽으로 제안한다.
- 큐 스크립트의 설정·모델·실행 단계를 진짜 서버에서 돌려 본 적이 없다. 재시도 줄(`retry 3 python -m pip install`, `retry 6 ... fetch-images`, `retry 3 hf download`)은 가짜 `hf`와 문자열 검사로만 확인했다. 가짜 `hf`로 확인한 것은 재시도뿐이고 실제 16.2GB 내려받기는 해 보지 않았다.
- 서버 `mpr-ivl`의 실제 상태가 로컬 복제와 같은지(패치 5개, venv 버전, `expA_inputs`, 이미지). 로컬 복제는 이 폴더의 다른 모델 시험에서 쓰던 `mpr-ivl` 재현본이다. 서버의 상태는 큐 스크립트의 설정 단계가 점검하고, 어긋나면 멈춘다.
- 네트워크 점검에 쓰는 URL(`resolve/<커밋>/model-00001-of-00004.safetensors`)이 실제로 응답하는지는 조회하지 않았다(파일 이름은 HF API 목록에서 확인). 404가 나오면 경고만 하고 계속한다.
- 시각 토큰 범위 768~1792는 REL 이미지에서 센 값이다. 실험 A는 같은 이미지 집합이라 같다고 보지만 실행 전에는 확인하지 못했다. 게이트가 어기면 멈춘다.
- 이 모델의 O/R 답이 2026-10-05 REL 실행의 같은 입력 답과 얼마나 일치하는지(앵커 비교)는 큐에 넣지 않았다. 분석 단계에서 할 수 있으나 입력 id 대응은 만들지 않았다(확인 필요). 하드웨어가 달라 일부는 다를 수 있다.
- 결과 해석(정확도, T 조건 효과)은 하지 않았다.
- 같은 맥북에 이전 세션에서 시험하다 남은 `queue_minicpm.sh` 프로세스 두 개가 떠 있다(로컬 가짜 작업 공간, 서버와 무관). 이 시험은 그 프로세스 이름에 걸리지 않게 패턴을 바꿔서 돌렸다.

## r1 수정 (2026-11-15, 로컬만, 서버 미업로드)
- 6×6 대기열이 일찍 STOP하면 QUEUE6_DONE이 안 생겨 이 대기열이 최대 168시간 기다리던 문제를 고침. queue_6x6.sh가 돌지 않고 마지막 상태 줄이 STOP·interrupted·FINISHED이면 두 번 확인 뒤 진행. 이후 GPU 유휴·다른 대기열 확인은 그대로.
- 로컬 시험 stop6 4개 통과, 기존 wait 시험 6개 통과. 실제 서버 실행은 미검증. 이전 판은 superseded/에 보관.
- 업로드·실행은 사용자 승인 뒤.

## r2 수정 (2026-11-15, 로컬만)
- 다른 대기열이 도는지 보는 pgrep 검사가 tmux 서버 프로세스 이름("tmux new-session ... bash run_all_overnight.sh")에 걸려 영원히 기다리던 문제(서버의 MiniCPM·6×6 대기열에서 실제로 발생)를 고침. 명령줄이 bash/sh로 시작하는 프로세스만 센다.
- 가짜 tmux 프로세스는 무시하고 실제 스크립트는 잡는 것을 로컬에서 확인. wait·stop6·others 시험 통과. 서버 실행은 미검증.

## r3 (2026-11-15)
- 서버 /etc/environment가 HF_HOME=/workspace/.hf_home를 정해 두어, smoke 단계가 모델을 못 찾고 실패(서버 실제 발생). smoke_expa·full_expa 안에서 캐시 변수를 직접 export. 잘못된 HF_HOME을 미리 둔 조건과 원래 조건 모두 real 시험 통과.
