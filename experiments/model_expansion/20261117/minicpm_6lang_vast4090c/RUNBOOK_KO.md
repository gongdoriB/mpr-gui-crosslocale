# MiniCPM-V-4.5 6개 언어 전체 결과: 서버 `vast4090c` 실행 안내

2026-11-17 작성, 같은 날 독립 검토 반영(수정 전 파일은 `superseded/`의 `*_v1_superseded`). 서버에는 아무것도 올리지도 실행하지도 않았다. 확인한 것은 `local_tests/`에, 확인하지 못한 것은 맨 아래 "미검증"에 있다.

## 무엇을 돌리나

MiniCPM-V-4.5는 지금까지 4개 언어(en, zh, th, ru) REL과 실험 A만 돌았다. 6개 언어(fr, ja 추가)로 채우려면 아래 다섯 가지가 필요하고, 큐 하나(`queue_minicpm_6lang.sh`)가 이 순서로 돌린다.

| 순서 | 내용 | 규모 | 결과 위치 (`/workspace/mpr-minicpm/` 아래) |
|---|---|---|---|
| 1 | 환경 준비(저장소, venv, 데이터, 모델), 6×6 smoke, REL smoke, 실험 A smoke + 게이트 | 216회 + 112회 + 162회 | `followup_rel_6lang/results/{rel6new,expA6}_minicpm_v45/smoke` |
| 2 | 일관성 점검: 예전 4개 언어 REL 추론 600개를 다시 돌려 예전 결과와 비교(보고만, 큐를 멈추지 않음) | 600회 | `.../consistency_minicpm_v45/consistency_agreement.json` |
| 3 | REL O/R/C/F `new_only`(질문 언어 또는 화면 언어가 fr, ja인 행) | 29,148행 = 26,057회 | `.../rel6new_minicpm_v45/full` |
| 4 | 실험 A 새 쌍, main 2,844쌍 × O/T/R + T_retry 148쌍 × 3 | 8,532회 + 444회 | `.../expA6_minicpm_v45/{full,retry}` |
| 5 | 6×6 전체 벤치마크(`queue_6x6.sh`를 그대로 호출, `MODELS=minicpm_v45`) | 79,020회 | `followup_rel_4lang/results/6x6_minicpm_v45` |

6×6이 가장 길어서 맨 뒤에 둔다. 3번과 4번의 입력은 InternVL3.5-8B, Qwen3-VL-8B가 돈 것과 같다(로컬에서 입력 id 목록 해시, 입력 파일 sha256, 데이터 sha256이 같음을 확인).

### 예전 MiniCPM 실행과 같은 조건

예전 실행(2026-11-16, `model_results_20261116/minicpm/`)의 `run_contract.json`과 대조했다.

- 모델 `openbmb/MiniCPM-V-4_5`, 리비전 `14a2004908140699317e2109e90573c1e6323c36`, 설정 `minicpm_v4_5_8b_4090.yaml` (sha256 `298953fe…`, 예전 실행에 쓴 파일과 바이트 단위로 같음)
- 배치 1, `strict_batch_size` false, SDPA 어텐션(비전 탑은 모델 코드가 eager로 강제), english_fixed 시스템 프롬프트, `mpr_label_only_v1`, 그리디 `max_new_tokens=2`, A–D 첫 토큰 점수
- 예전 실행이 쓴 묶음은 `minicpm_addon_20261115.tar.gz`다(예전 큐 로그 `addon_sha.log`의 마지막 확인 줄이 이 파일). 11월 14일 판이 아니다.
- 코드: 저장소 커밋 `6664400` + 어댑터 패치 1개(예전 `git_commit.txt`는 `723b80e`, 패치 적용 뒤 커밋이라 해시는 서버마다 달라질 수 있음). 추론 코드 13개 파일의 sha256이 예전 REL 실행 계약과 같은지 실행 전에 확인하고, 다르면 멈춘다.
- 패키지: 예전 `requirements.freeze.txt` 71개를 제약(constraints)으로 걸어 설치하고, 설치 뒤 핵심 패키지(torch, torchvision, transformers, accelerate, huggingface-hub, tokenizers, safetensors, pillow, numpy, timm, einops, sentencepiece, protobuf, qwen-vl-utils, pyyaml, triton, regex, av)가 하나라도 다르면 모델을 받기 전에 멈춘다. 나머지 차이는 `freeze_compare.txt`에 적기만 한다.
- 6×6 호출은 `queue_6x6.sh`의 `minicpm_v45` 항목 그대로다: `run_cross_locale.py --dimensions wf wi au ap ael rel --sample-unit semantic_items --seed 42 --batch-size 1 --score-labels --system-prompt-mode english_fixed`. 입력은 79,020개 고유 id이고 id 목록 sha256 `22ac60aa…`이 InternVL3.5-8B 6×6 실행과 같다(로컬 확인).

6개 언어 실행기(`run_rel_followup_6lang.py`)는 InternVL/Qwen용으로 이미 검토된 복사본에서 두 곳만 고쳤다: 허용 모델 계열에 `minicpmv` 추가, 코드 해시 목록에 `minicpmv.py` 추가. 배치·점수·파싱 코드는 4개 언어 실행기와 같다.

## 올릴 파일 (모두 `/workspace`에)

| 파일 | 크기 | sha256 |
|---|---|---|
| `queue_minicpm_6lang.sh` | 약 28 KB | `cb5d22f3be234e4d7747c89dc2e3f45df3a09ba34fa63842183678925f26eecf` |
| `minicpm_6lang_addon_20261117.tar.gz` | 약 5.2 MB | `dbf726db2f9165e5a91d22058959453ddbab1dc2e9dde17fd09b4ad9b53bef05` |
| `minicpm_addon_20261115.tar.gz` | 456 KB | `e638bb79e9795f15ef3a2759f2efe12c1a6a313a40018e4405fe8221620fb150` |
| `sixbysix_addon_20261114.tar.gz` | 34 KB | `ab2585e1fd2d3b0cf81f9b203d9833634032b16335638f3ce5e023a7d050187a` |
| `queue_6x6.sh` | 34 KB | `d538b0cd799a653d1cc536716e953168db86a15c3c346cbe659bd41a833a805a` |
| `internvl25_rel6_frja_images_20261116.tar` (이미 서버에 있음) | 749 MB | `5b80850fbf38c51042a9eff54226e13bd4026c0d88d6688e4d1d926b82e60754` |

위 다섯 개는 `upload/`에 있고 `SHA256SUMS.minicpm_6lang`(같은 폴더와 맨 위 폴더)에 해시가 있다. 서버에 이미 있을 수 있는 `/workspace/SHA256SUMS`를 덮어쓰지 않으려고 이름을 바꿨다. 큐가 시작할 때 모든 파일의 sha256을 다시 확인한다. 이미지 묶음이 서버에 없으면 허깅페이스에서 받는다.

## 실행

```bash
cd /workspace && sha256sum -c SHA256SUMS.minicpm_6lang   # upload/ 안의 파일 기준(이미지 tar 줄은 맨 위 폴더 파일에만 있음)
tmux new -s mc6
bash queue_minicpm_6lang.sh --plan                # 아무것도 바꾸지 않고 단계, 규모, 배치 크기를 출력
DELETE_QWEN3_8B_CACHE=1 bash queue_minicpm_6lang.sh
tail -f /workspace/QUEUE_C2_STATUS                # 다른 tmux 창에서
```

`DELETE_QWEN3_8B_CACHE=1`은 다른 시작 점검(다른 큐 없음, GPU 비어 있음, 네트워크, 파일 sha256, Python, 4090)이 모두 통과한 뒤 디스크 점검 바로 앞에서 `/workspace/mpr-gui-crosslocale/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct` 폴더 하나만 지운다(약 17GB, 결과는 이미 로컬에 검증해서 저장됨). 점검 하나라도 실패하면 지우지 않는다(로컬 시험으로 확인). 다른 폴더, `queue_results/qwen3_8b_frja`, `QUEUE_C_STATUS`, `QUEUE_C_DONE`은 건드리지 않는다. 변수를 주지 않으면 아무것도 지우지 않는다. 지금 빈 공간 37GB로는 시작 전 디스크 점검(44GB)에서 멈추므로 변수가 사실상 필요하다.

GPU를 쓰는 프로세스가 있거나 GPU 메모리가 1.5GB 이상 쓰이고 있거나, 다른 큐 스크립트가 돌고 있으면 아무것도 시작하지 않고 `FAILED steps: gpu`(또는 `others`)로 끝난다. 기다리지 않는다. 도중에 끊기면 같은 명령을 다시 치면 이어서 한다(저장소, venv, 예측 캐시를 재사용). 이름은 예전 큐와 겹치지 않는다: 상태 `QUEUE_C2_STATUS`, 완료 표시 `QUEUE_C2_DONE`(내용 `OK`, `OK (non-fatal: ...)`, `FAILED steps: ... | not run: ...`).

환경변수: `FORCE_SLOW_NET=1`(네트워크 20 MB/s 미만이어도 진행. 안쪽 `queue_6x6.sh` 호출에는 이 변수를 항상 1로 넘긴다: 시작할 때 이 큐가 이미 네트워크를 점검했고 모델과 데이터는 이미 받은 뒤라서, 몇 시간 뒤 안쪽 점검이 느린 순간에 걸려 6×6이 시작하자마자 끝나는 것을 막기 위해서다), `IGNORE_ETA=1`, `SKIP_CONSISTENCY=1`, `SKIP_6X6=1`, `NO_CONSTRAINTS=1`(예전처럼 제약 없이 pip), `KEEP_MODEL_CACHE`(기본 1, 아래 "모델 캐시" 참조), `REL_H`(기본 7), `EXPA_H`(3), `CC_H`(1), `MIN_FREE_GB`(44).

## 단계별 동작과 실패 규칙

1. 시작 점검: GPU 비어 있음, 다른 큐 없음, (선택) Qwen 캐시 삭제, 네트워크(huggingface.co 20 MB/s 미만이면 중단, torch가 아직 없으면 download.pytorch.org도 본다), 파일 sha256, Python 3.12, RTX 4090, 디스크.
2. `setup_env`(90분 제한): `git clone`(5번 재시도), 커밋 `6664400`, 예전 묶음 풀기와 패치 적용(`HEAD`가 기준 커밋 + 1개인지 확인), venv, pip 설치(제약 설치 3번 재시도, 계속 실패하면 제약 없이 한 번 더. 어느 쪽이 쓰였는지 `QUEUE_C2_STATUS`의 `pip install path:` 줄과 `pip_path.txt`에 남는다), 핵심 패키지 비교(제약 없이 깔린 경우에도 이 비교가 판단하고 핵심 패키지가 다르면 멈춘다), 4개 언어 데이터 풀기·이미지·검증. 실패하면 이후 전부 건너뛴다.
3. `setup_6lang`(30분): 6개 언어 묶음 풀기 전에 파일 목록 검사(자기 파일 밖 경로가 있으면 거부), 묶음 안 SHA256SUMS 확인, fr/ja 이미지 729장 풀기와 sha256 검증(729가 맞는 값이다: 데이터 묶음의 fr/ja 이미지가 729개, 즉 fr 366 + ja 363이고, ja 3개 항목 `rel::2/setting_8`, `rel::3/note_8`, `rel::6/health_9`는 데이터에서 제외됐다. 로컬 `followup_rel_6lang/assets` 폴더에는 이 3개의 ja 파일이 남아 있어 732개로 세어지지만 어떤 조건에도 쓰이지 않고 묶음 tar에도 없다), 설정·시스템 프롬프트·코드 해시·패키지 버전을 예전 실행 계약과 대조. 실패하면 일관성/REL/실험 A만 건너뛰고 6×6은 계속 시도한다.
4. `setup_6x6`(3시간): `queue_6x6.sh --fn fn_setup mc`로 6×6 데이터 준비를 미리 한다. `/workspace/mpr-gui-crosslocale`에 6×6 원본 데이터가 완전하면 링크로 재사용하고, 없으면 2.6GB를 받는다(허깅페이스 429 제한 때 6분 간격 최대 5번). manifest sha256(`4dd919fc…`)이 다르면 중단. 실패하면 6×6만 건너뛴다. 6×6 데이터 문제를 몇 시간 뒤가 아니라 초반에 알기 위해 앞에 뒀다.
5. `model`(2시간 제한, 3번 재시도): `hf download`, 안전텐서 4조각 합계 17,391,890,184바이트 일치, 토크나이저·전처리기 로드, A–D가 한 토큰이고 id 32–35. 실패하면 추론 단계 전부(6×6 포함) 건너뛴다.
5b. `smoke_6x6`(1시간 제한): 모델 직후 `KEY=minicpm_v45 bash queue_6x6.sh --fn fn_smoke`(6×6 계획 79,020개 확인 + smoke 216회 + 게이트). 실제 모델과 `run_cross_locale.py`가 GPU에서 처음 도는 지점을 앞으로 당긴 것이다. 나중에 `queue_6x6.sh`가 통과한 `gate_6x6_smoke.json`을 보고 `smoke already passed`로 건너뛴다(로컬 통합 시험에서 확인). 실패하면 6×6 본 실행만 건너뛰고 REL, 실험 A, 일관성은 그대로 돈다(`FAILED steps: smoke_6x6`).
6. smoke REL(`--max-items 2`) + 게이트, smoke 실험 A(`--max-items 3`) + 게이트. 게이트: 모든 행 성공, 파싱 실패 50% 이하, 생성 라벨이 전부 같지 않음, A–D 확률 질량 중앙값 0.5 이상, 시각 토큰 576–640, 배치 1. smoke 정확도가 0.30 미만이면 경고만. REL smoke가 실패하면 일관성과 REL 본 실행을 건너뛰고, 실험 A smoke가 실패하면 실험 A 본 실행과 재번역만 건너뛴다. 서로 독립이라 한쪽 실패가 다른 쪽을 막지 않는다.
7. 일관성 점검(1시간): 실패해도 `OK (non-fatal: consistency failed)`로 계속. 결과 `consistency_agreement.json`은 Qwen3 큐가 낸 파일과 키가 같다. 합격 기준은 두지 않았다.
8. REL 본 실행(7시간 제한): smoke 중앙값 × 26,057이 제한보다 길면 시작하지 않고 `STOP step full_rel`을 남긴다(`IGNORE_ETA=1`로 강행). 끝나면 게이트(성공 26,057/29,148행, 언어별 보고).
9. 실험 A 본 실행(3시간) + 게이트(O/R 답을 REL 본 실행의 같은 질문 답과 비교, 일치율 0.90 미만이면 실패), 이어서 T_retry 444회(1시간).
10. 6×6: `queue_6x6.sh`가 smoke 216회 + 게이트, 본 실행 79,020회(제한 20시간, ETA 점검), 게이트, 요약, 포장, 성공하면 모델 캐시 삭제까지 한다. 이 큐의 상태 줄은 `QUEUE6_STATUS`에도 쌓이고 `QUEUE_C2_STATUS`에 `[queue_6x6]` 표시로 복사된다.

실패 규칙 한 줄: 시작 점검·`setup_env`·`model` 실패는 그에 기대는 모든 단계를 막고, 그 뒤의 REL, 실험 A, 일관성, 6×6은 서로 독립이라 하나가 실패해도 나머지를 계속 돌린다(단 smoke 게이트는 자기 단계의 본 실행을 막는다). 단계가 실패해도 마무리(포장, 완료 표시)는 항상 한다. Ctrl-C나 kill로 끊으면 완료 표시를 만들지 않는다.

## 일관성 점검 줄을 사람이 본다

일관성 점검은 보고만 하고 큐를 멈추지 않는다(멈추는 하한을 두지 않았다). 그래서 시작 후 약 1~1.5시간 뒤(환경 준비, 모델 받기, smoke 3개, 일관성 5분 뒤) 사람이 아래 줄을 확인한다.

```bash
grep -E "consistency: label agreement|6x6 smoke gate|smoke gate" /workspace/QUEUE_C2_STATUS
```

`consistency: label agreement 0.9xxx (…of 600) accuracy old … new …` 줄에서 일치율이 0.90 미만이면 큐를 멈추고(아래 중단 절차) 원인을 먼저 본다. 이미지, 프롬프트, 환경 중 하나가 예전 실행과 다르다는 신호일 수 있고, 이 상태로 REL과 6×6을 몇 시간 더 돌리면 쓸 수 없는 결과를 얻게 된다. 0.90은 임의로 잡은 기준이다. 같은 설정의 Qwen3-VL-8B 점검은 이 서버에서 0.9917이었다. 같은 줄에 정확도(예전/지금)도 나온다. `6x6 smoke gate` 줄(`passed`, 정확도, 예상 시간)도 이때 같이 본다.

## 중단 절차

큐만 멈추면 이미 돌고 있는 추론 프로세스(GPU 사용)가 남을 수 있다. 순서대로 한다.

```bash
# 1) 큐를 멈춘다: 큐가 도는 tmux 창에서 Ctrl-C (또는 아래)
pkill -TERM -f 'queue_minicpm_6lang.sh'; pkill -TERM -f 'queue_6x6.sh'
# 2) 남은 추론 프로세스를 끈다
pkill -f 'run_rel_followup_6lang|run_rel_T_6lang|run_cross_locale|consistency_run'
# 3) 확인: GPU가 비었는지, 완료 표시가 없는지
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ls /workspace/QUEUE_C2_DONE /workspace/QUEUE6_DONE 2>&1
```

중단하면 `QUEUE_C2_DONE`을 만들지 않는다. 같은 명령(`bash queue_minicpm_6lang.sh`)을 다시 치면 이어서 한다(예측 캐시, 끝난 게이트 재사용). 중단 시점의 프로세스가 `run_cross_locale.py`라면 6×6 큐(`queue_6x6.sh`)도 같이 멈춰 있어야 한다(1번에서 둘 다 끈다).

## `queue_6x6.sh`가 `/workspace`에 만드는 것

6×6은 검토된 `queue_6x6.sh`를 수정 없이 부른다. 아래가 생긴다(로컬 통합 시험에서 실제로 만들어진 것을 확인했다).

| 경로 | 내용 |
|---|---|
| `/workspace/QUEUE6_STATUS` | 6×6 큐 상태 줄(같은 줄이 `QUEUE_C2_STATUS`에 `[queue_6x6]`로 복사된다) |
| `/workspace/QUEUE6_DONE` | 6×6 큐가 끝났다는 표시(성공 여부와 무관, 성공은 아래 `COMPLETE`로 본다) |
| `/workspace/queue_logs6/` | 단계별 로그(`setup_mc.log`, `minicpm_v45_download.log`, `_smoke.log`, `_full.log`, `_post.log`, `audit_release_*.log` 등) |
| `/workspace/sixbysix_addon/` | 6×6 묶음 푼 것(게이트 스크립트, 패치, 설정) |
| `/workspace/.queue6.lock` | 중복 실행 방지 잠금 |
| `/workspace/queue_results/net_check.txt` | 네트워크 점검 값 |
| `/workspace/queue_results/6x6_minicpm_v45/` | 결과 묶음 `6x6_minicpm_v45_<날짜>_<시각>[_PARTIAL].tar.gz`, `SHA256SUMS`, 성공하면 `COMPLETE` |
| `/workspace/queue_results/disk_end6.log`, `QUEUE6_STATUS.txt`, `queue_logs6/` | 6×6 큐가 끝날 때 복사하는 디스크 상태, 상태 파일, 로그 |
| `/workspace/mpr-minicpm/followup_rel_4lang/results/6x6_minicpm_v45/` | 원본 결과(`full_6x6.jsonl` 등) |
| `/workspace/mpr-minicpm/01_code/data/raw/mpr_gui_bench`, `01_code/data/manifests/mpr_gui_manifest.jsonl` | 6×6 원본 데이터(링크 또는 받은 것)와 manifest |

기존 `/workspace/SHA256SUMS`, `QUEUE_C_*`, `queue_results/qwen3_8b_frja`는 건드리지 않는다.

## 모델 캐시

안쪽 6×6 호출에는 `KEEP_MODEL_CACHE=1`을 기본으로 넘긴다. 6×6이 성공해도 모델 캐시(17.4GB)를 지우지 않는다. 일부 단계가 실패한 뒤 다시 돌릴 때 17GB를 다시 받지 않게 하려는 것이다. 대신 끝난 뒤 디스크 17GB가 계속 차 있으니, 결과를 모두 받고 sha256을 확인한 뒤 인스턴스를 지울 거면 상관없고, 디스크를 비우고 싶으면 `rm -rf /workspace/mpr-minicpm/.cache/huggingface`를 사람이 한다. 큐가 성공 뒤 스스로 지우게 하려면 `KEEP_MODEL_CACHE=0`을 준다.

## 결과 포장 (단계마다)

`/workspace/queue_results/minicpm_6lang/`에 다음이 쌓인다. 각 포장마다 `.tar.gz`, `.tar.gz.sha256`, 안에 `results_files.sha256`(파일별 sha256), 밖에 `…files.sha256`, 전체 목록 `PACKAGES_SHA256.txt`.

| 포장 | 시점 | 내용 |
|---|---|---|
| `01_setup_smoke_consistency` | 일관성 점검 직후 | 6×6 smoke, REL/실험 A smoke, 일관성 결과, 패키지 비교, 로그 |
| `02_rel_new_only` | REL 본 실행 직후 | `rel6new_minicpm_v45` 전체 + 일관성 |
| `03_expA_new_pairs` | 실험 A + 재번역 직후 | `expA6_minicpm_v45` 전체 |
| `6x6/` + `minicpm_6lang_04_6x6_*.files.sha256` | 6×6 직후 | 6×6 큐의 tar.gz(같은 디스크 안에서 하드링크)와 파일별 sha256 |
| `00_final` | 항상 마지막 | 상태, 로그, pip freeze, nvidia-smi, 디스크 |

일찍 받고 싶으면 02 포장이 생긴 뒤 `scp`로 받아도 된다. 결과를 모두 받고 sha256을 확인하기 전에 인스턴스를 지우지 않는다. 인스턴스를 끄는 판단은 사람이 한다.

## 시간 추정 (이 서버에서 재지 않은 추정치)

근거는 예전 MiniCPM 실행의 `inference_runtime_ms`다(`model_results_20261116/minicpm/unpacked/*/full/predictions.jsonl`, 배치 1, RTX 4090): REL 18,432회 중앙값 0.493초·평균 0.512초, 실험 A 5,688회 중앙값 0.489초·평균 0.504초. 시간은 평균으로 계산했다.

| 단계 | 계산 | 시간 |
|---|---|---|
| 환경 준비(torch 등 설치, 이미지, 6×6 데이터) | 예전 설정이 약 10분(캐시 있을 때)~수십 분 | 약 0.5~1시간 |
| 모델 받기 17.4GB | 20~100 MB/s | 약 3~15분 |
| smoke 2개 + 모델 적재 | | 약 10~15분 |
| 일관성 600회 | × 0.512 | 약 5분 |
| REL 26,057회 | × 0.512 | 약 3.7시간 |
| 실험 A 8,532회 + 444회 | × 0.504 | 약 1.3시간 |
| 6×6 79,020회 | × 0.49~0.60 (6×6은 MiniCPM으로 돈 적이 없어 가정) | 약 10.8~13.2시간 |
| 합계 | | 약 17~20시간 |

단계 제한은 REL 7시간, 실험 A 3시간, 6×6 20시간이다. smoke 속도로 계산한 예상 시간이 제한을 넘으면 그 단계는 시작하지 않는다. fr/ja 질문이 길어 속도가 달라질 수 있다(미검증).

## 디스크 계획 (`/workspace` 65GB, 지금 37GB 비어 있음)

| 항목 | 크기 |
|---|---|
| Qwen3-VL-8B 캐시 삭제로 확보(선택) | +약 17GB → 약 54GB |
| MiniCPM 모델 캐시 | 17.4GB (6×6이 성공하면 마지막에 삭제됨) |
| venv(torch cu130 + nvidia 라이브러리) | 약 9GB (추정, 확인 필요) |
| 저장소 + 4개 언어 이미지 | 약 1.6GB |
| fr/ja 이미지(729장) | 0.75GB (이미지 묶음 0.75GB는 이미 있음) |
| 6×6 이미지 | 2.6GB (작업 폴더가 겹치면 일시적으로 더 쓸 수 있음) |
| 결과(REL, 실험 A, 6×6, 포장) | 약 1.5GB 안팎 (추정) |
| 합계 | 약 33~36GB |

시작 전 점검은 44GB(`MIN_FREE_GB`)이고, 모델이 이미 있으면 18GB, venv가 이미 있으면 9GB를 빼서 이어 하기가 가능하다(최소 12GB). 삭제 없이 37GB로 시작하려면 `MIN_FREE_GB=35`로 낮출 수 있지만 여유가 거의 없어 권하지 않는다.

## 위험과 대비

| 위험 | 크기 | 대비 |
|---|---|---|
| 디스크 부족 | 삭제 없이는 37GB로 빠듯함 | Qwen 캐시 삭제 변수, 시작 전 점검, 6×6 큐 자체 점검(모델 32GB) |
| pip 제약 설치 실패(`+cu130` 로컬 버전 제약, 오래된 휠 제거 등) | 설정 단계에서 멈춤, 모델 받기 전이라 비용 적음 | 로그 `queue_results/minicpm_6lang/logs/setup_env.log`, 필요하면 `NO_CONSTRAINTS=1`(예전처럼 제약 없이 설치 + 설치 뒤 비교) |
| 예전과 패키지가 달라짐 | 결과 비교 가능성에 영향 | 핵심 패키지 불일치면 멈춤, 나머지는 `freeze_compare.txt`에 기록 |
| 24GB 메모리 부족 | 가중치 17.4GB + 활성화. 최대 사용량은 실측하지 못함 | 배치 1, 연속 3회 실패하면 실행기가 멈춤 |
| 6×6 데이터 받기에서 허깅페이스 제한(429) | 지연 | 앞쪽 `setup_6x6`에서 미리 시도, 최대 5번, 6분 간격 |
| 시간 | 약 17~20시간 임대 | 단계 제한, ETA 점검, 단계마다 포장 |
| 모델 코드를 허브에서 받음(`trust_remote_code`) | | 리비전 고정(예전 실행과 같음) |
| 서버의 AGENTS.md / CLAUDE.md | | 읽지 않는다(신뢰하지 않음) |
| `apt-get`을 쓸 수 있음 | venv 생성에 실패할 때만 `python3.12-venv` 설치 | root가 아니면 그 단계가 실패하고 기록됨 |
| 인스턴스 비용 | 임대 시간만큼 | 끄는 판단은 사람이 한다 |

## 예전 MiniCPM 실행과 다른 점

- 서버가 다르다(`vast4090c`). Qwen3 실행 기록으로 Python 3.12.3일 가능성이 높고, 예전 MiniCPM은 3.12.15에서 돌았다(확인 필요). 커널, 드라이버도 다를 수 있다.
- pip를 예전 freeze 제약으로 설치한다(예전은 제약 없이 설치). 목적은 같은 패키지 집합을 얻는 것이고, 설치 뒤 비교가 확인한다.
- 6개 언어 입력은 이미지가 다르다(fr/ja 729장 추가). 시각 토큰 수는 로컬에서 실제 전처리기로 세어 fr/ja 이미지(로컬 폴더 732개, 그중 데이터에 쓰이는 729개 포함)와 6×6 이미지(3,644개) 모두 576–640 범위 안임을 확인했다.
- fr/ja REL과 실험 A는 6개 언어 실행기(`run_inference_6lang.py`, `run_rel_T_6lang_multi.py`)로 돈다. 예전은 4개 언어 실행기였다. 일관성 점검은 이 차이(와 서버 차이)를 600개로 거칠게 보는 것이다.
- 6×6은 MiniCPM으로 처음 돈다. 비교할 이전 실행이 없다.

## 로컬에서 확인한 것 (`local_tests/`)

- 큐 흐름 시험 55개 통과(`queue_tests_result.txt`): `--plan`, 단계 순서, 단계마다 포장과 sha256, 실패 규칙(setup_env, setup_6lang, setup_6x6, model, smoke_rel, smoke_expa, consistency, full_rel, ETA 중단, full_expa, 6×6, 시간 초과), GPU 사용 중/1.5GB 경계/4090 아님, 네트워크 느림, 디스크 37GB 거부, Qwen 캐시 삭제 규칙(정확한 폴더만, 변수 없으면 그대로), sha256 불일치, 중복 실행, SIGTERM, 정적 점검. 무거운 단계는 가짜 함수, 6×6 큐는 가짜(`fake_6x6.sh`)이고 나머지는 실제 스크립트다.
- 실제 `queue_6x6.sh`(수정 없음, sha `d538b0cd…`) 통합 시험 33개 통과(`6x6_integration_result.txt`, `run_6x6_integration.sh`): 이 큐가 부르는 방식 그대로(`--fn fn_setup mc`, `KEY=minicpm_v45 --fn fn_smoke`, `MODELS=minicpm_v45 … bash queue_6x6.sh`)를 `setup_env` 직후 상태의 가짜 `/workspace`(어댑터 패치 커밋까지 있는 저장소, 스텁 venv)에서 돌렸다. 가짜는 hf, nvidia-smi, curl, df, timeout, flock과 모델(`Q6_TEST=1` mock)뿐이고 코드 경로는 실제다. 통과: patch-id 확인, venv_ok, 6×6 원본 데이터 재사용(링크)/내려받기 훅, manifest 복사/`audit_release.sh` 생성과 sha(`4dd919fc…`), config 확인, 79,020개 계획과 id 목록 sha, smoke 216 + 게이트, `smoke already passed`, ETA 점검, 본 실행(mock 79,020행) + 게이트 + 요약, 포장과 `COMPLETE`, `KEEP_MODEL_CACHE=1`일 때 캐시 유지, `QUEUE6_*` 파일 기록, ETA 중단/시간 초과 시 `_PARTIAL`, 재실행 이어 하기, GPU 대기 반복(1.5GB 이상이면 30번 기다린 뒤 STOP)과 4090 확인(`Q6_TEST` 없이), `FORCE_SLOW_NET=1`이 느린 네트워크 점검을 통과시키고 없으면 STOP하는 것. 로컬에서 실행할 수 없는 경로: 실제 모델 적재와 GPU 실행, 안전텐서 크기 검사(`Q6_TEST`에서 건너뜀), 실제 HF 내려받기와 429 제한, 실제 pip.
- 실제 데이터 파일로 확인한 규모(`mock_data_checks_result.txt`): REL new_only 26,057회/29,148행, 실험 A 8,532(2,844쌍)/444(148쌍), 일관성 600개(Qwen3와 같은 id), 6×6 79,020개. 입력 id 목록 해시가 InternVL3.5/Qwen3 실행과 같다. 가짜 모델(`--mock-model`)로 실행기와 게이트 전체를 통과시켰다.
- `bash -n`(큐, 시험 스크립트).

## 미검증

- 실제 서버에서의 모든 실행: `git clone`, venv, pip 설치(특히 제약 + `+cu130`), `hf download`, 실제 모델 적재, GPU 추론, 메모리 최대치, 실제 속도.
- `setup_env`, `setup_6lang`, `setup_6x6`, `model`은 흐름 시험에서 가짜 함수로 대체했다. 그 안의 개별 명령은 예전 큐(`queue_minicpm.sh`, `queue_qwen3_8b_frja.sh`)에서 가져왔지만 이 큐 안에서는 돌려 보지 않았다.
- 6×6 큐의 실제 GPU/모델 경로: 로컬 통합 시험(`local_tests/run_6x6_integration.sh`)은 `Q6_TEST=1`(가짜 모델)로 돌렸다. 실제 모델 적재, 안전텐서 크기 검사(`Q6_TEST`에서는 건너뜀), GPU 메모리, 실제 smoke 정확도는 확인하지 못했다.
- 6×6의 MiniCPM 시간(0.49~0.60초/회는 REL 실측에서 가져온 가정).
- venv 크기 약 9GB, 결과 크기는 추정.
- 서버 `/workspace/mpr-gui-crosslocale`에 6×6 원본 데이터가 있는지(없으면 2.6GB를 받는다).
- MiniCPM 모델 라이선스 조건은 읽지 않았다(확인 필요).
- 결과 해석(정확도, 시각 토큰 차이의 영향)은 하지 않았다.
