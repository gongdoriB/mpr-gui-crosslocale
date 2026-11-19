# vast4090b: 4B 두 모델의 6개 언어 전체 결과 (Qwen3-VL-4B, InternVL3.5-4B)

2026-11-17 작성. 서버에는 아무것도 올리거나 실행하지 않았다. 로컬 시험 결과는 `local_tests/`에 있고, 확인하지 못한 것은 맨 아래 "미검증"에 모았다.

## 무엇을 돌리나

B2 큐(`queue_b2_frja.sh`, 지금 UI-TARS 실험 A를 돌리는 중)가 끝난 뒤에 시작한다. 모델 둘에 대해 같은 순서로 세 가지를 돌리되, 우선순위가 높은 REL과 실험 A를 두 모델 모두 먼저 끝내고 6×6은 나중에 돌린다.

| 순서 | 모델 | 저장소 | 내용 | 규모 |
|---|---|---|---|---|
| 1a | Qwen3-VL-4B-Instruct (ebb281ec) | `/workspace/mpr-gui-crosslocale` | REL O/R/C/F `new_only` (질문 또는 화면 언어에 fr/ja가 있는 행) | 26,057회 (조건 행 29,148) |
| 1b | 같은 모델 | 같은 저장소 | 실험 A 새 쌍 main 2,844쌍 x O/T/R, T_retry 148쌍 x 3 | 8,532회 + 444회 |
| 2a | InternVL3.5-4B (481f6e32) | `/workspace/mpr-ivl` | REL `new_only` | 26,057회 |
| 2b | 같은 모델 | 같은 저장소 | 실험 A 새 쌍 + T_retry | 8,532회 + 444회 |
| 3 | Qwen3-VL-4B | `mpr-gui-crosslocale` | 6×6 전체 벤치마크 | 79,020회 |
| 4 | InternVL3.5-4B | `mpr-ivl` | 6×6 전체 벤치마크 | 79,020회 |

6×6은 이전 8B와 같은 `queue_6x6.sh`를 한 글자도 바꾸지 않고 모델 하나씩(`MODELS=<키>`) 부른다. 호출은 `run_cross_locale.py --dimensions wf wi au ap ael rel --sample-unit semantic_items --seed 42 --batch-size 1 --score-labels --system-prompt-mode english_fixed --device-map cuda:0 --resume`이다.

시작 전에는 기다리기만 한다. `/workspace/QUEUE_B2_DONE`이 생기고, 다른 큐 스크립트가 없고, GPU 계산 프로세스가 없고, GPU 메모리가 1,500 MiB 아래인 상태가 60초 간격으로 두 번 연속 확인될 때까지 아무것도 바꾸지 않고 내려받지도 않는다(최대 96시간, 넘기면 `FAILED steps: wait`로 끝난다). B2 큐의 판정이 `FAILED`여도 파일이 있으면 시작하고, 그 내용은 상태 파일에 남긴다.

## 이전 4B 실행과 같게 맞춘 조건

근거는 2026-11-14 같은 서버의 `sizes_uitars` 큐 결과 `model_results_20261114/{qwen3_vl_4b,internvl35_4b}/.../{rel_full,expa_full}/run_contract.json`이다. 이 네 파일을 그대로 애드온의 `contracts/`에 넣었고, 큐의 setup 단계가 서버의 설정·코드·패키지가 이 계약과 다르면(설치하지 않고) 멈춘다.

| 항목 | Qwen3-VL-4B | InternVL3.5-4B |
|---|---|---|
| 배치 크기 | 1 (REL, 실험 A) | 1 |
| strict_batch_size | true (REL `--strict-batch-size`, 실험 A도 true) | true |
| 설정 yaml sha256 | `4a3de63b…ec167544` (`qwen3_vl_4b_4090_channels_last.yaml`) | `6a3158fc…fb3dc6e8` (`internvl3_5_4b_4090.yaml`) |
| 어텐션 | sdpa (`patch_embed_memory_format: channels_last_3d`) | eager (`use_flash_attn: false`, max_num 8, 최대 2,304 토큰) |
| 시스템 프롬프트·템플릿 | english_fixed, cross_locale_system_v1 | 같음 |
| 생성 | 탐욕, max_new_tokens 2, 라벨 점수 on | 같음 |
| 패키지 | torch 2.9.1+cu130, transformers 4.57.6, accelerate 1.15.0, timm 1.0.31 등 (계약 값) | 같음 |
| 이전 실행 코드 커밋 | 6664400 | 9694fff (6664400 + InternVL 패치 5개) |

설정 yaml 전체 해시: Qwen `4a3de63bc0036755c7e4633be03544b3320fac0388c1fbc897aa50d2ec167544`, InternVL `6a3158fc8cac7214fe6edb9febbd64165274281513bcbb75aa24aa89fb3dc6e8`. 6×6 애드온이 저장소에 넣는 설정 파일도 이 두 값과 같다(로컬에서 비교).

이전 4B 실행과 다른 점은 다음과 같다.
- 입력 범위: 이전은 4개 언어 REL 전체(18,432회)와 실험 A 5,688회였고, 이번은 fr/ja가 들어가는 새 행만이다(REL 26,057회, 실험 A 8,532회 + 444회). 같은 8B 모델들의 fr/ja 작업과 같은 범위다.
- 실행기: REL은 `run_inference_6lang.py`(검토된 6개 언어판, 모델 어댑터는 저장소 것을 그대로 씀), 실험 A는 `run_rel_T_6lang_multi.py`(이전 `run_rel_T_generic.py`의 6개 언어판)다. 프롬프트 조립·파싱·점수는 같은 저장소 함수를 가져다 쓴다. fr/ja 새 쌍의 O/R 문항과 번역(T, T_retry) 입력은 B2 큐가 8B 모델들의 fr/ja 실험 A에 쓴 파일(`OR_gemma_conditions.jsonl`, `T_gemma_conditions.jsonl`, `T_gemma_retry_conditions.jsonl`, 잠금 파일 `expA6_lock.json`의 sha256)과 바이트가 같다. 그래서 8B와 4B가 같은 질문을 받는다.
- InternVL3.5 8B는 REL·실험 A가 strict false였지만 4B는 이전 실행이 true여서 4B는 true로 맞췄다. 8B와 4B의 결과 비교에서는 이 차이를 알고 있어야 한다.
- 6×6: 4B의 6×6은 이전에 돌린 적이 없다. 호출은 8B 6×6과 같고, 배치는 InternVL3.5-8B 6×6과 같은 1이다. Qwen3-VL-8B 6×6은 배치 4였으므로(`queue_6x6.sh` 머리말, 결과 파일의 `batch_size` 확인) Qwen 쪽 8B와 4B 사이에는 배치 차이가 있다.
- 6×6 입력은 8B 실행과 같은 입력이다: manifest sha256 `4dd919fc…`, 입력 ID 79,020개의 목록 sha256 `22ac60aa…`(로컬 dry-run으로 확인, 어긋나면 `queue_6x6.sh`가 멈춘다).

## 올릴 파일 (서버 `/workspace`)

로컬 `upload/` 폴더의 다섯 파일이다.

| 파일 | sha256 |
|---|---|
| `queue_b3_4b.sh` | `214539a32b89fc4fc31ca3c38cadcc1912f8084221dd4744a2babc601b647eee` |
| `b3_4b_addon_20261117.tar.gz` | `20155e4a801c0e92597b853c3eabda2bfc9c73b10be676b34868ca25eb9f51fd` |
| `queue_6x6.sh` | `d538b0cd799a653d1cc536716e953168db86a15c3c346cbe659bd41a833a805a` |
| `sixbysix_addon_20261114.tar.gz` | `ab2585e1fd2d3b0cf81f9b203d9833634032b16335638f3ce5e023a7d050187a` |
| `SHA256SUMS` | 위 네 줄을 담은 파일. 서버에서 `sha256sum -c SHA256SUMS`로 확인 |

fr/ja 이미지 tar는 필요 없다. `mpr-ivl`에는 InternVL2.5 REL6 큐가 풀어 둔 것이 있고, `mpr-gui-crosslocale`에는 B2 큐가 UI-TARS 때 복사해 두었다(없으면 큐가 `mpr-ivl`에서 복사하고 sha256으로 확인한다). `internvl25_rel6_frja_images_20261116.tar`가 `/workspace`에 남아 있으면 그것을 쓰고, sha256이 다르면 멈춘다.

## 실행

```
cd /workspace
sha256sum -c SHA256SUMS
tmux new -s b3            # B2 큐가 도는 tmux 말고 새 세션 (또는 Ctrl-b c)
bash queue_b3_4b.sh --plan      # 아무것도 바꾸지 않고 계획만 출력
bash queue_b3_4b.sh             # 시작하면 B2 큐가 끝날 때까지 기다린다
tail -f /workspace/QUEUE_B3_STATUS
```

B2 큐가 아직 도는 동안에 시작해도 된다. 기다리는 동안에는 저장소도 GPU도 건드리지 않는다. 같은 큐를 두 번 띄우면 두 번째는 "already running"으로 끝난다.

환경변수:

| 변수 | 뜻 |
|---|---|
| `ONLY="internvl35_4b"` | 모델 일부만 (순서는 고정) |
| `SKIP_6X6=1` | 1a~2b(REL, 실험 A)만. 끝나면 두 모델 캐시를 지운다 |
| `ONLY_6X6=1` | 6×6만 (REL·실험 A가 이미 끝났을 때, 또는 따로 돌릴 때) |
| `NO_WAIT=1` | `QUEUE_B2_DONE`을 기다리지 않는다(다른 큐·GPU 점검은 그대로) |
| `REL_H`, `EXPA_H`, `S6_H` | 단계 제한 시간(시간 단위). 기본: REL Qwen 6 / InternVL 8, 실험 A 3, 6×6 한 번 호출 Qwen 40 / InternVL 42 |
| `FORCE_SLOW_NET=1`, `IGNORE_ETA=1`, `KEEP_MODEL_CACHE=1`, `SKIP_6X6_DATA_CHECK=1` | 네트워크 점검(20 MB/s 미만)·ETA 중단·캐시 삭제·6×6 원자료 점검을 넘긴다 |

도중에 끊었으면(Ctrl-C 등) 같은 명령을 다시 치면 이어서 한다. 끝난 모델의 REL·실험 A는 건너뛰고(`COMPLETE`), 예측 파일은 `--resume`으로 이어 쓴다.

## 실패가 나면

모델 한정 실패(모델 내려받기·검사, smoke·게이트, 실행 오류, 시간 초과, ETA 중단, 포장 실패, 6×6 미완성)는 그 모델의 남은 단계만 멈추고 다음으로 간다. 한 모델의 REL이 실패해도 같은 모델의 실험 A와 6×6은 따로 돈다. 환경 실패는 큐 전체를 멈춘다. GPU가 사용 중이거나 4090이 아님, 다른 큐 실행, 네트워크 느림, 애드온·`queue_6x6.sh` sha256 불일치, 저장소나 venv 없음, setup 실패(설정·코드·패키지·데이터 불일치, 서버에 있는 애드온 파일이 다름), 다운로드할 디스크 부족, 단계 로그의 CUDA 치명 오류 문구, 실패 뒤 GPU가 풀리지 않음이다. 멈추면 `QUEUE_B3_DONE`에 `FAILED steps: ...`가 적힌다.

6×6 앞에는 큐가 읽기만 하는 점검을 한다. Qwen 저장소의 HEAD가 `66644001…`이 아니면(`queue_6x6.sh`가 `git checkout -f`를 하므로) 그 모델의 6×6을 하지 않고 이유를 적는다. InternVL 저장소는 `66644001…` 위에 정확히 패치 5개여야 한다. venv에 필요한 패키지가 이미 있어야 하고(`queue_6x6.sh`가 pip를 돌리는 일이 없게), 6×6 원자료는 있으면 검증해서 쓴다(Hugging Face 리비전 c1edb808, 3,685개 파일, GitHub e4f1cfcd, manifest sha `4dd919fc…`, manifest가 가리키는 이미지 3,643개가 모두 있고 비어 있지 않음). 다르면 건드리지 않고 멈춘다. 없으면 `queue_6x6.sh`가 2.6 GB를 받는다.

## 결과 위치

저장소 안 결과 폴더는 전부 새 폴더다. `followup_rel_4lang/results`의 기존 폴더는 건드리지 않는다.

- `/workspace/mpr-gui-crosslocale/followup_rel_6lang/results/{rel6new_qwen3_vl_4b,expA6_qwen3_vl_4b}`와 `.../followup_rel_4lang/results/6x6_qwen3_vl_4b`
- `/workspace/mpr-ivl/followup_rel_6lang/results/{rel6new_internvl35_4b,expA6_internvl35_4b}`와 `.../followup_rel_4lang/results/6x6_internvl35_4b`

받을 것은 `/workspace/queue_results/b3_4b/`다. 큰 단계가 하나 끝날 때마다 포장한다.

- `<키>/b3_<키>_01_rel_<날짜>.tar.gz` (REL), `02_expA` (실험 A + T_retry), 6×6은 `queue_6x6.sh`가 만든 `6x6_<키>_<날짜>.tar.gz`를 같은 폴더에 연결하고 `b3_<키>_03_6x6_<날짜>.files.sha256`(결과 파일별 sha256)을 쓴다. 각 tar 옆에 `.sha256`, 안에 `results_files.sha256`, 로그, git 커밋, pip freeze, nvidia-smi가 있다. 단계가 실패했으면 이름에 `_PARTIAL`이 붙는다.
- `PACKAGES_SHA256.txt` (모든 포장의 sha256), `b3_4b_00_final_<날짜>.tar.gz` (전체 로그·상태, 판정), `STATUS`.
- `/workspace/QUEUE_B3_STATUS`, `/workspace/QUEUE_B3_DONE`(`OK` 또는 `FAILED steps: ...`), 6×6 쪽은 `/workspace/queue_results/6x6_<키>/`와 `/workspace/QUEUE6_STATUS`.

`/workspace/queue_results`를 통째로 내려받은 뒤 인스턴스 정리는 사람이 정한다(큐는 인스턴스를 끄지 않는다).

## 시간 추정 (추정, 이 서버에서 재지 않음)

근거는 2026-11-14 같은 서버의 4B 실행에서 `predictions.jsonl`의 `inference_runtime_ms`를 중앙값으로 낸 값이다(전체 성공 행 기준). 새 언어의 프롬프트·이미지가 더 길면 달라질 수 있다.

| 모델 | 이전 실행 중앙값 REL / 실험 A (평균) | 단계 | 계산 | 시간 |
|---|---|---|---|---|
| Qwen3-VL-4B | 0.448 s / 0.455 s (0.471 / 0.485) | REL 26,057회 | x 0.448 | 3.2시간 |
| | | 실험 A 8,532회 + 444회 | x 0.455 | 1.1시간 + 3분 |
| | | 6×6 79,020회 | x 0.448 (가정) | 9.8시간 |
| InternVL3.5-4B | 0.669 s / 0.645 s (0.665 / 0.631) | REL 26,057회 | x 0.669 | 4.8시간 |
| | | 실험 A 8,532회 + 444회 | x 0.645 | 1.5시간 + 5분 |
| | | 6×6 79,020회 | x 0.669 (가정) | 14.7시간 |

6×6의 추론당 시간은 4B에서 잰 적이 없어서 REL 중앙값과 같다고 가정했다. 8B에서는 InternVL3.5 6×6이 0.738 s로 REL 중앙값(0.748 s)과 비슷했고 Qwen3 6×6은 배치 4였다. 모델 두 번 내려받기(각 9GB)·적재·smoke, 6×6 준비와 smoke를 더하면 약 2시간이라, 합계는 약 36~38시간이다. 1a~2b만이면 약 11~12시간이다. 시작 후에는 smoke 게이트의 `eta_full_hours`(`QUEUE_B3_STATUS`의 "smoke gate:" 줄)로 이 추정을 바로 확인할 수 있고, 크게 어긋나면 멈추고 다시 정하는 편이 낫다. 시간당 요금은 모르므로 비용은 계산하지 않았다(확인 필요).

## 디스크 계획 (여유 약 34~50GB)

모델 캐시는 Qwen3-VL-4B 8.9GB(8,875,719,344바이트), InternVL3.5-4B 약 9.5GB다. REL·실험 A 결과와 포장은 모델당 0.5~1.5GB로 추정(미검증), 6×6 결과는 모델당 약 0.4GB, 6×6 원자료는 서버에 없을 때만 2.6GB다. Qwen 캐시는 6×6을 위해 InternVL을 받는 동안에도 남겨 두는데, 그때 남은 여유가 30GB(=20+10) 미만이면 먼저 지운다(6×6 때 다시 받는다). 다운로드 전에 여유가 20GB 미만이면 큐가 멈춘다. 각 모델의 캐시는 그 모델의 6×6이 성공하면 `queue_6x6.sh`가 지운다. 실패하거나 시간 초과인 모델의 캐시는 남는다(같은 명령으로 재개할 때 다시 받지 않도록). 결과 파일은 지우지 않는다.

## 위험

| 위험 | 크기 | 대비 |
|---|---|---|
| 아주 긴 실행 | 약 36~38시간, 요금 확인 필요 | `SKIP_6X6=1`로 먼저 REL·실험 A만(약 11시간) 돌리고 6×6은 나중에 `ONLY_6X6=1`로 돌릴 수 있다. tmux에서 Ctrl-C로 멈춰도 재개된다 |
| B2 큐가 끝나지 않거나 DONE이 안 생김 | 최대 96시간 기다림 | `WAIT_MAX_H`로 조정. 기다리는 동안은 아무것도 바꾸지 않는다. B2가 이미 끝났는데 DONE이 없으면 `NO_WAIT=1` |
| 서버 저장소가 예상과 다름 | setup(설정·코드 해시·패키지 버전·애드온 파일 바이트 일치)과 6×6 점검이 다운로드 전에 멈춘다 | `FAILED steps: setup_<모델>` 또는 `six_setup_<모델>`과 `queue_results/b3_4b/logs/` |
| 6×6 원자료를 새로 받아야 함 | 2.6GB, Hugging Face 요청 제한(429) 가능 | `queue_6x6.sh`가 작업자 2개, 6분 간격으로 최대 5번 다시 시도한다 |
| 24GB 메모리 부족 | 이전 4B 실행과 같은 설정이라 같은 범위로 보지만 fr/ja 화면은 처음 | 배치 1, 연속 3회 실패 시 실행기가 멈춘다. CUDA 오류 문구가 로그에 있으면 큐가 멈춘다 |
| `queue_6x6.sh`가 저장소 설정 yaml을 덮어씀 | Qwen 쪽 `fam_cfgs`에 UI-TARS 등 다른 설정도 있어 내용이 다르면 복사한다. B2 큐가 쓴 UI-TARS 설정은 같은 파일이라 같을 것으로 본다 | 내용이 다르면 UI-TARS 결과를 다시 돌릴 때 설정이 바뀔 수 있다. 미검증 |
| 결과 덮어쓰기 | 새 폴더만 만든다. 애드온은 파일이 없을 때만 복사하고, 있는 파일은 바이트가 같아야 한다 | 로컬 시험에서 B2 애드온을 먼저 풀어 둔 저장소에 대해 확인 |
| 디스크 | 여유 20GB 미만이면 다운로드 전에 멈춘다 | 모델 캐시 자동 정리(위) |

## 로컬에서 확인한 것

시험 스크립트는 `local_tests/run_b3_queue_tests.sh`이고 결과 파일은 같은 폴더의 `*_run.txt`다. 가짜 `/workspace`에 가짜 `hf`/`nvidia-smi`/`curl`/`timeout`/`df`/`flock`을 붙였고 네트워크는 막았다. 실제 GPU 추론, 실제 모델 가중치, 서버는 쓰지 않았다.

검사 110개를 전부 통과했고 실패는 0개다(`local_tests/queue_tests_result.txt`). 묶음별로 보면 다음과 같다.

- 계획·정적 검사 7개: `--plan`은 아무것도 만들지 않고, `bash -n`과 파이썬 컴파일, 업로드용 큐와 애드온 sha256 일치, `pip install`·`git checkout/reset/clean/am`이 큐에 없고 `rm -rf`는 큐 자신의 캐시·임시 폴더에만 쓴다는 것도 정적으로 확인(가드 묶음).
- 흐름 29개(단계를 가짜로 바꾸고 나머지는 실제 큐): 17단계 순서(REL·실험 A 두 모델 다음 6×6 두 모델), 단계별 포장과 sha256, 재실행 때 건너뛰기, `SKIP_6X6`/`ONLY_6X6`, 디스크 25GB에서 Qwen 캐시 정리와 15GB에서 멈춤, 모델 한정 실패 때 다음으로 계속(REL 실패·smoke 실패·내려받기 실패·ETA 중단·시간 초과·6×6 미완성·6×6 준비 실패), 환경 실패 때 전체 중단(setup 실패, CUDA 오류 문구, 실패 뒤 GPU 점유, 6×6 중 CUDA 문구, venv 없음), 이미 끝난 게이트 인식, 중복 실행 거부, SIGTERM.
- 기다리기 8개: `QUEUE_B2_DONE`이 없으면 대기, GPU가 차 있으면 대기, B2 큐 프로세스가 남아 있으면 대기, 기다리는 동안 아무 단계도 실행되지 않음, 한도를 넘기면 `FAILED steps: wait`, B2 판정이 FAILED여도 시작, `NO_WAIT=1`, `--plan`은 기다리지 않음.
- 가드 13개: GPU 프로세스·메모리·4090 아님·다른 큐 실행, 네트워크 느림과 `FORCE_SLOW_NET`, 404, 애드온·`queue_6x6.sh`·이미지 tar sha256 불일치.
- 개수 8개(로컬에 있는 실제 6개 언어 데이터와 6×6 데이터, 모델 없이 dry-run): 두 모델 모두 REL `new_only` 26,057회 / 조건 행 29,148 / 366문항 / 언어 en zh th ru fr ja / 배치 1, 실험 A main 2,844쌍 x O/T/R = 8,532회(조건마다 2,844), T_retry 148쌍 x 3 = 444회, 6×6 입력 ID 79,020개(고유 79,020, 의미 문항 2,195, ID 목록 sha256 `22ac60aa…`이 8B 6×6 실행과 같음).
- 실제 흐름 25개 + 이어하기 1개 + 설정 불일치 11개 + 6×6 가드 8개: 가짜 `/workspace`에 `mpr-gui-crosslocale`(커밋 6664400)와 `mpr-ivl`(6664400 + 패치 5개)의 로컬 복원본, B2 애드온을 먼저 풀어 둔 상태, 실제 데이터를 놓고 실제 `setup`(애드온 복사, 이전 4B 계약과 설정·코드 해시·패키지 버전 비교, 4개·6개 언어 데이터 sha256 확인), 실제 `model` 검사(가짜 캐시: Qwen은 실제 바이트 수 8,875,719,344, 토크나이저 A-D = 32..35), 실제 실행 스크립트와 게이트(모의 모델), 실제 `queue_6x6.sh`(모의 모델, 79,020행 전체와 게이트)를 끝까지 돌렸다. 확인한 것: 두 모델 모두 REL 26,057 게이트 통과(strict true, 배치 1), 실험 A 8,532와 T_retry 444 통과(O/R 답을 같은 모델의 REL 결과와 비교하는 점검까지 실행), 6×6 79,020 통과, B2 애드온이 이미 서버에 넣어 둔 파일은 바이트 그대로, 4개 언어 폴더의 파일 목록과 원본 해시는 그대로, 결과 폴더는 새 것만 생김, 6×6 원자료는 Qwen 저장소 것을 검증해서 쓰고 InternVL 저장소는 링크, 모델 캐시는 6×6 성공 뒤 삭제, 한 모델만 다시 돌려도 이미 끝난 결과를 다시 만들지 않음, 9,000행에서 잘린 REL 예측을 이어서 26,057행으로 완성(앞 9,000행 그대로). 부정 시험: 서버 파일이 애드온과 다름, 이미 풀린 애드온 설정이 변경됨, 코드 해시 불일치, torch 버전 불일치, qwen-vl-utils 없음, 모델 바이트 수 불일치(Qwen 정확값, InternVL 범위), 토크나이저 불일치, 4개 언어 이미지 없음, fr 이미지 손상, venv 없음, Qwen 저장소 HEAD가 다름, InternVL 패치가 4개뿐, 6×6 원자료의 리비전이 다름, manifest 이미지가 비어 있음, 6×6 원자료가 아예 없을 때(받는다고 표시, 여유 8GB 미만이면 멈춤).

## 미검증

- 실제 GPU 추론, 24GB 메모리 사용량, 실제 속도. 위 시간은 이전 실행의 중앙값에서 곱한 추정이다. fr/ja 질문·화면이 이전 4개 언어보다 길어 느려질 수 있다. 6×6의 추론당 시간은 4B에서 잰 적이 없다.
- 실제 Hugging Face 내려받기와 속도. 시험은 가짜 `hf`와 가짜 캐시(토크나이저는 Qwen3-VL-4B 것을 InternVL3.5-4B의 대용으로 사용)로만 했다. Qwen3-VL-4B의 safetensors 바이트 수(8,875,719,344, 2개 조각)는 로컬 Hugging Face 캐시의 실제 파일 크기다. InternVL3.5-4B의 safetensors 정확한 바이트 수와 조각 수는 모르고, 검사는 8.0~10.5GB 범위뿐이다(HF API 합계 9.48GB는 이전 런북 값). 확인 필요.
- 서버 `vast4090b`의 실제 상태. `mpr-gui-crosslocale`의 HEAD가 66644001이고 `mpr-ivl`이 그 위 패치 5개이고 venv에 torch 2.9.1+cu130, transformers 4.57.6, accelerate 1.15.0, timm 1.0.31, qwen_vl_utils(Qwen), timm과 einops(InternVL)가 있다는 가정이다. B2 큐와 InternVL2.5 REL6 큐가 그 저장소에 넣어 둔 애드온 파일이 이 애드온의 같은 경로 파일과 바이트가 같다는 가정도 같다. 로컬에서는 B2 애드온 tar를 먼저 풀어 둔 저장소로 확인했지만 서버 파일 자체는 보지 않았다. 다르면 다운로드 전에 setup 또는 6×6 준비 단계가 멈춘다.
- 6×6 원자료가 서버에 있는지는 모른다. 있으면 `01_code/data/raw/mpr_gui_bench`가 위 리비전과 같은지, manifest가 sha `4dd919fc…`인지, 이미지가 빠짐없이 있는지 확인해서 쓰고, 없으면 `queue_6x6.sh`가 2.6GB를 받는다. 서버의 Hugging Face 요청 제한 여부는 모른다.
- `queue_6x6.sh`는 시험 모드(`Q6_TEST=1`: 모의 모델, 가짜 venv)로만 돌렸다. 서버의 GNU `timeout`/`flock`/`df -BG`/`pgrep` 동작과 `queue_6x6.sh` 자신의 설정 파일 덮어쓰기(위험 표 참고)는 확인하지 못했다.
- 6×6 게이트의 토큰 범위(Qwen 프롬프트 토큰 800~2,700, InternVL 시각 토큰 1,792~2,304)는 8B 실행 값이다. 4B에서 6×6의 이미지가 이 범위를 벗어나면 smoke 게이트가 멈춘다. 이전 4B REL 게이트는 같은 범위로 통과했지만 fr/ja 화면과 6×6 전체 이미지는 확인하지 못했다.
- 결과 tar 크기(모델당 0.5~1.5GB는 추정). 실험 A 게이트의 이상 징후 기준(파싱 실패 0.35% 이하 등)은 이전 빌드 값이다.
- 이전 4B 4개 언어 결과와 새 6개 언어 결과를 같은 입력에서 맞춰 보는 일관성 점검(MiniCPM 큐에는 있다)은 이 큐에 넣지 않았다. 요청에 없어서이고, 원하면 REL 일부를 다시 돌려 비교하는 단계를 더할 수 있다.
