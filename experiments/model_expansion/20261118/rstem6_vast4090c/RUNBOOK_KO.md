# R_stem / R_opts 6개 언어: 서버 `vast4090c` 실행 안내

2026-11-18 작성. 서버에는 아무것도 올리지도 실행하지도 않았다. 서버에서 한 일은 읽기 전용 확인(`ls`, `cat`, `sha256sum`, `df`, `nvidia-smi`, `pip freeze`)뿐이다. 로컬에서 확인한 것은 맨 아래 "로컬에서 확인한 것", 확인하지 못한 것은 "미검증"에 있다.

## 무엇을 돌리나

5개 모델이 같은 입력 `inputs/rstem6_conditions.jsonl`(18,960행 = 4,740쌍 × 4조건, sha256 `940ed67e…`, `rstem6_lock.json`에 고정)을 한 번씩 돈다. 한 쌍은 같은 화면에 O, R_stem(R 질문 줄기 + O 보기), R_opts(O 질문 줄기 + R 보기), R이다. 쌍은 158문항 × 언어 조합 30개(en, zh, th, ru, fr, ja 중 질문 언어 ≠ 화면 언어)다. 한 쌍의 네 조건은 항상 이어서 돈다.

| 순서 | 모델 | 저장소 (`/workspace/…`) | 배치 / strict | 모델 크기 | 기준이 된 예전 실행 |
|---|---|---|---|---|---|
| 1 | Qwen3-VL-8B (`Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd0`) | `mpr-gui-crosslocale` (있음) | 4 / 예 | 17.5 GB | 11-09 R_stem/R_opts(배치 4, strict), 11-07 REL 실행 계약 |
| 2 | MiniCPM-V-4.5 (`openbmb/MiniCPM-V-4_5` @ `14a20049`) | `mpr-minicpm` (있음, 모델 캐시도 이미 있음) | 1 / 아니오 | 17.4 GB | 11-16 REL 실행 계약 |
| 3 | UI-TARS-1.5-7B (`ByteDance-Seed/UI-TARS-1.5-7B` @ `683d002d`) | `mpr-gui-crosslocale` (있음) | 1 / 예 | **33.2 GB** | 11-14 REL 실행 계약 |
| 4 | InternVL3.5-8B (`OpenGVLab/InternVL3_5-8B` @ `9bb6a56a`) | `mpr-ivl` (**새로 만든다**) | 1 / 아니오 | 17.1 GB | 11-09 REL 실행 계약, 11-10 R_stem/R_opts(배치 1) |
| 5 | InternVL2.5-8B (`OpenGVLab/InternVL2_5-8B` @ `e9e4c0dc`) | `mpr-ivl` | 1 / 예 | 16.2 GB | 11-14 실험 A 실행 계약(이 모델은 REL, R_stem/R_opts를 돈 적이 없다) |

모델 적재, 프롬프트(`english_fixed` 시스템 프롬프트, `mpr_label_only_v1`), 그리디 `max_new_tokens=2`, A–D 첫 토큰 점수, 파싱은 예전 실행기와 같은 저장소 함수(`run_rel_followup.py`의 `predict_chunk` 등)를 그대로 부른다. 새 프롬프트는 없다. 모델마다 시작 전에 모델 설정 yaml, 시스템 프롬프트, 추론 코드 해시(12~13개 파일), 패키지 버전(torch, transformers, accelerate, torchvision, timm, sentencepiece, qwen-vl-utils, protobuf)이 위 "예전 실행 계약"과 같은지 본다. 다르면 그 모델은 GPU를 쓰기 전에 멈춘다.

Qwen3-VL-8B만 배치 4(한 쌍 = 한 배치)다. 11-09 R_stem/R_opts 실행이 그랬고 REL 계약(11-07)도 배치 4, strict이기 때문이다. 배치 1로 바꾸면 11-09 결과와 비교가 달라질 수 있다(배치 구성에 따른 수치 차이). 바꾸려면 코드가 아니라 결정이 필요하다(아래 "결정이 필요한 것").

### 입력 검사 (`run_rstem6.py`, 모든 모델 공통, 추론 전에 하나라도 어긋나면 중단)

- 입력 파일 sha256 = 잠금 값, 고정 REL 파일(`conditions_6lang.jsonl`, sha256 `63101abf…`) sha256 = 잠금 값.
- 모든 쌍에 O, R_stem, R_opts, R이 하나씩. 이미지 경로, 이미지 sha256, 정답, 문항, 언어, 분할이 네 조건에서 같음.
- O, R 행이 고정 REL의 O, R 행과 같음(질문, 줄기, 보기, 정답, 이미지 경로, 문항, 언어). 선택으로 `OR_gemma_conditions.jsonl`(sha256 `bef9894f…`)과도 같음을 본다(서버에는 이 파일을 올리지 않으므로 큐는 쓰지 않는다. 로컬에서는 통과).
- R_stem = R의 줄기 + O의 보기, R_opts = O의 줄기 + R의 보기, 모든 행 `question == 줄기 + " A: .. B: .. C: .. D: .."`.
- 이미지 948장 전부 sha256이 행의 값과 같음(`--check-images`, 설정 단계에서 한 번).
- `input_id` = `rel_rstem6::<pair_id>::<조건>`. `--resume`이면 쌍 단위로 다시 돈다(한 쌍의 네 조건 중 하나라도 성공이 아니면 네 조건 모두 다시).

## 큐가 하는 일

`queue_rstem6_c.sh` 하나가 모델 5개를 위 순서로 돈다. 모델마다:

1. **설정**(90분 제한): 애드온 묶음을 저장소의 `followup_rel_6lang/rstem6/`에만 푼다(목록에 다른 경로가 있으면 거부). 이미지와 고정 REL 파일 확인. 저장소가 `6664400`의 후손인지 확인. `env_check_rstem6.py`로 위 계약 대조.
   - `mpr-ivl`이 없으면(3번 모델 때) 만든다: `git clone`(5번 재시도) → `6664400` 체크아웃 → 서버에 있는 `/workspace/sixbysix_addon/ivl_patches/0001~0005` 적용(패치 id 확인) → venv → `requirements-qwen3-4090.txt`와 제약 파일 `internvl35_constraints_20261109.txt`(sha256 `a3d98e62…`)로 pip 설치. 예전 InternVL 실행이 쓴 방식과 같다(`queue_6x6.sh`의 `fn_setup ivl`, 6×6 데이터 부분만 뺐다). 이 저장소는 InternVL 패치 때문에 Qwen/MiniCPM 저장소와 코드 해시가 달라서 **따로 만들어야 한다**.
   - 이미지/고정 파일이 없는 저장소는 `mpr-minicpm`, `mpr-gui-crosslocale`에서 복사(`cp -n`, 덮어쓰지 않음) → 이미지 tar(`internvl25_rel6_frja_images_20261116.tar`, 서버에 있음) → 저장소의 `data_bundle.py`로 받기 순서로 채우고, 마지막에 이미지 전부를 sha256으로 확인한다.
2. **모델 받기**(150분 제한): 디스크 점검, `hf download`(3번 재시도), 안전텐서 조각 수와 바이트 합이 예전 실행 기록과 정확히 같은지(Qwen3 4개 17,534,339,512 / MiniCPM 4개 17,391,890,184 / InternVL3.5 4개 17,056,729,432 / UI-TARS 7개 33,168,747,240 / InternVL2.5 4개 16,150,800,432), 구조(UI-TARS, Qwen3, InternVL2.5), 토크나이저 A–D 토큰 id(InternVL2.5는 290,309,289,302, 나머지 32–35).
3. **smoke**(40분 제한): `--max-items 2` = 2문항 × 30 언어조합 × 4조건 = **240회** + 게이트.
4. **ETA 점검**: smoke 중앙값 × 18,960 이 본 실행 제한(시간)보다 길면 본 실행을 시작하지 않는다(`STOP … ETA`, `IGNORE_ETA=1`로 강행).
5. **본 실행** 18,960회(`--resume`) + 완료 게이트.
6. **포장**: `/workspace/queue_results/rstem6/<모델>/rstem6_<모델>_results_<날짜>.tar.gz` + `.sha256` + 파일별 sha256 목록(`…files.sha256`, 안에도 `results_files.sha256`). 실패한 모델은 이름 끝에 `_PARTIAL`. 전체 목록은 `PACKAGES_SHA256.txt`.
7. **모델 캐시 삭제**: 그 모델이 모든 단계를 통과하면 그 모델의 HF 캐시 폴더(`<저장소>/.cache/huggingface/hub/models--…`) 하나만 지운다. 경로 모양을 확인하고 심볼릭 링크는 지우지 않는다. `KEEP_MODEL_CACHE=1`이면 지우지 않는다. 실패한 모델의 캐시는 남기고, 다음 모델 받기에 자리가 모자랄 때만 지운다. **MiniCPM 캐시(17.4GB, 이전 큐가 남겨 둔 것)도 MiniCPM이 끝나면 지워진다.**

결과 위치: `<저장소>/followup_rel_6lang/results/rstem6_<모델>/{smoke,full}/`(새 폴더). 상태는 `/workspace/QUEUE_RSTEM6_STATUS`, 끝 표시는 `/workspace/QUEUE_RSTEM6_DONE`(`OK` 또는 `FAILED steps: …`). 예전 큐의 `QUEUE_C*`, `QUEUE6_*`는 건드리지 않는다.

### 게이트 규칙 (`check_rstem6_gate.py`)

- 모든 행 성공, 모든 쌍에 네 조건, 배치 모양(배치 4: 한 배치 = 한 쌍의 네 조건, 배치 1: 한 배치 = 입력 하나), 기록된 배치 크기 = 기대값.
- 파싱 실패: smoke 3개 이하, 본 실행 max(2, 0.35%) 이하.
- 공통 규칙(예전 REL/실험 A 게이트와 같은 파일): 파싱된 라벨이 모두 같지 않음, A–D 확률 질량 중앙값 ≥ 0.5, 시각 토큰/프롬프트 토큰 범위(모델 계열별), InternVL2.5와 Qwen3의 시스템 프롬프트 규칙. smoke 정확도가 0.30 미만이면 경고만.
- **기준 일치율 두 가지**(문제 = 중단, 공통 행이 30개 미만이면 경고만):
  - `rel_OR`: 5개 모델 모두. 이 입력의 O, R 질문은 같은 모델의 예전 REL O/R 행과 글자까지 같다(기준 파일을 만들 때 확인함). 같은 쌍의 예전 답과 라벨 일치율이 0.90 이상이어야 한다. 기준: Qwen3 5,682행(fr/ja 새 쌍), MiniCPM·UI-TARS·InternVL2.5 9,474행(4개 언어 + fr/ja), InternVL3.5 5,682행.
  - `rstem4lang`: Qwen3(11-09), InternVL3.5(11-10)만. 4개 언어 부분 1,896쌍 × 4조건 = 7,584행이 예전 R_stem/R_opts 실행의 같은 행이다(질문 글자 같음 확인). 일치율 0.90 이상이어야 하고, 본 실행에서는 조건별 정확도가 예전과 0.05 넘게 다르면 경고(예전 정확도: Qwen3 O 0.7015, R_stem 0.7769, R_opts 0.7321, R 0.8397 / InternVL3.5 O 0.6951, R_stem 0.7278, R_opts 0.7099, R 0.7505).
  - MiniCPM, UI-TARS, InternVL2.5에는 예전 R_stem/R_opts 결과가 없으므로 R_stem/R_opts 쪽 비교 기준이 없다(O, R 일치율만 본다).
- smoke의 일치율 0.90은 임의로 잡은 값이다(같은 설정 Qwen3-VL-8B 일관성 점검이 이 서버에서 0.9917이었다는 기록이 있다. MiniCPM 큐 RUNBOOK 참조). 배치 1 모델은 서버만 바뀌었으므로 거의 100%여야 한다.

## 올릴 파일과 실행

`upload/`의 두 파일을 `/workspace`에 올린다(`SHA256SUMS.rstem6`은 이름이 `/workspace/SHA256SUMS`와 겹치지 않게 지은 것).

| 파일 | 크기 | sha256 |
|---|---|---|
| `queue_rstem6_c.sh` | 약 32 KB | `71110d18e712b3b0890f367603caf050e54c92eece8bbaa246570d669e9c26da` |
| `rstem6_addon_20261118.tar.gz` | 약 0.9 MB | `eb747ca0f9607590b523edba54ca827e89e63739aa9e05ce5d3d4940c2f1a622` |

서버에 이미 있어야 하는 것(읽기 전용으로 있음을 확인함, 큐가 sha256을 다시 확인): `/workspace/sixbysix_addon/`(SHA256SUMS 14줄, 확인 통과), `internvl25_rel6_frja_images_20261116.tar`(749 MB, 없어도 되지만 있으면 쓴다), `mpr-gui-crosslocale`(커밋 `6664400`, venv, 6개 언어 데이터와 이미지), `mpr-minicpm`(어댑터 패치, 모델 캐시, venv).

```bash
cd /workspace && shasum -a 256 -c SHA256SUMS.rstem6    # 서버에서는 sha256sum -c
tmux new -s rstem6
bash queue_rstem6_c.sh --plan        # 아무것도 바꾸지 않고 모델 순서, 디스크 필요량, 제한 시간, 추정 시간을 출력
bash queue_rstem6_c.sh               # 본 실행
tail -f /workspace/QUEUE_RSTEM6_STATUS   # 다른 tmux 창에서
```

일부만 돌리려면 `ONLY="minicpm_v45 uitars_15_7b" bash queue_rstem6_c.sh`(순서는 고정). 환경변수: `KEEP_MODEL_CACHE=1`, `FORCE_SLOW_NET=1`, `IGNORE_ETA=1`, `NET_MIN_MBPS`(20), `FULL_H_<모델키>`(본 실행 제한 시간), `MIN_FREE_GB_<모델키>`.

GPU를 쓰는 프로세스가 있거나 GPU 메모리가 1.5GB 이상 쓰이고 있거나 다른 `queue_*.sh`가 돌고 있으면 기다리지 않고 `FAILED steps: gpu`(또는 `others`)로 끝난다. 도중에 끊기면 같은 명령을 다시 치면 이어서 한다(끝난 모델은 `COMPLETE` 표시로 건너뛰고, 중간 모델은 예측 캐시에서 이어 한다).

### 사람이 smoke 직후에 볼 줄

각 모델의 smoke가 끝나면 상태 파일에 한 줄이 남는다. 처음 두 모델(Qwen3, MiniCPM)은 사람이 눈으로 확인하고 나서 나머지를 두는 편이 안전하다.

```bash
grep -E "smoke gate:|complete gate:|STOP|FAILED" /workspace/QUEUE_RSTEM6_STATUS
```

`smoke gate:` 줄에 조건별 정확도, 파싱 실패, 중앙값 s/회, 예상 시간, 기준 일치율이 나온다. 일치율이 0.90 아래면 게이트가 이미 그 모델을 멈춘다.

## 시간 추정 (이 서버에서 재지 않은 추정, 미검증)

초당 값은 같은 모델이 4090에서 예전에 돈 실행의 `inference_runtime_ms`(성공 행, 쌍/배치 구성이 같은 실행)다. 이 입력은 fr/ja 질문이 60%라서(예전 실행은 4개 언어만인 것도 있다) 속도가 다를 수 있다.

| 모델 | 근거 실행 | 중앙값 / 평균 (s/회) | 18,960회 | 본 실행 제한 |
|---|---|---|---|---|
| Qwen3-VL-8B | 11-09 R_stem/R_opts 7,584회, 배치 4 | 0.615 / 0.773 | 3.2–4.1 h | 8 h |
| MiniCPM-V-4.5 | 11-17 실험 A 새 쌍 8,532회 | 0.512 / 0.539 | 2.7–2.8 h | 6 h |
| InternVL3.5-8B | 11-10 R_stem/R_opts 7,584회 / 11-16 실험 A 8,532회 | 0.751 / 0.732 · 0.762 / 0.743 | 3.9–4.0 h | 8 h |
| UI-TARS-1.5-7B | 11-16 실험 A 새 쌍 8,532회 | 1.004 / 1.039 | 5.3–5.5 h | 10 h |
| InternVL2.5-8B | 11-16 실험 A 전체 14,220회 | 0.290 / 0.378 | 1.5–2.0 h | 5 h |
| 합계 추론 | | | 약 17–18.4 h | |

환경 준비(`mpr-ivl` 새로 만들기 pip 설치 약 15–30분, 이미지 복사), 모델 받기(MiniCPM은 이미 있어서 4개, 합계 약 84 GB, 20–100 MB/s 가정 시 모델당 3–15분; UI-TARS는 33 GB라서 더 걸림), 모델 적재와 smoke(모델당 약 5–15분)를 더하면 **전체 약 19–21시간**이다. 임대 시간만큼 비용이 든다.

## 디스크 (지금 `/workspace` 65 GB 중 29 GB 비어 있음, 2026-11-18 읽기 전용 확인)

| 시점 | 빈 공간 (추정) |
|---|---|
| 시작 | 29 GB (MiniCPM 캐시 17 GB가 차 있음) |
| Qwen3 받은 직후 | 약 11.5 GB (최소 20 GB 있어야 시작: 모델 17.5 + 3) → 끝나면 캐시 삭제 → 29 GB |
| MiniCPM 끝, 캐시 삭제 | 약 46 GB |
| **UI-TARS 받기**: 33.2 GB(fp32) + 3 GB 여유 = 36 GB 필요 | 46 GB ≥ 36 이라 통과, 받은 직후 약 13 GB → 끝나면 캐시 삭제 → 약 46 GB |
| `mpr-ivl` 만들기(venv 약 5.2 GB, 저장소 약 0.3, 4개 언어 이미지 약 1.4, fr/ja 이미지 약 1.1 → 약 8 GB, 확인 필요) | 약 38 GB |
| InternVL3.5 받은 직후 / 끝난 뒤 (20+10 GB 필요) | 약 21 GB / 약 38 GB |
| InternVL2.5 | 약 22 GB 남음 |

UI-TARS가 가장 빠듯하다(여유 약 2 GB로 시작, 받은 뒤 약 5 GB). 계산이 어긋나 36 GB에 못 미치면 큐는 `STOP: only N GB free (need 36 GB for …)`로 그 시점에 멈추고(앞 모델들은 이미 포장됨) 더 받지 않는다. 그때 사람이 공간을 만들어야 한다(예: 받아 둔 결과를 내려받은 뒤 필요 없는 폴더 정리. 큐는 자기 모델 캐시 외에는 지우지 않는다). 결과 파일은 모델당 수십~수백 MB(추정)다.

## 위험

| 위험 | 크기 | 대비 / 확인 |
|---|---|---|
| 디스크 부족 | UI-TARS 직전 계산상 46 GB, 필요 36 GB (디스크 부족은 환경 실패라 큐 전체가 멈춘다) | 큐의 디스크 점검이 받기 전에 멈춘다. `--plan`과 `df -h /workspace`로 확인 |
| `mpr-ivl` pip 설치가 실패하거나 버전이 예전과 달라짐 | 4번 모델의 설정에서 멈춤(4, 5번 모델 못 돈다, 1·2·3번은 이미 끝남) | 제약 파일 사용, 설치 뒤 env_check가 패키지 버전을 계약과 대조. 로그 `queue_results/rstem6/logs/setup_internvl35_8b.log` |
| Qwen3 배치 4에서 메모리 부족 | `--strict-batch-size`라 자동으로 배치를 바꾸지 않고 멈춘다 | 11-09 실행(같은 4090 계열, 배치 4)은 통과했다. 멈추면 배치를 임의로 바꾸지 말고 알린다 |
| 4090 24 GB에서 UI-TARS 메모리 | 가중치가 fp32 33 GB지만 예전 실행(같은 설정)은 통과 | 설정 yaml이 예전과 byte 단위로 같음(sha256 `e16c3fc0…`) |
| HF 429/느린 네트워크 | 받기 지연 | 3번 재시도, 시작 시 네트워크 점검(20 MB/s 미만이면 시작하지 않음) |
| 중간에 끊김 | 수 시간 손실 | tmux, `--resume`(쌍 단위 재개), 모델마다 포장 |
| 같은 쌍에서 질문이 같은 경우 | 서로 다른 쌍 기준 32쌍(15문항, 4,740의 0.7%). 겹침 건수: R_stem=O 28, R_stem=R 10, R_opts=O 10, R_opts=R 28, O=R 6 | 오류가 아니라 계획(`plan.json`)에 기록만 한다. 4개 언어 실행기는 이런 경우 중단했다(그 입력에서는 0건). 해석 때 고려 |
| 서버 AGENTS.md / CLAUDE.md | 읽지 않는다 | |
| 인스턴스 비용 | 약 20시간 임대 | 끄는 판단은 사람이 한다. 결과를 모두 받고 sha256을 확인하기 전에 인스턴스를 지우지 않는다 |
| 삭제 | 큐가 지우는 것은 각 모델의 HF 캐시 폴더 하나(모델 끝난 뒤), 포장용 임시 폴더뿐 | `rm -rf`는 4곳뿐이고 모두 경로 확인이 있다(로컬 시험에서 다른 모델 캐시, 예전 결과, 예전 DONE 파일이 그대로임을 확인) |

## 중단하는 법

tmux 창에서 Ctrl-C가 가장 안전하다. 추론 프로세스가 남을 수 있으니 이어서:

```bash
pkill -TERM -f 'queue_rstem6_c\.sh'
pkill -f 'run_rstem6\.(py|sh)'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv    # 비었는지 확인
ls /workspace/QUEUE_RSTEM6_DONE                                              # 없어야 한다(중단하면 만들지 않는다)
```

같은 명령(`bash queue_rstem6_c.sh`)을 다시 치면 이어서 한다. 중단하면서 모델 캐시는 지워지지 않는다.

## 결과 받기

각 모델의 포장이 생기면 바로 받아도 된다(`scp`). `PACKAGES_SHA256.txt`와 포장별 `.sha256`, 안쪽 `results_files.sha256`로 확인한다. 결과를 모두 받고 sha256이 맞는 것을 확인한 뒤에 인스턴스를 지운다.

## 로컬에서 확인한 것 (`local_tests/`)

| 시험 | 결과 파일 | 내용 |
|---|---|---|
| 큐 흐름 52개 | `queue_tests_result.txt` | `--plan`, 모델 순서, 모델별 단계 순서, 포장과 sha256, 캐시 삭제와 범위, 이어 하기(완료 모델 건너뜀), 실패 규칙(smoke, 설정, 받기, ETA 중단, 시간 초과, CUDA 문구, 환경 실패), GPU 사용 중/1.5GB 경계/4090 아님/다른 큐/느린 네트워크/디스크 부족(19, 29, 35, 36 GB)/sha 불일치/저장소 없음, 중복 실행, SIGTERM, 정적 점검. 무거운 단계(setup, model, smoke, full)는 가짜 |
| 실제 설정 단계 11개 | `setup_local_result.txt` | 실제 `setup()`: 기존 저장소 통과(이미지 948장 sha256, 계약 대조), 이미지/고정 파일을 형제 저장소에서 복사, 이미지 하나를 망가뜨리면 설정 실패, `mpr-ivl`을 처음부터 만들기(clone + 패치 5개 적용 확인, pip는 오프라인이라 실패하는 것까지), 재실행 때 패치 id로 확인, 비어 있지 않은 폴더 거부 |
| 실행기/게이트 14개 | `runner_checks_result.txt` | 실제 이미지 전체 sha256, 입력 변조, 재조합 규칙 위반, 잘못된 배치 크기/설정, `--resume`, 기준 일치율 계산 |
| 모의 실행 5개 모델 | `mock_runs_result.txt` | 5개 모델 모두 `--mock-model`로 smoke 240회 + 본 실행 18,960회 + 두 게이트 통과(모델 키마다 해당 저장소 상태: 6664400 / +MiniCPM 패치 / +InternVL 패치 5개) |
| 계약 대조 5개 | `env_check_result.txt` | 5개 모델의 설정, 시스템 프롬프트, 추론 코드 해시가 예전 실행 계약과 같음(위 세 저장소 상태에서) |
| `bash -n` | | 큐, `run_rstem6.sh`, 시험 스크립트 |

## 미검증

- 서버에서의 모든 실행: `git clone`, pip 설치(`mpr-ivl` venv, 제약 파일), `hf download`와 실제 바이트 합 확인, 토크나이저 확인(`model()` 안의 파이썬 부분은 로컬에서 돌려 보지 않았다), 실제 모델 적재와 GPU 추론, 실제 속도, 메모리 최대치.
- 로컬 Mac 환경이라 계약의 일부 패키지 버전(torch `+cu130`, torchvision, timm, protobuf, sentencepiece, qwen-vl-utils)은 대조하지 못했다. 서버 venv(`mpr-gui-crosslocale`, `mpr-minicpm`)의 pip freeze에서는 torch 2.9.1+cu130, transformers 4.57.6, accelerate 1.15.0, torchvision 0.24.1+cu130, timm 1.0.31, sentencepiece 0.2.0, protobuf 6.33.6, qwen-vl-utils 0.0.14가 계약과 같은 것을 읽기 전용으로 확인했다(Python은 서버 3.12.3, 예전 일부 실행은 3.12.15이고 대조에서 제외).
- `mpr-ivl` 새 venv가 같은 버전으로 설치되는지(설치 뒤 `env_check`가 판정).
- 이미지 복사 크기(약 2.5 GB)와 venv 크기(약 5.2 GB는 다른 두 venv 기준 추정).
- 속도 추정(위 표), 디스크 계산(위 표).
- UI-TARS 7조각 첫 조각 파일 이름(`model-00001-of-00007.safetensors`)은 네트워크 점검 URL에만 쓰이고, 틀려도 경고만 한다.
- 결과 해석(정확도, 효과)은 하지 않았다.

## 변경 (2026-11-18, 검토 반영)

- 모델 순서를 Qwen3-VL-8B → MiniCPM-V 4.5 → **UI-TARS-1.5-7B → InternVL3.5-8B** → InternVL2.5-8B로 바꿨다. UI-TARS는 빈 공간 36GB가 필요한데, 새 `mpr-ivl`(약 8GB, 추정)을 만들기 전에 돌리면 여유가 약 10GB 더 생긴다. 위 표의 순서 칸은 바꾸기 전 것이다. 바꾸기 전 큐는 `queue_rstem6_c_v1_order_ivl_first.sh`로 남겼다.
- 큐 sha256: `5531d5110caf471d0c490a35f54c2e418ce790cf051fdb319d69249590384bc2` (upload/SHA256SUMS.rstem6 갱신).
- 위 표들(모델 순서, 디스크 흐름, 올릴 파일 sha, 위험 표)을 새 순서로 고쳤다. 바꾸기 전 안내서는 `RUNBOOK_KO_v1_order.md`.
- `git clone` 앞에 `GIT_TERMINAL_PROMPT=0`을 넣었다(인증을 물으며 멈추지 않게). 큐 머리말 주석의 순서도 고쳤다.
- 큐 시험을 새 순서로 다시 돌려 52개 모두 통과했다(`local_tests/queue_tests_result.txt`, 바꾸기 전 결과는 `queue_tests_result_v1_order.txt`). 디스크 시나리오 기대값 두 개를 새 순서에 맞게 고쳤다(29·35 GB에서 UI-TARS 앞에서 멈춤).
- 현재 큐 sha256: `71110d18e712b3b0890f367603caf050e54c92eece8bbaa246570d669e9c26da`.

## Qwen3 smoke를 사람이 볼 때 (검토 지적)

smoke 2문항에서 Qwen3 기준 라벨이 대부분 C라서, 거의 C만 내는 고장 난 실행도 일치율 0.90을 넘을 수 있다. Qwen3 smoke 게이트 줄을 보고 4개 언어 R_stem/R_opts 일치율이 0.97 이상인지 확인한다(11-09 실측: smoke 0.986, 본 실행 0.995). 낮으면 본 실행 도중 멈추고 알린다. 완료 게이트는 7,584 + 5,682행으로 다시 보므로 나쁜 결과가 최종 통과하지는 않는다.
