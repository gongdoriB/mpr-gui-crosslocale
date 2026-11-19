# vast4090b fr/ja 추가분 큐: 실행 순서 (로컬 준비만 끝남, 서버 미업로드)

2026-11-16 작성. 서버에는 아무것도 올리지도 실행하지도 않았다. 확인한 것은 `local_tests/`에, 확인하지 못한 것은 맨 아래 "미검증"에 있다.

## 두 가지 큐

| 파일 | 내용 | 비고 |
|---|---|---|
| `step1_ivl35_rel/queue_b2_step1_ivl35_rel.sh` | InternVL3.5-8B REL `new_only`만 (26,057건) | addon 없음. 서버에 이미 풀린 InternVL2.5 REL6 파일을 쓴다(그 파일들이 그대로 있어야 한다) |
| `queue_b2_frja.sh` + `b2_frja_addon_20261116.tar.gz` | 3개 모델 전체 (아래 순서) | 1단계가 끝나 있으면 InternVL3.5 REL은 건너뛴다 |

해시는 `SHA256SUMS`. 두 큐는 같은 상태 파일(`/workspace/QUEUE_B2_STATUS`, `QUEUE_B2_DONE`)과 같은 잠금 파일을 쓰므로 동시에 돌지 않는다. 1단계가 끝난 뒤 전체 큐를 시작하면 시작 때 `QUEUE_B2_DONE`을 지운다.

## 전체 큐가 하는 일 (순서 고정, `ONLY="..."`로 일부만)

| 순서 | 모델 | 저장소 | 내용 | 규모 |
|---|---|---|---|---|
| 1 | InternVL2.5-8B | `/workspace/mpr-ivl` | 실험 A **6개 언어 전체** main 쌍 (예전 4언어 1,896쌍 + 새 2,844쌍) x O/T/R, T_retry는 전부 | 14,220건, T_retry 262쌍 = 786건 |
| 2 | InternVL3.5-8B | `/workspace/mpr-ivl` | REL O/R/C/F `new_only` + 실험 A 새 쌍 (main) + T_retry | 26,057건 + 8,532건 + 444건 |
| 3 | UI-TARS-1.5-7B | `/workspace/mpr-gui-crosslocale` | REL `new_only` + 실험 A 새 쌍 (main) + T_retry | 26,057건 + 8,532건 + 444건 |

InternVL2.5의 4언어 부분은 예전 실험 A와 입력 id, 질문 글자, 렌더된 프롬프트(5,688건)가 바이트 단위로 같다(시험 D). 예전 실험 A는 첫 번째 Gemma 번역(`T_conditions.jsonl`, sha 0ebbed0f)을 썼으므로 그 파일을 그대로 쓴다. 4언어 T_retry는 `expA_translation_20261112/work/T_gemma_retry_conditions.jsonl`이고, 예전 T와 질문이 달라진 main 114쌍이 들어간다(이 114쌍에는 재번역뿐 아니라 첫 번역의 형식 오류 수정분이 섞여 있을 수 있다. 해석 시 주의, 미검증).

## 예전 실행과 같은 조건 (예전 run_contract.json에서 확인)

| 모델 | 예전 실행 | 배치 | strict_batch_size | 어텐션 | 설정 파일 sha256 |
|---|---|---|---|---|---|
| InternVL2.5-8B | 실험 A 2026-11-14 (`model_results_20261114/internvl25_expA/.../expa_full/run_contract.json`) | 1 | true | eager | c78cd7ea... |
| InternVL3.5-8B | REL 2026-11-09 (`rstem_ropts_internvl35_20261110/.../run_contract_internvl35_rel_20261109.json`), 실험 A 2026-11-14 (`expA_results_20261114/internvl35/.../full/run_contract.json`) | 1 (둘 다) | false (둘 다) | eager | 7fed517c... |
| UI-TARS-1.5-7B | REL, 실험 A 2026-11-14 (`model_results_20261114/uitars_15_7b/unpacked/sizes_uitars_15_7b/{rel_full,expa_full}/run_contract.json`) | 1 (둘 다) | true (둘 다) | sdpa | e16c3fc0... |

사용자 결정(예전과 같은 조건)대로 기본값은 전부 배치 1이다. 바꿀 수 있는 것은 `EXPA_BATCH_uitars_15_7b=3`(한 쌍 = 한 배치)뿐이고, InternVL 쪽은 코드가 배치 1만 받는다. REL은 배치 1 고정(완료 게이트가 배치 1을 요구).
실행 전 검사(`env_check.py`): 설정 yaml, system prompt, 추론 코드 해시, 패키지 버전(torch 2.9.1+cu130, transformers 4.57.6, accelerate 1.15.0, torchvision, timm, sentencepiece, qwen-vl-utils)이 예전 계약과 같아야 하고, 다르면 설치하지 않고 멈춘다. InternVL 두 모델의 기준 코드는 예전 실험 A와 같은 `mpr-ivl`(커밋 9694fff) 해시이고, UI-TARS는 `mpr-gui-crosslocale`(커밋 6664400) 해시다.

## 올릴 파일과 실행

`/workspace`에 `queue_b2_frja.sh`, `b2_frja_addon_20261116.tar.gz`(약 6.6 MB)을 올린다. fr/ja 이미지 tar는 필요 없다(InternVL 저장소에 이미 있고 UI-TARS 저장소에는 그것을 복사해 sha256으로 검증한다. 올려 두면 그것을 우선 쓴다).

```
cd /workspace
# 1단계(선택, 먼저): InternVL3.5 REL만
bash queue_b2_step1_ivl35_rel.sh --plan
bash queue_b2_step1_ivl35_rel.sh        # tmux 안에서
tail -f /workspace/QUEUE_B2_STATUS
# 전체
bash queue_b2_frja.sh --plan
bash queue_b2_frja.sh                   # tmux 안에서
```
다른 큐를 기다리지 않는다. 다른 `queue_*.sh`/`run_all_overnight.sh`가 돌거나 GPU를 쓰는 프로세스가 있거나 GPU 메모리가 1.5GB 이상 쓰이면 아무것도 시작하지 않고 `FAILED steps: others|gpu`로 끝난다.

## 실패 처리 규칙 (구현한 것)

- **모델 한정 실패**(모델 받기/검사 실패, smoke 또는 게이트 실패, 실행 오류, 시간 초과, ETA 중단, 포장 실패): 그 모델의 나머지 단계를 멈추고(smoke가 실패한 쪽의 본 실행만 건너뜀) GPU 비어 있음·4090·디스크 5GB 재확인을 통과하면 다음 모델로 간다. 그 모델 결과는 `_PARTIAL` 이름으로 포장하고 캐시는 남긴다(다음 다운로드에 자리가 모자라면 자동 삭제).
- **환경 실패**: GPU 사용 중/4090 아님/nvidia-smi 불가, 다른 큐 실행, 네트워크 20 MB/s 미만, addon·이미지 tar sha256 불일치, 저장소·venv 없음, setup 실패(venv·코드 해시·설정·데이터 불일치, 서버의 기존 addon 파일이 다름), 다음 다운로드용 디스크 부족, 단계 로그에 CUDA 치명 오류 문구(CUDA error, device-side assert, Xid, NVML 등), 실패 뒤 GPU가 풀리지 않음 등: 큐 전체를 멈춘다. 남은 모델은 시작하지 않고 `QUEUE_B2_DONE`에 `FAILED steps: ...`가 적힌다.
- 같은 명령을 다시 치면 이어서 한다(포장이 끝난 모델은 `COMPLETE` 표시로 건너뜀, 예측 캐시는 `--resume`).

## 결과 위치

`/workspace/mpr-ivl/followup_rel_6lang/results/{expA6_internvl25_8b, rel6new_internvl35_8b, expA6_internvl35_8b}`, `/workspace/mpr-gui-crosslocale/followup_rel_6lang/results/{rel6new_uitars_15_7b, expA6_uitars_15_7b}`: 전부 새 폴더이고 `followup_rel_4lang/results`에는 아무것도 쓰지 않는다. 각 `expA6_*` 아래 `full/`(O/T/R)와 `retry/`(O/T_retry/R), 포장은 `/workspace/queue_results/b2_frja/<모델>/b2_<모델>_results_<날짜>.tar.gz`(+`.sha256`, 안에 파일별 sha256 `results_files.sha256`).

## 시간 추정 (추정치, 이 서버에서 재지 않음)

근거는 예전 4090 실행의 `inference_runtime_ms` 중앙값이다: InternVL2.5 실험 A 0.282초(이 서버 REL6 0.292초), InternVL3.5 REL 0.748초, 실험 A 0.752초, UI-TARS REL 1.009초, 실험 A 0.977초. 새 언어 프롬프트가 더 길면 달라질 수 있다.

| 단계 | 계산 | 시간 |
|---|---|---|
| 1 InternVL2.5 실험 A 14,220건 + 786건 | x 0.29 | 약 1.1시간 + 4분 |
| 2 InternVL3.5 REL 26,057건 | x 0.75 | 약 5.4시간 |
| 2 InternVL3.5 실험 A 8,532건 + 444건 | x 0.75 | 약 1.8시간 + 6분 |
| 3 UI-TARS REL 26,057건 | x 1.01 | 약 7.3시간 |
| 3 UI-TARS 실험 A 8,532건 + 444건 | x 0.98 | 약 2.3시간 + 7분 |
| 모델 받기·적재·점검 | 16.2 / 약 17 / 33.2 GB | 합계 약 1시간 |
| 합계 | | 약 19시간 (1단계를 먼저 끝냈으면 전체 큐는 약 13.6시간) |

단계별 한도(기본): REL InternVL3.5 9시간, UI-TARS 11시간, 실험 A 3~4시간. smoke 중앙값으로 계산한 예상 시간이 한도보다 길면 그 단계를 시작하지 않는다(`IGNORE_ETA=1`로 강행, `REL_H`/`EXPA_H`로 한도 변경).

## 디스크 (시작 시 약 67GB 여유)

한 번에 모델 캐시 하나: InternVL2.5 16.2GB, InternVL3.5 약 17GB, UI-TARS 33.2GB(fp32 저장). 모델이 끝나면(포장 성공 + 모든 단계 통과) 그 모델 캐시를 지운다(`KEEP_MODEL_CACHE=1`이면 유지). 결과와 포장은 모델당 1.5GB 미만으로 추정(미검증: 예전 InternVL2.5 REL6 모의 실행이 약 0.5GB). 다운로드 전 여유 기준: 30 / 30 / 48GB(캐시가 이미 있으면 생략). 최대 사용량은 UI-TARS 단계의 약 36GB 증가분.

## 위험과 대비

| 위험 | 크기 | 대비 |
|---|---|---|
| 24GB 메모리 부족 | InternVL2.5 7타일, InternVL3.5 8타일(2,304 토큰), UI-TARS 2MP는 예전 4090 실행과 같은 설정이라 같은 범위 | 배치 1, 연속 3회 실패 시 실행기가 멈춤 |
| 시간 | 추정 약 19시간 | 단계 한도와 ETA 중단, 모델별로 끊어 이어 하기 가능 |
| 디스크 | UI-TARS 33GB | 다운로드 전 48GB 점검, 실패한 모델의 캐시 자동 정리 |
| 서버 저장소가 로컬 복원본과 다름 | 설정·코드 해시 검사에서 멈춤(다운로드 전) | `FAILED steps: setup_<모델>` |
| 기존 결과 덮어쓰기 | 새 폴더만 만든다. addon은 파일이 없을 때만 복사하고 이미 있는 파일은 바이트가 같아야 한다 | 시험에서 확인 |
| 인스턴스 비용 | 약 19시간 임대 | 끄는 판단은 사람이 한다 |

## 미검증 (2026-11-16)

- 실제 GPU 추론, 24GB 메모리 사용량, 실제 속도(위 시간은 추정).
- 실제 Hugging Face 내려받기. 시험은 가짜 `hf`와 가짜 캐시(토크나이저는 Qwen3-VL-4B 것을 대용)로만 했다. InternVL3.5-8B와 UI-TARS의 safetensors 정확한 바이트 수는 모른다(검사는 16.0~18.0GB, 32.5~34.0GB 범위). InternVL3.5의 `config.json` 구조는 출력만 하고 검사하지 않는다(확인 필요). InternVL2.5는 정확한 바이트(16,150,800,432), 아키텍처, 토큰 id를 검사한다.
- InternVL3.5/UI-TARS의 토큰 A-D id 32~35는 예전 예측 파일의 `label_token_ids`에서 확인한 값이다.
- 서버 `mpr-ivl`, `mpr-gui-crosslocale`의 실제 코드·venv 상태: 로컬 복원본이 예전 계약의 코드 해시와 같음을 시험했을 뿐이고, 서버는 setup 단계가 검사한다. UI-TARS가 이 서버(vast4090b)의 `mpr-gui-crosslocale`에서 예전에 돌았다고 가정했다(확인 필요). 다르면 setup이 멈춘다.
- UI-TARS 저장소에는 fr/ja 이미지가 없을 수 있다고 보고 `mpr-ivl`에서 복사하는 경로를 넣었다(시험함). 서버 `mpr-ivl`의 이미지 위치(`followup_rel_6lang/assets`)는 InternVL2.5 REL6 런북에서 가정했다.
- 실험 A의 이상 징후 게이트(파싱 실패 0.35% 이하 등)는 Qwen3 빌드의 규칙을 그대로 썼다. 예전 세 모델의 파싱 실패는 0건이었지만 fr/ja 입력에서의 값은 모른다.
- InternVL2.5 REL6 결과(`rel6_internvl25_8b`)가 서버에 있으면 실험 A의 O/R 답을 그것과 비교(앵커, 0.90 미만이면 게이트 실패). 없으면 비교하지 않고 경고만 한다.
