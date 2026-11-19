# 6×6 전체 벤치마크 큐 (RTX 4090, 추가 모델 7개)

작성 2026-11-14, 같은 날 리뷰 수정 반영(수정 전 파일은 `superseded/`에 `_v1`로 남겼다). 서버(Vast)에서는 아무것도 실행하지 않았다(미검증). 로컬에서는 가짜 `/workspace`에 가짜 `hf`/`nvidia-smi`/`curl`/`timeout`/`df`를 붙여 큐 흐름을 돌렸고, 6×6 데이터와 manifest는 실제로 받아서 고유 입력 79,020개를 확인했다. 실제 모델 로드와 GPU 추론은 한 번도 해보지 않았다.

## 돌리는 것

모델마다 호출은 `01_code/run_cross_locale.py --dimensions wf wi au ap ael rel --sample-unit semantic_items --seed 42 --batch-size 1 --score-labels --system-prompt-mode english_fixed --device-map cuda:0 --resume`이다. 고유 추론 79,020회(2,195문항 × 36 언어 조합). 이번 큐의 배치는 전부 1이다. 8B 실행과의 관계는 모델마다 다르다. InternVL3.5-8B 6×6은 배치 1로 돌았으니 같은 호출이다. Qwen3-VL-8B 6×6은 배치 1이 아니라 4였다. 근거는 `analysis_qwen3_20261108/raw/full_6x6.jsonl`의 79,020행 전부에서 `batch_size`가 4라는 것(`batch_runtime_ms` 2,186 ms가 4입력의 합이고 추론당 `runtime_ms` 평균은 596 ms)이다. `QWEN3_RTX4090_RESULTS_KO.md` 7절의 "Qwen3 6×6 … 생성 배치 1"이라는 표현은 본 실행 전 준비 단계의 계획이고 실제 실행 결과와 다르다. 이 문서의 이전 판은 이 표를 따라 "8B와 같은 호출, 배치 전부 1"이라고 썼는데 Qwen3 쪽은 틀렸다. 순서는 모델마다 내려받기, 계획 점검, smoke(`--sample-size 6`, 216회)와 게이트, 전체 실행과 게이트, 요약(`summarize_cross_locale`), 결과 묶기, 모델 캐시 정리다. 한 모델이 실패해도 다음 모델로 넘어간다.

기본 순서(환경변수 `MODELS`로 바꾼다):

| 순서 | 키 | 모델 | 저장소 | 고정 revision | 가중치 |
| --- | --- | --- | --- | --- | --- |
| 1 | qwen3_vl_4b | Qwen/Qwen3-VL-4B-Instruct | mpr-gui-crosslocale | ebb281ec70b05090aa6165b016eac8ec08e71b17 | 8.89 GB |
| 2 | internvl35_4b | OpenGVLab/InternVL3_5-4B | mpr-ivl | 481f6e32467eab4e922ccd7fd6cf420441a62331 | 9.48 GB |
| 3 | uitars_15_7b | ByteDance-Seed/UI-TARS-1.5-7B | mpr-gui-crosslocale | 683d002dd99d8f95104d31e70391a39348857f4e | 33.18 GB (fp32 저장) |
| 4 | minicpm_v45 | openbmb/MiniCPM-V-4_5 | mpr-minicpm | 14a2004908140699317e2109e90573c1e6323c36 | 17.4 GB |
| 5 | qwen3_vl_2b | Qwen/Qwen3-VL-2B-Instruct | mpr-gui-crosslocale | 89644892e4d85e24eaac8bacfd4f463576704203 | 4.27 GB |
| 6 | internvl35_2b | OpenGVLab/InternVL3_5-2B | mpr-ivl | 7d7bd7bcc35b77b391883ebd7f9686cfa0e25cc0 | 4.71 GB |
| 7 | internvl35_1b | OpenGVLab/InternVL3_5-1B | mpr-ivl | 2f71cf52542334823e48a46ffba0e2bc9add3446 | 2.14 GB |

설정 파일은 앞서 만든 `sizes_uitars`의 yaml 6개(`addon/sixbysix_addon/configs/`, 내용 그대로 복사)와 MiniCPM 어댑터 패치 안의 설정을 쓴다. 저장소는 팀원 저장소 `followup-rel-4lang-inference` 브랜치의 커밋 `666440011f538cd9336794c9965429de640843c3`에 맞춘다. InternVL용 `mpr-ivl`은 패치 0001~0005(8B InternVL 실행 때와 같은 파일, sha256 동일 확인), `mpr-minicpm`은 MiniCPM 패치 1개를 얹는다. 패키지는 `requirements-qwen3-4090.txt`(torch 2.9.1+cu130, transformers 4.57.6), InternVL은 여기에 `internvl35_constraints_20261109.txt` 제약을 더한다.

## 6×6 데이터와 manifest는 어디서 오나

| 항목 | 내용 |
| --- | --- |
| 이미지 | Hugging Face 데이터셋 `chenruihan/MPR-GUI-Bench` revision `c1edb808d424a2fa7bc4a2e601d39b431b04acd5`, 파일 3,685개, 약 2.6 GB(HF API 합계 2.64 GB, 로컬 폴더 2.5 GB) |
| 질문 | GitHub `chenruihan32/MPR-GUI-Bench` 커밋 `e4f1cfcd11ee0d0dfa8ee6a0a97c2c782ce21aba`의 `qas/` (sparse clone) |
| 받는 명령 | `python -m mpr_crosslocale.data.download --source-lock data/manifests/source_lock.json --download-workers 2` (저장소 `01_code/scripts/download_mpr_gui.sh`와 같은 모듈). 큐가 대신 부른다 |
| manifest | `scripts/audit_release.sh`가 `01_code/data/manifests/mpr_gui_manifest.jsonl`을 만든다(13,470행, 28 MB, 로컬에서 약 3초) |
| 저장소가 여러 개일 때 | 데이터는 한 번만 받는다. 다른 저장소는 `01_code/data/raw/mpr_gui_bench`를 심볼릭 링크로 걸고 manifest는 복사한다 |

고유 입력 79,020개 확인: 이번에 로컬에서 위 절차로 직접 받아 manifest를 만들고 `run_cross_locale.py --dry-run`을 돌렸다. 계획은 고유 ID 79,020개, 의미 문항 2,195개이고 약 6초 걸린다. manifest의 sha256은 `4dd919fc9ddfeee6b3b807bba79191614bf7a475bafebc401329497c2a61aa59`로, 2026-11-09 InternVL3.5-8B 6×6 실행의 `full_6x6.jsonl.contract.json`에 기록된 값과 같다. 입력 ID 79,020개의 목록과 순서도 그 계약 파일과 완전히 같다(로컬 비교). 큐는 이 두 값(manifest sha, ID 목록 sha256 `22ac60aa…7b4c8`)을 고정해 두고, 어긋나면 그 모델 계열 설정에서 멈춘다(`ALLOW_MANIFEST_DIFF=1`로만 우회). 즉 8B 실행과 같은 입력으로 돈다.

주의: 로컬에서 받을 때 Hugging Face가 익명 IP를 요청 제한(429)으로 막았다(작업자 4개로 받았을 때). 작업자 2개로 줄이고 몇 분 쉰 뒤 이어받으면 됐다. 큐의 데이터 다운로드는 작업자 2개로 시도하고, 실패하면 6분 쉬고 최대 5번까지 이어받는다.

## 올릴 파일과 실행

`sixbysix/`의 세 파일을 서버 `/workspace/`에 올린다: `queue_6x6.sh`, `sixbysix_addon_20261114.tar.gz`, `SHA256SUMS`.

```bash
cd /workspace && sha256sum -c SHA256SUMS
tmux new -s q6                       # 다른 큐가 tmux에 있으면 새 창(Ctrl-b c)
bash queue_6x6.sh --plan             # 아무것도 실행하지 않고 계획만 출력
bash queue_6x6.sh --net              # 네트워크 속도만 재 본다
bash queue_6x6.sh                    # 전체 (기본 순서)
```

환경변수:

| 변수 | 뜻 |
| --- | --- |
| `MODELS="qwen3_vl_4b internvl35_4b"` | 모델 순서/부분 선택 (`ONLY`도 같은 뜻) |
| `WAIT_FOR=/workspace/MINICPM_DONE` | 그 파일이 생길 때까지 기다린 뒤 시작(`WAIT_FOR_MAX_H`, 기본 72시간). 파일이 있기만 하면 된다. 내용은 `OK`, `FAILED steps: ...`, `STOPPED rc=1` 무엇이든 상관없다. 안 주면 기다리지 않는다. 파일이 생긴 뒤에도 `run_all_overnight.sh`가 아직 돌고 있으면 그것이 끝날 때까지 더 기다린다(앞 큐가 일찍 STOP하면 DONE 파일이 먼저 생기기 때문) |
| `FORCE_SLOW_NET=1` | 네트워크 점검이 20 MB/s 미만이어도 시작 (`NET_MIN_MBPS`로 기준 변경) |
| `KEEP_MODEL_CACHE=1` | 모델 캐시를 지우지 않는다 |
| `IGNORE_ETA=1` | smoke 시간으로 계산한 전체 실행 예상 시간이 제한 시간보다 길어도 시작한다 |
| `IGNORE_OTHER_QUEUES=1` | 다른 큐 스크립트가 도는 중이어도 멈추지 않는다 |

앞선 큐(`queue_sizes_uitars.sh`, `queue_minicpm.sh`, `run_all_overnight.sh`)가 아직 돌고 있으면 `WAIT_FOR`를 안 준 경우 `pgrep`으로 감지해 STOP한다. 기다리게 하려면 `WAIT_FOR=/workspace/MINICPM_DONE bash queue_6x6.sh`. 기다림이 끝나도 GPU 메모리 사용이 1.5 GB 미만이 될 때까지 최대 30분 더 본다.

시작 순서: (1) 기다리기, 4090 확인, GPU 비었는지 (2) 네트워크 점검 (3) 묶음 sha256 확인과 풀기 (4) 모델마다 아래 단계. 네트워크 점검은 huggingface.co(Qwen3-VL-4B의 safetensors 조각)와 download.pytorch.org(torch 2.9.1+cu130 휠)에서 각각 약 15초 동안 바이트 범위 요청을 보내 MB/s를 재고 `QUEUE6_STATUS`와 `queue_results/net_check.txt`에 적는다. 둘 중 하나가 20 MB/s 미만이거나 응답이 없으면 아무것도 받지 않고 STOP한다(torch 점검은 필요한 venv가 아직 없을 때만 기준에 넣는다). 테스트 URL이 HTTP 4xx를 돌려주면 속도를 모르는 것이라 경고만 하고 계속한다. 현재 서버에서 0.6 MB/s가 나온 것이 이 점검의 이유다. 로컬 Mac에서 실제 `curl`로 한 번 돌려 본 값은 HF 9.6, PyTorch 19.7 MB/s였다(점검 코드가 동작하는지 보려는 것이고 서버 값은 아니다).

모델마다:

1. 저장소 준비(계열당 한 번): 없으면 clone, 커밋 고정, 패치, venv와 패키지, 데이터, manifest. 이미 있으면 상태를 확인해 쓴다. `mpr-ivl`은 패치 5개가 `git patch-id`로 같은지, `mpr-minicpm`은 패치 1개가 같은지 확인하고, 다르면 그 계열을 건너뛴다. 기존 venv는 torch 2.9.1+cu130, transformers 4.57.6, 저장소 `mpr_crosslocale` 경로가 맞으면 그대로 쓴다.
2. 디스크 확인 후 고정 revision 내려받기(최대 3번, 한 번에 2시간, 30초·60초 간격). 이미 받은 파일과 받던 파일은 캐시에 남아 `hf download`가 이어 받는다. 이전 서버에서 `git clone` 중 gnutls 핸드셰이크 오류, 이미지 받기 중 SSL EOF가 났기 때문에 `git clone`(5번), `git fetch`(3번), `pip install`(각 3번), 데이터 받기(5번, 6분 간격), 모델 내려받기(3번)를 모두 다시 시도하게 했다.
3. 설정의 model_id와 revision 확인, 전체 계획 `--dry-run`(79,020, ID 해시).
4. smoke 216회와 게이트.
5. 전체 79,020회. 중간에 실패한 입력은 같은 단계 안에서 한 번 더 시도한다.
6. 게이트와 요약, 결과 묶기, 모델 캐시 정리(아래 규칙).

캐시 정리 규칙: 모델 하나가 끝까지 ok이면(smoke, 전체, 게이트) 그 모델 캐시를 지운다. 내려받기 실패, 시간 초과, 게이트 실패, ETA 중단처럼 끝나지 않았으면 캐시를 남겨 같은 명령으로 다시 돌릴 때 다시 받지 않게 한다. 다만 지금 남은 디스크 여유가 `MODELS` 순서에서 뒤에 오는 미완료 모델 중 가장 큰 최소 여유(아래 디스크 절)보다 작으면 지운다. 마지막 모델이면 뒤에 필요한 것이 없으니 남긴다. `KEEP_MODEL_CACHE=1`이면 성공해도 지우지 않는다.

게이트(`check_6x6_gate.py`): 계획 ID 전부에 성공 기록이 있고(마지막 기록 기준), 계획에 없는 성공 ID가 없고, 성공 행의 batch_size가 전부 1이고, 파싱 실패가 성공 행의 50%를 넘지 않을 것. 파싱 실패 수와 비율은 통과해도 기록한다(`gate_6x6_smoke.json`, `gate_6x6_full.json`). smoke 게이트 JSON에는 추론당 중앙값 시간과 그에 따른 전체 예상 시간(`eta_hours_full_at_median`, `eta_full_seconds`)이 남고 `QUEUE6_STATUS` 줄에도 나온다. 8B InternVL의 같은 실행은 파싱 실패가 41/79,020이었다. 11월 14일에 아래 규칙을 더했다(큐가 `--model <키>`를 넘긴다).
- 파싱된 생성 라벨이 전부 같으면 실패(파싱된 행 20개 이상일 때 검사). A–D 확률 질량(`label_logprobs`의 exp 합) 중앙값이 0.5 미만이면 실패(필드가 없으면 검사하지 않고 "not evaluated"로 기록).
- 토큰 수가 모델 계열 범위를 벗어나면 실패. Qwen3-VL은 시각 토큰 2,048 이하(2,097,152 / 32²), UI-TARS는 2,674 이하(2,097,152 / 28²), InternVL3.5는 1,792~2,304(8B 6×6 결과 `internvl35_glmm.csv`의 `visual_token_count`가 정확히 2,304(60,228행)와 1,792(18,792행) 두 값이다), MiniCPM은 576~640. 어댑터가 `visual_token_count`를 기록하지 않으면 보고만 한다. Qwen3와 UI-TARS 어댑터는 이 필드를 기록하지 않는다(8B Qwen3 6×6 결과 79,020행 전부 비어 있음). 그래서 이 둘은 `prompt_token_count`로 대신 본다. Qwen3는 800~2,700(8B 6×6에서 1,712~2,515), UI-TARS는 800~3,300(상한은 2,674 + 글 부분 여유로 계산한 값이고 실측이 아니다).
- MiniCPM 범위 576~640은 REL 이미지 1,464장에서 센 값이다. 6×6은 이미지가 더 많아서 이 범위를 벗어나는 이미지가 있을 수 있다(미검증). 그 때문에 smoke가 멈추면 `GATE_VISUAL_RANGE=lo,hi`를 주고 다시 실행해 값을 확인한다.
- smoke 정확도는 기록하고 0.30 미만이면 경고만 한다(우연 수준 0.25).
- ETA: smoke 중앙값 × 79,020(이미 끝낸 만큼 제외)이 그 모델의 전체 실행 제한 시간보다 길면 전체 실행을 시작하지 않고 `QUEUE6_STATUS`에 `full 6x6 STOPPED, not started: ETA ...`와 `full=STOPPED_ETA`를 남긴다(결과는 `_PARTIAL`로 묶인다). 아래 제한 시간 표의 값이 기준이다.

진행 확인:

```bash
tail -n 20 /workspace/QUEUE6_STATUS
ls /workspace/queue_logs6/
wc -l /workspace/mpr-*/followup_rel_4lang/results/6x6_*/full_6x6.jsonl    # 줄 수 / 79,020 (실패 기록 포함)
```

끝나면 `/workspace/QUEUE6_DONE`이 생기고 `QUEUE6_STATUS` 마지막 줄에 모델별 ok/NOT_COMPLETE가 나온다. 인스턴스는 스크립트가 끄거나 지우지 않는다. 같은 명령을 다시 치면 끝난 모델은 건너뛰고(`queue_results/6x6_<키>/COMPLETE`), 끝나지 않은 모델은 이어서 돈다(캐시가 남아 있으면 다시 내려받지 않는다. 위 캐시 규칙).

## 단계별 제한 시간과 시간 추정

모델당 추론은 79,020회다. 8B에서 실측한 값은 두 가지다. InternVL3.5-8B 6×6, 배치 1: 추론당 평균 0.738초(`runtime_ms` 합계 16.2시간, 입력 구간 0.72~0.75초로 차원 간 차이가 작음). Qwen3-VL-8B 6×6, 배치 4: 추론당 약 0.6초(13.1시간, 결과 파일의 `runtime_ms` 평균 596 ms). 작은 모델과 UI-TARS, MiniCPM의 값은 측정한 적이 없고 아래는 모두 가정이다(미검증). `sizes_uitars`의 추정 범위를 이어받았다. 이미지 전처리와 약 2,000토큰 prefill 같은 고정 비용이 있어서 파라미터 수에 비례해 줄지 않는다고 보았다. MiniCPM은 시각 토큰이 약 576개로 적지만 언어 모델이 8B라 범위를 넓게 잡았다.

| 모델 | 가정 s/추론 (배치 1) | 전체 79,020회 | smoke 제한 | 전체 실행 제한 |
| --- | --- | --- | --- | --- |
| qwen3_vl_4b | 0.45~0.65 | 9.9~14.3 h | 45분 | 24시간 |
| internvl35_4b | 0.45~0.70 | 9.9~15.4 h | 45분 | 26시간 |
| uitars_15_7b | 0.70~1.00 | 15.4~22.0 h | 1시간 | 36시간 |
| minicpm_v45 | 0.30~0.60 | 6.6~13.2 h | 1시간 | 20시간 |
| qwen3_vl_2b | 0.30~0.45 | 6.6~9.9 h | 45분 | 16시간 |
| internvl35_2b | 0.30~0.50 | 6.6~11.0 h | 45분 | 18시간 |
| internvl35_1b | 0.25~0.40 | 5.5~8.8 h | 45분 | 14시간 |
| 합계 | | 60.5~94.6 h | | |

제한 시간은 가정 상한의 약 1.6~1.8배이고, 넘기면 그 단계가 끝나고(`TIMEOUT`) 부분 결과가 `_PARTIAL` tar로 묶인다. 다른 단계 제한: 저장소 준비 2시간, 내려받기 3번 × 2시간(전체 한도 6시간 10분), 게이트와 요약 1시간. 모델 내려받기와 로딩(10~30분, UI-TARS는 더 걸릴 수 있음), 새 인스턴스의 패키지 설치와 데이터 받기(첫 계열에서 30분 안팎 + HF 제한이 걸리면 6분 단위로 더 걸림)를 더하면 7개 전부 약 65~100시간, 즉 3~4일이다(미검증 추정). 시간당 요금을 몰라 비용은 계산하지 않았다(확인 필요). 전부 돌릴지는 `MODELS`로 고를 수 있고, 어느 모델을 돌릴지는 이 문서가 정하지 않는다. 시작 후에는 첫 모델 smoke의 `eta_full_h`로 실측 기반 추정을 볼 수 있다. 그 값이 위 범위를 크게 벗어나면 멈추고 다시 정하는 것이 좋다.

## 디스크

| 항목 | 크기 |
| --- | --- |
| venv (torch cu130 등) | 저장소당 약 8 GB로 가정(미검증). 새 인스턴스는 계열별 최대 3개 = 약 24 GB |
| 저장소 clone | 저장소당 약 1.5 GB (로컬 clone 크기) |
| 6×6 이미지 | 2.5~2.6 GB, 한 번만 (다른 저장소는 링크) |
| manifest | 28 MB (저장소마다 복사) |
| 모델 가중치 | 위 표. 끝까지 마친 모델은 지운다. 실패나 시간 초과 모델은 디스크가 허용하면 남긴다(위 캐시 규칙) |
| 결과 | 모델당 `full_6x6.jsonl` 약 0.3 GB (8B InternVL 실행은 308 MB), 계약 파일 4 MB. tar.gz는 줄어들 텐데 크기는 미검증(모의 실행 tar는 7 MB) |

새 인스턴스에서 가장 큰 순간은 UI-TARS를 받을 때다. 그때 venv 2개(약 16 GB) + 저장소 3~4.5 GB + 데이터 2.6 GB + UI-TARS 33.2 GB = 약 56~57 GB에 임시 파일을 더한 값이다. 키별 최소 여유 공간은 모델 파일 × 1.3 + 8 GB로 잡았다(qwen3_vl_4b 20, internvl35_4b 20, uitars 52, minicpm 32, qwen3_vl_2b 14, internvl35_2b 14, internvl35_1b 12 GB). 새 venv를 만들기 전에는 25 GB가 필요하다. 모자라면 그 모델을 건너뛰고 다음으로 넘어간다. 이 큐는 기존 8B 모델 캐시(Qwen3-VL-8B 17.55 GB, InternVL3.5-8B 17.07 GB)를 지우지 않는다. 디스크가 부족해서 건너뛰었다면 8B 결과를 내려받은 뒤 직접 지우고 다시 실행한다.

## 위험

| 위험 | 크기 | 대처 |
| --- | --- | --- |
| 시간과 비용 | 7개 전부 약 3~4일, 요금 확인 필요 | `MODELS`로 일부만. tmux에서 Ctrl-C로 멈추고 같은 명령으로 재개 |
| 네트워크가 느림 | 현재 서버 0.6 MB/s였음 | 점검이 STOP한다. 다른 인스턴스로 옮기거나 `FORCE_SLOW_NET=1`(권장하지 않음) |
| HF 요청 제한(429) | 데이터 3,685개 파일 | 작업자 2개, 6분 간격 재시도. 계속 막히면 본인 HF 토큰을 `export HF_TOKEN=...`으로 주고 다시 실행(선택) |
| 실제 모델 로드 실패 | 그 모델의 smoke에서 멈추고 다음 모델로 진행 | `queue_logs6/<키>_smoke.log`를 보내준다. 설정을 임의로 고치지 않는다 |
| 24GB OOM | 시각 토큰이 많은 모델(InternVL 2,304, UI-TARS 최대 약 2,674)은 긴 입력에서 가능. 실측하지 못함(미검증) | OOM은 코드가 그대로 중단한다(재시도 대상이 아님). 같은 입력에서 반복되면 알려주고 판단을 받는다 |
| UI-TARS가 라벨만 출력하지 않음 | GUI 에이전트용이라 파싱 실패가 많을 수 있음 | 50% 초과면 게이트가 막고 smoke 결과는 tar에 남는다 |
| 전체 실행이 제한 시간에 걸림 | 부분 결과(`_PARTIAL`) | 같은 명령으로 재개(캐시가 남아 있으면 다시 받지 않는다). 제한은 상한의 1.6~1.8배로 잡았다. smoke 시간으로 미리 계산해 안 끝날 것 같으면 시작하지 않는다(ETA 중단) |
| 게이트가 새 규칙으로 멈춤 | 모든 라벨이 같음, A–D 질량 0.5 미만, 토큰 수가 범위 밖 | `QUEUE6_STATUS`의 `PROBLEMS` 줄과 `gate_6x6_smoke.json`을 본다. 실제 모델 이상일 수도, 범위 가정이 틀렸을 수도 있다. 임의로 풀지 말고 알려준다 |
| 모델이 8B와 다른 배치/경로 | Qwen3-VL-8B 6×6은 배치 4(결과 파일 `batch_size` 전부 4로 확인), 이번 모델은 전부 배치 1 | 비교에서 배치로 인한 작은 수치 차이가 섞일 수 있다고 적는다. InternVL3.5-8B는 배치 1이라 같다. 배치 4와 1의 차이는 `sizes_uitars` 큐의 "8B 배치 1 점검"이 일부 입력으로 따로 잰다 |
| 기존 저장소 상태가 예상과 다름 | `mpr-ivl`/`mpr-minicpm`의 커밋이 패치와 다르면 그 계열 건너뜀 | `queue_logs6/setup_<계열>.log`의 이유를 보고 알려준다. 저장소를 지우지 않는다 |
| 다른 큐와 겹침 | GPU 동시 사용 = OOM | `WAIT_FOR`, `pgrep` 점검, GPU 사용량 점검 |
| MiniCPM 6×6 경로 | 어댑터는 REL 실행기에서 검증됨. 6×6 실행기(`run_cross_locale.py`)의 `minicpmv` 분기는 임의 가중치 소형 모델로만 돌려 봄 | smoke에서 확인됨 |
| 모델 라이선스 | MiniCPM 등 사용 조건 | 확인 필요 (읽지 않았다) |

## 로컬에서 한 검사

환경: Mac(arm64), `local_tests/` 아래 스크립트, 서버와 같은 torch 2.9.1/transformers 4.57.6의 CPU 환경(가짜 가중치/모의 모델).

- `bash -n queue_6x6.sh build_addon.sh local_tests/run_6x6_tests.sh` 통과, `bash queue_6x6.sh --plan` 출력 확인.
- 6개 yaml 설정과 MiniCPM 설정으로 `run_cross_locale.py --dry-run`(전체 계획 79,020, ID 해시 일치)을 새로 clone한 저장소에서 확인. `--mock-model`로 smoke 216과 전체 79,020을 돌려 게이트와 요약이 통과하는 것까지 확인(모델마다 약 20초).
- `python local_tests/test_gate.py ...` (결과 `local_tests/gate_unit_result.txt`): 정상, 재개 후 성공, 파싱 실패 10%와 정확히 50% 통과, 61%·실패 행 1개·행 누락·배치 3·계획에 없는 ID·계획 크기 불일치·전체 79,020 중 10개 누락은 실패 판정. 잘린 마지막 줄이 있어도 읽는다.
- 실제 MiniCPM 어댑터에 임의 가중치 소형 모델(이전 작업의 `mc/tiny`)을 붙여 `run_cross_locale.py`로 REL 2입력을 CPU에서 돌려 행이 성공으로 기록되는 것까지 확인했다. 임의 가중치라 출력이 라벨이 아니어서 게이트가 파싱 실패 100%로 실패 판정을 냈다(의도한 동작).
- 가짜 `/workspace` 흐름 시험(`bash local_tests/run_6x6_tests.sh <fresh|existing|net|wait|fail>`):
  - `fresh`: 빈 작업 공간에서 3개 계열 모두 clone, 커밋 고정, `git am`(InternVL 5개, MiniCPM 1개), 설정 복사, 데이터(다운로드 모듈은 로컬에서 받아 둔 데이터를 복사하는 가짜), 실제 `audit_release.sh`로 manifest 생성(sha 일치), 나머지 저장소는 링크와 복사. 3개 모델 완료.
  - `existing`: 기존 저장소 3개가 있는 상태(`mpr-ivl`은 패치 5개 커밋, `mpr-minicpm`은 패치 1개 커밋, `mpr-gui-crosslocale`은 데이터와 manifest 보유)에서 패치 확인(patch-id)과 재사용으로 7개 모델 전부 완료, `QUEUE6_DONE` 생성.
  - `net`: 속도 0.4 MB/s에서 STOP(QUEUE6_DONE 없음), `FORCE_SLOW_NET=1`이면 진행, PyTorch 쪽만 느려도 STOP.
  - `wait`: `WAIT_FOR` 파일이 생길 때까지 기다린 뒤 시작.
  - `fail`: 다운로드 실패 모델 건너뛰기, 전체 단계 강제 TIMEOUT 후 `_PARTIAL` tar(qwen3_vl_2b는 38,481 → 43,454 → 79,020으로 세 번 나눠 재개해 완료, qwen3_vl_4b는 27,948 → 79,020), 디스크 부족 건너뛰기, 이미 끝난 모델 건너뛰기, 재개 뒤 `COMPLETE`.
- 한 번은 로컬 Mac에서 진짜 `curl`로 `--net`을 돌려 점검 코드가 동작하는 것을 확인했다.
- 11월 14일 수정 뒤 다시 한 검사:
  - `python3 local_tests/test_gate.py <check_6x6_gate.py> <임시폴더>`: 기존 12개에 새 규칙 19개(라벨 전부 같음, A–D 질량, 계열별 토큰 범위 경계, 필드 없음 보고, 정확도 경고, ETA 계산, `GATE_VISUAL_RANGE`)를 더해 31개 모두 통과(`gate_unit_result.txt`).
  - 새 흐름 시험 `bash local_tests/run_6x6_tests.sh <stopnet|retry|cache|eta|waitany>`(PASS/FAIL 표시, 결과 `queue6_new_tests_result.txt`): 네트워크 느림 STOP과 4xx 경고, 내려받기 재시도와 코드의 재시도 줄 확인, 시간 초과·내려받기 실패 때 캐시 유지와 재실행 때 재전송 없음, 성공 후 삭제, 디스크 부족 때 삭제, ETA 중단과 `IGNORE_ETA=1`, `WAIT_FOR` 내용이 `FAILED steps: ...`·`STOPPED rc=1`이어도 시작하고 `run_all_overnight.sh`가 도는 동안 더 기다림. 30개 모두 통과.
  - 기존 `existing`(7개 모델 모두 완료), `fresh`(3개 계열 새로 설치 후 3개 완료), `net`, `wait`, `fail`도 수정판으로 다시 돌렸다(`fail`에서 실패 모델의 캐시가 이제 남는다).
  - 가짜 `timeout`이 GNU처럼 두 단위 값을 거부하게 고쳤다.
- 묶음 sha256(11월 14일 수정판): `sixbysix_addon_20261114.tar.gz` = `ab2585e1fd2d3b0cf81f9b203d9833634032b16335638f3ce5e023a7d050187a`(`queue_6x6.sh`의 `ADDON_SHA`와 같다), `queue_6x6.sh`는 `SHA256SUMS`에 있다. 이전 판은 `fb182895…2157`이고 `superseded/`에 있다.

## 검증하지 못한 것 (미검증)

- 실제 모델 가중치 로드, GPU 메모리 최대치, 실제 속도와 출력 품질(파싱 실패율). UI-TARS와 MiniCPM이 6×6 프롬프트에서 라벨을 내는지.
- 서버에서 `pip install -r requirements-qwen3-4090.txt`(+InternVL 제약)가 새 venv에 그대로 설치되는지, `python3.12-venv` 설치 경로(`apt`), 서버의 GNU `timeout`/`flock`/`df -BG` 동작. 로컬은 가짜 명령으로 시험했고 venv는 가짜(Q6_TEST) 대체를 썼다.
- 새 인스턴스에서 GitHub clone과 `hf download`의 실제 속도, HF 요청 제한이 서버 IP에도 걸리는지.
- 시간 추정(위 표의 가정), 결과 tar 크기, venv 크기.
- 다운로드 스크립트는 로컬에서 실제로 돌려 데이터를 받았지만, 서버에서 같은 revision이 그대로인지는 확인하지 않았다(revision 고정으로 같아야 함).
- 결과 해석은 하지 않았다.

## 받을 결과

`/workspace/queue_results/6x6_<키>/6x6_<키>_YYYYMMDD_HHMM.tar.gz`와 `SHA256SUMS`(`_PARTIAL`이 붙으면 일부 단계 실패), `queue_results/QUEUE6_STATUS.txt`, `net_check.txt`, `queue_logs6/`. tar 안: `smoke_6x6.jsonl`, `full_6x6.jsonl`(+계약, 계획), `gate_6x6_*.json`, `full_6x6_summary.json/csv`, 로그, `package_meta/`(설정, git 커밋, pip freeze, manifest sha, nvidia-smi, 디스크, 묶음 sha). 다 받은 뒤 인스턴스 정리는 사용자가 정한다(Stop은 디스크 요금이 계속 나가므로 Destroy).
