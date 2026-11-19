# 실험 A(O/T/R) 6개 언어 확장 입력 (2026-11-15 ~ 11-16)

`../expA_translation_20261112`(en, zh, th, ru)와 같은 방법으로, fr 또는 ja가 들어간 새 방향쌍만 만들었다. 기존 폴더의 코드는 복사해서 고쳤고 원본은 바꾸지 않았다.

## 규모

- 문항: 기존과 같은 170개(본 분석 158 + 예비 12).
- 새 방향쌍 3,060개(질문 언어 ≠ 화면 언어, fr 또는 ja 포함): 본 분석 2,844(158 × 18), 예비 216.
- 이름 자리 14,004곳(본 분석 12,852), 번역 묶음 3,060개.
- 모델 하나당 새로 돌릴 행: O·T·R 각 3,060 = 9,180행.

## 만든 순서

1. `make_inputs.py`: 새 쌍 목록(`work/pairs_new.jsonl`)과 이름 목록(`work/span_inventory.jsonl`). 이름 위치는 `../followup_rel_6lang`의 `reference_mappings.jsonl`에서 가져온다.
2. `translate_llm.py`: Gemma 3 12B(mlx 4bit, 리비전 86cc6a8d, greedy)로 화면 없이 번역. 지시문은 기존과 같고 언어 이름에 French, Japanese만 더했다. 결과 `work/translations_gemma.jsonl`(sha256 5e8ec43c…), 형식 깨짐 0건이라 parsefail 재시도는 없다.
3. `build_T_6lang.py`: T 조건 생성. R과 같은 자리에 화면 글자 대신 번역을 넣는다. 쓰기 전에 같은 코드로 화면 글자를 넣어 저장된 R 질문을 글자 그대로 재현하는지 모든 쌍에서 확인했다(3,060쌍 통과). 같은 쌍의 O·R 행도 `work/OR_gemma_conditions.jsonl`로 함께 쓴다.
4. `similarity_labse.py`: 번역과 화면 이름의 LaBSE 유사도(리비전 836121a0, 기존과 같음). `work/T_gemma_similarity.jsonl`.
5. `retry_untranslated.py --mode untranslated`: 문자 체계가 다른데 번역이 원문과 같게 나온 이름을 한 번 더 번역. 민감도 분석용 T_retry에만 쓴다. 이 단계는 `work/T_gemma_first_spans.jsonl`을 읽는데, parsefail 재시도가 없어서 `T_gemma_spans.jsonl`을 그대로 복사한 파일이다.

재시도 결과(2026-11-16): 262개 이름 모두 형식 정상, 그중 68개는 다시 해도 원문 그대로였다. 재시도 번역을 넣어 `--revert-collisions`로 만든 `work/T_gemma_retry_conditions.jsonl`(3,060쌍, R 재현 통과, 되돌린 충돌 0건)에서 "번역 안 됨" 판정은 본 분석 303곳 → 119곳. O·R 행은 T_gemma와 같은 파일이 다시 써졌다(`OR_gemma_retry_conditions.jsonl`, 바이트 동일).

해시: `SHA256SUMS`(입력·번역·조건·유사도 파일).

## 기존 코드에서 바꾼 점

- 일본어 화면은 가나와 한자가 섞이므로, "번역 안 됨" 판정에서 CJK·히라가나·가타카나를 모두 일본어 문자로 본다(`same_script`). fr은 en과 같은 라틴 문자라서 en↔fr에서는 이 판정이 생기지 않는다.
- 그림 경로: 화면이 fr/ja면 `followup_rel_6lang/assets/...`, 그 밖이면 `followup_rel_4lang/assets/...`. 서버에는 두 폴더의 그림이 모두 있어야 한다. 그림 sha256은 이 컴퓨터의 파일로 새로 계산했다.
- 행마다 `source_issue`, `eligible_without_source_adjudication`을 붙였다(fr/ja 원문 대응 주의 문항 표시).

## 결과 요약 (본 분석 2,844쌍, 이름 자리 12,852곳)

| 번역 vs 화면 이름 | 자리 수 |
|---|---|
| 글자 그대로 같음 | 4,819 |
| 대소문자·따옴표 등만 다름 | 392 |
| 다름 | 7,641 |

- 기존 4개 언어(1,944쌍, 8,568곳)는 같음 3,012 / 정규화 같음 311 / 다름 5,245였다.
- 번역 후 보기 A–D가 서로 겹치게 된 쌍 4개(원래는 모두 달랐음): calendar_6 fr→zh, clock_10 fr→en, clock_9 th→ja, phone_5 fr→ru. 분석 규칙에서 처리 방법을 정해야 한다(확인 필요). → 2026-11-16 정정: main은 3쌍뿐이다(clock_10 fr→en은 예비). clock_9는 `rel::3/clock_9.jpg`다. 처리는 `paper_plan_20261112/분석규칙_v1_부록2` §4를 따른다.
- "번역 안 됨" 판정 303곳은 대부분 WeChat, DND, Goodnotes 같은 상표·약어나 날짜다. 기존 4개 언어(27곳)보다 많은 것은 일본어 화면이 라틴 문자 상표를 그대로 쓰는 경우가 많기 때문으로 보인다(해석, 미검증).
- 뜻 보존 부분집합: 모든 자리가 같음/정규화 같음이거나 LaBSE 0.7 이상인 본 분석 쌍은 1,245개로 셌다. 이는 기존 708쌍 정의를 내가 다시 구현한 개수이고, 고정된 분석 코드로 다시 세어 대조해야 한다(미검증). → 2026-11-16 폐기: 공식 정의(다름 이름이 하나 이상 있어야 함)로 다시 세면 1,040쌍이다. 1,245는 이름이 모두 같은 205쌍을 잘못 넣은 값이다. 부록2 §5 참조.

## 아직 안 한 것

- NLLB 비교 번역(기존에는 번역기 일치도 참고용). 모델이 이 컴퓨터에 받아져 있는지 확인 필요.
- 6개 언어 분석 규칙 부록(결과 보기 전 고정).
- 서버 실행용 묶음과 sha 목록(서버 실행은 승인 전).
