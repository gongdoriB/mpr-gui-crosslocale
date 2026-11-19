# REL O/R/C/F 6개 언어 확장 (2026-11-15)

`followup_rel_4lang`(en, zh, th, ru)에 fr, ja를 더한 입력 묶음. 4개 언어 폴더는 그대로 두고 코드를 복사해 고쳤다.

## 무엇이 같은가

`data/rel_all_366/conditions_6lang.jsonl`에서 질문·화면 언어가 모두 en/zh/th/ru인 행 23,424개는 생성 쪽 `../followup_rel_4lang/data/rel_all_366/conditions_4lang.jsonl`(sha256 aa403674…)과 바이트까지 같다(2026-11-15 대조). 서버 실행에 쓴 파일(sha256 e5aa6cd0…)과는 생성 쪽 파일에만 있는 `object_extraction`(이름 추출 점검 기록) 필드 하나가 다르고, 질문·보기·이미지 경로·정답·행 순서는 같다(2026-11-16 정정, 실행기 모의 대조에서 4개 언어 부분 프롬프트 18,432개 동일). 의존 문항 파일(`data/variants/conditions_6lang.jsonl`)의 4개 언어 부분 19,328행도 같다. 따라서 이미 받은 4개 언어 결과는 새로 돌릴 필요가 없고, 새로 돌릴 행만 따로 뽑으면 된다.

## 규모

| | 4개 언어 | 6개 언어 | 새로 돌릴 행 |
|---|---|---|---|
| 의존 문항 302 | 19,328 | 43,356 | 24,028 |
| 독립 문항 64 | 4,096 | 9,216 | 5,120 |
| 합계 | 23,424 | 52,572 | 29,148 |

- 의존 문항 3개(원래 인덱스 117, 172, 244)는 일본어 질문에 두 번째 이름(Display & Brightness, Set reminder, Headphone Audio Levels)이 없다. 이 세 문항은 일본어가 들어간 쌍을 만들지 않았다(문항당 11쌍 × 4조건 = 132행 제외). `generation_report.json`의 `excluded_item_locales`에 기록.
- 6개 언어 모두 원문 정답 위치가 같다(`cross_locale_gold_disagreements` 0).

## fr/ja 이름 위치를 정한 방법

- 따옴표로 둘러싼 이름은 기존과 같은 규칙으로 자동 추출.
- 따옴표 없는 보기의 꾸밈말("Bouton", "ボタン" 등)을 떼는 규칙: `frja_wrap_rules.py`.
- 기존 4개 언어에서 손으로 정한 칸(의존 83, 독립 31)과, fr/ja의 따옴표 개수가 영어와 다른 칸(29)은 `frja_alignment.json`에 영어 이름 순서대로 fr/ja 표현을 적었다. 이 143칸의 초안은 언어 모델(Claude Sonnet)로 만들었고, 모든 표현이 원문에 그대로 있는지와 개수가 맞는지만 기계로 검사했다. 사람이 읽고 확인하지는 않았다(미검증).
- 규칙과 겹치는 예외 3칸(Baignoire/バスタブ, Taille du texte, 17jours)은 손으로 고정했다.

## fr/ja에서 새로 표시한 원문 대응 주의

기존 `SOURCE_ISSUES`/`ISSUES`는 바꾸지 않고, 새 목록을 따로 두었다(`FRJA_SOURCE_ISSUES`, `FRJA_ISSUES`, `data/variants/source_issues_frja.json`). 해당 행은 `eligible_without_source_adjudication=false`.

| 문항 | 언어 | 내용 |
|---|---|---|
| 의존 87 | ja | 보기 D가 再生停止中(재생 정지 중), Do Not Disturb가 아님 |
| 의존 99 | fr, ja | Home/Transit 보기가 번역 대응이 아님(ja 交通状況 = 교통 상황) |
| 의존 221 | fr | 보기 A가 Actualiser(새로고침) |
| 독립 22 | ja | 보기 D가 始める(시작), Plant가 아님 |
| 독립 27 | fr, ja | Close all 자리에 ×만 있음 |
| 독립 56 | fr, ja | 숫자가 다름(EN 72, FR 77, JA 74) |

## 원본과 검증

- `sources/qas/rel_el_fr.jsonl`, `rel_el_ja.jsonl`: MPR-GUI-Bench GitHub 커밋 e4f1cfcd에서 받음. sha256이 `mpr-gui-crosslocale-rel-annotation-ui/01_code/docs/RELEASE_AUDIT.md`의 값과 일치(29e532a39eb4…, 77591f6f3829…). en 파일도 로컬 사본과 같은 해시.
- `assets/images/*/fr`, `*/ja`: HF 리비전 c1edb808에서 732장(751MB). `data/assets_required_frja.json`.
- 확인 예시: booking_10의 ja 이름 「よく使われているその他の通貨」는 영어 "Suggested for you"와 뜻이 다르지만, 일본어 화면의 같은 자리에 실제로 그 글자가 있다(스크린숏 확인). R 조건이 화면 글자를 가져오는 설계대로다.

## 아직 안 한 것

- fr/ja 이름 정렬의 사람 검수: 하지 않기로 했다. 검수할 사람이 없어 4개 언어 때와 같이 기계 검사만으로 진행한다(사용자 결정, 2026-11-16, Obsidian `08 사람 검수 없이 진행하는 결정과 한계`). 논문에는 미검수로 적는다.
- 서버용 "새 행만" 조건 파일과 실행 스크립트 연결(서버 실행은 승인 전).
- 6개 언어용 분석 규칙 부록(결과 보기 전에 고정).

범위: 6개 언어 확장은 REL O/R/C/F와 실험 A만 한다. grounding은 진행하지 않기로 했다(사용자 확인, 2026-11-15).
