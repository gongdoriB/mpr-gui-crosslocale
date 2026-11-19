"""Builds app_cluster_KO.md from the CSVs (numbers only)."""
import csv
from pathlib import Path
D = Path("${PROJECT_ROOT}/night_runs/20261115_0106")
A = list(csv.DictReader(open(D / "app_cluster_expA.csv")))
A2 = list(csv.DictReader(open(D / "app_cluster_expA_seed20261114.csv")))
R = list(csv.DictReader(open(D / "app_cluster_rel.csv")))


def tab(rows):
    L = ["| 모델 | 범위 | 쌍/문항/앱 | 대비 | 추정 | 문항 CI | 앱 CI | 폭비 | 0 판정 변화 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda a, b: "[%+.2f, %+.2f]" % (float(a), float(b))
        L.append("| %s | %s | %s/%s/%s | %s | %+.2f | %s | %s | %.2f | %s |" % (
            r["model"], r["scope"], r["pairs"], r["items"], r["apps"], r["contrast"], float(r["estimate"]),
            f(r["item_ci_lo"], r["item_ci_hi"]), f(r["app_ci_lo"], r["app_ci_hi"]), float(r["width_ratio"]),
            "**변화**" if r["zero_call_changed"] == "True" else "-"))
    return L


def chg(rows, name):
    c = [r for r in rows if r["zero_call_changed"] == "True"]
    L = ["**%s**: %d행 중 %d행에서 0 포함 여부가 달라짐." % (name, len(rows), len(c))]
    for r in c:
        L.append("- %s / %s / %s: 문항 CI %s, 앱 CI %s" % (r["model"], r["scope"], r["contrast"],
                 "0 포함" if r["item_zero"] == "True" else "0 제외", "0 포함" if r["app_zero"] == "True" else "0 제외"))
    return L


wr = lambda rows: (min(float(r["width_ratio"]) for r in rows), max(float(r["width_ratio"]) for r in rows))
L = ["# 앱 군집 부트스트랩 확장 (탐색, 사후 분석)", "",
     "코드: `night_runs/20261115_0106/app_cluster_run.py` (표는 `make_md.py`). 실험 A는 analyze_expA.py의 item_matrix, bootstrap, contrasts를, REL은 run_frozen_analysis.py의 rel_quartets, rel_boot를 import해 썼고 원본 파일은 수정하지 않았다.",
     "앱 식별: parallel_id에 정규식 `^rel::\\d+/(.+)_\\d+\\.[A-Za-z]+$`의 첫 묶음(README 3절). 대소문자 유지, `setting`/`settings` 별개.",
     "실험 A: seed 20261113, B=10000, 문항 점수는 문항 안 쌍 평균, 앱 군집은 앱 복원추출 후 문항 평균(task1_task3_expA.py cluster_boot 방식). REL: seed 42, B=10000, 의존 문항 층, 앱 복원추출 후 쌍 합산 평균(rel_boot 방식).",
     "재현 확인: Qwen3-8B 실험 A 전체 4화면을 seed 20261114로 돌려 기존 task3_cluster_bootstrap_expA_qwen3.csv와 6개 구간 모두 일치(repro_check.txt). 원 README 3절의 실제 seed는 20261114였고, 요청대로 본 표는 20261113이다. 20261114 결과는 `app_cluster_expA_seed20261114.csv`에 따로 있다.",
     "문항 단위 CI는 각 모델 results.json 값과 소수 첫째 자리까지 일치(실험 A, REL 모두 불일치 0; rel_item_check.txt). 단위는 %p, 폭비 = 앱 CI 폭 / 문항 CI 폭.",
     "앱이 독립 군집이라는 가정은 확인하지 않았다. 앱 수가 적어(범위에 따라 약 20-26) percentile 구간이 불안정할 수 있다. 표본은 선택된 158문항(REL 의존 문항 301개)이다.", ""]
L += chg(A, "실험 A (seed 20261113)") + [""] + chg(A2, "실험 A (seed 20261114, 참고)") + [""] + chg(R, "REL (seed 42)") + [""]
a, b = wr(A); c, d = wr(R)
L += ["폭비 범위: 실험 A %.2f-%.2f, REL %.2f-%.2f." % (a, b, c, d), "", "## 실험 A 전체 표 (쌍/문항/앱)"] + tab(A) + ["", "## REL 전체 표 (의존 문항 층; 쌍=quartet 수)"] + tab(R)
(D / "app_cluster_KO.md").write_text("\n".join(L) + "\n")
print("\n".join(L[7:20]))
