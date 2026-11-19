import json, pandas as pd, os
O = "${PROJECT_ROOT}/analysis_plan_followup/"
Q = "${TMPDIR}/work"
T = dict(x.split("|", 1) for x in open(O + "task_tables_for_report.txt").read().split("\n@@\n"))
j = json.load(open(O + "task4/task_link_reliability.json"))
t5 = json.load(open(O + "task5/review_counts_and_sensitivity.json"))
a = json.load(open(O + "task3/area_summary.json"))
s6 = pd.read_csv(O + "task6/sign_consistency_long.csv")
sm = pd.read_csv(O + "task6/sign_consistency_summary.csv")
rd = pd.read_csv(O + "task6/rel_dependency_bonus_diff_item_bootstrap.csv")

def pg(x):
    x = float(x)
    return f"{x:.2e}" if x < 1e-3 else f"{x:.4f}"

rows = []
for q in sm.quantity:
    d = s6[s6.quantity == q]
    cells = []
    for m in ["qwen3", "qwen", "internvl"]:
        r = d[d.model == m]
        cells.append(f"{r.iloc[0].estimate:+.2f} [{r.iloc[0].ci_low:+.2f}, {r.iloc[0].ci_high:+.2f}]" if len(r) else "해당없음")
    s = sm[sm.quantity == q].iloc[0]
    rows.append(f"| {q} | " + " | ".join(cells) + f" | {'일치' if s.all_same_sign else '불일치'} ({s.signs}) | {'Qwen3만 존재' if q.startswith('grounding') else ''} |")
T6 = "\n".join(["| 지표 (단위 %p) | Qwen3-VL-8B | Qwen2.5-VL-7B | InternVL2.5-8B | 추정치 부호 (Qwen3/Qwen2.5/InternVL) | 비고 |", "|---|---|---|---|---|---|"] + rows)

# task 7 tables
D7a = O + "task7/rq1_scoring/"
lrt = pd.read_csv(D7a + "rq1_matched_lrt_scoring_correct.csv").iloc[1]
bon = pd.read_csv(D7a + "rq1_adjusted_concordance_bonus_scoring_correct.csv").iloc[0]
fe = pd.read_csv(D7a + "rq1_fixed_effects_scoring_correct.csv"); fe = fe[fe.term == "matched_num"].iloc[0]
dg = pd.read_csv(D7a + "rq1_diagnostics_scoring_correct.csv")
lrt_g = pd.read_csv(Q + "rq1_generation/rq1_matched_lrt_generation_correct.csv").iloc[1]
bon_g = pd.read_csv(Q + "rq1_generation/rq1_adjusted_concordance_bonus_generation_correct.csv").iloc[0]
fe_g = pd.read_csv(Q + "rq1_generation/rq1_fixed_effects_generation_correct.csv"); fe_g = fe_g[fe_g.term == "matched_num"].iloc[0]
T7a = "\n".join(["| 결과변수 | matched OR [95% CI] | LRT χ²(1) | p | 보정 일치 보너스 %p [95% CI] | n행 | 특이 적합 |", "|---|---|---|---|---|---|---|",
    f"| scoring_correct (신규) | {fe.odds_ratio:.3f} [{fe.odds_ratio_low:.3f}, {fe.odds_ratio_high:.3f}] | {lrt.Chisq:.2f} | {pg(lrt['Pr(>Chisq)'])} | {100*bon.estimate:+.2f} [{100*bon['conf.low']:+.2f}, {100*bon['conf.high']:+.2f}] | 79,020 | {dg.singular.tolist()} |",
    f"| generation_correct (기존, 비교용) | {fe_g.odds_ratio:.3f} [{fe_g.odds_ratio_low:.3f}, {fe_g.odds_ratio_high:.3f}] | {lrt_g.Chisq:.2f} | {pg(lrt_g['Pr(>Chisq)'])} | {100*bon_g.estimate:+.2f} [{100*bon_g['conf.low']:+.2f}, {100*bon_g['conf.high']:+.2f}] | 79,020 | - |"])

D7b = O + "task7/core5_all_fast/"
T7b = "(Task 7b 결과 파일 없음)"
if os.path.exists(D7b + "core5_rq2_matched_dimension_lrt_generation_correct.csv"):
    l = pd.read_csv(D7b + "core5_rq2_matched_dimension_lrt_generation_correct.csv")
    b = pd.read_csv(D7b + "core5_rq2_concordance_bonus_by_dimension_generation_correct.csv")
    dgn = pd.read_csv(D7b + "core5_diagnostics_generation_correct.csv")
    T7b_l = l.to_markdown(index=False) if hasattr(l, "to_markdown") else l.to_string()
    T7b = "LRT 표 (원본 열 그대로):\n\n" + l.to_csv(index=False) + "\n보너스 (원본 열 그대로):\n\n" + b.to_csv(index=False) + "\n진단:\n\n" + dgn.to_csv(index=False)
open(O + "task7_tables_raw.txt", "w").write(T7b)

tl = j["ceiling"]
R = open(O + "report_template.md").read()
for k, v in dict(T1_dep=T["T1_dep"], T1_ind=T["T1_ind"], T2=T["T2"], T3info=T["T3info"], T3a=T["T3a"], T3b=T["T3b"], T4=T["T4"], T6=T6, T7a=T7a).items():
    R = R.replace("{{" + k + "}}", v)
open(O + "REPORT_KO.md", "w").write(R)
print("written")
