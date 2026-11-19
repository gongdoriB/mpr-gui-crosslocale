import pandas as pd, json
O="${PROJECT_ROOT}/analysis_plan_followup/"
def p(x):
    x=float(x); return "<2e-4" if x==0 else (f"{x:.2e}" if x<1e-3 else f"{x:.4f}")
def pg(x):
    x=float(x); return f"{x:.2e}" if x<1e-3 else f"{x:.4f}"
out=[]
h=pd.read_csv(O+"task1/holm_ORCF.csv")
out.append("T1_dep|"+"\n".join(["| 모델 | 대비 | 부트스트랩 %p [95% CI] | 부트 p(raw) | 부트 p(Holm) | GLMM OR [95% CI] | Wald p(raw) | Wald p(Holm) |","|---|---|---|---|---|---|---|---|"]+[f"| {r.model} | {r.contrast} | {r.boot_est_pp:+.2f} [{r.boot_lo:+.2f}, {r.boot_hi:+.2f}] | {p(r.boot_p_raw)} | {('<8e-4' if r.boot_p_holm==0 else p(r.boot_p_holm))} | {r.glmm_OR:.2f} [{r.glmm_lo:.2f}, {r.glmm_hi:.2f}] | {pg(r.glmm_p_raw)} | {pg(r.glmm_p_holm)} |" for r in h[h.dependency=="dependent"].itertuples()]))
out.append("T1_ind|"+"\n".join(["| 모델 | 대비 | 부트스트랩 %p [95% CI] | 부트 p(raw) | GLMM OR [95% CI] | Wald p(raw) |","|---|---|---|---|---|---|"]+[f"| {r.model} | {r.contrast} | {r.boot_est_pp:+.2f} [{r.boot_lo:+.2f}, {r.boot_hi:+.2f}] | {p(r.boot_p_raw)} | {r.glmm_OR:.2f} [{r.glmm_lo:.2f}, {r.glmm_hi:.2f}] | {pg(r.glmm_p_raw)} |" for r in h[h.dependency=="independent"].itertuples()]))
d=pd.read_csv(O+"task2/direction_lrt.csv")
nm={"RA_only":"reference_alignment:%s","CL_only":"context_localization:%s","joint":"두 항 동시(6 df)"}
out.append("T2|"+"\n".join(["| 모델 | 교호작용 대상 | 항 | χ² | df | p | 특이 적합(기본/확장) | 수렴 메시지 |","|---|---|---|---|---|---|---|---|"]+[f"| {r.model} | {r.factor} | {nm[r.term]%r.factor if '%' in nm[r.term] else nm[r.term]} | {r.chisq:.2f} | {int(r.df)} | {pg(r.p)} | {r.singular_base}/{r.singular_alt} | {'없음' if (pd.isna(r.conv_msg_base) and pd.isna(r.conv_msg_alt)) else str(r.conv_msg_base)+'/'+str(r.conv_msg_alt)} |" for r in d.itertuples()]))
a=pd.read_csv(O+"task3/grounding_by_area.csv"); ti=pd.read_csv(O+"task3/area_tertile_info.csv")
out.append("T3info|"+"\n".join(["| 삼분위 | 문항 | 면적비 범위 | 중앙값 | hit O | R | C | F | 유효율 O/R/C/F |","|---|---|---|---|---|---|---|---|---|"]+[f"| T{int(r.tertile)} | {r.n_items} | {r.area_min:.4f}–{r.area_max:.4f} | {r.area_median:.4f} | {r.hit_O:.3f} | {r.hit_R:.3f} | {r.hit_C:.3f} | {r.hit_F:.3f} | {r.valid_O:.3f}/{r.valid_R:.3f}/{r.valid_C:.3f}/{r.valid_F:.3f} |" for r in ti.itertuples()]))
def t3(df):
    return "\n".join(["| 삼분위 | 대비 | 문항 | 추정 %p | 95% CI |","|---|---|---|---|---|"]+[f"| {r.tertile} | {r.contrast} | {r.items} | {r.estimate_pp:+.2f} | {r.ci_low:+.2f} ~ {r.ci_high:+.2f} |" for r in df.itertuples()])
out.append("T3a|"+t3(a[a.definition=="item_mean_area"])); out.append("T3b|"+t3(a[a.definition=="item_x_gui_cell_area"]))
j=json.load(open(O+"task4/task_link_reliability.json")); 
out.append("T4|"+"\n".join(["| 항목 | grounding R−O | 객관식 R−O |","|---|---|---|",
 f"| 문항 점수 평균(%p) | {j['grounding_item_score']['mean_pp']:.2f} | {j['mc_item_score']['mean_pp']:.2f} |",
 f"| 문항 점수 SD(0~1 척도) | {j['grounding_item_score']['sd']:.3f} | {j['mc_item_score']['sd']:.3f} |",
 f"| 점수가 정확히 0인 문항 비율 | {j['grounding_item_score']['share_items_zero']:.3f} | {j['mc_item_score']['share_items_zero']:.3f} |",
 f"| 반분 상관 r (2000회 평균, 범위) | {j['grounding_r_half']['mean']:.3f} ({j['grounding_r_half']['min']:.3f}–{j['grounding_r_half']['max']:.3f}) | {j['mc_r_half']['mean']:.3f} ({j['mc_r_half']['min']:.3f}–{j['mc_r_half']['max']:.3f}) |",
 f"| Spearman-Brown 신뢰도 (평균, 범위, 2.5–97.5%) | {j['grounding_SB']['mean']:.3f} ({j['grounding_SB']['min']:.3f}–{j['grounding_SB']['max']:.3f}; {j['grounding_SB']['p2_5']:.3f}–{j['grounding_SB']['p97_5']:.3f}) | {j['mc_SB']['mean']:.3f} ({j['mc_SB']['min']:.3f}–{j['mc_SB']['max']:.3f}; {j['mc_SB']['p2_5']:.3f}–{j['mc_SB']['p97_5']:.3f}) |"]))
open(O+"task_tables_for_report.txt","w").write("\n@@\n".join(out)); print("ok")
