import csv, json, math, os
R="${TMPDIR}/work"; O="${TMPDIR}/work"
A=R+"/01_code/results/analysis"
def rd(p): return list(csv.DictReader(open(p)))
def f(x,d=3): return f"{float(x):.{d}f}"
def pp(x): return f"{100*float(x):+.2f}"
def pv(x):
    x=float(x); return f"{x:.2e}" if x<1e-3 else f"{x:.4f}"
M={"qwen3":O+"/results/qwen3","qwen2.5":A+"/qwen","internvl2.5":A+"/internvl"}
def d(m,sub,pat):
    base=M[m]; 
    sub2={"rq1":"rq1_generation","rq2":"rq2_generation_fast","slope":"random_slope_generation_fast","core5":"core5_rq1_generation","td":"rel_text_dependency" if m=="qwen3" else "rel_text_dependency_adjudicated_v1"}[sub]
    return rd(f"{base}/{sub2}/{pat}_generation_correct.csv")
L=[]
def table(h,rows):
    L.append("| "+" | ".join(h)+" |"); L.append("|"+"---|"*len(h))
    for r in rows: L.append("| "+" | ".join(str(c) for c in r)+" |")
    L.append("")
L.append("# Qwen3-VL-8B 분석 수치 보고서 (생성 정답률 generation_correct)\n")
L.append("작성: 2026-11-08. 모든 수치는 기존 스크립트·명세로 계산한 값이며 해석은 포함하지 않는다. 경로 기준: `analysis_qwen3/`.\n")
L.append("## 1. 실행 내역\n")
L.append("- 환경: conda `mpr-r-analysis` (R 4.4.3, lme4 2.0.6, emmeans). Python 부분(부트스트랩·grounding)은 기본 `python3`(numpy 2.3.5, scipy 1.16.3) 사용 — 환경에는 scipy가 없었음.")
L.append("- SHA-256 검증(`sha_check.txt`): full_6x6 79,020행, full_rel results 23,424행, predictions 18,432행, grounding predictions 7,584행 — 4개 모두 MANIFEST와 일치.")
L.append("- 6x6 GLMM 표: `derived/qwen3_glmm.csv` 79,020행, 2,195 문항, 파싱 실패 24건, 채점 불일치 0 (`results/qwen3_validation.json`).")
L.append("- REL 4언어: prepare_glmm.py 사본(`rel4/prepare_glmm_qwen3.py`) → 17,168행·4,292 quartet(종속 3,576/독립 716), 원본 문제 400행 제외, 공개 summary.json 대비값 일치 검증 통과. check_glmm_input.R 사본 통과(설계행렬 14열 full rank). fit_glmm.R 사본으로 Qwen3 단독 적합(공식 불변).")
L.append("- 4언어(en,zh,th,ru) RQ1 민감도: 동일 공식, glmm_utils.R 사본에서 언어 4개·문항당 16행으로만 변경(**추가 민감도**, 사전 명세 아님). 35,120행.")
L.append("- 하드웨어: Qwen2.5 batch 32 B200, InternVL2.5 batch 1, Qwen3 batch 4 RTX 4090 — 배치/하드웨어 차이로 수치 차이가 생길 수 있음(미검증).\n")
L.append("## 2. 6x6 (RQ1/RQ2)\n")
dd=[r for r in rd(O+"/results/descriptive_bootstrap_qwen3.csv") if r["outcome"]=="generation_correct"]
L.append("### 2.1 RQ1 (random intercept, n=79,020)\n")
fe=[r for r in rd(O+"/results/qwen3/rq1_generation/rq1_fixed_effects_generation_correct.csv") if r["term"]=="matched_num"][0]
lrt=rd(O+"/results/qwen3/rq1_generation/rq1_matched_lrt_generation_correct.csv")[1]
bon=rd(O+"/results/qwen3/rq1_generation/rq1_adjusted_concordance_bonus_generation_correct.csv")[0]
fe4=[r for r in rd(O+"/results/qwen3/sens4lang_rq1_generation/rq1_fixed_effects_generation_correct.csv") if r["term"]=="matched_num"][0]
lrt4=rd(O+"/results/qwen3/sens4lang_rq1_generation/rq1_matched_lrt_generation_correct.csv")[1]
bon4=rd(O+"/results/qwen3/sens4lang_rq1_generation/rq1_adjusted_concordance_bonus_generation_correct.csv")[0]
fe5=[r for r in rd(O+"/results/qwen3/core5_rq1_generation/core5_rq1_fixed_effects_generation_correct.csv") if r["term"]=="matched_num"][0]
lrt5=rd(O+"/results/qwen3/core5_rq1_generation/core5_rq1_matched_lrt_generation_correct.csv")[1]
bon5=rd(O+"/results/qwen3/core5_rq1_generation/core5_rq1_adjusted_concordance_bonus_generation_correct.csv")[0]
def row(name,fe,lrt,bon): return [name,f(fe["odds_ratio"]),f"{f(fe['odds_ratio_low'])}–{f(fe['odds_ratio_high'])}",f(lrt["Chisq"],2),pv(lrt["Pr(>Chisq)"]),pp(bon["estimate"]),f"{pp(bon['conf.low'])} – {pp(bon['conf.high'])}"]
table(["모형","matched OR","95% CI","LRT χ²(1)","p","보정 일치 보너스 %p","95% CI %p"],[row("6x6 전체",fe,lrt,bon),row("Thai 제외 5x5 (en,zh,fr,ru,ja)",fe5,lrt5,bon5),row("4언어 en,zh,th,ru (추가 민감도)",fe4,lrt4,bon4)])
L.append("보정 확률 차이: item-marginal 표준화(15점 Gauss 구적), 2,000회 MVN draw, 문항 절편 SD plug-in. 상세 CSV: `results/qwen3/rq1_generation/`, `core5_rq1_generation/`, `sens4lang_rq1_generation/`.\n")
L.append("### 2.2 RQ2 차원 상호작용 (fast 설정)\n")
om=rd(O+"/results/qwen3/rq2_generation_fast/rq2_omnibus_lrt_generation_correct.csv")
table(["검정 블록","χ²","df","p"],[[om[i]["test_block"],f(om[i]["Chisq"],2),om[i]["Df"],pv(om[i]["Pr(>Chisq)"])] for i in (1,3,5,7)])
cb=rd(O+"/results/qwen3/rq2_generation_fast/rq2_concordance_bonus_by_dimension_generation_correct.csv")
tr=rd(O+"/results/qwen3/rq2_generation_fast/rq2_matched_trends_by_dimension_generation_correct.csv")
table(["차원","보너스 %p","95% CI %p","matched OR","95% CI","Holm p"],[[c["dimension"],pp(c["estimate"]),f"{pp(c['conf.low'])} – {pp(c['conf.high'])}",f(t["odds_ratio"],2),f"{f(t['odds_ratio_low'],2)}–{f(t['odds_ratio_high'],2)}",pv(t["p_value_holm"])] for c,t in zip(cb,tr)])
L.append("### 2.3 랜덤 기울기 민감도 (1 + matched || parallel_id, 차원별 matched 5-df LRT)\n")
sl=rd(O+"/results/qwen3/random_slope_generation_fast/random_slope_matched_lrt_generation_correct.csv")[1]
L.append(f"- matched×dimension LRT χ²(5)={f(sl['Chisq'],2)}, p={pv(sl['Pr(>Chisq)'])}.")
sd=rd(O+"/results/qwen3/random_slope_generation_fast/random_slope_diagnostics_generation_correct.csv")
L.append("- 진단: "+"; ".join(f"{r['model']}: singular={r['singular']}, RE SD={r['random_effect_sd']}" for r in sd)+" → 세 모형 모두 matched 기울기 분산이 경계(0)에서 추정된 특이 적합(InternVL 선행 결과와 같은 양상; Qwen2.5는 비특이).\n")
L.append("### 2.4 항목-클러스터 부트스트랩 (5,000회, seed 42; 문항 단위)\n")
sel=[r for r in dd if r["scope"] in("overall","dimension") and r["estimand"] in("accuracy","matched_minus_mismatched")]
table(["scope","level","estimand","추정치","95% CI","문항수"],[[r["scope"],r["level"],r["estimand"],pp(r["estimate"]) if "minus" in r["estimand"] else f(r["estimate"]),f"{pp(r['conf_low']) if 'minus' in r['estimand'] else f(r['conf_low'])} – {pp(r['conf_high']) if 'minus' in r['estimand'] else f(r['conf_high'])}",r["items"]] for r in sel])
L.append("(matched−mismatched는 %p.) 전체 CSV (generation_correct만 표시): `results/descriptive_bootstrap_qwen3.csv`.\n")
L.append("### 2.5 진단\n")
dg=[]
for sub,pat in [("rq1_generation","rq1_diagnostics"),("rq2_generation_fast","rq2_diagnostics"),("core5_rq1_generation","core5_diagnostics"),("sens4lang_rq1_generation","rq1_diagnostics")]:
    for r in rd(f"{O}/results/qwen3/{sub}/{pat}_generation_correct.csv"): dg.append([sub,r["model"],r["nobs"],r["singular"],r["max_absolute_gradient"][:8],r["hessian_min_eigenvalue"][:8],r["random_effect_sd"][:8],r["convergence_messages"] or "(없음)"])
table(["폴더","모형","n","singular","max|grad|","Hess min eig","RE SD","수렴 메시지"],dg)
L.append("## 3. REL 텍스트 의존성 GLMM (6언어 REL 차원, 366문항×36셀=13,176행; 명세: REL_TEXT_DEPENDENCY_GLMM_REPORT_KO.md)\n")
tl=rd(O+"/results/qwen3/rel_text_dependency/rel_textdep_lrt_generation_correct.csv")
ab=rd(O+"/results/qwen3/rel_text_dependency/rel_textdep_adjusted_bonuses_generation_correct.csv")
did=rd(O+"/results/qwen3/rel_text_dependency/rel_textdep_probability_did_generation_correct.csv")[0]
io=rd(O+"/results/qwen3/rel_text_dependency/rel_textdep_interaction_or_generation_correct.csv")[0]
L.append(f"- 공통 matched LRT χ²(1)={f(tl[1]['Chisq'],2)}, p={pv(tl[1]['Pr(>Chisq)'])}; **matched×text_dependency (주 검정)** χ²(1)={f(tl[3]['Chisq'],2)}, p={pv(tl[3]['Pr(>Chisq)'])}.")
L.append(f"- 상호작용 OR (종속/독립) {f(io['interaction_odds_ratio'],2)} (95% CI {f(io['interaction_or_low'],2)}–{f(io['interaction_or_high'],2)}).")
table(["의존성","보정 일치 보너스 %p","95% CI %p"],[[r["text_dependency"],pp(r["adjusted_concordance_bonus"]),f"{pp(r['bonus_conf_low'])} – {pp(r['bonus_conf_high'])}"] for r in ab])
L.append(f"- 종속−독립 보너스 차이 {pp(did['estimate'])} %p (95% CI {pp(did['conf.low'])} – {pp(did['conf.high'])}).")
td=rd(O+"/results/qwen3/rel_text_dependency/rel_textdep_diagnostics_generation_correct.csv")
L.append("- 진단: "+"; ".join(f"{r['model']} singular={r['singular']} msg={r['convergence_messages'] or '없음'}" for r in td)+"\n")
L.append("## 4. REL O/R/C/F 4언어 (RQ2/RQ3), Qwen3\n")
ct=rd(O+"/results/rel4_glmm/qwen3_contrasts.csv"); dg4=rd(O+"/results/rel4_glmm/qwen3_diagnostics.csv")[0]
L.append(f"GLMM: generation_correct ~ ref_align*context_loc*text_dep + query_lang + gui_lang + (1|parallel_id)+(1|pair_id); n={dg4['n_rows']}, 문항 {dg4['n_items']}, 쌍 {dg4['n_pairs']}, logLik={f(dg4['log_likelihood'],1)}, singular={dg4['singular']}, 수렴 메시지={dg4['convergence_message'] or '없음'}. 대비는 조건부 **오즈비**(%p 아님), Wald 95% CI.\n")
table(["의존성","대비","OR","95% CI","Wald p"],[[r["dependency"],r["contrast"],f(r["odds_ratio"],2),f"{f(r['or_ci_low'],2)}–{f(r['or_ci_high'],2)}",pv(r["p_value_wald"])] for r in ct])
L.append("### 4.1 문항 부트스트랩 %p (10,000회, seed 42, percentile; 의존성 층별로 문항 재표집)\n")
bt=rd(O+"/rel4/bootstrap_rel_contrasts.csv")
for m in ["qwen3","qwen","internvl"]:
    L.append(f"**{m}**\n")
    table(["의존성","대비","추정 %p","95% CI","문항","quartet"],[[r["dependency"],r["contrast"],f(r["estimate_pp"],2),f"{f(r['ci_low_pp'],2)} – {f(r['ci_high_pp'],2)}",r["items"],r["quartets"]] for r in bt if r["model"]==m and r["target_gui"]=="all"])
L.append("### 4.2 목표 GUI 언어별 R−O, F−R (탐색적·기술적, %p [95% CI])\n")
for m in ["qwen3"]:
    rows=[]
    for dep in ["dependent","independent"]:
        for c in ["R_minus_O","F_minus_R"]:
            rows.append([dep,c]+[ (lambda r: f"{f(r['estimate_pp'],1)} [{f(r['ci_low_pp'],1)}, {f(r['ci_high_pp'],1)}]")([x for x in bt if x['model']==m and x['dependency']==dep and x['contrast']==c and x['target_gui']==g][0]) for g in ["en","zh","th","ru"]])
    table(["의존성","대비","en","zh","th","ru"],rows)
L.append("(qwen/internvl 동일 수치는 `rel4/bootstrap_rel_contrasts.csv`.)\n")
L.append("## 5. Grounding (RQ4), 158문항 × 12쌍\n")
G=json.load(open(O+"/grounding_summary.json")); gc=rd(O+"/grounding_contrasts.csv")
L.append("- 엄격 파서 재계산은 러너 기록(valid_output, point_in_gt_box) 7,584행 전부와 일치. 관대 파서(괄호 없는 `x, y` 허용)는 17행을 복구.")
table(["조건","n","유효(엄격)","유효(관대)","hit(엄격)","hit율(엄격)","hit(관대)","hit율(관대)"],[[v["condition"],v["n"],v["strict_valid"],v["lenient_valid"],v["strict_hit"],f(v["strict_hit_rate"]),v["lenient_hit"],f(v["lenient_hit_rate"])] for v in G["validity_hits_by_condition"]])
table(["파서","대비","GUI","문항","추정 %p","95% CI","97.5% 구간 (프로토콜)"],[[r["parse"],r["contrast"],r["gui_language"],r["items"],f(r["estimate_pp"],2),f"{f(r['ci95_low'],2)} – {f(r['ci95_high'],2)}",f"{f(r['ci975_low'],2)} – {f(r['ci975_high'],2)}"] for r in gc])
L.append("부트스트랩: 문항 복원추출 10,000회, seed 20261103 (PROTOCOL_KO 명세). 문항당 12쌍 평균 후 문항 평균; GUI별은 문항당 3쌍. O→R 전이(쌍 수, 엄격): "+json.dumps(G["O_to_R_transitions_strict"])+". GUI×조건별 hit율: `grounding_by_gui_condition.csv`.\n")
L.append("## 6. 과제 연결 (grounding R−O vs 객관식 REL R−O, 종속 문항, 문항 수준)\n")
table(["파서","문항","쌍","grounding R−O %p","MC R−O %p","Pearson [95% CI]","Spearman [95% CI]"],[[t["parse"],t["items"],t["pairs_total"],t["mean_grounding_R_O_pp"],t["mean_MC_R_O_pp"],f"{t['pearson']} {t['pearson_ci95']}",f"{t['spearman']} {t['spearman_ci95']}"] for t in G["task_link"]])
L.append("기술 통계일 뿐이며 인과·매개 해석 없음. 두 과제의 쌍 단위 일치(parallel_id, query, gui)만 사용.\n")
L.append("## 7. 모델 비교 (동결 결과 vs Qwen3)\n")
rows=[]
for m in ["qwen2.5","internvl2.5","qwen3"]:
    try:
        sub=lambda s,p: d(m,s,p)
    except: pass
    fe_=[r for r in rd(f"{M[m]}/rq1_generation/rq1_fixed_effects_generation_correct.csv") if r["term"]=="matched_num"][0]
    l_=rd(f"{M[m]}/rq1_generation/rq1_matched_lrt_generation_correct.csv")[1]
    b_=rd(f"{M[m]}/rq1_generation/rq1_adjusted_concordance_bonus_generation_correct.csv")[0]
    c_=rd(f"{M[m]}/core5_rq1_generation/core5_rq1_adjusted_concordance_bonus_generation_correct.csv")[0]
    rb=[r for r in rd(f"{M[m]}/rq2_generation_fast/rq2_concordance_bonus_by_dimension_generation_correct.csv") if r["dimension"]=="rel"][0]
    ab_=rd(f"{M[m]}/{'rel_text_dependency' if m=='qwen3' else 'rel_text_dependency_adjudicated_v1'}/rel_textdep_adjusted_bonuses_generation_correct.csv")
    tl_=rd(f"{M[m]}/{'rel_text_dependency' if m=='qwen3' else 'rel_text_dependency_adjudicated_v1'}/rel_textdep_lrt_generation_correct.csv")[3]
    key={"qwen3":"qwen3","qwen2.5":"qwen","internvl2.5":"internvl"}[m]
    acc=[r for r in rd(O+"/results/descriptive_bootstrap_qwen3.csv") if r["outcome"]=="generation_correct" and r["scope"]=="overall" and r["estimand"]=="accuracy"][0] if m=="qwen3" else [r for r in rd(A+"/descriptive_bootstrap.csv") if r["model"]==key and r["outcome"]=="generation_correct" and r["scope"]=="overall" and r["estimand"]=="accuracy"][0]
    rows.append([m,f(acc["estimate"]),f(fe_["odds_ratio"],2),pp(b_["estimate"]),pp(c_["estimate"]),pp(rb["estimate"]),pp([x for x in ab_ if x["text_dependency"]=="dependent"][0]["adjusted_concordance_bonus"]),pp([x for x in ab_ if x["text_dependency"]=="independent"][0]["adjusted_concordance_bonus"]),pv(tl_["Pr(>Chisq)"])])
table(["모델","6x6 정확도","matched OR","6x6 보너스 %p","5x5 보너스 %p","REL 차원 보너스 %p","REL 종속 보너스 %p","REL 독립 보너스 %p","matched×dep LRT p"],rows)
cm=rd(R+"/followup_rel_4lang/results/glmm/combined_contrasts.csv")+rd(O+"/results/rel4_glmm/qwen3_contrasts.csv")
L.append("REL O/R/C/F 조건부 OR (GLMM, 종속):\n")
table(["모델","R−O","C−O","F−C","F−R","(F−C)−(R−O)"],[[m]+[f(x["odds_ratio"],2)+f" [{f(x['or_ci_low'],2)}, {f(x['or_ci_high'],2)}]" for c in ["R_minus_O","C_minus_O","F_minus_C","F_minus_R","interaction"] for x in cm if x["model"]==m and x["dependency"]=="dependent" and x["contrast"]==c] for m in ["qwen","internvl","qwen3"]])
L.append("REL O/R/C/F 조건부 OR (GLMM, 독립):\n")
table(["모델","R−O","C−O","F−C","F−R","(F−C)−(R−O)"],[[m]+[f(x["odds_ratio"],2)+f" [{f(x['or_ci_low'],2)}, {f(x['or_ci_high'],2)}]" for c in ["R_minus_O","C_minus_O","F_minus_C","F_minus_R","interaction"] for x in cm if x["model"]==m and x["dependency"]=="independent" and x["contrast"]==c] for m in ["qwen","internvl","qwen3"]])
L.append("부트스트랩 %p 비교는 §4.1 (세 모델 모두 동일 방법으로 재계산). 모델 간 공식 합동 비교(compare_models_*)는 README상 보류 분석이라 **미실행**. Grounding은 Qwen2.5·InternVL2.5의 본 실행 결과가 없어(파일럿만 존재) 비교 불가 — 해당없음. 모델 이름 대응: qwen=Qwen2.5-VL-7B, internvl=InternVL2.5-8B. 하드웨어/배치 차이(Qwen2.5 batch 32 B200, InternVL2.5 batch 1, Qwen3 batch 4 RTX 4090)는 결과에 미검증 영향 가능.\n")
L.append("## 8. 실패·건너뜀·주의\n")
L.append("- 실패한 적합 없음. 모든 비특이(random-slope 제외) 적합에서 수렴 메시지 없음.\n- 랜덤 기울기 세 모형은 특이 적합(기울기 SD≈0)이며 공식은 README대로 유지.\n- 4언어 RQ1 민감도는 사전 명세 밖의 추가 분석.\n- RQ2는 README의 fast 설정(사후 Hessian 계산 생략).\n- scoring_correct 모형, 5x5 RQ2, thai_specific, compare_models: README상 보류 → 미실행.\n- grounding 97.5% 구간은 프로토콜 명세, 95% 구간 병기. 부트스트랩 seed는 REL 42(선행 코드 관례), grounding 20261103(프로토콜).\n- 확인 필요: GLMM 합동 모델 간 통계 비교는 수행하지 않았으므로 모델 간 차이의 유의성은 이 보고서에서 주장하지 않음.\n")
open(O+"/REPORT.md","w").write("\n".join(L))
print(len(L))
