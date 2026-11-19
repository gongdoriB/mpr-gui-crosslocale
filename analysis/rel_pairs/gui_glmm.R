# Per-GUI-language GLMM for REL O/R/C/F (same fixed/random structure as followup_rel_4lang/analysis/fit_glmm.R,
# minus the gui_language term because each fit holds one GUI language). Odds-ratio contrasts with Wald 95% CI.
suppressPackageStartupMessages(library(lme4))
rows <- rbind(read.csv("glmm_rows_qwen3.csv"), read.csv("glmm_rows_qwen25_internvl25.csv"))
fixed <- ~ reference_alignment * context_localization * text_dependency + query_language
out <- list(); diag <- list()
for (m in c("qwen3", "qwen", "internvl")) for (g in c("en", "zh", "ru", "th")) {
  s <- rows[rows$model == m & rows$gui_language == g, ]
  s$text_dependency <- factor(s$text_dependency, levels = c("independent", "dependent"))
  qs <- sort(unique(s$query_language)); s$query_language <- factor(s$query_language, levels = qs)
  s$parallel_id <- factor(s$parallel_id); s$pair_id <- factor(s$pair_id)
  t0 <- Sys.time()
  fit <- glmer(update(fixed, generation_correct ~ . + (1 | parallel_id) + (1 | pair_id)), data = s, family = binomial,
               control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000)))
  beta <- fixef(fit); V <- as.matrix(vcov(fit))
  cell <- function(r, c, dep) { gd <- data.frame(reference_alignment = r, context_localization = c,
      text_dependency = factor(dep, levels = levels(s$text_dependency)), query_language = factor(qs[1], levels = qs))
      model.matrix(fixed, gd)[1, names(beta)] }
  for (dep in c("dependent", "independent")) {
    cs <- list(R_minus_O = cell(1,0,dep) - cell(0,0,dep), C_minus_O = cell(0,1,dep) - cell(0,0,dep),
               F_minus_R = cell(1,1,dep) - cell(1,0,dep), F_minus_C = cell(1,1,dep) - cell(0,1,dep))
    for (nm in names(cs)) { L <- cs[[nm]]; est <- sum(L * beta); se <- sqrt(drop(t(L) %*% V %*% L))
      out[[length(out) + 1]] <- data.frame(model = m, gui_language = g, dependency = dep, contrast = nm,
        odds_ratio = exp(est), or_ci_low = exp(est - 1.96 * se), or_ci_high = exp(est + 1.96 * se),
        p_value_wald = 2 * pnorm(-abs(est / se))) }
  }
  msgs <- paste(fit@optinfo$conv$lme4$messages, collapse = " | ")
  diag[[length(diag) + 1]] <- data.frame(model = m, gui_language = g, n_rows = nrow(s), singular = isSingular(fit),
    convergence_message = msgs, seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
  message(m, " ", g, " done")
}
write.csv(do.call(rbind, out), "gui_glmm_contrasts.csv", row.names = FALSE)
write.csv(do.call(rbind, diag), "gui_glmm_diagnostics.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "gui_glmm_session_info.txt")
