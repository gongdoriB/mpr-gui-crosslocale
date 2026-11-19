suppressPackageStartupMessages(library(lme4))
P <- "${PROJECT_ROOT}/analysis_rel_pairs/"
O <- "${PROJECT_ROOT}/analysis_plan_followup/task2/"
d <- rbind(read.csv(paste0(P,"glmm_rows_qwen3.csv"), stringsAsFactors=FALSE),
           read.csv(paste0(P,"glmm_rows_qwen25_internvl25.csv"), stringsAsFactors=FALSE)[, names(read.csv(paste0(P,"glmm_rows_qwen3.csv"), nrows=1))])
d <- d[d$text_dependency=="dependent",]
langs <- c("en","zh","th","ru")
d$query_language <- factor(d$query_language, levels=langs); d$gui_language <- factor(d$gui_language, levels=langs)
d$parallel_id <- factor(d$parallel_id); d$pair_id <- factor(d$pair_id)
d$y <- d$generation_correct; d$RA <- d$reference_alignment; d$CL <- d$context_localization
ctl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=200000L))
fit <- function(f, dat) glmer(as.formula(f), data=dat, family=binomial, nAGQ=1L, control=ctl)
base <- "y ~ RA*CL + query_language + gui_language + (1|parallel_id) + (1|pair_id)"
res <- list(); diag <- list()
for (m in c("qwen3","qwen","internvl")) {
  s <- d[d$model==m,]; cat(m, nrow(s), nlevels(droplevels(s$parallel_id)), "\n")
  b <- fit(base, s)
  for (fac in c("gui_language","query_language")) {
    forms <- list(RA_only=paste(base,"+ RA:",fac), CL_only=paste(base,"+ CL:",fac), joint=paste(base,"+ RA:",fac,"+ CL:",fac))
    for (nm in names(forms)) {
      f1 <- fit(forms[[nm]], s)
      a <- anova(b, f1)
      msg <- f1@optinfo$conv$lme4$messages
      res[[length(res)+1]] <- data.frame(model=m, factor=fac, term=nm, n_rows=nrow(s), n_items=nlevels(droplevels(s$parallel_id)),
        chisq=a$Chisq[2], df=a$Df[2], p=a[["Pr(>Chisq)"]][2], logLik_base=as.numeric(logLik(b)), logLik_alt=as.numeric(logLik(f1)),
        singular_base=isSingular(b), singular_alt=isSingular(f1),
        conv_msg_base=paste(b@optinfo$conv$lme4$messages, collapse="|"), conv_msg_alt=paste(msg, collapse="|"))
      cat(m, fac, nm, a$Chisq[2], a$Df[2], a[["Pr(>Chisq)"]][2], "\n"); flush.console()
      write.csv(do.call(rbind,res), paste0(O,"direction_lrt.csv"), row.names=FALSE)
    }
  }
}
