suppressMessages(library(lme4)); a<-commandArgs(TRUE); d<-read.csv(a[1])
m1<-glmer(mR~gfix+gui+qry+(1|item),d,binomial,control=glmerControl(optimizer="bobyqa"))
m0<-update(m1,.~.-gfix); print(anova(m0,m1)); s<-summary(m1)$coefficients["gfix",]
cat("OR",exp(s[1]),"CI",exp(s[1]+c(-1.96,1.96)*s[2]),"singular",isSingular(m1),"\n")
