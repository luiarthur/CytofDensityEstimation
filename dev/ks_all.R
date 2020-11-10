set.seed(0)
replace_inf = function(x, val=-10) ifelse(x == -Inf, val, x)

donor1 = read.csv('../data/TGFBR2/cytof-data/donor1.csv')
donor1 = donor1[sample(1:NROW(donor1), 20000, replace=FALSE), ]
markers = colnames(donor1)[-ncol(donor1)]

compute_ks = function(marker, rm_inf=FALSE) {
  yT = donor1[donor1$treatment == "TGFb", marker]
  yC = donor1[donor1$treatment != "TGFb", marker]

  log_yT = log(yT)
  log_yC = log(yC)
  if (rm_inf) {
    log_yT = log_yT[is.finite(log_yT)]
    log_yC = log_yC[is.finite(log_yC)]
  }

  ks_test_result = ks.test(log_yT, log_yC)
  pval = ks_test_result$p.value
  return(pval)
}

p = sapply(markers, compute_ks, rm_inf=F)
round(sort(p), 5)

# Proportion of zeros
prop0C = round(colMeans(donor1[donor1$treatment == "TGFb", ] == 0), 3)
prop0T = round(colMeans(donor1[donor1$treatment != "TGFb", ] == 0), 3)
sort(prop0C)
sort(prop0T)

# TEST. CD56, CD57, LAG3
marker = 'CD57'
yT = donor1[donor1$treatment == "TGFb", marker]
yC = donor1[donor1$treatment != "TGFb", marker]
log_yT = log(yT)
log_yC = log(yC)
r = range(c(log_yT[log_yT>-Inf], log_yC[log_yC>-Inf]))
hist(log_yT, breaks=100, xlim=r, border='transparent', col=rgb(1,0,0,.3))
hist(log_yC, breaks=100, xlim=r, border='transparent', col=rgb(0,0,1,.3), add=T)
