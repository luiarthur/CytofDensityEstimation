donor1 = read.csv('../data/TGFBR2/cytof-data/donor1.csv')

# marker = "CD16"
marker = "CD56"
yT = donor1[donor1$treatment == "TGFb", marker]
yC = donor1[donor1$treatment != "TGFb", marker]

replace_inf = function(x, val=-10) ifelse(x == -Inf, val, x)

log_yT = replace_inf(log(yT))
log_yC = replace_inf(log(yC))

(ks_test_result = ks.test(log_yT, log_yC))
pval = ks_test_result$p.value
plot(ecdf(log_yT), lwd=2, col='red', main=paste("KS test p-value: ", pval))
plot(ecdf(log_yC), lwd=2, col='blue', add=TRUE)

hist(log_yT, breaks=100, prob=T)
hist(log_yC, breaks=100, prob=T)

