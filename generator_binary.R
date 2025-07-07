correlation_matrix <- as.matrix(read_csv("Documents/xplanar_placement/src/correlation_matrix.csv", col_names = FALSE))
feat_probs <- as.array(as.matrix(read_csv("Documents/xplanar_placement/src/feat_probs.csv", col_names = FALSE)))
rownames(correlation_matrix) <- colnames(correlation_matrix)

print(correlation_matrix[25,39])

require(bindata)
commprob <- bincorr2commonprob(feat_probs, correlation_matrix)
commprob[commprob < 0] <- 0
res = rmvbin(100000, commonprob=commprob)
mu <- colMeans(res)
correl = cor(res)
print(correl[25,39])


# EP algorithm
require(simstudy)
library(Matrix)
corr_mat = ceiling(correlation_matrix*1e+8)/1e+8 # there are some problems with float numbers, rounding up helps
corr_mat = as.matrix(forceSymmetric(corr_mat))

res = genCorGen(n=100000,
                nvars=40, 
                params1=as.numeric(feat_probs), 
                dist="binary", 
                corMatrix=corr_mat, 
                wide=TRUE,
                method='ep')
res$id <- NULL
mu2 <- colMeans(res)
correl2 = cor(res)
print(correl2[25,39])


library(corrplot)
corrplot(ceiling(correlation_matrix*1e+14)/1e+14, method="color", diag=FALSE, title='corr rounded')
corrplot(correlation_matrix, method="color", diag=FALSE, title='corr origin')
corrplot(correl, method="color", diag=FALSE, title='bindata')
corrplot(correl2, method="color", diag=FALSE, title='EP method')