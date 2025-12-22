##############################################################
# To use the Luad dataset, you can download it from the GitHub 
# repository: https://github.com/mengyunwu2020/JRDNN-KM
##############################################################
load('luad.Rdata')

# Data processing and splitting
data <- luad$selected
label <- luad$labels

# Check for missing values and replace with 0
any_missing <- any(is.na(data))
print(any_missing)  # Print TRUE or FALSE
data[is.na(data)] <- 0
GENE<-colnames(data)
# Split data into three subsets based on labels
data1 <- data[label == "H1975", ]
data2 <- data[label == "H2228", ]
data3 <- data[label == "HCC827", ]

# Combine subsets into a list
datalist <- list(data1, data2, data3)

genes_selected=GENE

###CSCORE
library(CSCORE)
library(Seurat)
data_s1 <- CreateSeuratObject(counts = t(data1))
CSCORE_result <- CSCORE(data_s1, genes = GENE)
# Obtain CS-CORE co-expression estimates
CSCORE_coexp <- CSCORE_result$est

# Obtain BH-adjusted p values
CSCORE_p <- CSCORE_result$p_value
p_matrix_BH = matrix(0, length(genes_selected), length(genes_selected))
p_matrix_BH[upper.tri(p_matrix_BH)] = p.adjust(CSCORE_p[upper.tri(CSCORE_p)], method = "BH")
p_matrix_BH <- p_matrix_BH + t(p_matrix_BH)

# Set co-expression entires with BH-adjusted p-values greater than 0.05 to 0
CSCORE_coexp[p_matrix_BH > 0.05] <- 0
CSCORE_coexp[p_matrix_BH <= 0.05] <- 1
###
CSCORE_H1975<-CSCORE_coexp
CSCORE_H2228<-CSCORE_coexp
CSCORE_HCC827<-CSCORE_coexp
CSCORE_network<-list(CSCORE_H1975=CSCORE_H1975,CSCORE_H2228=CSCORE_H2228,CSCORE_HCC827=CSCORE_HCC827)
save(CSCORE_network,file = "CSCORE_network.Rdata")

###SPQN
library(spqn)

library(spqnData)
data(gtex.4k)
data_e1<-t(data1)
cor_m <- cor(data1)
ave_logrpkm <-rowMeans(data_e1)
cor_m_spqn <- normalize_correlation(cor_m, ave_exp=ave_logrpkm, ngrp=20, size_grp=10, ref_grp=10)

library(glasso)
S <- cor_m_spqn
rho <- 0.2
fit <- glasso(S, rho = rho)


adj_mat <- (abs(fit$wi) > 1e-6) * 1  
diag(adj_mat) <- 0

spqn_H1975<-adj_mat
spqn_H2228<-adj_mat
spqn_HCC827<-adj_mat

spqn_network<-list(spqn_H1975=spqn_H1975,spqn_H2228=spqn_H2228,spqn_HCC827=spqn_HCC827)
spqn_network_bi<-lapply(spqn_network, function(matrix) {
  # 将矩阵中所有大于 0.01 的元素替换为 1，其他替换为 0
  ifelse(matrix != 0, 1, 0)
})

save(spqn_network_bi,file = "spqn_network.Rdata")
