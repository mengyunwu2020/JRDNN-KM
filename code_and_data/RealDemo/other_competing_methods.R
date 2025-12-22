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

# Split data into three subsets based on labels
data1 <- data[label == "H1975", ]
data2 <- data[label == "H2228", ]
data3 <- data[label == "HCC827", ]

# Combine subsets into a list
datalist <- list(data1, data2, data3)

### Clustering using SC3
library(SC3)
library(SingleCellExperiment)

# Create SingleCellExperiment object
sce <- SingleCellExperiment(assays = list(counts = as.matrix(data)))

# Perform preprocessing and clustering using SC3
sce <- sc3(sce, ks = 3, biology = TRUE)

# View clustering results
sc3_clusters <- colData(sce)$sc3_3_clusters
print(sc3_clusters)

### Clustering using Seurat
library(Seurat)

# Create Seurat object
seurat_obj <- CreateSeuratObject(counts = data)

# Normalize data
seurat_obj <- NormalizeData(seurat_obj)

# Find highly variable features
seurat_obj <- FindVariableFeatures(seurat_obj)

# Scale data
seurat_obj <- ScaleData(seurat_obj)

# Principal Component Analysis
seurat_obj <- RunPCA(seurat_obj)

# Clustering using Seurat's method
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)

# View clustering results
seurat_clusters <- Idents(seurat_obj)
print(seurat_clusters)

### Modeling using BLGGM
library(BLGGM)

# Run BLGGM on LUAD data
# Set up parameters for BLGGM
data_matrix <- as.matrix(data)

# Gene number
gene_number <- nrow(data_matrix)
print(gene_number)  # Print the number of genes

# Cell number
cell_number <- ncol(data_matrix)
print(cell_number)  # Print the number of cells

# Set the number of cell types (assuming 3 for H1975, H2228, HCC827)
n_celltype <- 3

# Run BLGGM
Result <- BLGGM(data_matrix, n_celltype, num_iterations = 10000, num_threads = 10)

# Compare with true cell type labels
cell_table <- table(Result$cell_labels, label)
print(cell_table)

### Modeling using JGNsc
library(JGNsc)

# Run JGNsc on LUAD data
# Combine subsets into a list of matrices for JGNsc
observed.list <- lapply(datalist, as.matrix)

# Run JGNsc
jgnsc.result <- RunJGNsc(observed.list = observed.list, min.cell = 10, runNetwork = TRUE,
                              l1.vec = seq(1, 40, by = 3) / 100, l2.vec = seq(1, 10, by = 2) / 100)

### Modeling using JSEM
library(glasso)
library(grpreg)

# JSEM function definition
JSEM <- function(
  trainX,       
  trainY,        
  index,       
  lambda,    
  delta1 = NULL,
  delta2 = NULL,
  eps = 1e-06
){
  p = dim(trainX)[2]
  K = length(unique(trainY))
  
  lambda.grpreg = rep(lambda, p)
  
  # list observed data by model
  design = vector("list", K)
  Ahat = vector("list", K)
  for (k in 1:K){
    design[[k]] = trainX[which(trainY == k), ]
    Ahat[[k]] = matrix(0, p, p)
  }
  
  ## Start the loop for each node
  for (i in 1:p) {
    #cat("i= ", i, "\n")
    ## Create a block diagonal matrix for the variables
    list.x = vector("list", K)
    list.y = vector("list", K)
    for (k in 1:K) {
      list.x[[k]] = design[[k]][, -i]
      list.y[[k]] = design[[k]][, i]
    }
    #Duplicate columns for each k and add these extra columns to the end of the rearranged design X;
    #In the meantime, add the variable indices to the list of all variables.
    #Need to reorganize the estimate in the end.
    X = as.matrix(bdiag(list.x))
    new_X = X
    Y = unlist(list.y)
    
    ## To get the index label 
    myindex = sort.variables(index[[i]])
    X = X[, myindex$x.index]
    
    fit = grpreg(X, Y, myindex$v.index, family = "gaussian", penalty = "grLasso", lambda = lambda.grpreg[i])
    coeff = fit$beta[-1, ]    
    
    if (!is.null(delta1)){
      coeff = thr.group(coeff, myindex$v.index, delta1)
    }
    if (!is.null(delta2)){
      coeff = thr.coord(coeff, myindex$v.index, delta2)
    }
    
    tmp = matrix(coeff, nrow = K)
    for (j in 1:dim(tmp)[2]){
      tmp[, j] = tmp[myindex$b.index[, j], j]
    }
    
    for (k in 1:K){
      Ahat[[k]][i, -i] = tmp[k,]
    }     
  }
  
  # Symmetrize the thresholded coefficients; 
  # Get the symmetric adjacency matrices; 
  Theta = lapply(Ahat, FUN = symmetrize)
  Ahat = lapply(Theta, function(u) {u[abs(u)>eps] = 1; return(u)})
  
  return(list(Adj = Ahat, 
              lambda = lambda.grpreg))
}  

# Prepare data for JSEM
trainX <- as.matrix(data)
trainY <- label

# Assuming a node-specific grouping structure index
# Here, we use a simple placeholder for the index. You should replace this with an appropriate structure.
index <- matrix(1, nrow = n_celltype, ncol = ncol(trainX))

# Set the penalty parameter
lambda <- 0.1

# Run JSEM on LUAD data
jsem.result <- JSEM(trainX = trainX, trainY = trainY, index = index, lambda = lambda)

### Modeling using GENIE3
library(GENIE3)

# Run GENIE3 on each subset of LUAD data
for (i in 1:length(datalist)) {
  cat("Running GENIE3 for Dataset", i, "\n")

  genie3_result <- GENIE3(datalist[[i]])
  
  cat("Time taken for Dataset", i, ":", t2 - t1, "\n")
  # Print or save the result if needed
  print(genie3_result)
}


