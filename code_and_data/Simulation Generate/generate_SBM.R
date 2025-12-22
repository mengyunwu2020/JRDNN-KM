
suppressPackageStartupMessages(library(igraph))

# =========================
# 1) SBM block (10x10) WITHOUT degree capping
# =========================
# Generate ONE SBM block adjacency (0/1, symmetric, no self-loops).
generate_sbm_block <- function(
    n = 10,
    block.sizes = c(1, 1, 8),
    P = matrix(c(
      0.0, 0.2, 0.8,
      0.2, 0.0, 0.8,
      0.8, 0.8, 0.1
    ), nrow = 3, byrow = TRUE),
    directed = FALSE, seed = NULL
) {
  if (!is.null(seed)) set.seed(seed)
  g <- sample_sbm(n = n, pref.matrix = P, block.sizes = block.sizes, directed = directed)
  A <- as.matrix(as_adjacency_matrix(g, sparse = FALSE))
  diag(A) <- 0
  A <- (A + t(A)) > 0
  A * 1
}

# Assemble 10 SBM blocks on the diagonal (total 100 nodes)
build_blockdiag_sbm <- function(B = 10, block_size = 10,
                                block.sizes = c(1, 1, 8),
                                P = matrix(c(
                                  0.0, 0.2, 0.8,
                                  0.2, 0.0, 0.8,
                                  0.8, 0.8, 0.1
                                ), nrow = 3, byrow = TRUE),
                                directed = FALSE, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  total <- B * block_size
  adj_global <- matrix(0, total, total)
  for (b in 1:B) {
    M <- generate_sbm_block(n = block_size, block.sizes = block.sizes, P = P,
                            directed = directed, seed = NULL)
    idx <- ((b - 1) * block_size + 1):(b * block_size)
    adj_global[idx, idx] <- M
  }
  adj_global
}

# =========================
# 2) Manual BFS on adjacency (robust & version-agnostic)
# =========================
manual_bfs <- function(A_block, root = 1) {
  p <- nrow(A_block)
  visited <- rep(FALSE, p)
  father  <- integer(p); father[] <- 0
  order   <- integer(0)
  q <- root
  visited[root] <- TRUE
  while (length(q) > 0) {
    v <- q[1]; q <- q[-1]
    order <- c(order, v)
    neigh <- which(A_block[v, ] == 1)
    for (u in sort(neigh)) {
      if (!visited[u]) {
        visited[u] <- TRUE
        father[u]  <- v
        q <- c(q, u)
      }
    }
  }
  list(order = order, father = father)
}

# =========================
# 3) Neighbor-driven mean: x1..x4 are THIS NODE'S neighbors (up to 4)
# =========================
f_mu_from_neighbor_values <- function(vals, type = c("simple","medium","complex")) {
  type <- match.arg(type)
  if (is.null(vals) || length(vals) == 0) return(0)
  if (is.vector(vals)) vals <- cbind(vals)
  k <- ncol(vals)
  x1 <- vals[, 1, drop = TRUE]
  if (k == 1) return(x1^3)
  
  x2 <- vals[, 2, drop = TRUE]
  if (k == 2) return(x1^3 + 0.5 * x1 * x2)
  
  x3 <- vals[, 3, drop = TRUE]
  if (k == 3) {
    if (type == "simple")  return(x1^3 + 0.5 * x1 * x2 + sqrt(pmax(x3, 0)))
    if (type == "medium")  return(x1^3 + 0.5 * x1 * x2 + 0.2 * exp(x3))
    if (type == "complex") return(x1^3 + 0.5 * x1 * x2 + 0.2 * exp(x3))
  }
  
  x4 <- vals[, 4, drop = TRUE]  # k >= 4
  if (type == "simple")  return(x1^3 + 0.5 * x1 * x2 + sqrt(pmax(x3, 0)) + (x3 + x4)^2)
  if (type == "medium")  return(x1^3 + 0.5 * x1 * x2 + 0.2 * exp(x3) + log(pmax(x4, 1e-6)))
  if (type == "complex") return(x1^3 + 0.5 * x1 * x2 + 0.2 * exp(x3) + sin(x4))
}

# =========================
# 4) Simulate ONE block via manual BFS across ALL components
#    - For disconnected SBM blocks, start at 1, then any unvisited node starts a new BFS.
#    - Each component root ~ N(init_mean, 1); others: f(neighbors) + N(0, noise_sd^2)
# =========================
simulate_block_from_adj <- function(A_block, n, f_type = c("simple","medium","complex"),
                                    init_mean = 0, noise_sd = 0.1) {
  f_type <- match.arg(f_type)
  p <- ncol(A_block)
  Xb <- matrix(0, nrow = n, ncol = p)
  generated <- rep(FALSE, p)
  
  for (root in 1:p) {
    if (generated[root]) next
    bfs <- manual_bfs(A_block, root = root)
    order  <- bfs$order
    father <- bfs$father
    
    # Seed for this (sub)component
    Xb[, root] <- rnorm(n, mean = init_mean, sd = 1)
    generated[root] <- TRUE
    
    # Others in the component
    for (v in order[order != root]) {
      neigh <- which(A_block[v, ] == 1)
      # father first if exists, then remaining neighbors ascending
      if (father[v] != 0 && father[v] %in% neigh) {
        neigh <- c(father[v], setdiff(sort(neigh), father[v]))
      } else {
        neigh <- sort(neigh)
      }
      # use neighbors already generated (at least father)
      avail <- neigh[generated[neigh]]
      if (length(avail) > 4) avail <- avail[1:4]
      vals <- if (length(avail) == 0) NULL else Xb[, avail, drop = FALSE]
      mu <- f_mu_from_neighbor_values(vals, type = f_type)
      Xb[, v] <- mu + rnorm(n, 0, noise_sd)
      generated[v] <- TRUE
    }
  }
  colnames(Xb) <- paste0("V", 1:p)
  Xb
}

# =========================
# 5) Dropout (same API as before)
# =========================
dropout_range <- function(level = c("none", "low", "medium", "high")) {
  level <- match.arg(level)
  switch(level,
         "none"   = c(0, 0),
         "low"    = c(0.10, 0.30),
         "medium" = c(0.20, 0.40),
         "high"   = c(0.30, 0.50))
}

apply_dropout <- function(X, level = c("none", "low", "medium", "high"),
                          per_col = TRUE, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  level <- match.arg(level)
  rng <- dropout_range(level)
  if (all(rng == 0)) return(X)
  
  n <- nrow(X); p <- ncol(X)
  if (per_col) {
    probs <- runif(p, min = rng[1], max = rng[2])      # per-column dropout prob
    keep  <- matrix(runif(n * p), n, p) >
      matrix(probs, n, p, byrow = TRUE)         # TRUE = keep
  } else {
    prob  <- runif(1, min = rng[1], max = rng[2])      # single rate
    keep  <- matrix(runif(n * p) > prob, n, p)
  }
  X * keep
}

# =========================
# 6) Full pipeline: build SBM block-diagonal, simulate data, apply dropout
# =========================
simulate_data_from_blockdiag <- function(adj_global, B = 10, block_size = 10, n,
                                         f_type = c("simple","medium","complex"),
                                         block_means = 0, noise_sd = 0.1,
                                         dropout = c("none","low","medium","high"),
                                         per_col = TRUE,
                                         seed = NULL, dropout_seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  f_type  <- match.arg(f_type)
  dropout <- match.arg(dropout)
  
  if (length(block_means) == 1L) block_means <- rep(block_means, B)
  stopifnot(length(block_means) == B)
  
  blocks <- vector("list", B)
  for (b in 1:B) {
    idx  <- ((b - 1) * block_size + 1):(b * block_size)
    Ablk <- adj_global[idx, idx, drop = FALSE]
    blocks[[b]] <- simulate_block_from_adj(Ablk, n,
                                           f_type = f_type,
                                           init_mean = block_means[b],
                                           noise_sd = noise_sd)
  }
  X <- do.call(cbind, blocks)
  colnames(X) <- paste0("X", 1:ncol(X))
  
  # Apply dropout AFTER generation
  X <- apply_dropout(X, level = dropout, per_col = per_col, seed = dropout_seed)
  X
}

# =========================
# Example
# =========================
# 1) Build 100x100 block-diagonal SBM adjacency (no degree cap)
P <- matrix(c(
  0.0, 0.2, 0.8,
  0.2, 0.0, 0.8,
  0.8, 0.8, 0.1
), nrow = 3, byrow = TRUE)

adj_global <- build_blockdiag_sbm(B = 10, block_size = 10,
                                  block.sizes = c(1,1,8), P = P,
                                  directed = FALSE, seed = 123)

# Optional quick look at block degrees:
# for (b in 1:10) {
#   idx <- ((b-1)*10+1):(b*10)
#   cat("Block", b, "max degree:", max(rowSums(adj_global[idx, idx])), "\n")
# }

# 2) Simulate data with neighbor-based rules + dropout
n_samples <- 1000
means10   <- rep(0.5, 10)
X_medium <- simulate_data_from_blockdiag(adj_global,
                                         B = 10, block_size = 10, n = n_samples,
                                         f_type = "medium",
                                         block_means = means10, noise_sd = 0.1,
                                         dropout = "low", per_col = TRUE,
                                         seed = 1, dropout_seed = 42)

# X_medium is n x 100; rows=samples, cols=variables.
