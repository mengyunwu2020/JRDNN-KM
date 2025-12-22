
# ---------------------------------------
# Nonlinear link functions: simple/medium/complex
# ---------------------------------------
f_sc <- function(x, type = c("simple", "medium", "complex")) {
  type <- match.arg(type)
  cbrt <- function(z) sign(z) * abs(z)^(1/3)  # real cube root
  if (type == "simple") {
    cbrt(x)
  } else if (type == "medium") {
    0.25 * exp(x) + cbrt(x)
  } else { # complex
    3 * sin(x) + 0.25 * exp(x) + cbrt(x)
  }
}

# ---------------------------------------
# Block-level adjacency generators (matrix-only)
# ---------------------------------------
# Star block: node 1 is the hub; row/col 1 (except [1,1]) are 1s
generate_star_block <- function(n) {
  m <- matrix(0, n, n)
  if (n >= 2) {
    m[1, 2:n] <- 1
    m[2:n, 1] <- 1
  }
  m
}

# Chain block: |i - j| == 1 entries are 1s
generate_chain_block <- function(n) {
  m <- matrix(0, n, n)
  idx <- abs(row(m) - col(m)) == 1
  m[idx] <- 1
  m
}

# Assemble a block-diagonal adjacency with B blocks (random star/chain by p_star)
build_block_diag_starchain <- function(B = 10, block_size = 10, p_star = 0.5, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  total <- B * block_size
  adj_global <- matrix(0, total, total)
  block_type <- character(B)
  
  for (b in 1:B) {
    is_star <- runif(1) < p_star
    block_type[b] <- if (is_star) "star" else "chain"
    
    M <- if (is_star) generate_star_block(block_size) else generate_chain_block(block_size)
    idx <- ((b - 1) * block_size + 1):(b * block_size)
    adj_global[idx, idx] <- M
  }
  list(adj = adj_global, block_type = block_type)
}

# ---------------------------------------
# Data generators per block (arbitrary initial mean)
# ---------------------------------------
# Star block data:
#   x1 ~ N(mean, 1), others: f(x1) + eps, eps ~ N(0, noise_sd^2)
simulate_star_block <- function(n, block_size, f_type = "simple", mean = 0, noise_sd = 0.1) {
  X <- matrix(0, n, block_size)
  x1 <- rnorm(n, mean, 1)  # var = 1
  X[, 1] <- x1
  if (block_size >= 2) {
    mu <- f_sc(x1, type = f_type)
    X[, 2:block_size] <- matrix(rep(mu, block_size - 1), nrow = n) +
      matrix(rnorm(n * (block_size - 1), 0, noise_sd), nrow = n)
  }
  X
}

# Chain block data:
#   x1 ~ N(mean, 1), for j >= 2: x_j = f(x_{j-1}) + eps
simulate_chain_block <- function(n, block_size, f_type = "simple", mean = 0, noise_sd = 0.1) {
  X <- matrix(0, n, block_size)
  X[, 1] <- rnorm(n, mean, 1)  # var = 1
  if (block_size >= 2) {
    for (j in 2:block_size) {
      X[, j] <- f_sc(X[, j - 1], type = f_type) + rnorm(n, 0, noise_sd)
    }
  }
  X
}

# ---------------------------------------
# Dropout utilities
# ---------------------------------------
# Map level to a (lo, hi) range
dropout_range <- function(level = c("none", "low", "medium", "high")) {
  level <- match.arg(level)
  switch(level,
         "none"   = c(0, 0),
         "low"    = c(0.10, 0.30),
         "medium" = c(0.20, 0.40),
         "high"   = c(0.30, 0.50))
}

# Apply dropout to a data matrix X.
# If per_col = TRUE: draw a separate dropout prob for each column from the range.
# Otherwise: draw one dropout prob for the whole matrix.
apply_dropout <- function(X, level = c("none", "low", "medium", "high"),
                          per_col = TRUE, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  level <- match.arg(level)
  rng <- dropout_range(level)
  if (all(rng == 0)) return(X)
  
  n <- nrow(X); p <- ncol(X)
  
  if (per_col) {
    probs <- runif(p, min = rng[1], max = rng[2])           # length p
    keep  <- matrix(runif(n * p), n, p) > matrix(probs, n, p, byrow = TRUE)  # TRUE=keep
  } else {
    prob  <- runif(1, min = rng[1], max = rng[2])           # scalar
    keep  <- matrix(runif(n * p) > prob, n, p)
  }
  X * keep
}

# ---------------------------------------
# Assemble full dataset by blocks, then optional dropout
# ---------------------------------------
# Inputs:
#   block_type: character vector length B with entries "star"/"chain"
#   n: samples per block
#   block_size: variables per block
#   f_type: "simple" | "medium" | "complex"
#   mean: initial mean for x1 in all blocks (scalar; can be vectorized later if needed)
#   noise_sd: sd of Gaussian noise
#   dropout: "none" | "low" | "medium" | "high"
#   per_col: if TRUE, sample a separate dropout prob for each column
simulate_data_from_blocks <- function(block_type, n, block_size,
                                      f_type = "simple", mean = 0, noise_sd = 0.1,
                                      dropout = c("none", "low", "medium", "high"),
                                      per_col = TRUE,
                                      seed = NULL, dropout_seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  dropout <- match.arg(dropout)
  B <- length(block_type)
  
  blocks <- vector("list", B)
  for (b in 1:B) {
    if (block_type[b] == "star") {
      blocks[[b]] <- simulate_star_block(n, block_size, f_type = f_type, mean = mean, noise_sd = noise_sd)
    } else {
      blocks[[b]] <- simulate_chain_block(n, block_size, f_type = f_type, mean = mean, noise_sd = noise_sd)
    }
  }
  X <- do.call(cbind, blocks)
  colnames(X) <- paste0("V", seq_len(ncol(X)))
  
  # Apply dropout after data generation
  X <- apply_dropout(X, level = dropout, per_col = per_col, seed = dropout_seed)
  X
}

# ---------------------------------------
# Example usage
# ---------------------------------------
# 1) Build random adjacency + block types
res <- build_block_diag_starchain(B = 10, block_size = 10, p_star = 0.5, seed = 123)

# 2) Generate data (choose f and dropout level)
n_samples <- 1000
X_low_dropout <- simulate_data_from_blocks(res$block_type, n = n_samples, block_size = 10,
                                           f_type = "complex", mean = 0, noise_sd = 0.1,
                                           dropout = "low", per_col = TRUE,
                                           seed = 1, dropout_seed = 42)

# Alternatives:
# X_none   <- simulate_data_from_blocks(res$block_type, n_samples, 10, "simple", 0, 0.1, "none",   TRUE, 1, 42)
# X_medium <- simulate_data_from_blocks(res$block_type, n_samples, 10, "medium", 0, 0.1, "medium", TRUE, 1, 42)
# X_high   <- simulate_data_from_blocks(res$block_type, n_samples, 10, "complex",0, 0.1, "high",   TRUE, 1, 42)

# Final objects:
# - res$adj, res$block_type
# - X_low_dropout: simulated data matrix with dropout applied
