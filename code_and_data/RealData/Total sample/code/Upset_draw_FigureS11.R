##############################################################
########################   Figure S11   #######################
##############################################################

########################## Note ##############################
## Please set the working directory to the source file 
## location.
##############################################################

library(ComplexUpset)
library(ggplot2)
library(patchwork)

# Function: Construct binary presence matrix based on inferred edge overlaps between methods
generate_bin_df <- function(overlap_matrix) {
  method_names <- colnames(overlap_matrix)
  n_methods <- length(method_names)
  
  # Extract the number of edges per method (diagonal of the overlap matrix)
  edge_counts <- diag(overlap_matrix)
  
  # Initialize edge sets for each method
  edge_sets <- vector("list", n_methods)
  next_edge_id <- 1
  
  # Initialize the first method's edge IDs
  edge_sets[[1]] <- seq(next_edge_id, length.out = edge_counts[1])
  next_edge_id <- next_edge_id + edge_counts[1]
  
  # Iteratively assign edges to remaining methods to reflect observed overlaps
  for (i in 2:n_methods) {
    edge_sets[[i]] <- integer(0)
    
    for (j in 1:(i - 1)) {
      # Required overlap between method i and j
      required <- overlap_matrix[i, j]
      current <- length(intersect(edge_sets[[i]], edge_sets[[j]]))
      
      if (current < required) {
        shortage <- required - current
        candidates <- setdiff(edge_sets[[j]], edge_sets[[i]])
        remaining_capacity <- edge_counts[i] - length(edge_sets[[i]])
        
        # Assign edges based on overlap structure
        to_add <- min(shortage, length(candidates), remaining_capacity)
        if (to_add > 0) {
          selected <- sample(candidates, size = to_add, replace = FALSE)
          edge_sets[[i]] <- c(edge_sets[[i]], selected)
        }
      }
    }
    
    # Fill in remaining edges with unique ones to match edge count
    current_size <- length(edge_sets[[i]])
    needed <- edge_counts[i]
    
    if (current_size < needed) {
      new_edges <- seq(next_edge_id, length.out = needed - current_size)
      edge_sets[[i]] <- c(edge_sets[[i]], new_edges)
      next_edge_id <- next_edge_id + length(new_edges)
    }
  }
  
  # Construct binary matrix for presence of each edge in methods
  all_edges <- unique(unlist(edge_sets))
  bin_mat <- matrix(0, nrow = length(all_edges), ncol = n_methods)
  colnames(bin_mat) <- method_names
  
  for (idx in seq_along(all_edges)) {
    edge_id <- all_edges[idx]
    for (m in seq_len(n_methods)) {
      if (edge_id %in% edge_sets[[m]]) bin_mat[idx, m] <- 1
    }
  }
  
  return(as.data.frame(bin_mat))
}


# Function: Generate UpSet plot based on binary presence data and subgroup name
plot_upset <- function(bin_df, method_names, title_text) {
  queries <- lapply(method_names, function(m) upset_query(set = m, fill = "#F56D70"))
  
  upset(
    bin_df,
    method_names,
    wrap = TRUE,
    width_ratio = 0.15,
    height_ratio = 0.5,
    n_intersections = 30,
    min_size = 3,
    base_annotations = list(
      'Intersection size' = intersection_size(
        counts = FALSE,
        mapping = aes(fill = 'bars_color')
      ) + scale_fill_manual(values = c('bars_color' = '#3b4a79'), guide = 'none')
    ),
    set_sizes = (
      upset_set_size() +
        ylab('Methods')
    ),
    queries = queries
  ) + ggtitle(title_text) +
    theme(plot.title = element_text(hjust = 0.5))
}


# ----------- Define method names ------------
method_names <- c("JRDNN-KM", "BLGGM", "JGNsc", "JSEM", "SpQN", 
                  "Normalisr", "GENIE3", "locCSN", "CS-CORE")


# ----------- Subgroup 1 (H1975) ------------
data1 <- matrix(c(
  218, 59, 96, 103, 62, 75, 74, 77, 68,
  59, 165, 83, 102, 45, 99, 59, 53, 62,
  96, 83, 333, 148, 76, 145, 54, 71, 44,
  103, 102, 148, 355, 66, 71, 94, 115, 31,
  62, 45, 76, 66, 200, 87, 48, 51, 96,
  75, 99, 145, 71, 87, 333, 59, 82, 48,
  74, 59, 54, 94, 48, 59, 164, 117, 75,
  77, 53, 71, 115, 51, 82, 117, 284, 93,
  68, 62, 44, 31, 96, 48, 75, 93, 161), 
  byrow = TRUE, nrow = 9)

colnames(data1) <- method_names
rownames(data1) <- method_names
bin_df1 <- generate_bin_df(data1)
plot1 <- plot_upset(bin_df1, method_names, "A    Overlapping in Subgroup 1 H1975")


# ----------- Subgroup 2 (H2228) ------------
data2 <- matrix(c(
  225, 105, 132, 157, 84, 69, 98, 85, 103,
  105, 196, 71, 121, 68, 73, 64, 49, 97,
  132, 71, 347, 137, 94, 125, 68, 65, 59,
  157, 121, 137, 386, 102, 135, 76, 52, 64,
  84, 68, 94, 102, 205, 95, 71, 57, 83,
  69, 73, 125, 135, 95, 247, 67, 73, 91,
  98, 64, 68, 76, 71, 67, 158, 108, 81,
  85, 49, 65, 52, 57, 73, 108, 301, 143,
  103, 97, 59, 64, 83, 91, 81, 143, 257), 
  byrow = TRUE, nrow = 9)

colnames(data2) <- method_names
rownames(data2) <- method_names
bin_df2 <- generate_bin_df(data2)
plot2 <- plot_upset(bin_df2, method_names, "B    Overlapping in Subgroup 2 H2228")


# ----------- Subgroup 3 (HCC827) ------------
data3 <- matrix(c(
  188, 64, 59, 82, 56, 99, 85, 63, 58,
  64, 182, 94, 97, 68, 104, 65, 56, 51,
  59, 94, 288, 83, 69, 97, 38, 52, 49,
  82, 97, 83, 310, 103, 165, 76, 52, 54,
  56, 68, 69, 103, 214, 152, 46, 103, 54,
  99, 104, 97, 165, 46, 370, 35, 58, 39,
  85, 65, 38, 76, 103, 35, 132, 92, 67,
  63, 56, 52, 52, 103, 58, 92, 274, 78,
  58, 51, 49, 54, 54, 39, 67, 78, 140),
  byrow = TRUE, nrow = 9)

colnames(data3) <- method_names
rownames(data3) <- method_names
bin_df3 <- generate_bin_df(data3)
plot3 <- plot_upset(bin_df3, method_names, "C    Overlapping in Subgroup 3 HCC827")


# ----------- Common set across subgroups ------------
data4 <- matrix(c(
  70, 31, 24, 29, 24, 36, 11, 35, 38,
  31, 62, 36, 20, 19, 20, 14, 38, 25,
  24, 36, 84, 43, 24, 45, 9, 26, 22,
  29, 20, 43, 103, 33, 52, 14, 25, 17,
  24, 19, 24, 33, 73, 41, 10, 13, 19,
  36, 20, 45, 52, 10, 113, 21, 34, 17,
  11, 14, 9, 14, 13, 21, 35, 12, 15,
  35, 38, 26, 25, 13, 34, 12, 77, 22,
  38, 25, 22, 17, 19, 17, 15, 22, 69),
  byrow = TRUE, nrow = 9)

colnames(data4) <- method_names
rownames(data4) <- method_names
bin_df4 <- generate_bin_df(data4)
plot4 <- plot_upset(bin_df4, method_names, "D    Common shared by all subgroups")


# Combine and export plots
final_plot <- (plot1 + plot2) / (plot3 + plot4)