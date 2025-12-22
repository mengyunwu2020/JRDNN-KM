
##############################################################
##################   Figure S9   ############################
##############################################################

########################## Note ##############################
## Please set the working directory to the source file 
## location.
##############################################################


library(here)

load(here("RealData","Total Sample", "result_data", "Estimated_networks", "adjmatrix_liver.Rdata"))


library(ggpubr)
library(igraph)
library(ggplot2)
library(ggraph)
library(gridExtra)
library(network)
library(ggnet)
library(cowplot)


Dendritic <- as.matrix(adj_liver$Dendritic)
Erythroblast <- as.matrix(adj_liver$Erythroblast)
Endothelial <- as.matrix(adj_liver$Endothelial)
Kupffer <- as.matrix(adj_liver$Kupffer)
Tcell <- as.matrix(adj_liver$Tcell)



degree_Dendritic <- rowSums(Dendritic)
degree_Erythroblast <- rowSums(Erythroblast)
degree_Endothelial <- rowSums(Endothelial)
degree_Kupffer <- rowSums(Kupffer)
degree_Tcell <- rowSums(Tcell)


hub_Dendritic <- names(sort(degree_Dendritic, decreasing = TRUE))[1:30]
hub_Erythroblast <- names(sort(degree_Erythroblast, decreasing = TRUE))[1:30]
hub_Endothelial <- names(sort(degree_Endothelial, decreasing = TRUE))[1:30]
hub_Kupffer <- names(sort(degree_Kupffer, decreasing = TRUE))[1:2]
hub_Tcell <- names(sort(degree_Tcell, decreasing = TRUE))[1:1]


all_hub_intersection <- Reduce(intersect, list(hub_Dendritic, hub_Erythroblast, hub_Endothelial, hub_Kupffer, hub_Tcell))


kupffer_tcell_intersection <- intersect(hub_Kupffer, hub_Tcell)


cat("各自前30个hub节点的交集: ", all_hub_intersection, "\n")
cat("Kupffer和Tcell前30个hub节点的交集: ", kupffer_tcell_intersection, "\n")




# Ensure row and column names are consistent
rownames(Erythroblast) <- rownames(Dendritic)
colnames(Erythroblast) <- colnames(Dendritic)
rownames(Endothelial) <- rownames(Dendritic)
colnames(Endothelial) <- colnames(Dendritic)
rownames(Kupffer) <- rownames(Dendritic)
colnames(Kupffer) <- colnames(Dendritic)
rownames(Tcell) <- rownames(Dendritic)
colnames(Tcell) <- colnames(Dendritic)

# Identify common edges
H_common <- (Dendritic == 1) & (Erythroblast == 1) & (Endothelial == 1) & (Kupffer == 1) & (Tcell == 1)
H_common <- ifelse(H_common == TRUE, 1, 0)
sum(H_common == TRUE)
# Identify common edges between Kupffer and Tcell
H_common_kupffer_tcell <- (Kupffer == 1) & (Tcell == 1)
H_common_kupffer_tcell <- ifelse(H_common_kupffer_tcell == TRUE, 1, 0)

# Create igraph objects
graph_Dendritic <- graph_from_adjacency_matrix(Dendritic, mode = "undirected", diag = FALSE)
graph_Erythroblast <- graph_from_adjacency_matrix(Erythroblast, mode = "undirected", diag = FALSE)
graph_Endothelial <- graph_from_adjacency_matrix(Endothelial, mode = "undirected", diag = FALSE)
graph_Kupffer <- graph_from_adjacency_matrix(Kupffer, mode = "undirected", diag = FALSE)
graph_Tcell <- graph_from_adjacency_matrix(Tcell, mode = "undirected", diag = FALSE)
graph_H_common <- graph_from_adjacency_matrix(H_common, mode = "undirected", diag = FALSE)

# Set seed for reproducibility
set.seed(10)

# Define community colors
community_colors <- c("#4DAF4A", "#377EB8", "#CD534CFF", "red", "#984EA3", "#FFFF33", "#A65628", "#F781BF", "#999999", "#E41A1C", "#377EB8", "#4DAF4A")


# Function to process graphs
process_graph <- function(graph, H_common, community_colors, title, labels, H_common_kupffer_tcell = NULL) {
  community <- cluster_louvain(graph, resolution = 0.5)
  membership <- membership(community)
  community_sizes <- sizes(community)
  
  common_edges <- which(H_common == 1, arr.ind = TRUE)
  common_nodes <- unique(c(rownames(H_common)[common_edges[, 1]], colnames(H_common)[common_edges[, 2]]))
  
  if (!is.null(H_common_kupffer_tcell)) {
    common_edges_kupffer_tcell <- which(H_common_kupffer_tcell == 1, arr.ind = TRUE)
    common_nodes_kupffer_tcell <- unique(c(rownames(H_common_kupffer_tcell)[common_edges_kupffer_tcell[, 1]], colnames(H_common_kupffer_tcell)[common_edges_kupffer_tcell[, 2]]))
  } else {
    common_nodes_kupffer_tcell <- character(0)
  }
  
  top_communities <- order(community_sizes, decreasing = TRUE)[1:2]
  V(graph)$Type <- "Others"
  
  for (i in seq_along(top_communities)) {
    community_id <- as.numeric(top_communities[i])
    community_nodes <- V(graph)$name[membership == community_id]
    V(graph)$Type[V(graph)$name %in% community_nodes] <- paste("Community", i)
  }
  
  col <- setNames(community_colors[1:length(top_communities)], paste("Community", seq_along(top_communities)))
  col <- c(col, "Others" = "grey70", "Common" = "red", "Kupffer_Tcell" = "orange")
  
  V(graph)$color <- ifelse(V(graph)$name %in% common_nodes, "red", 
                           ifelse(V(graph)$name %in% common_nodes_kupffer_tcell, "orange", "grey70"))
  
  degree_values <- degree(graph)
  hub_nodes <- names(sort(degree_values, decreasing = TRUE))[1:5]
  V(graph)$size <- ifelse(V(graph)$name %in% hub_nodes, 5, 1)  # Increase size of hub nodes
  
  E(graph)$color <- "grey70"
  E(graph)$width <- 0.3
  
  for (i in seq_along(top_communities)) {
    community_id <- as.numeric(top_communities[i])
    community_edges <- which(membership[ends(graph, E(graph))[,1]] == community_id & membership[ends(graph, E(graph))[,2]] == community_id)
    E(graph)$color[community_edges] <- community_colors[i]
    E(graph)$width[community_edges] <- 0.3  # 设置社区边的宽度为1
  }
  
  
  # Highlight common edges
  common_edges <- data.frame(
    node1 = rownames(H_common)[common_edges[, 1]],
    node2 = colnames(H_common)[common_edges[, 2]]
  )
  
  # Highlight common edges
  for (i in seq_len(nrow(common_edges))) {
    node1 <- common_edges[i, "node1"]
    node2 <- common_edges[i, "node2"]
    
    # Find the edge in the graph corresponding to these nodes
    edge_idx <- which((ends(graph, E(graph))[,1] == node1 & ends(graph, E(graph))[,2] == node2) |
                        (ends(graph, E(graph))[,1] == node2 & ends(graph, E(graph))[,2] == node1))
    
    if(length(edge_idx) > 0) {
      E(graph)$width[edge_idx] <- 0.8
    }
  }
  
  
  
  if (!is.null(H_common_kupffer_tcell)) {
    common_edges_kupffer_tcell <- data.frame(
      node1 = rownames(H_common_kupffer_tcell)[which(H_common_kupffer_tcell == 1, arr.ind = TRUE)[, 1]],
      node2 = colnames(H_common_kupffer_tcell)[which(H_common_kupffer_tcell == 1, arr.ind = TRUE)[, 2]]
    )

    for (i in seq_len(nrow(common_edges_kupffer_tcell))) {
      node1 <- common_edges_kupffer_tcell[i, "node1"]
      node2 <- common_edges_kupffer_tcell[i, "node2"]
      

      edge_idx <- which((ends(graph, E(graph))[,1] == node1 & ends(graph, E(graph))[,2] == node2) |
                          (ends(graph, E(graph))[,1] == node2 & ends(graph, E(graph))[,2] == node1))
      
      if(length(edge_idx) > 0) {
        E(graph)$width[edge_idx] <- 0.8  
      }
    }
  }
  

  degree_values <- degree(graph)
  hub_nodes <- names(sort(degree_values, decreasing = TRUE))[1:10]
  V(graph)$size <- ifelse(V(graph)$name %in% hub_nodes, 3, 1) 
  
  
  layout_fr <- layout_with_fr(graph, dim = 3, niter = 5000, grid = "nogrid")
  
  p0 <- ggraph(graph, layout = "manual", x = layout_fr[, 1], y = layout_fr[, 2]) +
    geom_edge_link(aes(color = I(color), width = I(width)), show.legend = FALSE) +
    geom_node_point(aes(color = I(color), size = I(size)), alpha = 1) +
    geom_node_text(aes(label = ifelse(name %in% hub_nodes, name, "")), size = 2, color = "black", repel = TRUE, max.overlaps = Inf) +
    scale_color_manual(values = col, breaks = c(names(col))) +
    scale_size_continuous(range = c(1, 5)) + 
    theme_void() +
    theme(
      plot.background = element_blank(),  
      panel.background = element_blank(), 
      legend.background = element_blank(), 
      legend.box.background = element_blank(),
      legend.position = "none",
      plot.title = element_text(hjust = 0.5)
    ) + labs(title = title)
  
  

  for (i in seq_along(top_communities)) {
    community_id <- as.numeric(top_communities[i])
    community_nodes <- V(graph)$name[membership == community_id]
    community_center <- layout_fr[match(community_nodes, V(graph)$name), , drop = FALSE]
    if (nrow(community_center) > 0) {
      p0 <- p0 + annotate("text", x = mean(community_center[, 1]), y = mean(community_center[, 2]), label = paste("Community", i), size = 0, color = community_colors[i])#size = 2
    }
  }
  
  ggsubnet <- function(target, edge_color, title = "Community") {
    subnet <- induced_subgraph(graph, target)
    subnet_nodes <- V(subnet)$name
    
    subnet_edges <- E(subnet)
    subnet_common_edges <- which(sapply(subnet_edges, function(e) {
      endpoints <- ends(subnet, e)
      common <- H_common[V(graph)$name == endpoints[1], V(graph)$name == endpoints[2]] == 1
      common
    }))
    

    common_gene_names <- unique(as.vector(ends(subnet, subnet_common_edges)))
    
    E(subnet)$color <- edge_color
    
    col <- c("Community" = edge_color, "Common" = "red", "Kupffer_Tcell" = "orange")
    
    V(subnet)$color <- ifelse(V(subnet)$name %in% common_gene_names, "red", 
                              ifelse(V(subnet)$name %in% common_nodes_kupffer_tcell, "orange", "grey70"))
    
    set.seed(21)
    f <- ggraph(subnet, layout = "fr") +
      geom_edge_link(aes(color = I(color)), edge_alpha = 0.9, show.legend = FALSE) +
      geom_node_point(aes(color = I(color)), size = 2, alpha = 1) +  # 调整节点大小和透明度，节点颜色设置为灰色或金色
      geom_node_text(aes(label = ifelse(name %in% common_gene_names| name %in% common_nodes_kupffer_tcell, name, "")), repel = TRUE, size = 2, max.overlaps = Inf)  + # 标注金色边相连的基因名称
      scale_color_manual(values = col, breaks = c("Community", "Common", "Kupffer_Tcell")) +
      theme_void() +
      theme(plot.title = element_text(hjust = 0.5, size = 10)) +
      labs(title = title)
    return(f)
  }
  

  plots <- lapply(seq_len(min(2, length(top_communities))), function(i) {
    community_id <- as.numeric(top_communities[i])
    community_nodes <- V(graph)$name[membership == community_id]
    ggsubnet(community_nodes, community_colors[i], title = paste("Community", i))
  })
  
  
  labels <- labels
  
  final_plot <- ggpubr::ggarrange(
    ggpubr::ggarrange(p0, nrow = 1, labels = labels[1], font.label = list(size = 10, hjust = -1, vjust = 1)),
    ggpubr::ggarrange(plots[[1]], plots[[2]], nrow = 1, ncol = 2, labels = c(labels[2], ""), font.label = list(size = 10, hjust = -1, vjust = 1)),
    nrow = 2, ncol = 1, heights = c(1.5, 1)
  )
  return(final_plot)
}

# Process each graph
final_plot_with_title1 <- process_graph(graph_Dendritic, H_common, community_colors, "Subgroup 1 (Dendritic)", c("A", "B"))
final_plot_with_title2 <- process_graph(graph_Erythroblast, H_common, community_colors, "Subgroup 2 (Erythroblast)", c("C", "D"))
final_plot_with_title3 <- process_graph(graph_Endothelial, H_common, community_colors, "Subgroup 3 (Endothelial)", c("E", "F"))
final_plot_with_title4 <- process_graph(graph_Kupffer, H_common, community_colors, "Subgroup 4 (Kupffer)", c("G", "H"), H_common_kupffer_tcell)
final_plot_with_title5 <- process_graph(graph_Tcell, H_common, community_colors, "Subgroup 5 (T cell)", c("I", "J"), H_common_kupffer_tcell)

# Collect plots into a list and remove NULL plots
plots_list <- list(final_plot_with_title1, final_plot_with_title2, final_plot_with_title3, final_plot_with_title4, final_plot_with_title5)
plots_list <- plots_list[!sapply(plots_list, is.null)]

# Combine non-NULL plots
final_combined_plot <- ggpubr::ggarrange(plotlist = plots_list, nrow = 1, ncol = length(plots_list))

# Add overall title
final_combined_plot_with_title <- ggpubr::annotate_figure(
  final_combined_plot
  #top = text_grob("Liver Gene Network", size = 16, face = "bold")
)

# Print the final combined plot with title
print(final_combined_plot_with_title)
ggsave("Liver Gene Network.png", final_combined_plot, width = 10, height = 6, bg = "white")


