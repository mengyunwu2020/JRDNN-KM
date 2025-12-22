
##############################################################
##################   Figure S8   ############################
##############################################################

########################## Note ##############################
## Please set the working directory to the source file 
## location.
##############################################################



library(here)

load(here("RealData", "Total Sample","result_data", "Estimated_networks", "adjmatrix_mescs.Rdata"))

library(igraph)
library(ggplot2)
library(ggraph)
library(gridExtra)
library(network)
library(ggnet)
library(cowplot)
library(ggpubr)

data2i <- as.matrix(adj_mescs$data2i)
a2i <- as.matrix(adj_mescs$H228)
lif <- as.matrix(adj_mescs$lif)

degree_data2i <- rowSums(data2i)
degree_a2i <- rowSums(a2i)
degree_lif <- rowSums(lif)


hub_data2i <- names(sort(degree_data2i, decreasing = TRUE))[1:30]
hub_a2i <- names(sort(degree_a2i, decreasing = TRUE))[1:30]
hub_lif <- names(sort(degree_lif, decreasing = TRUE))[1:30]
all_hub_intersection <- Reduce(intersect, list(hub_data2i, hub_a2i, hub_lif))

cat("各自前30个hub节点的交集: ", all_hub_intersection, "\n")


H_common <- (data2i == 1) & (a2i == 1) & (lif == 1)
H_common <- ifelse(H_common == TRUE, 1, 0)
sum(H_common == TRUE)

graph_data2i <- graph_from_adjacency_matrix(data2i, mode = "undirected", diag = FALSE)
graph_a2i <- graph_from_adjacency_matrix(a2i, mode = "undirected", diag = FALSE)
graph_lif <- graph_from_adjacency_matrix(lif, mode = "undirected", diag = FALSE)

set.seed(10)

community_colors <- c("#4DAF4A", "#377EB8", "#CD534CFF", "red", "#984EA3", "#FFFF33", "#A65628", "#F781BF", "#999999", "#E41A1C", "#377EB8", "#4DAF4A")


process_graph <- function(graph, H_common, community_colors, title, labels) {
  community <- cluster_louvain(graph, resolution =0.8)
  membership <- membership(community)
  community_sizes <- sizes(community)
  
  common_edges <- which(H_common == 1, arr.ind = TRUE)
  common_nodes <- unique(c(rownames(H_common)[common_edges[, 1]], colnames(H_common)[common_edges[, 2]]))
  
  top_communities <- order(community_sizes, decreasing = TRUE)[1:2]
  V(graph)$Type <- "Others"
  
  for (i in seq_along(top_communities)) {
    community_id <- as.numeric(top_communities[i])
    community_nodes <- V(graph)$name[membership == community_id]
    V(graph)$Type[V(graph)$name %in% community_nodes] <- paste("Community", i)
  }
  
  col <- setNames(community_colors[1:length(top_communities)], paste("Community", seq_along(top_communities)))
  col <- c(col, "Others" = "grey70", "Common" = "red", "Hub" = "gold")
  
  V(graph)$color <- ifelse(V(graph)$name %in% common_nodes, "red", "grey70")

  E(graph)$color <- "grey70"
  E(graph)$width <- 0.3
  
  for (i in seq_along(top_communities)) {
    community_id <- as.numeric(top_communities[i])
    community_edges <- which(membership[ends(graph, E(graph))[,1]] == community_id & membership[ends(graph, E(graph))[,2]] == community_id)
    E(graph)$color[community_edges] <- community_colors[i]
    E(graph)$width[community_edges] <- 0.3  
  }
  
  

  common_edges <- data.frame(
    node1 = rownames(H_common)[common_edges[, 1]],
    node2 = colnames(H_common)[common_edges[, 2]]
  )
  

  for (i in seq_len(nrow(common_edges))) {
    node1 <- common_edges[i, "node1"]
    node2 <- common_edges[i, "node2"]
    
    edge_idx <- which((ends(graph, E(graph))[,1] == node1 & ends(graph, E(graph))[,2] == node2) |
                        (ends(graph, E(graph))[,1] == node2 & ends(graph, E(graph))[,2] == node1))
    
    if(length(edge_idx) > 0) {
      E(graph)$width[edge_idx] <- 0.8
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
      print("Edge Endpoints:")
      print(endpoints)  
      
      common <- H_common[V(graph)$name == endpoints[1], V(graph)$name == endpoints[2]] == 1
      print("Is Common Edge:")
      print(common)  
      common
    }))
    

    common_gene_names <- unique(as.vector(ends(subnet, subnet_common_edges)))
    E(subnet)$color <- edge_color
    col <- c("Community" = edge_color, "Common" = "red")

    V(subnet)$color <- ifelse(V(subnet)$name %in% common_gene_names, "red", "grey70")
    

    set.seed(21)
    f <- ggraph(subnet, layout = "fr") +
      geom_edge_link(aes(color = I(color)), edge_alpha = 0.9, show.legend = FALSE) +
      geom_node_point(aes(color = I(color)), size = 2, alpha = 1) +  # 调整节点大小和透明度，节点颜色设置为灰色或金色
      geom_node_text(aes(label = ifelse(name %in% common_gene_names, name, "")), repel = TRUE, size = 2, max.overlaps = Inf)  + # 标注金色边相连的基因名称
      scale_color_manual(values = col, breaks = c("Community", "Common")) +
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


final_plot_with_title1 <- process_graph(graph_data2i, H_common, community_colors, "Subgroup 1 (2i)", c("A", "B"))
final_plot_with_title2 <- process_graph(graph_a2i, H_common, community_colors, "Subgroup 2 (a2i)", c("C", "D"))
final_plot_with_title3 <- process_graph(graph_lif, H_common, community_colors, "Subgroup 3 (lif)", c("E", "F"))



final_combined_plot <- ggpubr::ggarrange(
  ggpubr::ggarrange(final_plot_with_title1, final_plot_with_title2, final_plot_with_title3, nrow = 1, ncol = 3)
)

final_combined_plot_with_title <- ggpubr::annotate_figure(
  final_combined_plot,
  top = text_grob("mESCs Gene Network", size = 16, face = "bold")
)

print(final_combined_plot_with_title)
ggsave("mESCs Gene Network.png", final_combined_plot, width = 10, height = 6, bg = "white")

