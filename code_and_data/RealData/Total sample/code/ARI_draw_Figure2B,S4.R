##############################################################
########################   Figure 2B   #######################
##############################################################

########################## Note ##############################
## Please set the working directory to the source file 
## location.
##############################################################
library(here)

library(png)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(grid)
library(cowplot)


data_jrdnn  <- read.csv(here("RealData","Total Sample", "result_data", "subgroup_results_JRDNN_KM.csv"))
data_blggm  <- read.csv(here("RealData","Total Sample", "result_data", "subgroup_results_BLGGM.csv"))
data_sc3    <- read.csv(here("RealData","Total Sample", "result_data", "subgroup_results_SC3.csv"))
data_seurat <- read.csv(here("RealData","Total Sample", "result_data", "subgroup_results_Seurat.csv"))

process_data <- function(data) {
  colnames(data)[colnames(data) == "X1"] <- "1"
  colnames(data)[colnames(data) == "X2"] <- "2"
  colnames(data)[colnames(data) == "X3"] <- "3"
  colnames(data)[colnames(data) == "X4"] <- "4"
  colnames(data)[colnames(data) == "X5"] <- "5"
  colnames(data)[colnames(data) == "X6"] <- "6"
  
  data$Dataset <- factor(c(rep("LUAD", 3), rep("PBMC", 3), rep("mESCs", 3), rep("Mouse Liver", 5), rep("Mouse Uterus", 6)),
                         levels = c("LUAD", "PBMC", "mESCs", "Mouse Liver","Mouse Uterus"))
  
  data_long <- data %>%
    pivot_longer(cols = c('1','2','3','4','5','6'), 
                 names_to = 'Subgroup', 
                 values_to = 'Proportion') %>%
    filter(Proportion > 0)
  
  return(data_long)
}

data_long_jrdnn <- process_data(data_jrdnn)
data_long_blggm <- process_data(data_blggm)
data_long_sc3 <- process_data(data_sc3)
data_long_seurat <- process_data(data_seurat)

custom_colors <- c("red", "#4DAF4A", "#377EB8", "#FFFF33", "orange","#984EA3")

create_barplot <- function(data_long, dataset) {
  ggplot(data_long %>% filter(Dataset == dataset), 
         aes(x = Group, y = Proportion, fill = Subgroup)) +
    geom_col(position = "stack", width = 0.7) +
    scale_fill_manual(values = custom_colors) +
    theme_minimal() +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      axis.text.x = element_text(size = 24, angle = 45, hjust = 1, margin = margin(t = 5)),
      # 为y轴文本添加angle=45，实现倾斜
      axis.text.y = element_text(size = 24, angle = 45, hjust = 1, margin = margin(r = 2)),
      legend.position = "none",
      panel.spacing = unit(0.01, "lines"),
      strip.text = element_text(size = 24)
    ) +
    coord_flip() +
    facet_wrap(~ Dataset, scales = "free", ncol = 1)
}

create_colorbar <- function(dataset) {
  subgroup_labels <- switch(dataset,
                            "LUAD" = c("Subgroup 1", "Subgroup 2", "Subgroup 3"),
                            "PBMC" = c("Subgroup 1", "Subgroup 2", "Subgroup 3"),
                            "mESCs" = c("Subgroup 1", "Subgroup 2", "Subgroup 3"),
                            "Mouse Liver" = c("Subgroup 1", "Subgroup 2", "Subgroup 3", "Subgroup 4", "Subgroup 5"),
                            "Mouse Uterus" = c("Subgroup 1", "Subgroup 2", "Subgroup 3", "Subgroup 4", "Subgroup 5", "Subgroup 6")
  )
  
  num_colors <- length(subgroup_labels)
  colors <- custom_colors[1:num_colors]
  
  dummy_plot <- ggplot() +
    scale_color_manual(values = colors, labels = subgroup_labels) +
    geom_point(aes(x = 1, y = 1, color = factor(subgroup_labels)), 
               shape = 15, size = 0) +
    guides(color = guide_legend(override.aes = list(size = 4))) +
    theme_void() +
    theme(legend.position = "top",
          legend.title = element_blank(),
          legend.text = element_text(size = 20),
          legend.box.margin = margin(t = 0, b = 0, r = 80),
          legend.spacing.x = unit(0.5, "cm"),
          legend.key.size = unit(0.8, "cm"))
  
  return(dummy_plot)
}

plots_with_legends_jrdnn <- lapply(levels(data_long_jrdnn$Dataset), function(dataset) {
  bar_plot <- create_barplot(data_long_jrdnn, dataset)
  colorbar <- create_colorbar(dataset)
  wrap_plots(bar_plot, colorbar, ncol = 1, heights = c(10, 1))
})
plots_jrdnn <- wrap_plots(plots_with_legends_jrdnn, ncol = 2)

print(plots_jrdnn)
ggsave("ARI_Proportion.png", plots_jrdnn, width = 30, height = 30, dpi = 300)







##############################################################
########################   Figure S4   #######################
##############################################################


plots_with_legends_blggm <- lapply(levels(data_long_blggm$Dataset), function(dataset) {
  bar_plot <- create_barplot(data_long_blggm, dataset)
  colorbar <- create_colorbar(dataset)
  wrap_plots(bar_plot, colorbar, ncol = 1, heights = c(10, 1))
})
plots_blggm <- wrap_plots(plots_with_legends_blggm, ncol = 5) + 
  plot_annotation(
    title = "BLGGM",
    theme = theme(plot.title = element_text(hjust = 0.5, size = 24, face = "bold"))
  )

plots_with_legends_sc3 <- lapply(levels(data_long_sc3$Dataset), function(dataset) {
  bar_plot <- create_barplot(data_long_sc3, dataset)
  colorbar <- create_colorbar(dataset)
  wrap_plots(bar_plot, colorbar, ncol = 1, heights = c(10, 1))
})
plots_sc3 <- wrap_plots(plots_with_legends_sc3, ncol = 5) + 
  plot_annotation(
    title = "SC3",
    theme = theme(plot.title = element_text(hjust = 0.5, size = 24, face = "bold"))
  )

plots_with_legends_seurat <- lapply(levels(data_long_seurat$Dataset), function(dataset) {
  bar_plot <- create_barplot(data_long_seurat, dataset)
  colorbar <- create_colorbar(dataset)
  wrap_plots(bar_plot, colorbar, ncol = 1, heights = c(10, 1))
})
plots_seurat <- wrap_plots(plots_with_legends_seurat, ncol = 5) + 
  plot_annotation(
    title = "Seurat",
    theme = theme(plot.title = element_text(hjust = 0.5, size = 24, face = "bold"))
  )

plot1_with_label <- ggdraw() + 
  draw_plot(plots_blggm) +
  draw_label("A", x = 0, y = 1, hjust = -0.1, vjust = 1, size = 24, fontface = "bold")

plot2_with_label <- ggdraw() + 
  draw_plot(plots_sc3) +
  draw_label("B", x = 0, y = 1, hjust = -0.1, vjust = 1, size = 24, fontface = "bold")

plot3_with_label <- ggdraw() + 
  draw_plot(plots_seurat) +
  draw_label("C", x = 0, y = 1, hjust = -0.1, vjust = 1, size = 24, fontface = "bold")

final_plot <- plot_grid(
  plot1_with_label,
  plot2_with_label,
  plot3_with_label,
  ncol = 1,
  rel_heights = c(1, 1, 1)
)

print(final_plot)
ggsave("ARI_others.png", final_plot, width = 40, height = 25, dpi = 300)
