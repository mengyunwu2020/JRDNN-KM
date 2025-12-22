
##############################################################
########################   Figure 2A   #######################
##############################################################

########################## Note ##############################
## Please set the working directory to the source file 
## location.
##############################################################


library(ggplot2)
library(cowplot)


ari_methods <- rep(c('JRDNN-KM', 'BLGGM', 'SC3', 'Seurat'), each=5)
ari_cell_types <- rep(c('LUAD', 'mESCs', 'Mouse Liver', 'PBMC', 'Mouse Uterus'), times=4)
ari_values <- c(1.000, 0.946, 0.729, 0.953, 0.86,
                0.904, 0.724, 0.762, 0.82, 0.81, 
                0.635, 0.584, 0.650, 0.69, 0.673, 
                0.692, 0.498, 0.672, 0.74, 0.634)
ari_data <- data.frame(Methods = ari_methods, Cell_Types = ari_cell_types, ARI = ari_values)


nmi_methods <- rep(c('JRDNN-KM', 'BLGGM', 'SC3', 'Seurat'), each=5)
nmi_cell_types <- rep(c('LUAD', 'mESCs', 'Mouse Liver', 'PBMC', 'Mouse Uterus'), times=4)
nmi_values <- c(1.000, 0.949, 0.873, 0.953, 0.913,
                0.926, 0.863, 0.886, 0.882, 0.868, 
                0.757, 0.716, 0.763, 0.785, 0.768, 
                0.791, 0.678, 0.774, 0.821, 0.739)
nmi_data <- data.frame(Methods = nmi_methods, Cell_Types = nmi_cell_types, NMI = nmi_values)


color_palette <- c('JRDNN-KM' = '#377eb8', 'BLGGM' = '#4daf4a', 'SC3' = '#e41a1c', 'Seurat' = '#ff7f00')


ari_data$Cell_Types <- factor(ari_data$Cell_Types, levels = c('LUAD', 'PBMC', 'mESCs', 'Mouse Liver', 'Mouse Uterus'))
ari_data$Methods <- factor(ari_data$Methods, levels = c('JRDNN-KM', 'BLGGM', 'SC3', 'Seurat'))

nmi_data$Cell_Types <- factor(nmi_data$Cell_Types, levels = c('LUAD', 'PBMC', 'mESCs', 'Mouse Liver', 'Mouse Uterus'))
nmi_data$Methods <- factor(nmi_data$Methods, levels = c('JRDNN-KM', 'BLGGM', 'SC3', 'Seurat'))


p_ari <- ggplot(ari_data, aes(x = Cell_Types, y = ARI, fill = Methods)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
  scale_fill_manual(values = color_palette) +
  theme_bw() +
  labs(y = "Adjusted Rand Index (ARI)") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    legend.title = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 16),
    axis.text.y = element_text(size = 16),    
    axis.text.x = element_blank(),  
    axis.ticks.x = element_blank(), 
    legend.text = element_text(size = 16),
    legend.position = "top",
    legend.direction = "horizontal",  
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )


p_nmi <- ggplot(nmi_data, aes(x = Cell_Types, y = NMI, fill = Methods)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
  scale_fill_manual(values = color_palette) +
  theme_bw() +
  labs(x = "Datasets", y = "Normalized Mutual Information (NMI)") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    legend.title = element_blank(),
    legend.position = "none",  
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.y = element_text(size = 16),    
    axis.text.x = element_text(angle = 45, hjust = 1, size = 16),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )



combined_plot <- plot_grid(
  p_ari,
  p_nmi,
  ncol = 1,
  rel_heights = c(0.4, 0.45)  
)


print(combined_plot)

ggsave("ARI_NMI_Combined_Plots.png", plot = combined_plot, width = 10, height = 12, dpi = 300)    








