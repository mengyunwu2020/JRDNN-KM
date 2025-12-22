
##############################################################
########################   Figure S2   #######################
##############################################################

########################## Note ##############################
## Please set the working directory to the source file 
## location.
##############################################################
library(here)

load(here("RealData","Total Sample", "result_data", "centers_evolution.RData"))
load(here("RealData","Total Sample", "result_data", "all_experiments.Rdata"))


library(Rtsne)
library(ggplot2)
library(patchwork)
library(dplyr)

set.seed(123)
n_dim <- 100
n_clusters <- 3
n_iter <- 20
n_samples <- 3000
n_experiments <- 10





plot_data <- bind_rows(all_experiments)


summary_data <- plot_data %>%
  group_by(cluster, iter) %>%
  summarise(
    dim1_median = mean(dim1),
    dim1_lower = quantile(dim1, 0.025),
    dim1_upper = quantile(dim1, 0.975),
    dim2_median = mean(dim2),
    dim2_lower = quantile(dim2, 0.025),
    dim2_upper = quantile(dim2, 0.975),
    .groups = "drop"
  )


plot_list <- lapply(1:n_clusters, function(k) {
  class_data <- filter(summary_data, cluster == paste0("Class", k))
  raw_data <- filter(plot_data, cluster == paste0("Class", k))
  
  ggplot() +
   
    geom_point(data = raw_data, 
               aes(x = iter, y = dim1, color = "Dimension 1"), 
               alpha = 0.15, position = position_jitter(width = 0.2)) +
    geom_point(data = raw_data, 
               aes(x = iter, y = dim2, color = "Dimension 2"), 
               alpha = 0.15, position = position_jitter(width = 0.2)) +
   
    geom_ribbon(data = class_data,
                aes(x = iter, ymin = dim1_lower, ymax = dim1_upper),
                fill = "#E69F00", alpha = 0.3) +
    
    geom_ribbon(data = class_data,
                aes(x = iter, ymin = dim2_lower, ymax = dim2_upper),
                fill = "#56B4E9", alpha = 0.3) +
    
    geom_line(data = class_data,
              aes(x = iter, y = dim1_median, color = "Dimension 1"),
              linewidth = 1.2) +
    geom_line(data = class_data,
              aes(x = iter, y = dim2_median, color = "Dimension 2"),
              linewidth = 1.2) +
    scale_color_manual(values = c("Dimension 1" = "#E69F00",
                                  "Dimension 2" = "#56B4E9")) +
    labs(title = paste("Class", k, "Center Evolution (95% Range)"),
         x = "Iteration",
         y = "t-SNE Coordinate Value",
         color = "Dimension") +
    theme_bw() +
    theme(legend.position = "bottom",
          plot.title = element_text(size = 14),
          axis.text = element_text(size = 14))
})


final_plot <- wrap_plots(plot_list, ncol = 3) + 
  plot_annotation(
    theme = theme(plot.title = element_text(hjust = 0.5, size = 14))
  )

final_plot


