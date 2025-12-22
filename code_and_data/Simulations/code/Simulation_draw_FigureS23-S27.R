########################## Note ##############################
## Root directory is set to: code_and_data
## Please ensure all data folders are within this root.
##############################################################

##############################################################
##################    Figure S23, Figure S24    ##############
##############################################################



library(ggplot2)
library(dplyr)
library(readxl)
library(tidyr)
library(patchwork)
library(png)
library(grid)
library(here) # Load here package


custom_colors <- c(
  "JRDNN-KM" = "#377EB8",   # Blue
  "SC3" = "red",            # Red
  "BLGGM" = "#4DAF4A",      # Green
  "Seurat" = "orange",      # Orange
  "GENIE3" = "#FFCC00",     # Purple
  "JGNsc" = "#80B7FF",      # Dark Orange
  "JSEM" = "#778899",       # Grey
  "locCSN" = "#8FE506",     # Light Green
  "SpQN" = "#F781BF",       # Pink
  "Normalisr" = "#984EA3",  # Violet
  "CS-CORE" = "#A65628"     # Brown
)


ari_methods <- c("JRDNN", "SC3", "BLGGM", "Seurat")
f1_methods <- c("JRDNN", "BLGGM",  "JGNsc", "JSEM","SpQN","Normalisr","GENIE3", "locCSN","CS-CORE")


load_and_process_data <- function(base_path, subfolders) {
  all_data <- list()
  
  for (folder in subfolders) {
    
    files <- list.files(
      path = file.path(base_path, folder), 
      pattern = "*.xlsx", 
      full.names = TRUE
    )
    files <- files[!grepl("^~\\$", basename(files))]  
    
    if (length(files) == 0) next
    
    data_list <- lapply(files, function(file) {
      data <- read_excel(file)
      method <- tools::file_path_sans_ext(basename(file))
      data <- data %>% mutate(Method = method)
    })
    
    combined_data <- bind_rows(data_list)
    
    parts <- unlist(strsplit(folder, "_"))
    long_data <- combined_data %>%
      pivot_longer(cols = -Method, names_to = "Setting", values_to = "Value") %>%
      mutate(NetworkType = parts[2], Level = parts[3], Complexity = parts[4])
    
    all_data[[folder]] <- long_data
  }
  
  return(bind_rows(all_data))
}


# Modification: Use here() for paths relative to code_and_data root
balance_base_path <- here("Simulations", "result_data", "simulation result", "balance")
balance_subfolders <- c(
  "balance_scalefree_high_complex", "balance_scalefree_high_medium", 
  "balance_scalefree_high_simple", "balance_scalefree_low_complex", 
  "balance_scalefree_low_medium", "balance_scalefree_low_simple", 
  "balance_scalefree_medium_complex", "balance_scalefree_medium_medium", 
  "balance_scalefree_medium_simple", 
  
  "balance_starchain_high_complex", "balance_starchain_high_medium", 
  "balance_starchain_high_simple", "balance_starchain_low_complex", 
  "balance_starchain_low_medium", "balance_starchain_low_simple", 
  "balance_starchain_medium_complex", "balance_starchain_medium_medium", 
  "balance_starchain_medium_simple", 
  
  "balance_twohubs_high_complex", "balance_twohubs_high_medium", 
  "balance_twohubs_high_simple", "balance_twohubs_low_complex", 
  "balance_twohubs_low_medium", "balance_twohubs_low_simple", 
  "balance_twohubs_medium_complex", "balance_twohubs_medium_medium", 
  "balance_twohubs_medium_simple"
)

imbalance_base_path <- here("Simulations", "result_data", "simulation result", "imbalance")
imbalance_subfolders <- c(
  
  "imbalance_scalefree_high_complex", "imbalance_scalefree_high_medium", 
  "imbalance_scalefree_high_simple", "imbalance_scalefree_low_complex", 
  "imbalance_scalefree_low_medium", "imbalance_scalefree_low_simple", 
  "imbalance_scalefree_medium_complex", "imbalance_scalefree_medium_medium", 
  "imbalance_scalefree_medium_simple", 
  
  "imbalance_starchain_high_complex", "imbalance_starchain_high_medium", 
  "imbalance_starchain_high_simple", "imbalance_starchain_low_complex", 
  "imbalance_starchain_low_medium", "imbalance_starchain_low_simple", 
  "imbalance_starchain_medium_complex", "imbalance_starchain_medium_medium", 
  "imbalance_starchain_medium_simple", 
  
  "imbalance_twohubs_high_complex", "imbalance_twohubs_high_medium", 
  "imbalance_twohubs_high_simple", "imbalance_twohubs_low_complex", 
  "imbalance_twohubs_low_medium", "imbalance_twohubs_low_simple", 
  "imbalance_twohubs_medium_complex", "imbalance_twohubs_medium_medium", 
  "imbalance_twohubs_medium_simple"
)

balance_processed_data <- load_and_process_data(balance_base_path, balance_subfolders)
imbalance_processed_data <- load_and_process_data(imbalance_base_path, imbalance_subfolders)

filtered_data_ARI <- balance_processed_data %>%
  filter(Method %in% ari_methods) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(NetworkType = case_when(
    NetworkType == "starchain" ~ "star-chain",
    NetworkType == "scalefree" ~ "scale-free",
    NetworkType == "twohubs" ~ "stochastic-block",  
    TRUE ~ NetworkType
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM", "SC3", "Seurat"))) %>%
  filter(Setting == "ARI") %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Complexity = factor(Complexity, levels = c("simple", "medium", "complex"))) %>%
  mutate(NetworkType = factor(NetworkType, levels = c("star-chain", "scale-free", "stochastic-block")))  


filtered_data_F1 <- balance_processed_data %>%
  filter(Method %in% f1_methods) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(NetworkType = case_when(
    NetworkType == "starchain" ~ "star-chain",
    NetworkType == "scalefree" ~ "scale-free",
    NetworkType == "twohubs" ~ "stochastic-block",  
    TRUE ~ NetworkType
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM",  "JGNsc", "JSEM","SpQN","Normalisr","GENIE3", "locCSN","CS-CORE")))  %>%
  filter(Setting == "F1") %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Complexity = factor(Complexity, levels = c("simple", "medium", "complex"))) %>%
  mutate(NetworkType = factor(NetworkType, levels = c("star-chain", "scale-free", "stochastic-block")))  


imbalance_filtered_data_ARI <- imbalance_processed_data %>%
  filter(Method %in% ari_methods) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(NetworkType = case_when(
    NetworkType == "starchain" ~ "star-chain",
    NetworkType == "scalefree" ~ "scale-free",
    NetworkType == "twohubs" ~ "stochastic-block",  
    TRUE ~ NetworkType
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM", "SC3", "Seurat"))) %>%
  filter(Setting == "ARI") %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Complexity = factor(Complexity, levels = c("simple", "medium", "complex"))) %>%
  mutate(NetworkType = factor(NetworkType, levels = c("star-chain", "scale-free", "stochastic-block")))  


imbalance_filtered_data_F1 <- imbalance_processed_data %>%
  filter(Method %in% f1_methods) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(NetworkType = case_when(
    NetworkType == "starchain" ~ "star-chain",
    NetworkType == "scalefree" ~ "scale-free",
    NetworkType == "twohubs" ~ "stochastic-block",  
    TRUE ~ NetworkType
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM",  "JGNsc", "JSEM","SpQN","Normalisr","GENIE3", "locCSN","CS-CORE"))) %>%
  filter(Setting == "F1") %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Complexity = factor(Complexity, levels = c("simple", "medium", "complex"))) %>%
  mutate(NetworkType = factor(NetworkType, levels = c("star-chain", "scale-free", "stochastic-block")))  



balance_plot_ARI <- ggplot(filtered_data_ARI, aes(x = Level, y = Value, fill = Method, color = Method)) +
  geom_boxplot(position = position_dodge(width = 0.5), width = 0.4, outlier.size = 0.5) +
  facet_grid(rows = vars(NetworkType), cols = vars(Complexity), scales = "free_y") +  
  labs(title = "Balance",
       x = "Zero Inflation Level",
       y = "ARI Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.position = "none",
    legend.title = element_blank(),
    strip.text = element_text(size = 12)
  ) +
  scale_fill_manual(values = custom_colors) +
  scale_color_manual(values = custom_colors)+
  scale_y_continuous(breaks = c(0.875, 0.9, 0.925, 0.95, 0.975, 1), limits = c(0.875, 1))  

imbalance_plot_ARI <- ggplot(imbalance_filtered_data_ARI, aes(x = Level, y = Value, fill = Method, color = Method)) +
  geom_boxplot(position = position_dodge(width = 0.5), width = 0.4, outlier.size = 0.5) +
  facet_grid(rows = vars(NetworkType), cols = vars(Complexity), scales = "free_y") +  
  labs(title = "Imbalance",
       x = "Zero Inflation Level",
       y = "ARI Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.position = "right",
    legend.title = element_blank(),
    strip.text = element_text(size = 12)
  ) +
  scale_fill_manual(values = custom_colors) +
  scale_color_manual(values = custom_colors)+
  scale_y_continuous(breaks = c(0.875, 0.9, 0.925, 0.95, 0.975, 1), limits = c(0.875, 1))  

balance_plot_F1 <- ggplot(filtered_data_F1, aes(x = Level, y = Value, fill = Method, color = Method)) +
  geom_boxplot(position = position_dodge(width = 0.5), width = 0.4, outlier.size = 0.5) +
  facet_grid(rows = vars(NetworkType), cols = vars(Complexity), scales = "free_y") +
  labs(title = "Balance",
       x = "Zero Inflation Level",
       y = "F1 Score Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    legend.position = "none",
    legend.title = element_blank(),
    strip.text = element_text(size = 12)
  ) +
  scale_fill_manual(values = custom_colors) +
  scale_color_manual(values = custom_colors)+
  scale_y_continuous(breaks = c(0.3,0.4, 0.5, 0.6, 0.7, 0.8), limits = c(0.3, 0.8))  

imbalance_plot_F1 <- ggplot(imbalance_filtered_data_F1, aes(x = Level, y = Value, fill = Method, color = Method)) +
  geom_boxplot(position = position_dodge(width = 0.5), width = 0.4, outlier.size = 0.5) +
  facet_grid(rows = vars(NetworkType), cols = vars(Complexity), scales = "free_y") +
  labs(title = "Imbalance",
       x = "Zero Inflation Level",
       y = "F1 Score Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12), 
    axis.text.y = element_text(size = 12), 
    legend.position = "right",
    legend.title = element_blank(),
    strip.text = element_text(size = 12)
  ) +
  scale_fill_manual(values = custom_colors) +
  scale_color_manual(values = custom_colors)+
  scale_y_continuous(breaks = c(0.3,0.4, 0.5, 0.6, 0.7, 0.8), limits = c(0.3, 0.8))  

final_plot_ARI <- 
  (balance_plot_ARI / imbalance_plot_ARI) +  
  plot_annotation(tag_levels = "A")          

final_plot_F1 <- 
  (balance_plot_F1 / imbalance_plot_F1) +    
  plot_annotation(tag_levels = "A")          


# Save figures using here() path
ggsave(filename = here("Simulations", "result_data", "ARI.png"), 
       plot = final_plot_ARI, 
       width = 10, 
       height = 10, 
       dpi = 800)

ggsave(filename = here("Simulations", "result_data", "F1.png"), 
       plot = final_plot_F1, 
       width = 10, 
       height = 10, 
       dpi = 800)


##############################################################
##################    Figure S25, S26    #####################
##############################################################


filtered_data_Recall <- balance_processed_data %>%
  filter(Method %in% c("BLGGM", "GENIE3", "JGNsc", "JRDNN", "JSEM", "locCSN","SpQN","Normalisr","CS-CORE")) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(NetworkType = case_when(
    NetworkType == "starchain" ~ "star-chain",
    NetworkType == "scalefree" ~ "scale-free",
    NetworkType == "twohubs" ~ "stochastic-block",  
    TRUE ~ NetworkType
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM",  "JGNsc", "JSEM","SpQN","Normalisr","GENIE3", "locCSN","CS-CORE"))) %>%
  filter(Setting == "Recall") %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Complexity = factor(Complexity, levels = c("simple", "medium", "complex"))) %>%
  mutate(NetworkType = factor(NetworkType, levels = c("star-chain", "scale-free", "stochastic-block")))  


filtered_data_Precision <- balance_processed_data %>%
  filter(Method %in% c("BLGGM", "GENIE3", "JGNsc", "JRDNN", "JSEM", "locCSN","SpQN","Normalisr","CS-CORE")) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(NetworkType = case_when(
    NetworkType == "starchain" ~ "star-chain",
    NetworkType == "scalefree" ~ "scale-free",
    NetworkType == "twohubs" ~ "stochastic-block",  
    TRUE ~ NetworkType
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM",  "JGNsc", "JSEM","SpQN","Normalisr","GENIE3", "locCSN","CS-CORE"))) %>%
  filter(Setting == "Precision") %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Complexity = factor(Complexity, levels = c("simple", "medium", "complex"))) %>%
  mutate(NetworkType = factor(NetworkType, levels = c("star-chain", "scale-free", "stochastic-block")))  



imbalance_filtered_data_Recall <- imbalance_processed_data %>%
  filter(Method %in% c("BLGGM", "GENIE3", "JGNsc", "JRDNN", "JSEM", "locCSN","SpQN","Normalisr","CS-CORE")) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(NetworkType = case_when(
    NetworkType == "starchain" ~ "star-chain",
    NetworkType == "scalefree" ~ "scale-free",
    NetworkType == "twohubs" ~ "stochastic-block", 
    TRUE ~ NetworkType
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM",  "JGNsc", "JSEM","SpQN","Normalisr","GENIE3", "locCSN","CS-CORE"))) %>%
  filter(Setting == "Recall") %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Complexity = factor(Complexity, levels = c("simple", "medium", "complex"))) %>%
  mutate(NetworkType = factor(NetworkType, levels = c("star-chain", "scale-free", "stochastic-block")))  


imbalance_filtered_data_Precision <- imbalance_processed_data %>%
  filter(Method %in% c("BLGGM", "GENIE3", "JGNsc", "JRDNN", "JSEM", "locCSN","SpQN","Normalisr","CS-CORE")) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(NetworkType = case_when(
    NetworkType == "starchain" ~ "star-chain",
    NetworkType == "scalefree" ~ "scale-free",
    NetworkType == "twohubs" ~ "stochastic-block",  
    TRUE ~ NetworkType
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM",  "JGNsc", "JSEM","SpQN","Normalisr","GENIE3", "locCSN","CS-CORE"))) %>%
  filter(Setting == "Precision") %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Complexity = factor(Complexity, levels = c("simple", "medium", "complex"))) %>%
  mutate(NetworkType = factor(NetworkType, levels = c("star-chain", "scale-free", "stochastic-block"))) 


balance_plot_Recall <- ggplot(filtered_data_Recall, aes(x = Level, y = Value, fill = Method, color = Method)) +
  geom_boxplot(position = position_dodge(width = 0.5), width = 0.4, outlier.size = 0.5) +
  facet_grid(rows = vars(NetworkType), cols = vars(Complexity), scales = "free_y") +  
  labs(title = "Balance",
       x = "Zero Inflation Level",
       y = "Recall Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),  
    axis.text.y = element_text(size = 12),  
    legend.position = "none",
    legend.title = element_blank(),
    strip.text = element_text(size = 12)
  ) +
  scale_fill_manual(values = custom_colors) +
  scale_color_manual(values = custom_colors)+
  scale_y_continuous(breaks = c(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), limits = c(0.4, 0.9))  



imbalance_plot_Recall <- ggplot(imbalance_filtered_data_Recall, aes(x = Level, y = Value, fill = Method, color = Method)) +
  geom_boxplot(position = position_dodge(width = 0.5), width = 0.4, outlier.size = 0.5) +
  facet_grid(rows = vars(NetworkType), cols = vars(Complexity), scales = "free_y") +  
  labs(title = "Imbalance",
       x = "Zero Inflation Level",
       y = "Recall Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.position = "right",
    legend.title = element_blank(),
    strip.text = element_text(size = 12)
  ) +
  scale_fill_manual(values = custom_colors) +
  scale_color_manual(values = custom_colors)+
  scale_y_continuous(breaks = c(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), limits = c(0.4, 0.9))  



balance_plot_Precision <- ggplot(filtered_data_Precision, aes(x = Level, y = Value, fill = Method, color = Method)) +
  geom_boxplot(position = position_dodge(width = 0.5), width = 0.4, outlier.size = 0.5) +
  facet_grid(rows = vars(NetworkType), cols = vars(Complexity), scales = "free_y") + 
  labs(title = "Balance",
       x = "Zero Inflation Level",
       y = "Precision Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.position = "none",
    legend.title = element_blank(),
    strip.text = element_text(size = 12)
  ) +
  scale_fill_manual(values = custom_colors) +
  scale_color_manual(values = custom_colors)+
  scale_y_continuous(breaks = c(0.3,0.4, 0.5, 0.6, 0.7, 0.8), limits = c(0.3, 0.8)) 


imbalance_plot_Precision <- ggplot(imbalance_filtered_data_Precision, aes(x = Level, y = Value, fill = Method, color = Method)) +
  geom_boxplot(position = position_dodge(width = 0.5), width = 0.4, outlier.size = 0.5) +
  facet_grid(rows = vars(NetworkType), cols = vars(Complexity), scales = "free_y") +  
  labs(title = "Imbalance",
       x = "Zero Inflation Level",
       y = "Precision Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.position = "right",
    legend.title = element_blank(),
    strip.text = element_text(size = 12)
  ) +
  scale_fill_manual(values = custom_colors) +
  scale_color_manual(values = custom_colors)+
  scale_y_continuous(breaks = c(0.3,0.4, 0.5, 0.6, 0.7, 0.8), limits = c(0.3, 0.8))  

final_plot_Recall <- 
  (balance_plot_Recall / imbalance_plot_Recall) +
  plot_annotation(tag_levels = "A")  

final_plot_Precision <- 
  (balance_plot_Precision / imbalance_plot_Precision) +
  plot_annotation(tag_levels = "A")  


final_plot_Recall
final_plot_Precision

ggsave(here("Simulations", "result_data", "Recall.png"), final_plot_Recall, width = 10, height = 10, dpi = 800)
ggsave(here("Simulations", "result_data", "Precision.png"), final_plot_Precision, width = 10, height = 10, dpi = 800)



##############################################################
####################    Figure S27    ########################
##############################################################

custom_colors <- c(
  "JRDNN-KM" = "#377EB8",   # Blue
  "SC3" = "red",            # Red
  "BLGGM" = "#4DAF4A",      # Green
  "Seurat" = "orange",      # Orange
  "GENIE3" = "#FFCC00",     # Purple
  "JGNsc" = "#80B7FF",      # Dark Orange
  "JSEM" = "#778899",       # Grey
  "locCSN" = "#8FE506",     # Light Green
  "SpQN" = "#F781BF",       # Pink
  "Normalisr" = "#984EA3",  # Violet
  "CS-CORE" = "#A65628"     # Brown
)


ari_methods <- c("JRDNN", "SC3", "BLGGM", "Seurat")
other_methods <- c("JRDNN", "BLGGM", "GENIE3", "JGNsc", "JSEM", "locCSN","SpQN","Normalisr","CS-CORE")


# Modification: Use here() for paths relative to code_and_data root
base_path <- here("Simulations", "result_data", "simulation result", "linear")

subfolders <- c(
  "n=3000_high", "n=3000_low", "n=3000_median",
  "n=6000_high", "n=6000_low", "n=6000_median",
  "n=9000_high", "n=9000_low", "n=9000_median"
)


load_and_process_data <- function(base_path, subfolders) {
  all_data <- list()
  
  for (folder in subfolders) {
    files <- list.files(path = file.path(base_path, folder), pattern = "*.xlsx", full.names = TRUE)
    
    
    data_list <- lapply(files, function(file) {
      data <- read_excel(file)
      method <- tools::file_path_sans_ext(basename(file))
      data <- data %>% mutate(Method = method)
    })
    
    
    combined_data <- bind_rows(data_list)
    
    
    parts <- unlist(strsplit(folder, "_"))
    long_data <- combined_data %>%
      pivot_longer(cols = -Method, names_to = "Setting", values_to = "Value") %>%
      mutate(SampleSize = parts[1], Level = parts[2])
    
    all_data[[folder]] <- long_data
  }
  
  return(bind_rows(all_data))
}


processed_data <- load_and_process_data(base_path, subfolders)
processed_data <- processed_data %>%
  mutate(Level = replace(Level, Level == "median", "medium"))


filtered_data_ARI <- processed_data %>%
  filter(Method %in% ari_methods) %>%
  filter(Setting == "ARI") %>%
  mutate(SampleSize = factor(SampleSize, levels = c("n=3000", "n=6000", "n=9000"))) %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Method = factor(Method, levels = ari_methods)) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM", "SC3", "Seurat")))



filtered_data_F1 <- processed_data %>%
  filter(Method %in% other_methods) %>%
  filter(Setting == "F1") %>%
  mutate(SampleSize = factor(SampleSize, levels = c("n=3000", "n=6000", "n=9000"))) %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Method = factor(Method, levels = other_methods)) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM",  "JGNsc", "JSEM","SpQN","Normalisr","GENIE3", "locCSN","CS-CORE")))






filtered_data_Recall <- processed_data %>%
  filter(Method %in% other_methods) %>%
  filter(Setting == "Recall") %>%
  mutate(SampleSize = factor(SampleSize, levels = c("n=3000", "n=6000", "n=9000"))) %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Method = factor(Method, levels = other_methods)) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM",  "JGNsc", "JSEM","SpQN","Normalisr","GENIE3", "locCSN","CS-CORE")))





filtered_data_Precision <- processed_data %>%
  filter(Method %in% other_methods) %>%
  filter(Setting == "Precision") %>%
  mutate(SampleSize = factor(SampleSize, levels = c("n=3000", "n=6000", "n=9000"))) %>%
  mutate(Level = factor(Level, levels = c("low", "medium", "high"))) %>%
  mutate(Method = factor(Method, levels = other_methods)) %>%
  mutate(Method = case_when(
    Method == "JRDNN" ~ "JRDNN-KM",
    TRUE ~ Method
  )) %>%
  mutate(Method = factor(Method, levels = c("JRDNN-KM", "BLGGM",  "JGNsc", "JSEM","SpQN","Normalisr","GENIE3", "locCSN","CS-CORE")))





plot_metric <- function(data, y_label,  y_breaks, y_limits,show_legend = FALSE) {
  p <- ggplot(data, aes(x = Level, y = Value, fill = Method, color = Method)) +
    geom_boxplot(position = position_dodge(width = 0.5), width = 0.4, outlier.size = 0.5) +
    facet_grid(rows = vars(SampleSize), cols = vars(Setting), scales = "free_y") +
    labs(x = "Zero Inflation Level", y = y_label) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(size = 12),  
      axis.text.y = element_text(size = 12),                        
      legend.title = element_blank(),
      legend.position = ifelse(show_legend, "left", "none"), 
      strip.text = element_text(size = 12)
    ) +
    scale_fill_manual(values = custom_colors) +
    scale_color_manual(values = custom_colors) +
    scale_y_continuous(breaks = y_breaks, limits = y_limits)  
  
  return(p)
}

plot_ARI <- plot_metric(
  filtered_data_ARI, 
  y_label = "ARI Value", 
  y_breaks = c(0.95, 0.96, 0.97, 0.98, 0.99, 1),  
  y_limits = c(0.95, 1), 
  show_legend = TRUE
)

plot_F1 <- plot_metric(
  filtered_data_F1, 
  y_label = "F1 Score Value", 
  y_breaks = c(0.5, 0.6, 0.7, 0.8),  
  y_limits = c(0.5, 0.8)
)
plot_Recall <- plot_metric(
  filtered_data_Recall, 
  y_label = "Recall Value", 
  y_breaks = c(0.5, 0.6, 0.7,0.8, 0.9, 1),  
  y_limits = c(0.5, 1), 
  show_legend = TRUE
)
plot_Precision <- plot_metric(
  filtered_data_Precision, 
  y_label = "Precision Value", 
  y_breaks = c(0.4,0.5, 0.6, 0.7, 0.8),  
  y_limits = c(0.4, 0.8)
)

combined_plot <- (plot_ARI | plot_F1) / (plot_Recall | plot_Precision)

combined_plot <- combined_plot + 
  plot_annotation(
    tag_levels = 'A'
  ) 
combined_plot
ggsave(filename = here("Simulations", "result_data", "ARI_F1_Recall_Precision_linear.png"), plot = combined_plot, width = 12, height = 10, dpi = 800)


