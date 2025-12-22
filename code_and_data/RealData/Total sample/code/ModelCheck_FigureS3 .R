
##############################################################
##################   Figure S3   ############################
##############################################################

########################## Note ##############################
## Please set the working directory to the source file 
## location.
##############################################################




# ==============================================================================
# 0. Environment Setup
# ==============================================================================
library(tidyverse)
library(ggplot2)
library(outliers) 
library(scales)
library(patchwork) 
library(here) # Load here package

# Define paths using here()
# Root directory is 'code_and_data'
base_dir <- here("RealData","Total Sample", "result_data", "model checking")
datasets_list <- c("luad", "pbmc", "mescs", "liver", "uterus")

# ==============================================================================
# 1. Core Utility Functions (Unchanged)
# ==============================================================================
apply_int <- function(vec) {
  vec <- as.numeric(vec)
  vec_clean <- na.omit(vec)
  n <- length(vec_clean)
  if(n < 3) return(vec) 
  ranks <- rank(vec_clean, na.last = "keep")
  qnorm((ranks - 0.375) / (n + 0.25))
}

calc_corr_pvalue <- function(vec_p, vec_r) {
  if(sd(vec_p, na.rm=T) == 0 || sd(vec_r, na.rm=T) == 0) return(NA)
  test <- tryCatch(cor.test(vec_p, vec_r, method="pearson", use="complete.obs"), error=function(e) NA)
  ifelse(is.null(test), NA, test$p.value)
}

calc_ks_pvalue <- function(vec) {
  if(length(na.omit(vec)) < 3) return(NA)
  test <- tryCatch(ks.test(vec, "pnorm", mean(vec, na.rm=T), sd(vec, na.rm=T)), error=function(e) NA)
  ifelse(is.null(test), NA, test$p.value)
}

calc_grubbs_pvalue <- function(vec) {
  if(length(na.omit(vec)) < 3) return(NA)
  test <- tryCatch(grubbs.test(vec), error=function(e) NA)
  ifelse(is.null(test), NA, test$p.value)
}

# ==============================================================================
# 2. Plotting Function (Visual Adjustments)
# ==============================================================================

process_and_plot <- function(dataset_name, base_path) {
  
  # --- Load Data ---
  # Assuming files are directly inside the 'model checking' folder
  # e.g., .../model checking/luad_modelcheck.Rdata
  rdata_file <- file.path(base_path, paste0(dataset_name, "_modelcheck.Rdata"))
  message(paste("Processing:", dataset_name, "..."))
  
  if(!file.exists(rdata_file)) {
    message(paste("File not found:", rdata_file))
    return(NULL)
  }
  
  local_env <- new.env()
  load(rdata_file, envir = local_env)
  
  obj_name <- ls(local_env)[grepl("modelcheck", ls(local_env))][1]
  if(is.na(obj_name)) obj_name <- ls(local_env)[1]
  main_list <- local_env[[obj_name]]
  
  data <- main_list$data
  res <- main_list$res
  predict <- main_list$predict
  
  # --- Calculate Statistics ---
  common <- intersect(names(data), intersect(names(predict), names(res)))
  
  df_clean <- map_dfr(common, function(dname) {
    p_df <- as.data.frame(predict[[dname]])
    r_df <- as.data.frame(res[[dname]])
    d_df <- as.data.frame(data[[dname]])
    vars <- colnames(p_df)
    
    map_dfr(vars, function(v) {
      p1 <- calc_corr_pvalue(p_df[[v]], r_df[[v]])
      p2 <- calc_ks_pvalue(r_df[[v]])
      raw_no_zero <- d_df[[v]][d_df[[v]] != 0]
      p3 <- if(length(raw_no_zero) < 3) NA else calc_grubbs_pvalue(apply_int(raw_no_zero))
      
      bind_rows(
        tibble(Dataset=dname, Variable=v, P_Raw=p1, Test_Type="1. Independence"),
        tibble(Dataset=dname, Variable=v, P_Raw=p2, Test_Type="2. Residual Normality"),
        tibble(Dataset=dname, Variable=v, P_Raw=p3, Test_Type="3. Outlier Detection")
      )
    })
  })
  
  # --- Correction & Renaming ---
  plot_data <- df_clean %>%
    group_by(Dataset, Test_Type) %>%
    mutate(P_Value = case_when(
      grepl("Independence|Normality", Test_Type) ~ p.adjust(P_Raw, method="fdr"),
      TRUE ~ P_Raw
    )) %>%
    ungroup() %>%
    mutate(Dataset = case_when(
      Dataset == "CD19B" ~ "CD19+ B",
      Dataset == "CD4CD25TReg" ~ "CD4+/CD25 T Reg",
      Dataset == "CD56NK" ~ "CD56+ NK",
      Dataset == "data2i" ~ "2i",
      TRUE ~ Dataset
    ))
  
  # --- Plot Configuration ---
  Y_LIMIT_MAX <- 3.0
  SIG_LINE <- -log10(0.05)
  
  p <- ggplot(plot_data, aes(x = Variable, y = -log10(P_Value), color = Dataset, shape = Dataset)) +
    geom_jitter(width = 0.4, height = 0, size = 1.8, alpha = 0.6, stroke = 0.3) + # Slightly larger points
    geom_hline(yintercept = SIG_LINE, linetype = "dashed", color = "red", size = 0.8) +
    
    facet_wrap(~Test_Type, scales = "free_x", ncol = 3) +
    
    labs(
      # --- Modification 1: Remove top Title, move data name to Y axis ---
      title = NULL,
      y = toupper(dataset_name), 
      x = NULL,
      color = "Subgroups",
      shape = "Subgroups"
    ) +
    
    scale_y_continuous(
      limits = c(0, Y_LIMIT_MAX),
      oob = scales::squish,
      breaks = c(0, round(SIG_LINE, 2), 2, Y_LIMIT_MAX),
      expand = expansion(mult = c(0.05, 0.05))
    ) +
    
    # --- Modification 2: Increase overall font size (base_size 12 -> 15) ---
    theme_bw(base_size = 15) + 
    
    theme(
      # Remove subplot title area
      plot.title = element_blank(),
      
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      
      # Axis tick label size
      axis.text.y = element_text(size = 12),
      
      # --- Modification: Increase axis title size + rotate 90 degrees ---
      # Rotate 90 degrees (angle=90) to ensure consistent left width for alignment
      axis.title.y = element_text(angle = 90, vjust = 0.5, face = "bold", size = 16, margin = margin(r = 10)),
      
      # Facet label font size
      strip.text = element_text(face = "bold", size = 12),
      
      # --- Modification 3: Legend alignment and font size ---
      legend.position = "right", 
      
      # [Key] Align legend to the left:
      # This forces "Subgroups" to stick to the right edge of the plot area,
      # ensuring fixed position regardless of legend content width.
      legend.justification = "left", 
      
      legend.title = element_text(size = 14, face = "bold"), # Increase legend title size
      legend.text = element_text(size = 12),                 # Increase legend text size
      legend.margin = margin(l = 5, r = 5),
      
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      
      # Maintain plot margins
      plot.margin = margin(t = 5, r = 5, b = 20, l = 5)
    )
  
  return(p)
}

# ==============================================================================
# 3. Generation & Stacking
# ==============================================================================

plot_list <- map(datasets_list, function(ds) {
  process_and_plot(ds, base_dir)
})
plot_list <- plot_list[!sapply(plot_list, is.null)]

if(length(plot_list) > 0) {
  
  # Vertical stacking
  final_plot <- wrap_plots(plot_list, ncol = 1) 
  
  print(final_plot)
  
} else {
  message("No plots generated.")
}

# ==============================================================================
# 4. Save Image
# ==============================================================================

# Run this line to save
ggsave(
  filename = file.path(base_dir, "model_Diagnostics_noadjusted.png"),
  plot = final_plot,
  width = 13,  # Width increased to accommodate larger legend
  height = 18, # Height increased to accommodate larger fonts and spacing
  dpi = 300
)
message(paste("Plot saved to:", file.path(base_dir, "model_Diagnostics_noadjusted.png")))