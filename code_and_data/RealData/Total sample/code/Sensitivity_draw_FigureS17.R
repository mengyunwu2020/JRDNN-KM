##############################################################
####################   Figure S17   #########################
##############################################################
library(here)
library(ggplot2)
library(readxl)
library(patchwork)
library(reshape2)
library(grid)
library(dplyr)

# ===================== PART 1: Basic Settings =====================

base_path <- here( "RealData", "Total Sample", "result_data", "Sensitive")

# Define dataset colors (5 datasets with corresponding colors)
custom_colors <- c(
  "LUAD" = "#377EB8",        # Blue
  "PBMC" = "#4DAF4A",        # Green
  "mESCs" = "red",           # Red
  "Mouse Liver" = "orange",  # Orange
  "Mouse Uterus" = "#FFCC00" # Gold
)


custom_shapes <- c(
  "2" = 21,  # Hollow circle (Depth=2)
  "4" = 24,  # Hollow triangle (Depth=4)
  "7" = 8    # Star (Depth=7)
)


depth_mapping <- c("3" = "2", "5" = "4", "8" = "7")

# Dataset mapping
dataset_info <- data.frame(
  short_name = c("luad", "pbmc", "mescs", "uterus", "liver"),
  full_name = c("LUAD", "PBMC", "mESCs", "Mouse Liver", "Mouse Uterus"),
  stringsAsFactors = FALSE
)

# ===================== PART 2: Read & Process ARI/NMI Data =====================
combined_data <- data.frame(
  Metric = character(),
  Depth = numeric(),
  Width = numeric(),
  Dataset = factor(),
  Value = numeric(),
  stringsAsFactors = FALSE
)

for (i in 1:nrow(dataset_info)) {
  short_name <- dataset_info$short_name[i]
  full_name <- dataset_info$full_name[i]
  
  for (metric in c("ari", "nmi")) {
    file_path <- here(base_path, short_name, paste0(metric, ".xlsx"))
    
    if (!file.exists(file_path)) {
      warning(paste("⚠️ File does not exist:", file_path))
      next
    }
    
    tryCatch({
      df <- read_excel(
        file_path, 
        col_names = c("Depth", "Width0.5", "Width1", "Width2"),
        skip = 1,
        n_max = 3
      )
      
      df$Depth <- as.numeric(df$Depth)
      df$Width0.5 <- as.numeric(df$Width0.5)
      df$Width1 <- as.numeric(df$Width1)
      df$Width2 <- as.numeric(df$Width2)
      

      if (!all(df$Depth %in% c(3,5,8))) {
        warning(paste("⚠️ Abnormal Depth values in", file_path, "Actual values:", paste(df$Depth, collapse = ",")))
        next
      }
      

      df$Depth <- as.numeric(depth_mapping[as.character(df$Depth)])
      
      df_long <- reshape2::melt(
        df,
        id.vars = "Depth",
        measure.vars = c("Width0.5", "Width1", "Width2"),
        variable.name = "Width",
        value.name = "Value"
      )
      
      df_long$Width <- as.numeric(gsub("Width", "", df_long$Width))
      df_long$Metric <- toupper(metric)
      df_long$Dataset <- factor(full_name, levels = names(custom_colors))
      
      combined_data <- rbind(combined_data, df_long)
      message(paste("✅ Successfully read:", file_path))
      
    }, error = function(e) {
      warning(paste("❌ Failed to read:", file_path, "Error:", substr(e$message, 1, 100)))
    })
  }
}

combined_data <- combined_data %>%
  mutate(
    Depth = factor(Depth, levels = c(2,4,7)),
    Width = factor(Width, levels = c(0.5,1,2)),
    Dataset = factor(Dataset, levels = names(custom_colors))
  )

# ===================== PART 3: Read & Process Precision/Recall/F1 Data =====================
# Define ground truth order
gt_order <- c("CORUM", "PPI(F)", "PPI(P)", "PerturbAtlas")

# Depth/Width combinations 
depth_width_combinations <- expand.grid(
  Depth = c(2, 4, 7),
  Width = c(0.5, 1, 2)
) %>% 
  arrange(Depth, Width) %>%
  mutate(
    Depth_Group = paste0("Depth = ", Depth),
    Width_Label = paste0("Width = ", Width),
    Combination = paste0("D", Depth, "W", Width),
    Combination = factor(Combination, levels = paste0("D", Depth, "W", Width)),
    Depth_Group = factor(Depth_Group, levels = paste0("Depth = ", c(2,4,7))),
    Width_Label = factor(Width_Label, levels = paste0("Width = ", c(0.5,1,2)))
  )

# Ground truth mapping
gt_mapping <- data.frame(
  folder_name = c("CORUM", "ppi(f)", "ppi(p)", "perturb"),
  display_name = c("CORUM", "PPI(F)", "PPI(P)", "PerturbAtlas"),
  stringsAsFactors = FALSE
)

# Initialize metrics data frame
combined_metrics_data <- data.frame(
  Metric = character(),
  GroundTruth = factor(),
  Depth = numeric(),
  Width = numeric(),
  Depth_Group = factor(),
  Width_Label = factor(),
  Combination = factor(),
  Dataset = factor(),
  Value = numeric(),
  stringsAsFactors = FALSE
)

# Define metrics
metrics <- c("precision", "recall", "f1")

read_count <- 0
for (i in 1:nrow(dataset_info)) {
  short_name <- dataset_info$short_name[i]
  full_name <- dataset_info$full_name[i]
  cat("🔄 Processing dataset：", short_name, "→", full_name, "\n")
  
  for (gt_idx in 1:nrow(gt_mapping)) {
    gt_folder <- gt_mapping$folder_name[gt_idx]
    gt_display <- gt_mapping$display_name[gt_idx]
    
    for (metric in metrics) {
      file_path <- here(base_path, short_name, gt_folder, paste0(metric, ".xlsx"))
      cat("   📄 Checking file：", file_path, "\n")
      
      if (!file.exists(file_path)) {
        warning(paste("⚠️ File does not exist:", file_path))
        next
      }
      
      tryCatch({
        df <- read_excel(
          file_path, 
          col_names = c("Depth", "Width0.5", "Width1", "Width2"),
          skip = 1,
          n_max = 3
        )
        
        df$Depth <- as.numeric(df$Depth)
        df$Width0.5 <- as.numeric(df$Width0.5)
        df$Width1 <- as.numeric(df$Width1)
        df$Width2 <- as.numeric(df$Width2)
        

        if (!all(df$Depth %in% c(3,5,8))) {
          warning(paste("⚠️ Abnormal Depth values:", file_path, "Actual values:", paste(df$Depth, collapse = ",")))
          next
        }
        

        df$Depth <- as.numeric(depth_mapping[as.character(df$Depth)])
        
        for (d in c(2,4,7)) {
          depth_row <- df[df$Depth == d, ]
          depth_group <- paste0("Depth = ", d)
          
          for (w in c(0.5, 1, 2)) {
            width_col <- paste0("Width", w)
            value <- as.numeric(depth_row[[width_col]])
            if (is.na(value)) value <- 0
            
            width_label <- paste0("Width = ", w)
            comb_label <- paste0("D", d, "W", w)
            
            new_row <- data.frame(
              Metric = metric,
              GroundTruth = factor(gt_display, levels = gt_order),
              Depth = d,
              Width = w,
              Depth_Group = factor(depth_group, levels = paste0("Depth = ", c(2,4,7))),
              Width_Label = factor(width_label, levels = paste0("Width = ", c(0.5,1,2))),
              Combination = factor(comb_label, levels = depth_width_combinations$Combination),
              Dataset = factor(full_name, levels = names(custom_colors)),
              Value = value,
              stringsAsFactors = FALSE
            )
            
            combined_metrics_data <- rbind(combined_metrics_data, new_row)
            read_count <- read_count + 1
          }
        }
        
        message(paste("✅ Successfully read:", file_path, "Data rows:", nrow(df)*3))
        
      }, error = function(e) {
        warning(paste("❌ Failed to read:", file_path, "Error:", substr(e$message, 1, 100)))
      })
    }
  }
}

# Clean up metrics data
combined_metrics_data <- combined_metrics_data %>%
  mutate(
    Depth = factor(Depth, levels = c(2,4,7)),
    Width = factor(Width, levels = c(0.5,1,2))
  )

# ===================== PART 4: Common Theme (Unified Style) =====================
common_theme <- theme_minimal() +
  theme(
    # Axis settings
    axis.text.x = element_text(size = 14, angle = 0, hjust = 0.5, margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 14, margin = margin(0,0,0,0)),
    axis.title = element_text(size = 16, face = "bold", margin = margin(0,0,0,0)),
    axis.ticks.length = unit(0.1, "cm"),
    # Facet settings (NO GAPS between dataset subplots)
    strip.text.x = element_text(size = 10,  face = "bold", margin = margin(0,0,0,0)),
    strip.background.x = element_rect(fill = "lightgray", color = NA),
    panel.spacing.x = unit(0, "cm"),
    # Legend settings (hide all subplot legends)
    legend.position = "none",
    # Panel/Plot settings
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
    # Plot tags
    plot.tag = element_text(size = 16, face = "bold"),
    plot.tag.position = c(0.01, 0.99),
    # Panel spacing
    panel.spacing = unit(0.2, "cm")
  )

# ===================== PART 5: Create ARI/NMI Plots =====================
# ARI Plot
plot_ari <- ggplot(combined_data[combined_data$Metric == "ARI", ], 
                   aes(
                     x = Width,
                     y = Value,
                     color = Dataset,
                     shape = Depth,
                     fill = "white"
                   )) +
  geom_point(
    size = 3,
    alpha = 0.9,
    stroke = 0.8
  ) +
  facet_wrap(~Dataset, nrow = 1) +
  scale_color_manual(values = custom_colors) +
  scale_shape_manual(values = custom_shapes) +
  scale_fill_manual(values = "white", guide = "none") +
  scale_y_continuous(limits = c(0.7, 1.01), breaks = seq(0.7, 1, 0.05), name = "ARI Value") +
  scale_x_discrete(name = "Width") +
  common_theme

# NMI Plot
plot_nmi <- ggplot(combined_data[combined_data$Metric == "NMI", ], 
                   aes(
                     x = Width,
                     y = Value,
                     color = Dataset,
                     shape = Depth,
                     fill = "white"
                   )) +
  geom_point(
    size = 3,
    alpha = 0.9,
    stroke = 0.8
  ) +
  facet_wrap(~Dataset, nrow = 1) +
  scale_color_manual(values = custom_colors) +
  scale_shape_manual(values = custom_shapes) +
  scale_fill_manual(values = "white", guide = "none") +
  scale_y_continuous(limits = c(0.7, 1.01), breaks = seq(0.7, 1, 0.05), name = "NMI Value") +
  scale_x_discrete(name = "Width") +
  common_theme 

# ===================== PART 6: Manual Legend Panel =====================
# 1. Build legend data
depth_legend_data <- data.frame(
  x = 1,
  y = 3:1,
  Depth = factor(c(2,4,7), levels = c(2,4,7)),
  label = paste("Depth =", c(2,4,7))
)

# Dataset color legend data
dataset_legend_data <- data.frame(
  x = 2,
  y = 5:1,
  Dataset = factor(names(custom_colors), levels = names(custom_colors)),
  label = names(custom_colors)
)

# 2. Create manual legend panel
manual_legend_plot <- ggplot() +
  # Depth shape legend
  geom_point(data = depth_legend_data, 
             aes(x = x, y = y, shape = Depth), 
             size = 5,
             color = "black", 
             fill = "white",
             stroke = 1) +
  geom_text(data = depth_legend_data, 
            aes(x = x + 0.2, y = y, label = label), 
            hjust = 0, 
            size = 4, 
            fontface = "bold") +
  # Dataset color legend
  geom_rect(data = dataset_legend_data,
            aes(xmin = x - 0.1, xmax = x + 0.1, 
                ymin = y - 0.2, ymax = y + 0.2,
                fill = Dataset),
            color = "black") +
  geom_text(data = dataset_legend_data, 
            aes(x = x + 0.2, y = y, label = label), 
            hjust = 0, 
            size = 4, 
            fontface = "bold") +
  # Style settings
  scale_shape_manual(values = custom_shapes) +
  scale_fill_manual(values = custom_colors) +
  # Axis range
  xlim(0.5, 3) +
  ylim(0, 6) +
  # Title
  labs(title = "Legend") +
  # Theme
  theme_minimal() +
  theme(
    # Hide axes
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    # Title style
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5, margin = margin(b = 10)),
    # Margin matching main plot
    plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
    # Transparent background
    plot.background = element_blank()
  )

# ===================== PART 7: Create Precision/Recall/F1 Plots =====================
# Helper function for metric plots
create_gt_metric_plot <- function(gt_name, metric_name, metric_label, y_lim = c(0.7, 1.01)) {
  plot_data <- combined_metrics_data %>%
    filter(GroundTruth == gt_name, Metric == metric_name)
  
  ggplot(plot_data, aes(
    x = Width,
    y = Value,
    color = Dataset,
    shape = Depth,
    fill = "white"
  )) +
    geom_point(
      size = 3,
      alpha = 0.9,
      stroke = 0.8
    ) +
    facet_wrap(~Dataset, nrow = 1) +
    scale_color_manual(values = custom_colors) +
    scale_shape_manual(values = custom_shapes) +
    scale_fill_manual(values = "white", guide = "none") +
    scale_y_continuous(
      limits = y_lim, 
      breaks = seq(y_lim[1], y_lim[2], by = 0.1),
      name = metric_label
    ) +
    scale_x_discrete(name = "Width") +
    common_theme
}

# -------------------------- CORUM Row --------------------------
plot_corum_precision <- create_gt_metric_plot("CORUM", "precision", "Precision", y_lim = c(0.1, 0.9))
plot_corum_recall <- create_gt_metric_plot("CORUM", "recall", "Recall", y_lim = c(0.1, 0.9))
plot_corum_f1 <- create_gt_metric_plot("CORUM", "f1", "F1 Score", y_lim = c(0.1, 0.9))

# Create prominent title for CORUM row
corum_title <- ggplot() + 
  geom_text(aes(x = 1, y = 1, label = "CORUM"), 
            size = 6, fontface = "bold", color = "black") +
  theme_void() +
  theme(plot.margin = margin(10, 0, 10, 0, "pt"))

row_corum <- (corum_title / (plot_corum_precision + plot_corum_recall + plot_corum_f1)) +
  plot_layout(heights = c(0.1, 1), widths = c(1,1,1))

# -------------------------- PPI(F) Row --------------------------
plot_ppif_precision <- create_gt_metric_plot("PPI(F)", "precision", "Precision", y_lim = c(0.35, 0.85))
plot_ppif_recall <- create_gt_metric_plot("PPI(F)", "recall", "Recall", y_lim = c(0.35, 0.85))
plot_ppif_f1 <- create_gt_metric_plot("PPI(F)", "f1", "F1 Score", y_lim = c(0.35, 0.85))

# Create prominent title for PPI(F) row
ppif_title <- ggplot() + 
  geom_text(aes(x = 1, y = 1, label = "PPI(F)"), 
            size = 6, fontface = "bold", color = "black") +
  theme_void() +
  theme(plot.margin = margin(10, 0, 10, 0, "pt"))

row_ppif <- (ppif_title / (plot_ppif_precision + plot_ppif_recall + plot_ppif_f1)) +
  plot_layout(heights = c(0.1, 1), widths = c(1,1,1))

# -------------------------- PPI(P) Row --------------------------
plot_ppip_precision <- create_gt_metric_plot("PPI(P)", "precision", "Precision", y_lim = c(0.35, 0.85))
plot_ppip_recall <- create_gt_metric_plot("PPI(P)", "recall", "Recall", y_lim = c(0.35, 0.85))
plot_ppip_f1 <- create_gt_metric_plot("PPI(P)", "f1", "F1 Score", y_lim = c(0.35, 0.85))

# Create prominent title for PPI(P) row
ppip_title <- ggplot() + 
  geom_text(aes(x = 1, y = 1, label = "PPI(P)"), 
            size = 6, fontface = "bold", color = "black") +
  theme_void() +
  theme(plot.margin = margin(10, 0, 10, 0, "pt"))

row_ppip <- (ppip_title / (plot_ppip_precision + plot_ppip_recall + plot_ppip_f1)) +
  plot_layout(heights = c(0.1, 1), widths = c(1,1,1))

# -------------------------- PerturbAtlas Row --------------------------
plot_perturb_precision <- create_gt_metric_plot("PerturbAtlas", "precision", "Precision", y_lim = c(0, 0.85))
plot_perturb_recall <- create_gt_metric_plot("PerturbAtlas", "recall", "Recall", y_lim = c(0, 0.85))
plot_perturb_f1 <- create_gt_metric_plot("PerturbAtlas", "f1", "F1 Score", y_lim = c(0, 0.85))

# Create prominent title for PerturbAtlas row
perturb_title <- ggplot() + 
  geom_text(aes(x = 1, y = 1, label = "PerturbAtlas"), 
            size = 6, fontface = "bold", color = "black") +
  theme_void() +
  theme(plot.margin = margin(10, 0, 10, 0, "pt"))

row_perturb <- (perturb_title / (plot_perturb_precision + plot_perturb_recall + plot_perturb_f1)) +
  plot_layout(heights = c(0.1, 1), widths = c(1,1,1))

# ===================== PART 8: Combine All Plots =====================
# Top row: ARI + NMI + manual legend panel (width ratio 1:1:1)
top_row <- plot_ari + plot_nmi + manual_legend_plot +
  plot_layout(widths = c(1,1,1)) +
  plot_annotation(title = "ARI & NMI", 
                  theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5, margin = margin(b = 10))))

# Combine all rows (top row + 4 GT rows)
final_combined_plot <- (top_row / row_corum / row_ppif / row_ppip / row_perturb) +
  plot_layout(heights = c(1, 1.2, 1.2, 1.2, 1.2)) +
  plot_annotation(
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.spacing = unit(1, "cm")
    )
  )

final_combined_plot




