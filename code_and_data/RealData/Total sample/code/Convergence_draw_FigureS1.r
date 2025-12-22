##############################################################
########################   Figure S1   #######################
##############################################################

########################## Note ##############################
## Please set the working directory to the source file 
## location.
##############################################################

library(here)


load(here("RealData","Total Sample", "result_data", "loss_data.RData"))

iterations <- 1:100




mean_loss <- apply(loss_data, 1, mean)
ci_lower <- apply(loss_data, 1, quantile, probs = 0.025)
ci_upper <- apply(loss_data, 1, quantile, probs = 0.975)

library(ggplot2)
ggplot(data.frame(Iteration = iterations, 
                  Mean = mean_loss,
                  Lower = ci_lower,
                  Upper = ci_upper), 
       aes(x = Iteration)) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), 
              fill = "grey70", alpha = 0.5) +
  geom_line(aes(y = Mean), 
            color = "blue", linewidth = 1.2) +
  scale_y_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  ) +
  labs(
    x = "Iteration Number",
    y = "Objective Function Value "
  ) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(size = 15, face = "bold"),
    axis.text = element_text(size = 13),
    axis.ticks.length = unit(0.2, "cm"),
    axis.ticks = element_line(linewidth = 0.8)
  )

