library("ggplot2")
library("tidyr")
library("dplyr")

path = "C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/papers/MMM-paper/results/breakdown"
setwd(path)
data  = read.table("breakdownDataTmp.csv", sep=",", header=T)
data$tot_time = data$tot_time / 1000

data$other = data$other +  data$tot_time - (data$comp + data$comm_reduce + data$comm_copy + data$comm_other + data$other) 

# make a table with the data to divide by
div_data = data[data$overlap == "OFF.",]
div_data$overlap = NULL
div_data$comp = NULL
div_data$comm_reduce = NULL
div_data$comm_copy = NULL
div_data$comm_other = NULL
div_data$other = NULL
names(div_data)[names(div_data) == "tot_time"] <- "divby"

# join those tables 
data <- inner_join(data, div_data)

#divide to normalize
data$comp = data$comp / data$divby
data$comm_reduce = data$comm_reduce / data$divby
data$comm_copy = data$comm_copy / data$divby
data$comm_other = data$comm_other / data$divby
data$other = data$other / data$divby
data$tot_time = data$tot_time / data$divby

#remove the divby column
data$divby = NULL

# melt to have different times in the same column
data = gather(data, overheadtype, time, comp:tot_time)

# remove total column for overlap = "OFF.", remove all other columns for overlap = "ON."
data <- data %>% 
  group_by(setup, overlap, p, m, n, k) %>%
  filter( ((overlap == "ON.") & (overheadtype == "tot_time")) |  ((overlap == "OFF.") & (overheadtype != "tot_time")) ) %>%
  ungroup() %>%
  as.data.frame()

w = 10
aspRatio = 1/3
pdf(file="breakdownPlot2.pdf",
    width = w, height = w*aspRatio)



# 
# p = ggplot(violinData, aes(x = algorithm2, y = flops, fill = algorithm2)) +
#   geom_violin() +
#   facet_grid(.~case) +
#   scale_y_continuous("% peak performance", limits = c(0,100)) +
#   scale_fill_discrete(labels = annotl)+
#   theme_bw(17) +
#   # annotate("label", x = 0.3, y = 0.8, label = "from left to right: ")  +
#   # ylim(0, 110) +
#   theme(axis.title.x=element_blank(),
#         axis.text.x=element_blank(),
#         axis.ticks.x=element_blank(),
#         legend.position = c(0.6,0.935),
#         legend.title=element_blank(),
#         legend.text=element_text(size=17)
#   ) +
#   guides(fill=guide_legend(nrow=1,byrow=TRUE),
#          keywidth=19.5,
#          keyheight=2.9,
#          default.unit="inch")

p = ggplot(data=data, aes(x=interaction(overlap, p, m, n, k, sep="::", drop=T), y=time, fill=overheadtype)) +
  geom_bar(stat="identity") +
  facet_wrap(~setup, nrow=1, scales="free") +
  scale_y_continuous("Breakdown of Runtime (normalized)") +
  scale_x_discrete("Test Case") +
  theme(axis.text.x = element_text(angle = 90)) +
  theme(axis.ticks.x = element_blank()) +
  theme_bw(17) +
  # annotate("label", x = 0.3, y = 0.8, label = "from left to right: ")  +
  # ylim(0, 110) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.position = c(0.6,0.935),
        legend.title=element_blank(),
        legend.text=element_text(size=17)
  ) +
  guides(fill=guide_legend(nrow=1,byrow=TRUE),
         keywidth=19.5,
         keyheight=2.9,
         default.unit="inch")

print(p)
dev.off()


