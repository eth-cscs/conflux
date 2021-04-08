
library(ggplot2)
library(ggrepel)
library(reshape2)
library(dplyr)
library(hash)
library(gmodels)



#-------------------------SETUP----------------------#
#path = "C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/conflux_cpp_2/results/conflux/benchmarks" #getwd()
#path = "/mnt/c/gk_pliki/uczelnia/doktorat/performance_modelling/repo/conflux_cpp_2/results/conflux/benchmarks" #getwd()
#exp_filename = "benchmarks.csv"
exp_filename = "/../benchmarks.csv"
#exp_filename = "rawData_old.csv"
#setwd(path)
source("SPCL_Stats.R")
source("plot_settings.R")
#source(paste(path, "scripts/SPCL_Stats.R", sep="/"))
#source(paste(path, "scripts/plot_settings.R", sep="/"))
scalings = c("$2^{17}$", "$2^{14}$", '$2^{13} \\cdot \\sqrt{P}$')
matrixShapes = c("lu","cholesky")
variantPlots = c("commVol", "time")
algorithms = c("conflux","mkl","slate","candmc")
annotl = c("COnfLUX/ConfCHOX (this work) ", "MKL [cite] ","SLATE [cite] ","CANDMC/CAPITAL [cite] ")
importantCols = c("algorithm","N_base","library", "P","N","value","unit") #, "V","commVolModel","commModelRatio")

results <- data.frame(matrix(ncol = 5, nrow = 0))
rescolNames <- append(algorithms, "scenario", after = 0)
colnames(results) <- rescolNames
GFLOPSperNode = 1209
S = 5e+09



#--------------------END of SETUP---------------------#


statistics = 0

annotCoord = list()
DataColumnsHash = hash()
DataColumnsHash['commVol'] = c("P", "algLabel", "V")
DataColumnsHash['time'] = c("P", "algLabel", "value")
DataColumnsHash['flops'] = c("P", "algLabel", "flops")



#--------------------PREPROCESSING-----------------------------#
#rawData = read.table(exp_filename, header = T, sep = ',',fill = TRUE, stringsAsFactors=FALSE)
rawData <- read.csv(file=paste(getwd(), exp_filename, sep = ""), sep=",", stringsAsFactors=FALSE, header=TRUE)


rawData[rawData$N_base == "-" & rawData$type == "strong",]$N_base <- rawData[rawData$N_base == "-" & rawData$type == "strong",]$N
rawData[rawData$N_base == "-" & rawData$type == "weak",]$N_base <- rawData[rawData$N_base == "-" & rawData$type == "weak",]$N / sqrt(rawData[rawData$N_base == "-" & rawData$type == "weak",]$P)
if (nrow(rawData[str_cmp("capital", rawData$library),]) > 0) {
  rawData[str_cmp(rawData$library, "capital"),]$library = "candmc"
}
if (nrow(rawData[str_cmp("psychol", rawData$library),]) > 0) {
  rawData[str_cmp("psychol", rawData$library),]$library = "conflux"
}

rawData$case = paste(rawData$algorithm, rawData$N_base,sep ="_")


# rawData = rawData[rawData$S < Inf,]
# rawData$V = rawData$V/3

setups <- rawData[rawData$library == "conflux",]
setups <-setups[c("P", "N", "value")]
#options("scipen"=100, "digits"=4)
write.table(setups, file = "setups.csv",sep = " ",col.names = FALSE, row.names = FALSE)

rawData$algLabel = ifelse(rawData$library == "mkl", annotl[2],
                          ifelse(rawData$library == "candmc", annotl[4],
                                 ifelse(rawData$library == "conflux", annotl[1],
                                        annotl[3])
                          )
)
rawData$algLabel <- as.factor(rawData$algLabel)
rawData$algLabel = factor(rawData$algLabel, levels = levels(rawData$algLabel)[c(2,4,3,1)])


rawData$mShape = rawData$algorithm
rawData$scaling = 'type 1 ERROR!'
rawData$scaling = ifelse(rawData$N_base == 16384, '$2^{14}$',
                         ifelse(rawData$N_base == 131072, '$2^{17}$',
                                ifelse(rawData$N_base == 8192, '$2^{13} \\cdot \\sqrt{P}$',
                                       ifelse(rawData$N_base == 1024, '$2^{10} \\cdot \\sqrt{P}$', 'ERROR!')
                                )
                         )
)


# rawData[rawData$blocksize == "", ]$blocksize = 1
# time_data <- rawData[rawData$unit == "time",]
# time_data <- time_data[complete.cases(time_data),]
# comm_data <- rawData[rawData$unit == "bytes",]
# comm_data <- comm_data[complete.cases(comm_data),]
# fastest_blocks <- as.data.frame(time_data %>% group_by(algorithm, library, N, N_base, P) %>% summarise_each(list(min), value))
# comm_min_blocks <- as.data.frame(comm_data %>% group_by(algorithm, library, N, N_base, P) %>% summarise_each(list(min), value))
# rows_to_remove = c()

# for (row in 1:nrow(time_data)) {
#   cur_blocksize = time_data[row, "blocksize"]
#   cur_p = time_data[row, "P"]
#   cur_n = time_data[row, "N"]
#   cur_n_base = time_data[row, "N_base"]
#   cur_lib = time_data[row, "library"]
#   cur_val = time_data[row, "value"]
#   cur_alg = time_data[row, "algorithm"]
#   all_values = time_data[time_data$algorithm == cur_alg & time_data$library == cur_lib & time_data$N == cur_n & time_data$N_base == cur_n_base & time_data$P == cur_p & time_data$blocksize == cur_blocksize,]$value
#   best_block_value = fastest_blocks[fastest_blocks$algorithm == cur_alg & fastest_blocks$library == cur_lib & fastest_blocks$N == cur_n & fastest_blocks$N_base == cur_n_base  &fastest_blocks$P == cur_p, ]$value
#   
#   smallest_com_vol = comm_min_blocks[comm_min_blocks$algorithm == cur_alg & comm_min_blocks$library == cur_lib & comm_min_blocks$N == cur_n & comm_min_blocks$N_base == cur_n_base  & comm_min_blocks$P == cur_p, ]$value
#   if (!identical(smallest_com_vol, numeric(0))){
#     time_data[row, "V"] = smallest_com_vol / cur_p * 1e-6
#   }
#   if (min(all_values) != best_block_value | cur_val > 1.3 * best_block_value){
#     rows_to_remove <- c(rows_to_remove, row)
#   }
# }
# filtered_data = time_data[-rows_to_remove, ]


rawData <- find_optimal_blocks(rawData)
rawData <- as.data.frame(rawData %>% group_by(algorithm, library, N, N_base, P, grid, unit, type, blocksize, case, algLabel, mShape, scaling) %>% summarise_each(list(mean)))
rawData$time = rawData$value


rawData$flops = 0
rawData[str_cmp(rawData$algorithm, "lu"),]$flops =
  200/3 * (rawData[str_cmp(rawData$algorithm, "lu"),]$N)^3 / (1e6 * (rawData[str_cmp(rawData$algorithm, "lu"),]$P/2) * rawData[str_cmp(rawData$algorithm, "lu"),]$value * GFLOPSperNode)

rawData[str_cmp(rawData$algorithm, "cholesky"),]$flops =
  100/3 * (rawData[str_cmp(rawData$algorithm, "cholesky"),]$N)^3 / (1e6 * (rawData[str_cmp(rawData$algorithm, "cholesky"),]$P/2) * rawData[str_cmp(rawData$algorithm, "cholesky"),]$value * GFLOPSperNode)


#
detach("package:reshape2",unload = TRUE)
detach("package:ggrepel",unload = TRUE)

if (('plyr' %in% (.packages()))) {
  detach("package:plyr",unload = TRUE)
}


#finding median and confidence intervals
dataSummary =
  rawData %>%
  group_by(N,P,library, mShape, scaling) %>%
  summarise(med_time = median(time)) %>% #, lci = ci(time)["CI lower"], uci = ci(time)["CI upper"], count = N()) %>%
  ungroup() %>%
  # filter(library == "conflux (this work) ") %>%
  as.data.frame()

remAlgs = c('candmc', 'slate', 'mkl')

#second-best library
dataSummary2 = reshape(dataSummary,timevar="library",idvar=c("N","P","mShape"),direction="wide")
dataSummary2[is.na(dataSummary2)] <- 99999999
dataSummary2<-dataSummary2[!(dataSummary2$"med_time.conflux"==99999999),]
dataSummary2$secondBestTime = apply(dataSummary2[, c('med_time.candmc','med_time.slate','med_time.mkl')], 1, min)
dataSummary2$secondBestAlg = remAlgs[apply(dataSummary2[, c('med_time.candmc','med_time.slate','med_time.mkl')], 1, which.min)]
dataSummary2<-dataSummary2[!(dataSummary2$secondBestTime==99999999),]

dataSummary2$peak_flops <- 0
dataSummary2[str_cmp(dataSummary2$mShape, "LU"),]$peak_flops = 
  200/3 * (dataSummary2[str_cmp(dataSummary2$mShape, "LU"),]$N)^3 / (1e6 * (dataSummary2[str_cmp(dataSummary2$mShape, "LU"),]$P/2) * dataSummary2[str_cmp(dataSummary2$mShape, "LU"),]$med_time.conflux * GFLOPSperNode)

dataSummary2[str_cmp(dataSummary2$mShape, "Cholesky"),]$peak_flops = 
  100/3 * (dataSummary2[str_cmp(dataSummary2$mShape, "Cholesky"),]$N)^3 / (1e6 * (dataSummary2[str_cmp(dataSummary2$mShape, "Cholesky"),]$P/2) * dataSummary2[str_cmp(dataSummary2$mShape, "Cholesky"),]$med_time.conflux  * GFLOPSperNode)


dataSummary2$maxSpeedup = dataSummary2$secondBestTime / dataSummary2$"med_time.conflux"

importantCols = c('N', 'P', 'mShape', "secondBestAlg", "maxSpeedup", "peak_flops")
heatmap_data <- dataSummary2[importantCols]


heatmap_data[heatmap_data$secondBestAlg == "candmc",]$secondBestAlg <- "C"
heatmap_data[heatmap_data$secondBestAlg == "mkl",]$secondBestAlg <- "M"
heatmap_data[heatmap_data$secondBestAlg == "slate",]$secondBestAlg <- "S"


# --------------------- FILTERING THRESHOLD (in percent): ---------- #
threshold = 3


# ----------------- SPEEDUP ---------------------- #
h = 4
w = 4.8

# --- LU ----#
data <- heatmap_data[heatmap_data$mShape == "lu" & heatmap_data$peak_flops > threshold,]

min_speedup = 0.0
max_speedup = 3.1
pdf(file= "../lu_heatmap_labelled_filtered_compacted.pdf",  width = w, height = h)
p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
                           fill=maxSpeedup)) +
  geom_tile(aes(fill = maxSpeedup)) +
  geom_text(aes(label = paste(round(maxSpeedup,1), "x", sep = "")), position = position_nudge(y=0.2)) +
  geom_text(aes(label = paste("\n", secondBestAlg, sep = ""), fontface = "bold"), position = position_nudge(y=0.05)) +
  scale_x_discrete("Number of nodes") +
  scale_y_discrete("Matrix size") +
  scale_fill_gradient("", low = "white", high = "green") + #, limits = c(min_speedup,max_speedup)) +
  theme_bw(20) + theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle = 90))
print(p)
ggsave(file="../lu_heatmap_labelled_filtered_compacted.svg", plot=p, width=w, height=h)
dev.off()

# -cHOLESKY -#
data <- heatmap_data[heatmap_data$mShape == "cholesky" & heatmap_data$peak_flops > threshold,]

min_speedup = 0.0
max_speedup = 3.1
pdf(file= "../chol_heatmap_labelled_filtered_compacted.pdf",  width = w, height = h)
p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
                           fill=maxSpeedup)) +
  geom_tile(aes(fill = maxSpeedup)) +
  geom_text(aes(label = paste(round(maxSpeedup,1), "x", sep = "")), position = position_nudge(y=0.2)) +
  geom_text(aes(label = paste("\n", secondBestAlg, sep = ""), fontface = "bold"), position = position_nudge(y=0.05)) +
  scale_x_discrete("Number of nodes") +
  scale_y_discrete("Matrix size") +
  scale_fill_gradient("", low = "white", high = "green") + #, limits = c(min_speedup,max_speedup)) +
  theme_bw(20) + theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle = 90))
print(p)
ggsave(file="../chol_heatmap_labelled_filtered_compacted.svg", plot=p, width=w, height=h)
dev.off()



# --------------- PERFORMANCE -------------------- #
# --- LU ----#
data <- heatmap_data[heatmap_data$mShape == "lu" & heatmap_data$peak_flops > threshold,]

pdf(file= "../lu_heatmap_labelled_performance.pdf",  width = 4, height = 4)
p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
                           fill=peak_flops,
                           label=(paste(round(peak_flops,0), "%", sep ="") ))) +
  geom_tile(aes(fill = peak_flops)) +
  geom_text(fontface="bold") +
  scale_x_discrete("Number of nodes") +
  scale_y_discrete("Matrix size") +
  scale_fill_gradient("", low = "orange", high = "green") +
  theme_bw(20) +
  theme(legend.position = "none",
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  theme(axis.text.x = element_text(angle = 90))
print(p)
ggsave(file="../lu_heatmap_labelled_performance.svg", plot=p, width = 4, height = 4)
dev.off()

# -cHOLESKY -#
data <- heatmap_data[heatmap_data$mShape == "cholesky" & heatmap_data$peak_flops > threshold,]

pdf(file= "../chol_heatmap_labelled_performance.pdf",  width = 4, height = 4)
p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
                           fill=peak_flops,
                           label=(paste(round(peak_flops,0), "%", sep ="") ))) +
  geom_tile(aes(fill = peak_flops)) +
  geom_text(fontface="bold") +
  scale_x_discrete("Number of nodes") +
  scale_y_discrete("Matrix size") +
  scale_fill_gradient("", low = "orange", high = "green") +
  theme_bw(20) +
  theme(legend.position = "none",
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  theme(axis.text.x = element_text(angle = 90))
print(p)
ggsave(file="../chol_heatmap_labelled_performance.svg", plot=p, width = 4, height = 4)
dev.off()
# 
# 
# 
# 
# # 
# # data <- heatmap_data[heatmap_data$mShape == "lu" & heatmap_data$peak_flops > threshold,]
# # pdf(file= "../lu_heatmap_labelled_filtered.pdf",  width = 6, height = 5)
# # p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
# #                       fill=maxSpeedup,
# #                       label=(paste((round(maxSpeedup,1)),
# #                                   secondBestAlg, sep="\n") ))) +
# #   geom_tile(aes(fill = maxSpeedup)) +
# #   geom_text() +
# #   geom_text(fontface="bold") +
# #   # geom_point(aes(shape=as.factor(datasrc)), size=3) +
# #   scale_x_discrete("Number of nodes") +
# #   scale_y_discrete("Matrix size") +
# #   scale_fill_gradient("", low = "orange", high = "green", limits = c(min_speedup,max_speedup)) +
# #   theme_bw(20) + theme(legend.position = "none") +
# #   theme(axis.text.x = element_text(angle = 90))
# # print(p)
# # dev.off()
# 
# pdf(file= "../lu_heatmap_labelled_performance.pdf",  width = 4, height = 4)
# p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
#                            fill=peak_flops,
#                            label=(paste(round(peak_flops,0), "%", sep ="") ))) +
#   geom_tile(aes(fill = peak_flops)) +
#   geom_text() +
#   geom_text(fontface="bold") +
#   # geom_point(aes(shape=as.factor(datasrc)), size=3) +
#   scale_x_discrete("Number of nodes") +
#   scale_y_discrete("Matrix size") +
#   scale_fill_gradient("", low = "orange", high = "green") +
#   theme_bw(20) +
#   theme(legend.position = "none",
#         axis.title.y = element_blank(),
#         axis.text.y = element_blank(),
#         axis.ticks.y = element_blank()) +
#   theme(axis.text.x = element_text(angle = 90))
# print(p)
# ggsave(file="../lu_heatmap_labelled_performance.svg", plot=p, width = 4, height = 4)
# dev.off()
# 
# 
# # -------------------- plots with data filtered out based on achieved peak performance ------------ #
# 
# aspRatio = 1
# w = 4
# 
# h = 4
# w = 4
# 
# max_speedup = 1.8
# min_speedup = 0.4
# 
# data <- heatmap_data[heatmap_data$mShape == "cholesky" & heatmap_data$peak_flops > threshold,]
# pdf(file= "../cholesky_heatmap_labelled_filtered_compacted.pdf",  width = w, height = w*aspRatio)
# p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
#                       fill=maxSpeedup)) +
#                       # label=paste(round(maxSpeedup,1),
#                       #             secondBestAlg, sep="\n") )) +
#   geom_tile(aes(fill = maxSpeedup)) +
#   geom_text(aes(label = paste(round(maxSpeedup,1), "x", sep = "")), position = position_nudge(y=0.2)) +
#   geom_text(aes(label = paste("\n", secondBestAlg, sep = ""), fontface = "bold"), position = position_nudge(y=0.05)) +
#   # geom_text(fontface="bold") +
#   # geom_point(aes(shape=as.factor(datasrc)), size=max_speedup) +
#   scale_x_discrete("Number of nodes") +
#   scale_y_discrete("Matrix size") +
#   scale_fill_gradient("", low = "orange", high = "green", limits = c(min_speedup,max_speedup)) +
#   theme_bw(20) + 
#   theme(legend.position = "none",
#         axis.title.y = element_blank(),
#         axis.text.y = element_blank(),
#         axis.ticks.y = element_blank()) +
#   theme(axis.text.x = element_text(angle = 90))
# print(p)
# ggsave(file="../cholesky_heatmap_labelled_filtered_compacted.svg", plot=p, width=w, height=h)
# dev.off()
# 
# h = 4
# w = 4.8
# 
# max_speedup = 3.1
# min_speedup = 0.8
# 
# data <- heatmap_data[heatmap_data$mShape == "lu" & heatmap_data$peak_flops > threshold,]
# pdf(file= "../lu_heatmap_labelled_filtered_compacted.pdf",  width = w, height = h)
# p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
#                       fill=maxSpeedup
#                       # label=paste(round(maxSpeedup,1),
#                       #             secondBestAlg, sep="\n") )
#             )) +
#   geom_tile(aes(fill = maxSpeedup)) +
#   geom_text(aes(label = paste(round(maxSpeedup,1), "x", sep = "")), position = position_nudge(y=0.2)) +
#   geom_text(aes(label = paste("\n", secondBestAlg, sep = ""), fontface = "bold"), position = position_nudge(y=0.05)) +
#   #geom_text(fontface="bold") +
#   # geom_point(aes(shape=as.factor(datasrc)), size=max_speedup) +
#   scale_x_discrete("Number of nodes") +
#   scale_y_discrete("Matrix size") +
#   scale_fill_gradient("", low = "white", high = "green") + #, limits = c(min_speedup,max_speedup)) +
#   theme_bw(20) + theme(legend.position = "none") +
#   theme(axis.text.x = element_text(angle = 90))
# print(p)
# ggsave(file="../lu_heatmap_labelled_filtered_compacted.svg", plot=p, width=w, height=h)
# dev.off()
# #}

