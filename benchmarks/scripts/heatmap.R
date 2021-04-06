
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
annotl = c("COnfLUX/PsyChol (this work) ", "MKL [cite] ","SLATE [cite] ","CANDMC/CAPITAL [cite] ")
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
rawData[str_cmp(rawData$library, "capital"),]$library = "candmc"
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

#for (alg in matrixShapes){
data <- heatmap_data[heatmap_data$mShape == "cholesky",]
pdf(file= "../cholesky_heatmap_labelled.pdf",  width = w, height = w*aspRatio)
ggplot(data=data, aes(x=as.factor(P), y=as.factor(N), 
                      fill=maxSpeedup, 
                      label=paste(round(maxSpeedup,1), 
                                  secondBestAlg, sep="\n") )) +
  geom_tile(aes(fill = maxSpeedup)) +
  geom_text() +
  # geom_text(fontface="bold") +
  # geom_point(aes(shape=as.factor(datasrc)), size=3) +
  scale_x_discrete("Number of nodes") +
  scale_y_discrete("Matrix Size") +
  scale_fill_gradient("", low = "orange", high = "green") +
  theme_bw(20) + theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle = 90))
dev.off()


data <- heatmap_data[heatmap_data$mShape == "lu",]
pdf(file= "../lu_heatmap_labelled.pdf",  width = w, height = w*aspRatio)
ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
                      fill=maxSpeedup,
                      label=paste(round(maxSpeedup,1),
                                  secondBestAlg, sep="\n") )) +
  geom_tile(aes(fill = maxSpeedup)) +
  geom_text() +
  # geom_text(fontface="bold") +
  # geom_point(aes(shape=as.factor(datasrc)), size=3) +
  scale_x_discrete("Available Nodes") +
  scale_y_discrete("Matrix Size [N]") +
  scale_fill_gradient("", low = "orange", high = "green") +
  theme_bw(20) + theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle = 90))
dev.off()


# -------------------- plots with data filtered out based on achieved peak performance ------------ #
# --------------------- FILTERING THRESHOLD (in percent): ---------- #
threshold = 3

data <- heatmap_data[heatmap_data$mShape == "cholesky" & heatmap_data$peak_flops > threshold,]
pdf(file= "../cholesky_heatmap_labelled_filtered.pdf",  width = w, height = w*aspRatio)
ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
                      fill=maxSpeedup,
                      label=paste(round(maxSpeedup,1),
                                  secondBestAlg, sep="\n") )) +
  geom_tile(aes(fill = maxSpeedup)) +
  geom_text() +
  # geom_text(fontface="bold") +
  # geom_point(aes(shape=as.factor(datasrc)), size=3) +
  scale_x_discrete("Available Nodes") +
  scale_y_discrete("Matrix Size [N]") +
  scale_fill_gradient("", low = "orange", high = "green") +
  theme_bw(20) + theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle = 90))
dev.off()

data <- heatmap_data[heatmap_data$mShape == "lu" & heatmap_data$peak_flops > threshold,]
pdf(file= "../lu_heatmap_labelled_filtered.pdf",  width = w, height = w*aspRatio)
ggplot(data=data, aes(x=as.factor(P), y=as.factor(N),
                      fill=maxSpeedup,
                      label=paste(round(maxSpeedup,1),
                                  secondBestAlg, sep="\n") )) +
  geom_tile(aes(fill = maxSpeedup)) +
  geom_text() +
  # geom_text(fontface="bold") +
  # geom_point(aes(shape=as.factor(datasrc)), size=3) +
  scale_x_discrete("Available Nodes") +
  scale_y_discrete("Matrix Size [N]") +
  scale_fill_gradient("", low = "orange", high = "green") +
  theme_bw(20) + theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle = 90))
dev.off()
#}

