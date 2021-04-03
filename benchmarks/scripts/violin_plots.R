library(ggplot2)
library(ggrepel)
library(reshape2)
library(dplyr)
library(hash)
library(gmodels)



#-------------------------SETUP----------------------#
exp_name = ""
exp_filename = "/../benchmarks.csv"
scalings = c("weak", "strong")

variantPlots = c("time", "FLOPS", "commVol")
algorithms = c("LU", "Cholesky")

sizes_strong = c(16384, 131072)
sizes_weak = c(1024, 8192)
sizes <- hash()
sizes[["strong"]] <- sizes_strong
sizes[["weak"]] <- sizes_weak

libraries_chol = c("MKL [cite]", "SLATE [cite]", "CAPITAL [cite]", "PsyChol (this work)")
libraries_LU = c("MKL [cite]", "SLATE [cite]", "CANDMC [cite]", "CONFLUX (this work)")
libraries <- hash()
libraries[["LU"]] <- libraries_chol
libraries[["Cholesky"]] <- libraries_LU
annotl = c("COnfLUX / PsyChol (this work)", "SLATE [cite]","MKL [cite]","CANDMC[cite] / CAPITAL[cite]")
#varPlot = "FLOPS"

FLOPSperNode = 1209 


statistics = 0

annotCoord = list()


# 
# path = getwd() #"C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/papers/MMM-paper/results/generic" #getwd()
# exp_filename = "res.csv"
# #exp_filename = "rawData_old.csv"
# #setwd(path)
# source("SPCK_Stats.R")
# scalings = c("strong", "memory_p0", "memory_p1")
# matrixShapes = c("square","largeK", "largeM","flat")
# variantPlots = c("commVol", "time", "flops")
# # algorithms = c("carma","cyclops","cosma","scalapack")
# # annotl = c("CARMA [21]","CTF [48]","COSMA (this work)", "ScaLAPACK [14]")
# algorithms = c("cosma","scalapack","cyclops","carma")
# annotl = c("COSMA (this work) ", "ScaLAPACK [14] ","CTF [48] ","CARMA [21] ")
# importantCols = c("mShape","scaling","algorithm", "p","m","n","k","time","flops", "V","commVolModel","commModelRatio")
# 
# results <- data.frame(matrix(ncol = 5, nrow = 0))
# rescolNames <- append(algorithms, "scenario", after = 0)
# colnames(results) <- rescolNames
# GFLOPSperCore = 1209/36
# S = 5e+09
# 
# strongSquare = 16384
# strong1DShort = 17408
# strong1Dlong = 3735552
# strongFlatShort = 2^9
# strongFlatLong = 2^17
# weakFlatShort = 256


#-------plotting-------#
# DataColumnsHash = hash()
# DataColumnsHash['commVol'] = c("p", "algLabel", "V")
# DataColumnsHash['time'] = c("p", "algLabel", "time")
# DataColumnsHash['flops'] = c("p", "algLabel", "flops")


yLabelHash = hash()
yLabelHash['commVol'] = "MB communicated per processor"
yLabelHash['time'] = "total time [ms]"
yLabelHash['FLOPS'] = "% peak performance"
yscale = scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))

shapes = scale_shape_manual(values=c(15, 16, 17,18))
shapesColors = scale_color_manual(values = c("#7CAE00", "#00BFC4",  "#C77CFF",  "#F8766D"))
aspRatio = 0.75
w = 10
textSize = 30
pointSize = 5



# setwd("C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/conflux_cpp_2/results/conflux/benchmarks/scripts")
# prepare the data 
#setwd(paste("../",exp_name,sep =""))
source("SPCL_Stats.R")
rawData <- read.csv(file=paste(getwd(), exp_filename, sep = ""), sep=",", stringsAsFactors=FALSE, header=TRUE)

rawData <- rawData[!(rawData$N == 16384 & rawData$P > 500),]

rawData[rawData$N_base == "-" & rawData$type == "strong",]$N_base <- rawData[rawData$N_base == "-" & rawData$type == "strong",]$N
rawData[rawData$N_base == "-" & rawData$type == "weak",]$N_base <- rawData[rawData$N_base == "-" & rawData$type == "weak",]$N / sqrt(rawData[rawData$N_base == "-" & rawData$type == "weak",]$P)

#-------------annot data points----------------#
annotXHash = hash()
annotYHash = hash()
annotPointX1Hash = hash()
annotPointX2Hash = hash()
annotPointY2Hash = hash()

# annotXHash['square;strong;commVol'] = c(9000,712,1524,256)
# annotYHash['square;strong;commVol'] = c(50,5,70,20)
# annotPointX1Hash['square;strong;commVol'] = c(4096,1024,512,128)
# annotPointX2Hash['square;strong;commVol'] = c(7000,800,700,150)
# annotPointY2Hash['square;strong;commVol'] = c(47,8,67,23)
# 
# annotXHash['square;memory_p0;commVol'] = c(200,200,8000,3100)
# annotYHash['square;memory_p0;commVol'] = c(15,35,86,50)
# annotPointX1Hash['square;memory_p0;commVol'] = c(256,468,2196,2048)
# annotPointX2Hash['square;memory_p0;commVol'] = c(200,335,3500,2500)
# annotPointY2Hash['square;memory_p0;commVol'] = c(18,35,83,47)
# 
# annotXHash['square;memory_p1;commVol'] = c(180,200,4048,4396)
# annotYHash['square;memory_p1;commVol'] = c(22,5,77,36)
# annotPointX1Hash['square;memory_p1;commVol'] = c(128,433,4096,2048)
# annotPointX2Hash['square;memory_p1;commVol'] = c(150,335,3500,3050)
# annotPointY2Hash['square;memory_p1;commVol'] = c(26,5,73,32)
# 
# annotXHash['largeK;strong;commVol'] = c(4096,5300,5300,14000)
# annotYHash['largeK;strong;commVol'] = c(10,52,65,5)
# annotPointX1Hash['largeK;strong;commVol'] = c(4096,7236,3348,11988)
# annotPointX2Hash['largeK;strong;commVol'] = c(4096,6500,4000,11988)
# annotPointY2Hash['largeK;strong;commVol'] = c(13,50,68,7)
# 
# annotXHash['largeK;memory_p0;commVol'] = c(512,200,506,1224)
# annotYHash['largeK;memory_p0;commVol'] = c(42,5,74,32)
# annotPointX1Hash['largeK;memory_p0;commVol'] = c(1024,256,512,1024)
# annotPointX2Hash['largeK;memory_p0;commVol'] = c(900,200,400,900)
# annotPointY2Hash['largeK;memory_p0;commVol'] = c(45,8,70,28)
# 
# annotXHash['largeK;memory_p1;commVol'] = c(306,200,2024,800)
# annotYHash['largeK;memory_p1;commVol'] = c(35,5,55,24)
# annotPointX1Hash['largeK;memory_p1;commVol'] = c(128,256,1024,468)
# annotPointX2Hash['largeK;memory_p1;commVol'] = c(150,200,1200,600)
# annotPointY2Hash['largeK;memory_p1;commVol'] = c(35,8,52,22)



#--------------------END of SETUP---------------------#


statistics = 0

annotCoord = list()


#--------------generating the violion plot--------------#
# this is for full paper width
aspectRatio = 4.4
# this is for one column width
aspectRatio = 5.5
hg = 3

#violinData = rawData[rawData$unit == 'time',]
violinData <- find_optimal_blocks(rawData)
violinData$case = paste(violinData$algorithm, violinData$N_base,sep ="_")
#violinData = violinData[violinData$case != "memory_p2_FALSE",]
#violinData = violinData[violinData$case != "strong_TRUE",]
if (nrow(violinData[violinData$case == "lu_16384",]) > 0){
  violinData[violinData$case == "lu_16384",]$case = "LU, N=16,384"
  violinData[violinData$case == "lu_131072",]$case = "LU, N=131,072"
  violinData[violinData$case == "lu_8192",]$case = "LU, N=8,192 sqrt(P)"
}
violinData[violinData$case == "cholesky_16384",]$case = "Cholesky, N=16,384"
violinData[violinData$case == "cholesky_131072",]$case = "Cholesky, N=131,072"
violinData[violinData$case == "cholesky_8192",]$case = "Cholesky, N=8,192 sqrt(P)"

violinData$flops = 0
violinData[str_cmp(violinData$algorithm, "LU"),]$flops = 
  200/3 * (violinData[str_cmp(violinData$algorithm, "LU"),]$N)^3 / (1e6 * (violinData[str_cmp(violinData$algorithm, "LU"),]$P/2) * violinData[str_cmp(violinData$algorithm, "LU"),]$value * FLOPSperNode)

violinData[str_cmp(violinData$algorithm, "Cholesky"),]$flops = 
  100/3 * (violinData[str_cmp(violinData$algorithm, "Cholesky"),]$N)^3 / (1e6 * (violinData[str_cmp(violinData$algorithm, "Cholesky"),]$P/2) * violinData[str_cmp(violinData$algorithm, "Cholesky"),]$value * FLOPSperNode)


violinData[str_cmp(violinData$library, "capital"),]$library = "CANDMC/CAPITAL"
violinData[str_cmp(violinData$library, "candmc"),]$library = "CANDMC/CAPITAL"

if(nrow(violinData[str_cmp("conflux", violinData$library),]) > 0) {
  violinData[str_cmp("conflux", violinData$library),]$library = "COnlLUX/PsyChol"
}

if (nrow(violinData[str_cmp("psychol", violinData$library),]) > 0) {
  violinData[str_cmp("psychol", violinData$library),]$library = "COnlLUX/PsyChol"
}


# violinData = violinData[complete.cases(violinData),]

#filter out cases. We don't need strong scaling N = 1024?
violinData <- violinData[violinData$N_base != 1024,]


pdf(paste(getwd(), "/../barPlot2.pdf", sep = ""), height = hg, width = hg * aspectRatio)
violinData$library <- as.factor(violinData$library)
violinData$library2 = factor(violinData$library, levels = levels(violinData$library)[c(2,4,3,1)])

confluxData = violinData[violinData$library == "COnlLUX/PsyChol",]

p = ggplot(violinData, aes(x = library2, y = flops, fill = library2)) +
  geom_violin(width=TRUE) +
  #geom_boxplot(notch=TRUE) +
  facet_grid(.~case) +
  scale_y_continuous("% peak performance", limits = c(0,60)) +
  scale_fill_discrete(labels = annotl)+
  theme_bw(17) +
  # annotate("label", x = 0.3, y = 0.8, label = "from left to right: ")  +
  # ylim(0, 110) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.position = c(0.5,0.87),
        legend.title=element_blank(),
        legend.text=element_text(size=17)
  ) +
  guides(fill=guide_legend(nrow=1,byrow=TRUE),
         keywidth=19.5,
         keyheight=2.9,
         default.unit="inch")
print(p)
dev.off()

#----------end of generating the violion plot-----------


#--------- 1st page bat plot ----------------

violinData$library <- as.factor(violinData$library)
violinData$library2 = factor(violinData$library, levels = levels(violinData$library)[c(2,4,3,1)])

confluxData = violinData[violinData$library == "COnlLUX/PsyChol",]

#final_data <- find_statistics(violinData)

time_data <- violinData[violinData$unit == "time",]

relevant_cols <- c("algorithm", "library", "N", "case", "P", "flops")
time_data <- time_data[relevant_cols]

peak_flops <- as.data.frame(time_data %>% group_by(algorithm, case, library, N, P) %>% summarise_each(list(max), flops))
peak_flops$metric = "peak"
mean_flops <- as.data.frame(time_data %>% group_by(algorithm, case, library, N, P) %>% summarise_each(list(mean), flops))
mean_flops$metric = "mean"
final_data <- rbind(peak_flops, mean_flops)


pdf(paste(getwd(), "/../bars.pdf", sep = ""), height = 10, width = 10 * 2.1)

p = ggplot(final_data, aes(x = library, y = flops, fill = metric)) +
  geom_bar(stat="identity") +
  facet_grid(.~case) +
  scale_y_continuous("% peak performance", limits = c(0,75)) +
  scale_fill_discrete() + #labels = annotl)+
  theme_bw(17) +
  # annotate("label", x = 0.3, y = 0.8, label = "from left to right: ")  +
  # ylim(0, 110) +
  theme(legend.position = c(0.5,0.90),
        legend.title=element_blank(),
        legend.text=element_text(size=17),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)
  ) +
  guides(fill=guide_legend(nrow=1,byrow=TRUE),
         keywidth=19.5,
         keyheight=2.9,
         default.unit="inch")
print(p)
dev.off()
# --------end of 1st page bar plot----------