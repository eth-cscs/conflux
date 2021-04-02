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



setwd("C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/conflux_cpp_2/results/conflux/benchmarks/scripts")
# prepare the data 
#setwd(paste("../",exp_name,sep =""))
source("SPCL_Stats.R")
rawData <- read.csv(file=paste(getwd(), exp_filename, sep = ""), sep=",", stringsAsFactors=FALSE, header=TRUE)

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
#annotCoordX[["square_memory_p1_time"]] = 


#exp_filename = paste(exp_name,'.csv',sep="")
#setwd(paste(path,exp_name,sep =""))

# 
# #--------------------PREPROCESSING-----------------------------#
# 
# rawData = read.table(exp_filename, header = T, sep = ',',fill = TRUE, stringsAsFactors=TRUE)
# #rawData = rawData[rawData$S < Inf,]
# #rawData$V = rawData$V/3
# 
# setups <- rawData[rawData$library == "mkl",]
# setups <-setups[c("P", "N_base")] #, "n", "k", "S", "V")]
# #options("scipen"=100, "digits"=4)
# write.table(setups, file = "setups.csv",sep = " ",col.names = FALSE, row.names = FALSE)
# cosmaModel <-read.table("cosma_model.csv", header = T, sep = ' ',fill = TRUE)
# 
# #rawData[rawData$algorithm == "cosma",]$V = cosmaModel$V2 / 1000 * cosmaModel$p * 1000
# #rawData$p = rawData$p * 36
# 
# 
# #filtering CARMA by non powers of two
# rawDataTmp = rawData
# rawData = rawDataTmp[rawDataTmp$algorithm != "carma",]
# rawData = rbind(rawData,rawDataTmp[rawDataTmp$algorithm == "carma" & (0.5 - abs(0.5-log2(rawDataTmp$p)%%1)) < 0.05, ])
# 
# rawData$algLabel = ifelse(rawData$algorithm == "scalapack", annotl[2],
#                            ifelse(rawData$algorithm == "carma", annotl[4],
#                                   ifelse(rawData$algorithm == "cosma", annotl[1],
#                                          annotl[3])
#                            )
# )
# rawData$algLabel <- as.factor(rawData$algLabel)
# rawData$algLabel = factor(rawData$algLabel, levels = levels(rawData$algLabel)[c(2,4,3,1)])
# 
# rawData$mShape = ifelse(rawData$m == rawData$n  & rawData$m == rawData$k, 'square',
#                        ifelse(rawData$m == rawData$n  & rawData$m > rawData$k, 'flat',
#                               ifelse(rawData$m == rawData$n  & rawData$m < rawData$k, 'largeK',
#                                      ifelse(rawData$n == rawData$k  & rawData$m > rawData$k, 'largeM', 'ERROR!')
#                               )
#                        )
# )
# 
# 
# 
# rawData$scaling = 'type 1 ERROR!'
# 
# rawData$scaling = ifelse(grepl('strong',rawData$setup), 'strong',
#                          ifelse(grepl('weak_p0',rawData$setup), 'memory_p0',
#                                 'memory_p1')
# )
#                               
# # rawData$scaling = ifelse((rawData$m == strong1DShort & rawData$k == strong1Dlong) |
# #         (rawData$m == strong1Dlong & rawData$k == strong1DShort) |
# #         (rawData$m == strongFlatLong & rawData$k == strongFlatShort) |
# #         (rawData$m == strongSquare & rawData$k == strongSquare), 'strong',
# #         ifelse(rawData$m/rawData$S * rawData$n + rawData$m/rawData$S * rawData$k + rawData$k /rawData$S * rawData$n > 0.9 * rawData$p, 'memory_p0',
# #             ifelse(rawData$m /rawData$S * rawData$n + rawData$m /rawData$S * rawData$k + rawData$k /rawData$S * rawData$n < 1.1*rawData$S * rawData$p^(3/2), 'memory_p1','ERROR!')
# #      )
# # )
# 
#   
# #rawData$time = apply(rawData[c('t1','t2','t3')], 1, FUN=min)
# rawData$time = apply(rawData[c('t1')], 1, FUN=min)
# rawData$flops = 200* rawData$m * rawData$n * rawData$k / (rawData$time * 1e6) / (GFLOPSperCore * rawData$p)
# 
# #
# # detach("package:reshape2",unload = TRUE)
# # detach("package:ggrepel",unload = TRUE)
# # detach("package:ggplot2",unload = TRUE)
# # detach("package:plyr",unload = TRUE)
# 
# if (!('plyr' %in% (.packages()))) {
#   #finding median and confidence intervals
#   dataSummary =
#     rawData %>%
#     group_by(m,n,k,p,algorithm, mShape, scaling) %>%
#     summarise(med_time = median(time)) %>% #, lci = ci(time)["CI lower"], uci = ci(time)["CI upper"], count = n()) %>%
#     ungroup() %>%
#    # filter(algorithm == "COSMA (this work) ") %>%
#     as.data.frame()
#   
#   remAlgs = c('carma', 'cyclops', 'scalapack')
#   
#   #second-best algorithm
#   dataSummary2 = reshape(dataSummary,timevar="algorithm",idvar=c("m","n","k","p","mShape", "scaling"),direction="wide")
#   dataSummary2[is.na(dataSummary2)] <- 99999999
#   dataSummary2<-dataSummary2[!(dataSummary2$"med_time.cosma"==99999999),]
#   dataSummary2$secondBestTime = apply(dataSummary2[, c('med_time.carma','med_time.cyclops','med_time.scalapack')], 1, min)
#   dataSummary2$secondBestAlg = remAlgs[apply(dataSummary2[, c('med_time.carma','med_time.cyclops','med_time.scalapack')], 1, which.min)]
#   dataSummary2<-dataSummary2[!(dataSummary2$secondBestTime==99999999),]
#   dataSummary2$maxSpeedup = dataSummary2$secondBestTime / dataSummary2$"med_time.cosma"
#   
#   if(nrow(dataSummary2[dataSummary2$maxSpeedup < 1,])) {
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('Suspicious data!')
#     print(dataSummary2[dataSummary2$maxSpeedup < 1,])
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#   }
#   suspicious = dataSummary2[dataSummary2$maxSpeedup < 1,]
#   
#   statSummary = setNames(data.frame(matrix(ncol = 9, nrow = 0)), c("shape", "scaling", algorithms, "mean", "min", "max"))
# }
# statSummary = statSummary[0,]
# rawData = rawData[!(rawData$mShape == 'square' & rawData$scaling == 'memory_p0' & rawData$time > 100000),]
# 
# #--------------------END OF PREPROCESSING-----------------------------#
# 
# 
# #-------------communication model part---------#
# #fixing comm vol per process
# rawData$V = ifelse(rawData$algorithm == "carma", rawData$V / 2^floor(log2(rawData$p)) * 1e-6 / 2, rawData$V / rawData$p * 1e-6 / 2)
# 
# rawData$domSize_mn = (rawData$m / rawData$p * rawData$n * rawData$k)^(1/3)
# apply(data.frame((rawData$m / rawData$p * rawData$n * rawData$k)^(1/3), sqrt(rawData$S)), 1, FUN=min)
# 
# rawData$domSize_mn = ifelse(rawData$algorithm == "scalapack", (rawData$m / rawData$p * rawData$n)^(1/2),
#                             ifelse(rawData$algorithm == "carma",
#                                    apply(data.frame((rawData$m / rawData$p * rawData$n * rawData$k)^(1/3), (sqrt(S/3))), 1, FUN=min),
#                                    ifelse(rawData$algorithm == "cosma", 
#                                           apply(data.frame((rawData$m / rawData$p * rawData$n * rawData$k)^(1/3), sqrt(S)), 1, FUN=min),
#                                           apply(data.frame((rawData$m / rawData$p * rawData$n * rawData$k)^(1/3), sqrt(S/2)), 1, FUN=min))
#                             )
# )
# 
# rawData$domSize_k = rawData$m / rawData$p * rawData$n /  (rawData$domSize_mn)^2 * rawData$k
# rawData$commVolModel =  ifelse(rawData$algorithm == "scalapack",2*rawData$domSize_mn*rawData$domSize_k,
#                                ifelse(rawData$domSize_k == rawData$k, 2*rawData$domSize_mn*rawData$domSize_k,
#                                       (rawData$domSize_mn)^2 + 2*rawData$domSize_mn*rawData$domSize_k))*1e-6*8
# rawData$commModelRatio = rawData$V /rawData$commVolModel
# 
# rawData$checkDomSize = (rawData$domSize_mn)^2 * rawData$domSize_k  - (rawData$m / rawData$p * rawData$n  * rawData$k )
# 
# #filtering incorrect data
# #rawData = na.omit(rawData[rawData$commModelRatio > 0.5, ])
# #-------------end of communication model part---------#
# 
# 
# library(plyr)
# 
# for (i1 in 1:length(matrixShapes)) {
#   mShape = matrixShapes[i1]
#   dataFirst = rawData[rawData$mShape == mShape,]
#   for (i2 in 1:length(scalings)) {
#     scaling = scalings[i2]
#     data = dataFirst[dataFirst$scaling == scaling, ]
#     
#     if (nrow(data) == 0)
#       next
# 
#     for (i3 in 1:length(variantPlots)) {
#       variant = variantPlots[i3]
#   
#       setupName = paste(mShape, scaling, variant, sep=";")
#       finalData = data[DataColumnsHash[[variant]]]
#       finalData = finalData[complete.cases(finalData),]
#       ylabel = yLabelHash[[variant]]
#       yscale = scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
#     
#       #algorithms = c("carma","cyclops","cosma","scalapack")
#       #annotl = c("CARMA [21]","CTF [48]","COSMA (this work)", "ScaLAPACK [14]")
#       if (mShape == 'largeK' & scaling == 'strong' & variant == 'flops'){
#         aaa = 1
#       }
#       
#       #create statistics
#       if (variant == "commVol") {
#         cosmaAvg = mean(na.omit((finalData[finalData$algLabel == annotl[1],]$V)))
#         scaAvg = mean(na.omit((finalData[finalData$algLabel == annotl[2],]$V)))
#         cyclopsAvg = mean(na.omit((finalData[finalData$algLabel == annotl[3],]$V)))
#         carmaAvg = mean(na.omit((finalData[finalData$algLabel == annotl[4],]$V)))
#         statSummary[nrow(statSummary)+1,c('shape','scaling',algorithms)] = list(mShape, scaling, carmaAvg,cyclopsAvg,cosmaAvg,scaAvg)
#       } else if (variant == "time"){
#         gmMeanSpeedup = gm_mean(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
#         maxSpeedup    =     max(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
#         minSpeedup    =     min(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
#         statSummary[nrow(statSummary),c('mean', 'min', 'max')] = list(gmMeanSpeedup,minSpeedup,maxSpeedup)
#       }
#       
#     #  print(finalData[c("algLabel","p","V")])
#       annotX = annotXHash[[setupName]]
#       annotY = annotYHash[[setupName]]
#       annotPointX1 = annotPointX1Hash[[setupName]]
#       annotPointX2 = annotPointX2Hash[[setupName]]
#       annotPointY2 = annotPointY2Hash[[setupName]]
#       
#       annotPointY1 = c(finalData[finalData$p == annotPointX1[1] & finalData$algLabel == annotl[1],]$time[1],
#                        finalData[finalData$p == annotPointX1[2] & finalData$algLabel == annotl[2],]$time[1],
#                        finalData[finalData$p == annotPointX1[3] & finalData$algLabel == annotl[3],]$time[1],
#                        finalData[finalData$p == annotPointX1[4] & finalData$algLabel == annotl[4],]$time[1])
#       
#       if (is.null(annotX)) {
#         legendFlag = TRUE
#         plotAnnot = theme(legend.position = c(0.45,0.98),
#                           legend.title=element_blank(),
#                           legend.text=element_text(size=22),
#                           legend.direction="horizontal",
#                           text = element_text(size=textSize),
#                           aspect.ratio=aspRatio)
#       } else {
#         legendFlag = FALSE
#         plotAnnot = annotate("text", x = annotX, y = annotY, label = annotl, size=textSize/3) +
#           annotate("segment", x = annotPointX2[1], xend = annotPointX1[1],
#                    y = annotPointY2[1], yend = annotPointY1[1]) +
#           annotate("segment", x = annotPointX2[2], xend = annotPointX1[2],
#                    y = annotPointY2[2], yend = annotPointY1[2]) +
#           annotate("segment", x = annotPointX2[3], xend = annotPointX1[3],
#                    y = annotPointY2[3], yend = annotPointY1[3]) +
#           annotate("segment", x = annotPointX2[4], xend = annotPointX1[4],
#                    y = annotPointY2[4], yend = annotPointY1[4]) 
#       }
# 
#       
#       # plot the timers
#       pdf(file=paste("shape_",mShape,"_scale_", scaling,  "_var_", variant , ".pdf", sep=""),
#           width = w, height = w*aspRatio)
#       limit = yscale
#       
#       if (variant == "commVol") {
#         p1 = ggplot(mapping=aes(x=p, y=finalData$V, fill=finalData$algLabel, color=finalData$algLabel, shape=finalData$algLabel)) +
#           geom_point(data=finalData, size = 4, show.legend=legendFlag)
#         
#       } else {
#         if (variant == "time") {
#           finalData = ddply(finalData, ~ algLabel+p, summarize, min=min(time), max=max(time), mean=median(time))
#         } else {
#           finalData = ddply(finalData, ~ algLabel+p, summarize, min=min(flops), max=max(flops), mean=median(flops))
#         }
#      finalData$algLabel2 <- as.factor(finalData$algLabel)
#    #   finalData$algLabel2 = factor(finalData$algLabel, levels = levels(finalData$algLabel)[c(2,4,3,1)])
#      #
#        firstSeries =  finalData[finalData$algLabel2 == annotl[1],]
#        secondSeries = finalData[finalData$algLabel2 == annotl[2],]
#        thirdSeries =  finalData[finalData$algLabel2 == annotl[3],]
#        fourthSeries = finalData[finalData$algLabel2 == annotl[4],]
#        finalData = finalData[0,]
#        finalData = rbind(firstSeries,secondSeries,thirdSeries,fourthSeries)
#      #
#    finalData$algLabel = factor(finalData$algLabel, levels = levels(finalData$algLabel)[c(4,1,3,2)])
# 
#     finalData$algLabel2 = factor(finalData$algLabel, levels = levels(finalData$algLabel)[c(1,3,2,4)])
#         
#         #p1 <- ggplot(mapping=aes(x=p, y=mean, ymin=min, ymax=max, fill=algLabel, color=algLabel, shape=algLabel)) +
#         p1 <- ggplot(mapping=aes(x=p, y=mean, ymin=min, ymax=max, fill=algLabel, color=algLabel, shape=algLabel)) +
#          # geom_ribbon(data=finalData[finalData$algLabel != annotl[4],], alpha=0.3, show.legend=legendFlag)+
#           geom_point(data=finalData, size = 4, show.legend=legendFlag)# +
#         #  geom_errorbar(data=finalData[finalData$algLabel == annotl[4],], width=0.1, size=1, show.legend=legendFlag)
#       }
#       
#       p = p1 + 
#         shapes + 
#         expand_limits(y = 1) +
#         scale_x_continuous(trans='log2',labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
#         scale_y_continuous(trans='log2',labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
#         xlab("# of cores") +
#         ylab(ylabel)+ 
#         theme_bw(27) + 
#         plotAnnot
#       
#       print(p)
#       dev.off()
#     }
#   }    
# }
# 
# write.csv(rawData, file = "res.csv")


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
  #geom_violin() +
  geom_boxplot(notch=TRUE) +
  facet_grid(.~case) +
  scale_y_continuous("% peak performance", limits = c(0,55)) +
  scale_fill_discrete(labels = annotl)+
  theme_bw(17) +
  # annotate("label", x = 0.3, y = 0.8, label = "from left to right: ")  +
  # ylim(0, 110) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.position = c(0.5,0.90),
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

pdf(paste(getwd(), "/../barPlot2.pdf", sep = ""), height = hg, width = hg * aspectRatio)
violinData$library <- as.factor(violinData$library)
violinData$library2 = factor(violinData$library, levels = levels(violinData$library)[c(2,4,3,1)])

confluxData = violinData[violinData$library == "COnlLUX/PsyChol",]

final_data <- rbind(peak_flops, mean_flops)

p = ggplot(final_data, aes(x = library, y = flops, fill = library)) +
  geom_bar(stat="metric") +
  facet_grid(.~case) +
  scale_y_continuous("% peak performance", limits = c(0,55)) +
  scale_fill_discrete(labels = annotl)+
  theme_bw(17) +
  # annotate("label", x = 0.3, y = 0.8, label = "from left to right: ")  +
  # ylim(0, 110) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.position = c(0.5,0.90),
        legend.title=element_blank(),
        legend.text=element_text(size=17)
  ) +
  guides(fill=guide_legend(nrow=1,byrow=TRUE),
         keywidth=19.5,
         keyheight=2.9,
         default.unit="inch")
print(p)
dev.off()
# --------end of 1st page bar plot----------