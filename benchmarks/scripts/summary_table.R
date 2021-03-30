# library(ggplot2)
# library(ggrepel)
# library(reshape2)
# library(plyr)
# library(hash)
# #library("reshape2")
# 
# 
# exp_name = ""
# exp_filename = "benchmarks.csv"
# scalings = c("weak", "strong")
# 
# variantPlots = c("time", "FLOPS", "bytes")
# algorithms = c("Cholesky", "LU")
# 
# sizes_strong = c(16384, 131072)
# sizes_weak = c(1024, 8192)
# sizes <- hash()
# sizes[["strong"]] <- sizes_strong
# sizes[["weak"]] <- sizes_weak
# 
# libraries_chol = c("MKL [cite]", "SLATE [cite]", "CAPITAL [cite]", "PsyChol (this work)")
# libraries_LU = c("MKL [cite]", "SLATE [cite]", "CANDMC [cite]", "CONFLUX (this work)")
# libraries <- hash()
# libraries[["LU"]] <- libraries_chol
# libraries[["Cholesky"]] <- libraries_LU
# #annotl = c("candmc [21]","CTF [49]","conflux (this work)", "mkl [14]")
# #varPlot = "FLOPS"
# 
# FLOPSperNode = 1209 
# 
# 
# 
# #exp_filename = paste(exp_name,'.csv',sep="")
# setwd("C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/conflux_cpp_2/results/conflux/benchmarks/scripts")
# setwd(paste("../",exp_name,sep =""))
# source(paste(getwd(), "/scripts/SPCL_Stats.R", sep=""))
# 
# # prepare the data 
# rawData <- read.csv(file=exp_filename, sep=",", stringsAsFactors=FALSE, header=TRUE)
# 
# rawData <- rawData[rawData$unit == "time",]
# 
# rawData[str_cmp(rawData$library, "capital"),]$library = "candmc"
# # rawData[str_cmp(rawData$library, "capital"),]$library = "CANDMC/CAPITAL"
# # rawData[str_cmp(rawData$library, "candmc"),]$library = "CANDMC/CAPITAL"
# 
# # if(nrow(rawData[str_cmp("conflux", rawData$library),]) > 0) {
# #   rawData[str_cmp("conflux", rawData$library),]$library = "COnlLUX/PsyChol"
# # }
# 
# # if (nrow(rawData[str_cmp("psychol", rawData$library),]) > 0) {
# #   rawData[str_cmp("psychol", rawData$library),]$library = "COnlLUX/PsyChol"
# # }
# if (nrow(rawData[str_cmp("psychol", rawData$library),]) > 0) {
#   rawData[str_cmp("psychol", rawData$library),]$library = "conflux"
# }
# 
# rawData$case = paste(rawData$algorithm, rawData$N_base,sep ="_")
# 
# interesting_columns = c("case", "library", "N", "P", "value")
# data <- rawData[interesting_columns]
# data <- data[complete.cases(data),]
# meanMeasrs <- as.data.frame(data %>% group_by(case, library, N, P) %>% summarise_each(list(mean)))
# 
# 



library(ggplot2)
library(ggrepel)
library(reshape2)
library(dplyr)
library(hash)
library(gmodels)



#-------------------------SETUP----------------------#
path = "C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/conflux_cpp_2/results/conflux/benchmarks" #getwd()
exp_filename = "benchmarks.csv"
#exp_filename = "rawData_old.csv"
setwd(path)
source(paste(path, "scripts/SPCL_Stats.R", sep="/"))
scalings = c("strong_131072", "strong_16384", "weak_8192")
matrixShapes = c("LU","Cholesky")
variantPlots = c("commVol", "time", "flops")
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
#annotCoordX[["square_memory_p1_time"]] = 


#exp_filename = paste(exp_name,'.csv',sep="")
#setwd(paste(path,exp_name,sep =""))



#--------------------PREPROCESSING-----------------------------#

rawData = read.table(exp_filename, header = T, sep = ',',fill = TRUE, stringsAsFactors=TRUE)

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
#confluxModel <-read.table("conflux_model.csv", header = T, sep = ' ',fill = TRUE)

#rawData[rawData$algorithm == "conflux",]$V = confluxModel$V2 / 1000 * confluxModel$P * 1000
#rawData$P = rawData$P * 36


#filtering candmc by non powers of two
rawDataTmp = rawData
rawData = rawDataTmp[rawDataTmp$library != "candmc",]
rawData = rbind(rawData,rawDataTmp[rawDataTmp$library == "candmc" & (0.5 - abs(0.5-log2(rawDataTmp$P)%%1)) < 0.05, ])

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


rawData$scaling = ifelse(rawData$N_base == 16384, 'strong 16384',
                        ifelse(rawData$N_base == 131072, 'strong 131072',
                               ifelse(rawData$N_base == 8192, 'weak 8192',
                                      ifelse(rawData$N_base == 1024, 'weak 1024', 'ERROR!')
                               )
                        )
)


# rawData$scaling = ifelse((rawData$m == strong1DShort & rawData$k == strong1Dlong) |
#         (rawData$m == strong1Dlong & rawData$k == strong1DShort) |
#         (rawData$m == strongFlatLong & rawData$k == strongFlatShort) |
#         (rawData$m == strongSquare & rawData$k == strongSquare), 'strong',
#         ifelse(rawData$m/rawData$S * rawData$N + rawData$m/rawData$S * rawData$k + rawData$k /rawData$S * rawData$N > 0.9 * rawData$P, 'memory_p0',
#             ifelse(rawData$m /rawData$S * rawData$N + rawData$m /rawData$S * rawData$k + rawData$k /rawData$S * rawData$N < 1.1*rawData$S * rawData$P^(3/2), 'memory_p1','ERROR!')
#      )
# )

rawrawData <- rawData

rawData <- rawData[rawData$unit == "time",]
rawData <- rawData[complete.cases(rawData),]
rawData <- as.data.frame(rawData %>% group_by(algorithm, library, N, N_base, P, grid, unit, type, blocksize, case, algLabel, mShape, scaling) %>% summarise_each(list(mean)))


#rawData$time = apply(rawData[c('t1','t2','t3')], 1, FUN=min)
rawData$time = rawData$value

rawData = rawData[complete.cases(rawData),]

rawData$flops = 0
rawData[str_cmp(rawData$algorithm, "lu"),]$flops = 
  200/3 * (rawData[str_cmp(rawData$algorithm, "lu"),]$N)^3 / (1e6 * rawData[str_cmp(rawData$algorithm, "lu"),]$P * rawData[str_cmp(rawData$algorithm, "lu"),]$value * GFLOPSperNode)

rawData[str_cmp(rawData$algorithm, "cholesky"),]$flops = 
  100/3 * (rawData[str_cmp(rawData$algorithm, "cholesky"),]$N)^3 / (1e6 * rawData[str_cmp(rawData$algorithm, "cholesky"),]$P * rawData[str_cmp(rawData$algorithm, "cholesky"),]$value * GFLOPSperNode)


#
# detach("package:reshape2",unload = TRUE)
# detach("package:ggrepel",unload = TRUE)
# detach("package:ggplot2",unload = TRUE)
# detach("package:plyr",unload = TRUE)

if (!('plyr' %in% (.packages()))) {
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
  dataSummary2 = reshape(dataSummary,timevar="library",idvar=c("N","P","mShape", "scaling"),direction="wide")
  dataSummary2[is.na(dataSummary2)] <- 99999999
  dataSummary2<-dataSummary2[!(dataSummary2$"med_time.conflux"==99999999),]
  dataSummary2$secondBestTime = apply(dataSummary2[, c('med_time.candmc','med_time.slate','med_time.mkl')], 1, min)
  dataSummary2$secondBestAlg = remAlgs[apply(dataSummary2[, c('med_time.candmc','med_time.slate','med_time.mkl')], 1, which.min)]
  dataSummary2<-dataSummary2[!(dataSummary2$secondBestTime==99999999),]
  dataSummary2$maxSpeedup = dataSummary2$secondBestTime / dataSummary2$"med_time.conflux"
  
  if(nrow(dataSummary2[dataSummary2$maxSpeedup < 1,])) {
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Suspicious data!')
    print(dataSummary2[dataSummary2$maxSpeedup < 1,])
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
  }
  suspicious = dataSummary2[dataSummary2$maxSpeedup < 1,]
  
  statSummary = setNames(data.frame(matrix(ncol = 9, nrow = 0)), c("shape", "scaling", algorithms, "mean", "min", "max"))
}
statSummary = statSummary[0,]
rawData = rawData[!(rawData$mShape == 'square' & rawData$scaling == 'memory_p0' & rawData$time > 100000),]

#--------------------END OF PREPROCESSING-----------------------------#


#-------------communication model part---------#
#fixing comm vol per process
rawData$V = ifelse(rawData$library == "candmc", rawData$V / 2^floor(log2(rawData$P)) * 1e-6 / 2, rawData$V / rawData$P * 1e-6 / 2)

rawData$domSize_mn = (rawData$m / rawData$P * rawData$N * rawData$k)^(1/3)
apply(data.frame((rawData$m / rawData$P * rawData$N * rawData$k)^(1/3), sqrt(rawData$S)), 1, FUN=min)

rawData$domSize_mn = ifelse(rawData$library == "mkl", (rawData$m / rawData$P * rawData$N)^(1/2),
                            ifelse(rawData$library == "candmc",
                                   apply(data.frame((rawData$m / rawData$P * rawData$N * rawData$k)^(1/3), (sqrt(S/3))), 1, FUN=min),
                                   ifelse(rawData$library == "conflux", 
                                          apply(data.frame((rawData$m / rawData$P * rawData$N * rawData$k)^(1/3), sqrt(S)), 1, FUN=min),
                                          apply(data.frame((rawData$m / rawData$P * rawData$N * rawData$k)^(1/3), sqrt(S/2)), 1, FUN=min))
                            )
)

rawData$domSize_k = rawData$m / rawData$P * rawData$N /  (rawData$domSize_mn)^2 * rawData$k
rawData$commVolModel =  ifelse(rawData$library == "mkl",2*rawData$domSize_mn*rawData$domSize_k,
                               ifelse(rawData$domSize_k == rawData$k, 2*rawData$domSize_mn*rawData$domSize_k,
                                      (rawData$domSize_mn)^2 + 2*rawData$domSize_mn*rawData$domSize_k))*1e-6*8
rawData$commModelRatio = rawData$V /rawData$commVolModel

rawData$checkDomSize = (rawData$domSize_mn)^2 * rawData$domSize_k  - (rawData$m / rawData$P * rawData$N  * rawData$k )

#filtering incorrect data
#rawData = na.omit(rawData[rawData$commModelRatio > 0.5, ])
#-------------end of communication model part---------#


library(plyr)

for (i1 in 1:length(matrixShapes)) {
  mShape = matrixShapes[i1]
  dataFirst = rawData[rawData$mShape == mShape,]
  for (i2 in 1:length(scalings)) {
    scaling = scalings[i2]
    data = dataFirst[dataFirst$scaling == scaling, ]
    
    if (nrow(data) == 0)
      next
    
    for (i3 in 1:length(variantPlots)) {
      variant = variantPlots[i3]
      
      finalData = data[DataColumnsHash[[variant]]]
      finalData = finalData[complete.cases(finalData),]
      
      if (mShape == 'largeK' & scaling == 'strong' & variant == 'flops'){
        aaa = 1
      }
      
      #create statistics
      if (variant == "commVol") {
        confluxAvg = mean(na.omit((finalData[finalData$algLabel == annotl[1],]$V)))
        scaAvg = mean(na.omit((finalData[finalData$algLabel == annotl[2],]$V)))
        slateAvg = mean(na.omit((finalData[finalData$algLabel == annotl[3],]$V)))
        candmcAvg = mean(na.omit((finalData[finalData$algLabel == annotl[4],]$V)))
        statSummary[nrow(statSummary)+1,c('shape','scaling',algorithms)] = list(mShape, scaling, candmcAvg,slateAvg,confluxAvg,scaAvg)
      } else if (variant == "time"){
        gmMeanSpeedup = gm_mean(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
        maxSpeedup    =     max(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
        minSpeedup    =     min(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
        statSummary[nrow(statSummary),c('mean', 'min', 'max')] = list(gmMeanSpeedup,minSpeedup,maxSpeedup)
      }
    }
  }    
}

write.csv(rawData, file = "res.csv")
