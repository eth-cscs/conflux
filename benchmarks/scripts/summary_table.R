
library(ggplot2)
library(ggrepel)
library(reshape2)
library(dplyr)
library(hash)
library(gmodels)



#-------------------------SETUP----------------------#
#path = "C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/conflux_cpp_2/results/conflux/benchmarks" #getwd()
path = "/mnt/c/gk_pliki/uczelnia/doktorat/performance_modelling/repo/conflux_cpp_2/results/conflux/benchmarks" #getwd()
exp_filename = "benchmarks.csv"
#exp_filename = "rawData_old.csv"
setwd(path)
source(paste(path, "scripts/SPCL_Stats.R", sep="/"))
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

rawData = read.table(exp_filename, header = T, sep = ',',fill = TRUE, stringsAsFactors=FALSE)

rawData <- rawData[!(rawData$N == 16384 & rawData$P > 500),]

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
#confluxModel <-read.table("conflux_model.csv", header = T, sep = ' ',fill = TRUE)

#rawData[rawData$algorithm == "conflux",]$V = confluxModel$V2 / 1000 * confluxModel$P * 1000
#rawData$P = rawData$P * 36


#filtering candmc by non powers of two
# rawDataTmp = rawData
# rawData = rawDataTmp[rawDataTmp$library != "candmc",]
# rawData = rbind(rawData,rawDataTmp[rawDataTmp$library == "candmc" & (0.5 - abs(0.5-log2(rawDataTmp$P)%%1)) < 0.05, ])

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

rawrawData <- rawData

####################### 
# 
# rawData[rawData$blocksize == "", ]$blocksize = 1
# time_data <- rawData[rawData$unit == "time",]
# time_data <- time_data[complete.cases(time_data),]
# comm_data <- rawData[rawData$unit == "bytes",]
# comm_data <- comm_data[complete.cases(comm_data),]
# fastest_blocks <- as.data.frame(time_data %>% group_by(algorithm, library, N, P) %>% summarise_each(list(min), value))
# comm_min_blocks <- as.data.frame(comm_data %>% group_by(algorithm, library, N, P) %>% summarise_each(list(min), value))
# rows_to_remove = c()
# 
# for (row in 1:nrow(time_data)) {
#   cur_blocksize = time_data[row, "blocksize"]
#   cur_p = time_data[row, "P"]
#   cur_n = time_data[row, "N"]
#   cur_lib = time_data[row, "library"]
#   cur_val = time_data[row, "value"]
#   cur_alg = time_data[row, "algorithm"]
#   all_values = time_data[time_data$algorithm == cur_alg & time_data$library == cur_lib & time_data$N == cur_n & time_data$P == cur_p & time_data$blocksize == cur_blocksize,]$value
#   best_block_value = fastest_blocks[fastest_blocks$algorithm == cur_alg & fastest_blocks$library == cur_lib & fastest_blocks$N == cur_n &fastest_blocks$P == cur_p, ]$value
#   
#   smallest_com_vol = comm_min_blocks[comm_min_blocks$algorithm == cur_alg & comm_min_blocks$library == cur_lib & comm_min_blocks$N == cur_n & comm_min_blocks$P == cur_p, ]$value
#   if (!identical(smallest_com_vol, numeric(0))){
#     time_data[row, "V"] = smallest_com_vol / cur_p * 1e-6
#   }
#   if (min(all_values) != best_block_value | cur_val > 1.3 * best_block_value){
#     rows_to_remove <- c(rows_to_remove, row)
#   }
# }
# filtered_data = time_data[-rows_to_remove, ]


####################



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
detach("package:ggplot2",unload = TRUE)

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

statSummary = statSummary[0,]
rawData = rawData[!(rawData$mShape == 'square' & rawData$scaling == 'memory_p0' & rawData$time > 100000),]

#--------------------END OF PREPROCESSING-----------------------------#


#-------------communication model part---------#
#fixing comm vol per process
#rawData <- rawrawData[rawrawData$unit == "bytes", ]
#rawData$V = ifelse(rawData$library == "candmc", rawData$V / 2^floor(log2(rawData$P)) * 1e-6 / 2, rawData$value / rawData$P * 1e-6 / 2)
#rawData$V = rawData$value / rawData$P * 1e-6
# 
# rawData$domSize_mn = (rawData$m / rawData$P * rawData$N * rawData$k)^(1/3)
# apply(data.frame((rawData$m / rawData$P * rawData$N * rawData$k)^(1/3), sqrt(rawData$S)), 1, FUN=min)
# 
# rawData$domSize_mn = ifelse(rawData$library == "mkl", (rawData$m / rawData$P * rawData$N)^(1/2),
#                             ifelse(rawData$library == "candmc",
#                                    apply(data.frame((rawData$m / rawData$P * rawData$N * rawData$k)^(1/3), (sqrt(S/3))), 1, FUN=min),
#                                    ifelse(rawData$library == "conflux", 
#                                           apply(data.frame((rawData$m / rawData$P * rawData$N * rawData$k)^(1/3), sqrt(S)), 1, FUN=min),
#                                           apply(data.frame((rawData$m / rawData$P * rawData$N * rawData$k)^(1/3), sqrt(S/2)), 1, FUN=min))
#                             )
# )
# 
# rawData$domSize_k = rawData$m / rawData$P * rawData$N /  (rawData$domSize_mn)^2 * rawData$k
# rawData$commVolModel =  ifelse(rawData$library == "mkl",2*rawData$domSize_mn*rawData$domSize_k,
#                                ifelse(rawData$domSize_k == rawData$k, 2*rawData$domSize_mn*rawData$domSize_k,
#                                       (rawData$domSize_mn)^2 + 2*rawData$domSize_mn*rawData$domSize_k))*1e-6*8

rawData$commVolModel =  ifelse(rawData$library == "mkl" | rawData$library == "slate", (rawData$N)^2 / sqrt(rawData$P),
                               ifelse(rawData$library == "conflux", (rawData$N)^2 / (rawData$P)^(2/3),
                                      5*(rawData$N)^2 / (rawData$P)^(2/3)))*1e-6*8

rawData$commModelRatio = rawData$V /rawData$commVolModel


#filtering incorrect data
#rawData = na.omit(rawData[rawData$commModelRatio > 0.5, ])
#-------------end of communication model part---------#


library(plyr)

for (i1 in 1:length(matrixShapes)) {
  mShape = matrixShapes[i1]
  dataFirst = rawData[rawData$algorithm == mShape,]
  for (i2 in 1:length(scalings)) {
    scaling = scalings[i2]
    data = dataFirst[dataFirst$scaling == scaling, ]
    
    if (nrow(data) == 0)
      next
    
    for (i3 in 1:length(variantPlots)) {
      variant = variantPlots[i3]
      
      finalData = data[DataColumnsHash[[variant]]]
      
      if (mShape == 'cholesky' & variant == 'commVol'){
        aaa = 1
      }
      
      #create statistics
      if (variant == "commVol") {
        confluxAvg = mean(na.omit((finalData[finalData$algLabel == annotl[1],]$V)))
        scaAvg = mean(na.omit((finalData[finalData$algLabel == annotl[2],]$V)))
        slateAvg = mean(na.omit((finalData[finalData$algLabel == annotl[3],]$V)))
        candmcAvg = mean(na.omit((finalData[finalData$algLabel == annotl[4],]$V)))
        statSummary[nrow(statSummary)+1,c('shape','scaling',algorithms)] = list(mShape, scaling, confluxAvg, scaAvg, slateAvg, candmcAvg)
      } else if (variant == "time"){
        gmMeanSpeedup = gm_mean(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
        maxSpeedup    =     max(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
        minSpeedup    =     min(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
        statSummary[nrow(statSummary),c('mean', 'min', 'max')] = list(gmMeanSpeedup,minSpeedup,maxSpeedup)
      }
    }
  }    
}

write.csv(statSummary, file = "statSummary.csv")
write.csv(rawData, file = "res.csv")
