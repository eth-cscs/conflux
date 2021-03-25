#library(ggplot2)
#library(ggrepel)
#library(reshape2)
#library(plyr)
#library("reshape2")


path = getwd()
exp_filename = "comm_files_aggr.csv"
variants = c("strong", "memory_p0", "memory_p1")
sizes = c("square","tall")
variantPlots = c("commVol")
algorithms = c("cosma","scalapack","cyclops", "carma")
annotl = c("CARMA [21]","CTF [48]","COSMA (this work)", "ScaLAPACK [14]")

results <- data.frame(matrix(ncol = 5, nrow = 0))
rescolNames <- append(algorithms, "scenario", after = 0)
colnames(results) <- rescolNames

GFLOPSperCore = 1209/36

S = 5e+09


statistics = 0

annotCoord = list()
#annotCoordX[["square_memory_p1_time"]] = 


#exp_filename = paste(exp_name,'.csv',sep="")
#setwd(paste(path,exp_name,sep =""))
setwd(path)


# prepare the data 

rawData <- read.csv(file=exp_filename, sep=",", stringsAsFactors=FALSE, header=TRUE)
#rawData$p = rawData$p * 36
rawData$V = rawData$V / ((rawData$p)^(1/2)-1)^2 * 1e-6 / 2

#rawDataTmp = rawData
#rawData = rawDataTmp[rawDataTmp$algorithm != "carma",]
#rawData = rbind(rawData,rawDataTmp[rawDataTmp$algorithm == "carma" & log2(rawDataTmp$p)%%1 == 0, ])


rawData$domSize_mn = ifelse(rawData$algorithm == "scalapack", (rawData$m / rawData$p * rawData$n)^(1/2),
                            ifelse(rawData$algorithm == "carma", min((rawData$m / rawData$p * rawData$n * rawData$k)^(1/3), (sqrt(S/3))),
                                   ifelse(rawData$algorithm == "cosma", min((rawData$m / rawData$p * rawData$n * rawData$k)^(1/3), sqrt(S)),
                                          min((rawData$m  / rawData$p * rawData$n * rawData$k)^(1/3), (sqrt(S/2))))
                            )
)

rawData$domSize_k = rawData$m / rawData$p * rawData$n /  (rawData$domSize_mn)^2 * rawData$k
rawData$commVolModel =  ifelse(rawData$algorithm == "scalapack",2*rawData$domSize_mn*rawData$domSize_k,
                               ifelse(rawData$domSize_k == rawData$k, 2*rawData$domSize_mn*rawData$domSize_k,
                                      (rawData$domSize_mn)^2 + 2*rawData$domSize_mn*rawData$domSize_k))*1e-6*8
#rawData$commVolModel =  ((rawData$domSize_mn)^2 + 2*rawData$domSize_mn*rawData$domSize_k)*1e-6
rawData$commModelRatio = rawData$V /rawData$commVolModel

rawData$checkDomSize = (rawData$domSize_mn)^2 * rawData$domSize_k  - (rawData$m / rawData$p * rawData$n  * rawData$k )


for (i1 in 1:length(sizes)) {
  size = sizes[i1]
  if (size == 'square') {
    dataFirst = rawData[rawData$m == rawData$k,]
  } else {
    dataFirst = rawData[rawData$m < rawData$k,]
  }
  for (i2 in 1:length(variants)) {
    variant = variants[i2]
    if ((variant == 'strong') & (size == 'square')) {
      data = dataFirst[dataFirst$m == 16384 & dataFirst$k==16384,]
    }
    if ((variant == 'strong') & (size == 'tall')) {
      data = dataFirst[dataFirst$m == 8704 & dataFirst$k==933888,]
    }
    if ((variant == 'memory_p0') & (size == 'square')) {
      data = dataFirst[(dataFirst$p > 900* (3*dataFirst$m ^2) / S) & (dataFirst$p * S / (3*dataFirst$m ^2) < 901), ]
      #      data <- filter(dataFirst,(dataFirst["p"] > 900* (3*dataFirst["m"] ^2) / S) & (dataFirst["p"] * S / (3*dataFirst["m"] ^2) < 901))
    }
    if ((variant == 'memory_p0') & (size == 'tall')) {
      data = dataFirst[(dataFirst$p * S / (dataFirst$m ^2 + 2*dataFirst$m * dataFirst$k) > 1340) &
                         (dataFirst$p * S / (dataFirst$m ^2 + 2*dataFirst$m * dataFirst$k) < 1350), ]
    }
    if ((variant == 'memory_p1') & (size == 'square')) {
      data = dataFirst[((dataFirst$p)^(2/3) > 272* (3*dataFirst$m ^2) / S) & ((dataFirst$p)^(2/3) * S / (3*dataFirst$m ^2) < 273), ]
      #      data <- filter(dataFirst,(dataFirst["p"] > 900* (3*dataFirst["m"] ^2) / S) & (dataFirst["p"] * S / (3*dataFirst["m"] ^2) < 901))
    }
    if ((variant == 'memory_p1') & (size == 'tall')) {
      data = dataFirst[((dataFirst$p)^(2/3) * S / (dataFirst$m ^2 + 2*dataFirst$m * dataFirst$k) > 405) &
                         ((dataFirst$p)^(2/3) * S / (dataFirst$m ^2 + 2*dataFirst$m * dataFirst$k) < 407), ]
    }
    
    if (nrow(data) == 0)
      next
    
    Vscaling <- data[c("p", "algorithm", "V", "commVolModel")]
    m = data$m

    
    
    #create statistics
    #algorithms = c("cosma","scalapack","cyclops", "carma")
    cosmaAvg = mean(na.omit((Vscaling[Vscaling$algorithm == "cosma",]$V)))
    scaAvg = mean(na.omit((Vscaling[Vscaling$algorithm == "scalapack",]$V)))
    cyclopsAvg = mean(na.omit((Vscaling[Vscaling$algorithm == "cyclops",]$V)))
    carmaAvg = mean(na.omit((Vscaling[Vscaling$algorithm == "carma",]$V)))
    results[nrow(results)+1,] = list(paste("size_",size, "_var_", variant, sep=""), cosmaAvg,scaAvg,cyclopsAvg,carmaAvg)
  }
}

write.csv(rawData, file = "res.csv")

