library(reshape2)
library(dplyr)

#-------------------------SETUP----------------------#
path = getwd()
setwd(path)
exp_filename = "rawData.csv"
scalings = c("strong", "memory_p0", "memory_p1")
matrixShapes = c("square","largeK", "largeM","flat")
variantPlots = c("commVol", "time", "flops")
algorithms = c("carma","cyclops","cosma","scalapack")
annotl = c("CARMA [21]","CTF [48]","COSMA (this work)", "ScaLAPACK [14]")

results <- data.frame(matrix(ncol = 5, nrow = 0))
rescolNames <- append(algorithms, "scenario", after = 0)
colnames(results) <- rescolNames
GFLOPSperCore = 1209/36
S = 5e+09


strongSquare = 16384
strong1DShort = 17408
strong1Dlong = 3735552
strongFlatShort = 2^9
strongFlatLong = 2^17
weakFlatShort = 256

#--------------------END of SETUP---------------------#



#--------------------PREPROCESSING-----------------------------#

rawData <-read.table(exp_filename, header = T, sep = ',')
#rawData <- read.csv(file=exp_filename, sep=",", stringsAsFactors=FALSE, header=TRUE)

setups <- rawData[rawData$algorithm == "cosma",]
setups <-setups[c("p", "m", "n", "k", "S"),]
write.csv(setups, file = "setups.csv")
#rawData$p = rawData$p * 36

rawData$algLabel = ifelse(rawData$algorithm == "scalapack", annotl[4],
                          ifelse(rawData$algorithm == "carma", annotl[1],
                                 ifelse(rawData$algorithm == "cosma", annotl[3],
                                        annotl[2])
                          )
)

rawData$mShape = ifelse(rawData$m == rawData$n  & rawData$m == rawData$k, 'square',
                        ifelse(rawData$m == rawData$n  & rawData$m > rawData$k, 'flat',
                               ifelse(rawData$m == rawData$n  & rawData$m < rawData$k, 'largeK',
                                      ifelse(rawData$n == rawData$k  & rawData$m > rawData$k, 'largeM', 'ERROR!')
                               )
                        )
)



rawData$scaling = 'type 1 ERROR!'

rawData[rawData$mShape == 'flat',]$scaling = ifelse(rawData$m == strongFlatLong & rawData$k == strongFlatShort, 'strong',
                                                    ifelse(rawData$m > sqrt(rawData$S * rawData$p + 65536) - 5000 & rawData$k == weakFlatShort, 'memory_p0',
                                                           ifelse(rawData$m < sqrt(rawData$S * rawData$p^(2/3) + 65536) + 500 & rawData$k == weakFlatShort, 'memory_p1','ERROR!')
                                                    )
)

rawData$time = apply(rawData[c('t1','t2','t3')], 1, FUN=min)
rawData$flops = 200* rawData$m * rawData$n * rawData$k / (rawData$time * 1e6) / (GFLOPSperCore * rawData$p)


if (!('plyr' %in% (.packages()))) {
  #finding median and confidence intervals
  dataSummary =
    rawData %>%
    group_by(m,n,k,p,algorithm, mShape, scaling) %>%
    summarise(med_time = median(time)) %>% #, lci = ci(time)["CI lower"], uci = ci(time)["CI upper"], count = n()) %>%
    ungroup() %>%
    # filter(algorithm == "COSMA (this work) ") %>%
    as.data.frame()
  
  remAlgs = c('carma', 'cyclops', 'scalapack')
  
  #second-best algorithm
  dataSummary2 = reshape(dataSummary,timevar="algorithm",idvar=c("m","n","k","p","mShape", "scaling"),direction="wide")
  dataSummary2[is.na(dataSummary2)] <- 99999999
  dataSummary2<-dataSummary2[!(dataSummary2$"med_time.cosma"==99999999),]
  dataSummary2$secondBestTime = apply(dataSummary2[, c('med_time.carma','med_time.cyclops','med_time.scalapack')], 1, min)
  dataSummary2$secondBestAlg = remAlgs[apply(dataSummary2[, c('med_time.carma','med_time.cyclops','med_time.scalapack')], 1, which.min)]
  dataSummary2<-dataSummary2[!(dataSummary2$secondBestTime==99999999),]
  dataSummary2$maxSpeedup = dataSummary2$secondBestTime / dataSummary2$"med_time.cosma"
  
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
  
  
  statSummary = setNames(data.frame(matrix(ncol = 9, nrow = 0)), c("shape", "scaling", algorithms, "mean", "min", "max"))
}


#--------------------END OF PREPROCESSING-----------------------------#


#-------------communication model part---------#
#fixing comm vol per process
rawData$V = ifelse(rawData$algorithm == "carma", rawData$V / 2^floor(log2(rawData$p)) * 1e-6 / 2, rawData$V / rawData$p * 1e-6 / 2)

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
rawData$commModelRatio = rawData$V /rawData$commVolModel

rawData$checkDomSize = (rawData$domSize_mn)^2 * rawData$domSize_k  - (rawData$m / rawData$p * rawData$n  * rawData$k )

#filtering incorrect data
#rawData = na.omit(rawData[rawData$commModelRatio > 0.5, ])
#-------------end of communication model part---------#


#filtering CARMA by non powers of two
#rawDataTmp = rawData
#rawData = rawDataTmp[rawDataTmp$algorithm != "carma",]
#rawData = rbind(rawData,rawDataTmp[rawDataTmp$algorithm == "carma" & (0.5 - abs(0.5-log2(rawDataTmp$p)%%1)) < 0.3, ])

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
      
      setupName = paste(mShape, scaling, variant, sep=";")
      finalData <- data[DataColumnsHash[[variant]]]
      ylabel = yLabelHash[[variant]]
      yscale = scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
      
      #create statistics
      if (variant == "commVol") {
        cosmaAvg = mean(na.omit((finalData[finalData$algLabel == annotl[3],]$V)))
        scaAvg = mean(na.omit((finalData[finalData$algLabel == annotl[4],]$V)))
        cyclopsAvg = mean(na.omit((finalData[finalData$algLabel == annotl[2],]$V)))
        carmaAvg = mean(na.omit((finalData[finalData$algLabel == annotl[1],]$V)))
        statSummary[nrow(statSummary)+1,c('shape','scaling',algorithms)] = list(mShape, scaling, cosmaAvg,scaAvg,cyclopsAvg,carmaAvg)
      } else if (variant == "time"){
        gmMeanSpeedup = gm_mean(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
        maxSpeedup    =     max(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
        minSpeedup    =     min(dataSummary2[dataSummary2$mShape == mShape & dataSummary2$scaling == scaling,]$maxSpeedup)
        statSummary[nrow(statSummary),c('mean', 'min', 'max')] = list(gmMeanSpeedup,minSpeedup,maxSpeedup)
      }
    }
  }    
}

print(statSummary)

write.csv(rawData, file = "res.csv")
