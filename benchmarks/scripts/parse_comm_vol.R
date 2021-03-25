library(ggplot2)
library(ggrepel)
library(reshape2)
library(plyr)
#library("reshape2")


path = "C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/papers/MMM-paper/results/comm_volume" #getwd()
#exp_name = "weak48"
exp_filename = "comm_files_aggr.csv"
variants = c("strong", "memory_p0", "memory_p1")
sizes = c("square","tall")
variantPlots = c("commVol")
algorithms = c("cosma","scalapack","cyclops", "carma")
annotl = c("CARMA [21]","CTF [49]","COSMA (this work)", "ScaLAPACK [14]")

results <- data.frame(matrix(ncol = 5, nrow = 0))
rescolNames <- append(algorithms, "scenario", after = 0)
colnames(results) <- rescolNames

GFLOPSperCore = 1209/36

S = 5000000000


statistics = 0

annotCoord = list()
#annotCoordX[["square_memory_p1_time"]] = 


#exp_filename = paste(exp_name,'.csv',sep="")
#setwd(paste(path,exp_name,sep =""))
setwd(path)
source(paste(path, "../SPCL_Stats.R", sep="/"))


# prepare the data 

rawData <- read.csv(file=exp_filename, sep=",", stringsAsFactors=FALSE, header=TRUE)

# setups <- rawData[rawData$algorithm == "cosma",]
# setups <-setups[c("p", "m", "n", "k", "S")]
# write.csv(setups, file = "setups.csv")
#rawData$p = rawData$p * 36


#fixing comm vol per process
rawData$V = ifelse(rawData$algorithm == "carma", rawData$V / 2^floor(log2(rawData$p)) * 1e-6 / 2,
             ifelse(rawData$algorithm == "ctf", (rawData$V - 2*rawData$m*rawData$k) / 2^floor(log2(rawData$p)) * 1e-6 / 2,
                   rawData$V / rawData$p * 1e-6 / 2)
              )

rawData$domSize_mn = ifelse(rawData$algorithm == "scalapack", (rawData$m / rawData$p * rawData$n)^(1/2),
                      ifelse(rawData$algorithm == "carma", pmin((rawData$m / rawData$p * rawData$n * rawData$k)^(1/3), (sqrt(S/3))),
                        ifelse(rawData$algorithm == "cosma", pmin((rawData$m / rawData$p * rawData$n * rawData$k)^(1/3), sqrt(S)),
                               pmin((rawData$m  / rawData$p * rawData$n * rawData$k)^(1/3), (sqrt(S/2))))
                      )
                     )

#rawData$domSize_test = min((rawData$m / rawData$p * rawData$n * rawData$k)^(1/3), sqrt(S))
rawData$domSize_k = rawData$m / rawData$p * rawData$n /  (rawData$domSize_mn)^2 * rawData$k
rawData$commVolModel =  ifelse(rawData$algorithm == "scalapack",2*rawData$domSize_mn*rawData$domSize_k,
                          ifelse(rawData$domSize_k == rawData$k, 2*rawData$domSize_mn*rawData$domSize_k,
                               (rawData$domSize_mn)^2 + 2*rawData$domSize_mn*rawData$domSize_k))*1e-6*8
rawData$commModelRatio = rawData$V /rawData$commVolModel

rawData$checkDomSize = (rawData$domSize_mn)^2 * rawData$domSize_k  - (rawData$m / rawData$p * rawData$n  * rawData$k )

#filtering incorrect data
#rawData = na.omit(rawData[rawData$commModelRatio > 0.5, ])

#filtering CARMA by non powers of two
rawDataTmp = rawData
rawData = rawDataTmp[rawDataTmp$algorithm != "carma",]
rawData = rbind(rawData,rawDataTmp[rawDataTmp$algorithm == "carma" & (0.5 - abs(0.5-log2(rawDataTmp$p)%%1)) < 0.1, ])


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
      ylabel = "MB communicated per core"
      yscale = scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
      
      name = paste(size, variant, sep="_")
      
      
      #create statistics
      #algorithms = c("cosma","scalapack","cyclops", "carma")
      cosmaAvg = mean(na.omit((Vscaling[Vscaling$algorithm == "cosma",]$V)))
      scaAvg = mean(na.omit((Vscaling[Vscaling$algorithm == "scalapack",]$V)))
      cyclopsAvg = mean(na.omit((Vscaling[Vscaling$algorithm == "cyclops",]$V)))
      carmaAvg = mean(na.omit((Vscaling[Vscaling$algorithm == "carma",]$V)))
      results[nrow(results)+1,] = list(paste("size_",size, "_var_", variant, sep=""), cosmaAvg,scaAvg,cyclopsAvg,carmaAvg)
      
      print(Vscaling[c("algorithm","p","V")])
      
      aspRatio = 0.75
      w = 10
      textSize = 30
      pointSize = 5
      if (size == 'square' && variant == 'strong') {
        annotx = c(170,950,230,1200)
        annoty = c(110,360,70,240)
        annotPointX1 = c(252,468,512,468)
        
        annotPointX2 = c(200,680,390,680)
        annotPointY2 = c(120,370,80,240)
        limit = ylim(0, 90)
      } else if (size == 'square' && variant == 'memory_p0')  {
          annotx = c(210,250,1000,400)
          annoty = c(650,1200,300,900)
          annotPointX1 = c(252,468,972,468)
          annotPointX2 = c(200,335,972,400)
          annotPointY2 = c(600,1200,330,850)
      } else if (size == 'square' && variant == 'memory_p1')  {
        annotx = c(240,850,488,595)
        annoty = c(350,512,190,230)
        annotPointX1 = c(252,468,468,576)
        
        annotPointX2 = c(240,600,510,530)
        annotPointY2 = c(335,520,175,240)
        #algorithms = c("cosma","scalapack","cyclops", "carma")
      }
      else if (size == 'tall' && variant == 'strong')  {
        annotx = c(1300,1100,300,1000)
        annoty = c(800,2900,300,5000)
        annotPointX1 = c(972,900,468,468)
        
        annotPointX2 = c(1000,1000,350,550)
        annotPointY2 = c(750,2500,360,5000)
      }
      else if (size == 'tall' && variant == 'memory_p0')  {
        annotx = c(1024,200,400,560)
        annoty = c(300,300,40,720)
        annotPointX1 = c(972,288,252,468)
        
        annotPointX2 = c(900,200,300,530)
        annotPointY2 = c(256,360,47,680)
      }
      else if (size == 'tall' && variant == 'memory_p1')  {
        annotx = c(1024,200,400,980)
        annoty = c(65,205,40,224)
        annotPointX1 = c(972,144,576,468)
        
        annotPointX2 = c(1024,200,512,570)
        annotPointY2 = c(60,185,41,222)
      }
      else {
        next
      }
      
      annotPointY1 = c(Vscaling[Vscaling$p == annotPointX1[1] & Vscaling$algorithm == 'carma',]$V[1],
                      Vscaling[Vscaling$p == annotPointX1[2] & Vscaling$algorithm == 'cyclops',]$V[1],
                      Vscaling[Vscaling$p == annotPointX1[3] & Vscaling$algorithm == 'cosma',]$V[1],
                      Vscaling[Vscaling$p == annotPointX1[4] & Vscaling$algorithm == 'scalapack',]$V[1])
      
      # plot the timers
      pdf(file=paste("size_",size, "_var_", variant , ".pdf", sep=""),
          width = w, height = w*aspRatio)
      
      limit = yscale
      shapes = scale_shape_manual(values=c(15, 16, 17,18))
      shapesColors = scale_color_manual(values = c("#F8766D", "#7CAE00","#00BFC4",  "#C77CFF"))
      
      #data3 = ddply(Vscaling, ~ algorithm+p, summarize, min=min(time), max=max(time), mean=median(time))
      
      p <- ggplot(mapping=aes(x=p, y=Vscaling$V, fill=Vscaling$algorithm, color=Vscaling$algorithm, shape=Vscaling$algorithm)) +
        #+geom_ribbon(data=data3[data3$algorithm != "CARMA [21] ",], alpha=0.3, show.legend=FALSE)
        shapes + 
        geom_point(data=Vscaling, size = 4, show.legend=FALSE) +
     #   geom_line(aes(x=Vscaling$p, y=Vscaling$commVolModel, color=Vscaling$algorithm)) + #,show.legend=TRUE) +
        #geom_errorbar(data=data3[data3$algorithm == "CARMA [21] ",], width=0.1, size=1, show.legend=FALSE) 
        scale_x_continuous(trans='log2',
                           breaks = c(128,256,512,1024, 2048),
                           labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
        #scale_x_log2("# of cores", breaks=c(128, 256, 512, 1024, 2048, 4096, 8192, 16384)) 
        scale_y_continuous(trans='log2',
                       #    breaks=function(x) format(x, big.mark = ",", scientific = FALSE),
                          # breaks = c(32,45, 64, 90, 128,256,512,1024,2048), 
                          breaks = c(32,45, 64, 90, 128,181, 256,362, 512,724, 1024,1448,2048), 
                           labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
                           #breaks = function("log2", function(x) 2^x),
                           #labels = trans_format("log2", math_format(2^.x))) + #labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
       # scale_y_log10(ylabel) +
        xlab("# of cores") +
       # yscale +
        ylab(ylabel)+ 
        theme_bw(27) +
        # theme(legend.position = c(0.45,0.98),
        #       legend.title=element_blank(),
        #       legend.text=element_text(size=22),
        #       legend.direction="horizontal",
        #       text = element_text(size=textSize),
        #       aspect.ratio=aspRatio
        # )+
        annotate("text", x = annotx, y = annoty, label = annotl, size=textSize/3)+
        annotate("segment", x = annotPointX2[1], xend = annotPointX1[1],
                 y = annotPointY2[1], yend = annotPointY1[1]) +
        annotate("segment", x = annotPointX2[2], xend = annotPointX1[2],
                 y = annotPointY2[2], yend = annotPointY1[2]) +
        annotate("segment", x = annotPointX2[3], xend = annotPointX1[3],
                 y = annotPointY2[3], yend = annotPointY1[3]) +
        annotate("segment", x = annotPointX2[4], xend = annotPointX1[4],
                 y = annotPointY2[4], yend = annotPointY1[4])
      print(p)
      
      dev.off()
  }
}

write.csv(rawData, file = "res.csv")
