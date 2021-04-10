library(ggplot2)
library(ggrepel)
library(reshape2)
library(plyr)
library(hash)
library(forcats)
#library("reshape2")


exp_name = ""
#exp_filename = "benchmarks.csv"
exp_filename = "/../benchmarks.csv"
scalings = c("weak", "strong")

variantPlots = c("time", "FLOPS", "bytes")
algorithms = c("Cholesky", "LU")

sizes_strong = c(16384, 131072)
sizes_weak = c(1024, 8192)
sizes <- hash()
sizes[["strong"]] <- sizes_strong
sizes[["weak"]] <- sizes_weak

libraries_chol = c("MKL [35]", "SLATE [29]", "CAPITAL [34]", "COnfCHOX (this work)")
libraries_LU = c("MKL [35]", "SLATE [29]", "CANDMC [57]", "COnfLUX (this work)")
libraries <- hash()
libraries[["LU"]] <- libraries_LU
libraries[["Cholesky"]] <- libraries_chol
#annotl = c("CARMA [21]","CTF [49]","COSMA (this work)", "ScaLAPACK [14]")
#varPlot = "FLOPS"

# we use two MPI ranks per one node
FLOPSperNode = 1209 * 2


statistics = 0

annotCoord = list()
#annotCoordX[["square_memory_p1_time"]] = 


source("SPCL_Stats.R")


# prepare the data 
#rawData <- read.csv(file=exp_filename, sep=",", stringsAsFactors=FALSE, header=TRUE)
rawData <- read.csv(file=paste(getwd(), exp_filename, sep = ""), sep=",", stringsAsFactors=FALSE, header=TRUE)

# we use two MPI ranks per one node
rawData$P <- rawData$P / 2


rawData[str_cmp("conflux", rawData$library),]$library = "COnfLUX (this work)"
rawData[str_cmp("mkl", rawData$library),]$library = "MKL [35]"
rawData[str_cmp("slate", rawData$library),]$library = "SLATE [29]"
rawData[str_cmp("candmc", rawData$library),]$library = "CANDMC [57]"
rawData[str_cmp("psychol", rawData$library),]$library = "COnfCHOX (this work)"
rawData[str_cmp("capital", rawData$library),]$library = "CAPITAL [34]"


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO - FILTERING SMALL PEAK FLOPS RESULTS!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rawData <- rawData[!(rawData$N == 16384 & rawData$P > 500),]


rawData[rawData$N_base == "-" & rawData$type == "strong",]$N_base <- rawData[rawData$N_base == "-" & rawData$type == "strong",]$N
rawData[rawData$N_base == "-" & rawData$type == "weak",]$N_base <- rawData[rawData$N_base == "-" & rawData$type == "weak",]$N / sqrt(rawData[rawData$N_base == "-" & rawData$type == "weak",]$P)



filtered_time_data <- find_optimal_blocks(rawData)

for (variant in variantPlots){
  print(variant)
  if (variant == "FLOPS" | variant == "time") {
    df1 <- filtered_time_data
  }
  if (variant == "bytes"){
    df1 <- rawData[rawData$unit == "bytes",]
  }
  
  for (alg in algorithms){
    print(alg)
    df2 <- df1[str_cmp(df1$algorithm, alg),]
    for (scaling in scalings){
      print(scaling)
      if (alg == "LU" & scaling == "strong"){
        df2_5 <- df2
      }
      else {
        df2_5 <- df2[str_cmp(df2$type, scaling),]
      }
      for (size in sizes[[scaling]]){
        print(size)
        df3 <- df2_5[df2_5$N_base == size, ]
        
        if (nrow(df3) == 0)
          next
        
        if (alg == "Cholesky" & scaling == "weak" & size == 8192 & variant == "FLOPS"){
          a = 1
        }
        
        
        plot_data <- df3[c("P", "library", "value", "N")]
        
        if (variant == "time"){
          ylabel = "time [ms]"
          yscale = scale_y_log10(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
        }
        else if(variant == "FLOPS"){
          ylabel = "% peak performance"
          yscale = scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
          if (alg == "Cholesky"){
            plot_data$percent_peak = 100/3 * (plot_data$N)^3 / (1e6 * (plot_data$P/2) * plot_data$value * FLOPSperNode)
          } else {
            plot_data$percent_peak = 200/3 * (plot_data$N)^3 / (1e6 * (plot_data$P/2) * plot_data$value * FLOPSperNode)
          }
          plot_data$value <- plot_data$percent_peak
          
          # filtering out data with peak flops below the threshold
          plot_data <- plot_data[plot_data$percent_peak > 2,]
        }
        else{
          ylabel = "total communication volume [GB]"
          plot_data$value <- plot_data$value / 1e9
          yscale = scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
        }
        
        name = paste(alg, scaling, size, variant, sep="_")
        
        aspRatio = 0.65
        w = 10
        textSize = 30
        pointSize = 5
        # 
        # if (alg == 'LU' && size > 4000) {
        #   a = 1
        # }
        
        if (alg == 'LU' && scaling == 'weak') {
          annotx = c(9,8,48,128)
          annoty = c(12,30,25,40)
          annotPointX1 = c(64,16,64,64)

          annotPointX2 = c(18,10,40,120)
          annotPointY2 = c(12,27,24,37)
          limit = ylim(0, 90)
        } else if (alg == 'LU' & scaling == 'strong' & size == 16384 ) {
          annotx = c(6,12,20,45)
          annoty = c(5,35,16,30)
          annotPointX1 = c(16,4,8,16)
          
          annotPointX2 = c(9,8,11,32)
          annotPointY2 = c(7,33,16,27)
          #limit = ylim(0, 90)
        } else if (alg == 'Cholesky' & scaling == 'weak' & size == 8192 & variant == 'FLOPS' ){
          annotx = c(64,8,9,256)
          annoty = c(20,25,14,36)
          annotPointX1 = c(64,16,8,64)

          annotPointX2 = c(100,10,6,128)
          annotPointY2 = c(18,27,12,34)

        }
        else if (alg == 'Cholesky' & scaling == 'strong' & size == 131072 & variant == 'FLOPS' ){
          annotx = c(38,512,128,230)
          annoty = c(26,31,22,56)
          annotPointX1 = c(16,512,512,64)

          annotPointX2 = c(23,490,256,120)
          annotPointY2 = c(25,28,21,53)
        }
        else if (alg == 'Cholesky' & scaling == 'strong' & size == 16384 & variant == 'FLOPS' ){
          annotPointX1 = c(8,16,64,4)
          
          annotx = c(6,40,36,26)
          annoty = c(3,25,3,45)

          annotPointX2 = c(6,28,50,9)
          annotPointY2 = c(5,22,5,45)
        }
        else if (alg == 'LU' && scaling == 'strong' & size == 131072)  {
          annotPointX1 = c(16,128,128,64)
          
          annotx = c(30,70,420,230)
          annoty = c(18,7,30,50)
          
          annotPointX2 = c(22,100,240,110)
          annotPointY2 = c(21,10,28,48)
        }
        else {
          next
        }
        annotPointX1 = annotPointX1 / 2
        annotx = annotx / 2
     #   annoty = annoty / 2
        annotPointX2 = annotPointX2 / 2
      #  annotPointY2 = annotPointY2 / 2

        annotPointY1 = c(plot_data[plot_data$P == annotPointX1[1] & plot_data$library == 'MKL [35]',]$value[1],
                        plot_data[plot_data$P == annotPointX1[2] & plot_data$library == 'SLATE [29]',]$value[1],
                        plot_data[plot_data$P == annotPointX1[3] & (plot_data$library == 'CANDMC [57]' | plot_data$library == 'CAPITAL [34]'),]$value[1],
                        plot_data[plot_data$P == annotPointX1[4] & (plot_data$library == 'COnfLUX (this work)' | plot_data$library == 'COnfCHOX (this work)'),]$value[1])

        # plot the timers
        pdf(file=paste("../", name, ".pdf", sep=""),
            width = w, height = w*aspRatio)
        
        limit = yscale
        shapes = scale_shape_manual(values=c(15, 16, 17,18))
        shapesColors = scale_color_manual(values = c("#F8766D", "#7CAE00","#00BFC4",  "#C77CFF"))
        
        data3 = ddply(plot_data, ~ library+P+N, summarize, min=min(value), max=max(value), mean=median(value))
        
        scaling_label = paste(scaling, "scaling", sep = " ")
        if (variant == "strong"){
          size_label = paste("N = ", size, sep = " ")
        }
        else {
          size_label = paste("N = ", size, " * sqrt(P)", sep = " ")
        }
        
        
        data3 <- data3 %>% mutate(library = fct_relevel(library, 
                                            libraries[[alg]]))
        
        if (variant == "bytes") {
          p <- ggplot(mapping=aes(x=P, y=min, fill=library, color=library, shape=library)) +
           # geom_ribbon(data=data3[data3$library != "CARMA [21] ",], alpha=0.3, show.legend=TRUE)+
            shapes + 
            geom_point(data=data3, size = 4, show.legend=TRUE) +
            geom_errorbar(data=data3[data3$library == "CARMA [21] ",], width=0.1, size=1, show.legend=TRUE) +
            scale_x_continuous(trans='log2',labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
            #scale_x_log2("# of cores", breaks=c(128, 256, 512, 1024, 2048, 4096, 8192, 16384)) +
            # scale_y_log10(ylabel) +
            xlab("Number of nodes") +
            yscale +
            ggtitle(paste(alg, scaling, size, sep=" ")) +
            ylab(ylabel) +
            theme_bw(27) +
          annotate("text", x = annotx, y = annoty, label = libraries[[alg]], size=textSize/3) +
          annotate("segment", x = annotPointX2[1], xend = annotPointX1[1],
                   y = annotPointY2[1], yend = annotPointY1[1]) +
          annotate("segment", x = annotPointX2[2], xend = annotPointX1[2],
                   y = annotPointY2[2], yend = annotPointY1[2]) +
          annotate("segment", x = annotPointX2[3], xend = annotPointX1[3],
                   y = annotPointY2[3], yend = annotPointY1[3]) +
          annotate("segment", x = annotPointX2[4], xend = annotPointX1[4],
                   y = annotPointY2[4], yend = annotPointY1[4])
        }
        else {
          p <- ggplot(mapping=aes(x=P, y=mean, ymin=min, ymax=max, fill=library, color=library, shape=library)) +
            geom_ribbon(data=data3[data3$library != "CARMA [21] ",], alpha=0.3, show.legend=FALSE)+
            shapes + 
            geom_point(data=data3, size = 4, show.legend=FALSE) +
            geom_errorbar(data=data3[data3$library == "CARMA [21] ",], width=0.1, size=1, show.legend=FALSE) +
            scale_x_continuous(trans='log2',labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
            #scale_x_log2("# of cores", breaks=c(128, 256, 512, 1024, 2048, 4096, 8192, 16384)) +
            # scale_y_log10(ylabel) +
            xlab("# of nodes") +
            yscale +
          #  ggtitle(paste(alg, scaling, size, sep=" ")) +
            ylab(ylabel) +
            theme_bw(27) +
            annotate("text", x = annotx, y = annoty, label = libraries[[alg]], size=textSize/3) +
            annotate("segment", x = annotPointX2[1], xend = annotPointX1[1],
                     y = annotPointY2[1], yend = annotPointY1[1]) +
            annotate("segment", x = annotPointX2[2], xend = annotPointX1[2],
                     y = annotPointY2[2], yend = annotPointY1[2]) +
            annotate("segment", x = annotPointX2[3], xend = annotPointX1[3],
                     y = annotPointY2[3], yend = annotPointY1[3]) +
            annotate("segment", x = annotPointX2[4], xend = annotPointX1[4],
                     y = annotPointY2[4], yend = annotPointY1[4])
        }
        print(p)
        
        dev.off()
      }
    }
  }
}



# -------------- DEBUGGING. CHECKING THE DATA ---------------#


# ------------------- END OF DEBUGGING. ---------------------#

