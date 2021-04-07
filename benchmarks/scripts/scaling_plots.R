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

libraries_chol = c("MKL [cite]", "SLATE [cite]", "CAPITAL [cite]", "PsyChol (this work)")
libraries_LU = c("MKL [cite]", "SLATE [cite]", "CANDMC [cite]", "COnfLUX (this work)")
libraries <- hash()
libraries[["LU"]] <- libraries_LU
libraries[["Cholesky"]] <- libraries_chol
#annotl = c("CARMA [21]","CTF [49]","COSMA (this work)", "ScaLAPACK [14]")
#varPlot = "FLOPS"

FLOPSperNode = 1209 


statistics = 0

annotCoord = list()
#annotCoordX[["square_memory_p1_time"]] = 


source("SPCL_Stats.R")


# prepare the data 
#rawData <- read.csv(file=exp_filename, sep=",", stringsAsFactors=FALSE, header=TRUE)
rawData <- read.csv(file=paste(getwd(), exp_filename, sep = ""), sep=",", stringsAsFactors=FALSE, header=TRUE)


rawData[str_cmp("conflux", rawData$library),]$library = "COnfLUX (this work)"
rawData[str_cmp("mkl", rawData$library),]$library = "MKL [cite]"
rawData[str_cmp("slate", rawData$library),]$library = "SLATE [cite]"
rawData[str_cmp("candmc", rawData$library),]$library = "CANDMC [cite]"
rawData[str_cmp("psychol", rawData$library),]$library = "PsyChol (this work)"
rawData[str_cmp("capital", rawData$library),]$library = "CAPITAL [cite]"


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
      df2_5 <- df2[str_cmp(df2$type, scaling),]
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
        
        
        
        
        #  m = data$m
        #  plot_data$time = 200* data$m * data$n * data$k / (data$time * 1e6) / (GFLOPSperCore * data$p)
        
        name = paste(alg, scaling, size, variant, sep="_")
        
        
        # print(plot_data[c("algorithm","p","time")])
        
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
        } else if (alg == 'Cholesky' & scaling == 'weak' & size == 8192 & variant == 'bytes' ){
          annotx = c(180,200,4048,4396)
          annoty = c(22,5,77,36)
          annotPointX1 = c(128,433,4096,2048)

          annotPointX2 = c(150,335,3500,3050)
          annotPointY2 = c(26,5,73,32)

        }
        else if (size == 'tall' && variant == 'strong')  {
          annotx = c(10,5300,5300,14000)
          annoty = c(10,52,65,5)
          annotPointX1 = c(4,16,64,15)

          annotPointX2 = c(4096,6500,4000,11988)
          annotPointY2 = c(13,50,68,7)
        }
        else if (size == 'tall' && variant == 'memory_p1')  {
          annotx = c(512,200,506,1224)
          annoty = c(42,5,74,32)
          annotPointX1 = c(1024,256,512,1024)

          annotPointX2 = c(900,200,400,900)
          annotPointY2 = c(45,8,70,28)
        }
        else if (size == 'tall' && variant == 'memory_p2')  {
          annotx = c(306,200,2024,800)
          annoty = c(35,5,55,24)
          annotPointX1 = c(128,256,1024,468)

          annotPointX2 = c(150,200,1200,600)
          annotPointY2 = c(35,8,52,22)
        }
        else {
          next
        }

        annotPointY1 = c(plot_data[plot_data$P == annotPointX1[1] & plot_data$library == 'MKL [cite]',]$value[1],
                        plot_data[plot_data$P == annotPointX1[2] & plot_data$library == 'SLATE [cite]',]$value[1],
                        plot_data[plot_data$P == annotPointX1[3] & plot_data$library == 'CANDMC [cite]',]$value[1],
                        plot_data[plot_data$P == annotPointX1[4] & plot_data$library == 'COnfLUX (this work)',]$value[1])

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
            xlab("# of nodes") +
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

