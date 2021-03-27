library(ggplot2)
library(ggrepel)
library(reshape2)
library(plyr)
library(hash)
#library("reshape2")


exp_name = ""
exp_filename = "benchmarks.csv"
scalings = c("weak", "strong")

variantPlots = c("time", "FLOPS", "commVol")
algorithms = c("Cholesky", "LU")

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
#annotl = c("CARMA [21]","CTF [49]","COSMA (this work)", "ScaLAPACK [14]")
#varPlot = "FLOPS"

FLOPSperNode = 1209 


statistics = 0

annotCoord = list()
#annotCoordX[["square_memory_p1_time"]] = 


#exp_filename = paste(exp_name,'.csv',sep="")
setwd("C:/gk_pliki/uczelnia/doktorat/performance_modelling/repo/conflux_cpp_2/results/conflux/benchmarks/scripts")
setwd(paste("../",exp_name,sep =""))
source(paste(getwd(), "/scripts/SPCL_Stats.R", sep=""))


# prepare the data 
rawData <- read.csv(file=exp_filename, sep=",", stringsAsFactors=FALSE, header=TRUE)

for (alg in algorithms){
  print(alg)
  df1 <- rawData[str_cmp(rawData$algorithm, alg),]
  for (scaling in scalings){
    print(scaling)
    for (size in sizes[[scaling]]){
      print(size)
      df2 <- df1[df1$N_base == size, ]
      
      for (variant in variantPlots){
        print(variant)
        df3 <- df2[str_cmp(df2$unit, variant),]
        if (variant == "FLOPS") {
          df3 <- df2[str_cmp(df2$unit, "time"),]
        }
        
        if (nrow(df3) == 0)
          next
        
        
        
        plot_data <- df3[c("P", "library", "value", "N")]
        
        if (variant == "time"){
          ylabel = "time [ms]"
          yscale = scale_y_log10(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
        }
        else if(variant == "FLOPS"){
          ylabel = "% peak performance"
          yscale = scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
          if (alg == "Cholesky"){
            plot_data$percent_peak = 100/3 * (plot_data$N)^3 / (1e6 * plot_data$P * plot_data$value * FLOPSperNode)
          } else {
            plot_data$percent_peak = 200/3 * (plot_data$N)^3 / (1e6 * plot_data$P * plot_data$value * FLOPSperNode)
          }
          plot_data$value <- plot_data$percent_peak
        }
        else{
          ylabel = "total communication volume [GB]"
          yscale = scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
        }
        
        
        
        
        #  m = data$m
        #  plot_data$time = 200* data$m * data$n * data$k / (data$time * 1e6) / (GFLOPSperCore * data$p)
        
        name = paste(alg, scaling, size, variant, sep="_")
        
        
        # print(plot_data[c("algorithm","p","time")])
        
        aspRatio = 0.75
        w = 10
        textSize = 30
        pointSize = 5
        
        # if (size == 'square' && variant == 'strong') {
        #   annotx = c(9000,712,1524,256)
        #   annoty = c(50,5,70,20)
        #   annotPointX1 = c(4096,1024,512,128)
        #   
        #   annotPointX2 = c(7000,800,700,150)
        #   annotPointY2 = c(47,8,67,23)
        #   limit = ylim(0, 90)
        # } else if (size == 'square' && variant == 'memory_p1')  {
        #     annotx = c(200,200,8000,3100)
        #     annoty = c(15,35,86,50)
        #     annotPointX1 = c(256,468,2196,2048)
        #     annotPointX2 = c(200,335,3500,2500)
        #     annotPointY2 = c(18,35,83,47)
        # } else if (size == 'square' && variant == 'memory_p2')  {
        #   annotx = c(180,200,4048,4396)
        #   annoty = c(22,5,77,36)
        #   annotPointX1 = c(128,433,4096,2048)
        #   
        #   annotPointX2 = c(150,335,3500,3050)
        #   annotPointY2 = c(26,5,73,32)
        #   
        # }
        # else if (size == 'tall' && variant == 'strong')  {
        #   annotx = c(4096,5300,5300,14000)
        #   annoty = c(10,52,65,5)
        #   annotPointX1 = c(4096,7236,3348,11988)
        #   
        #   annotPointX2 = c(4096,6500,4000,11988)
        #   annotPointY2 = c(13,50,68,7)
        # }
        # else if (size == 'tall' && variant == 'memory_p1')  {
        #   annotx = c(512,200,506,1224)
        #   annoty = c(42,5,74,32)
        #   annotPointX1 = c(1024,256,512,1024)
        #   
        #   annotPointX2 = c(900,200,400,900)
        #   annotPointY2 = c(45,8,70,28)
        # }
        # else if (size == 'tall' && variant == 'memory_p2')  {
        #   annotx = c(306,200,2024,800)
        #   annoty = c(35,5,55,24)
        #   annotPointX1 = c(128,256,1024,468)
        #   
        #   annotPointX2 = c(150,200,1200,600)
        #   annotPointY2 = c(35,8,52,22)
        # }
        # else {
        #   next
        # }
        # 
        # annotPointY1 = c(plot_data[plot_data$p == annotPointX1[1] & plot_data$algorithm == 'CARMA [21] ',]$time[1],
        #                 plot_data[plot_data$p == annotPointX1[2] & plot_data$algorithm == 'CTF [45] ',]$time[1],
        #                 plot_data[plot_data$p == annotPointX1[3] & plot_data$algorithm == 'COSMA (this work) ',]$time[1],
        #                 plot_data[plot_data$p == annotPointX1[4] & plot_data$algorithm == 'ScaLAPACK [14] ',]$time[1])
        # 
        # plot the timers
        pdf(file=paste(name, ".pdf", sep=""),
            width = w, height = w*aspRatio)
        
        limit = yscale
        shapes = scale_shape_manual(values=c(15, 16, 17,18))
        shapesColors = scale_color_manual(values = c("#F8766D", "#7CAE00","#00BFC4",  "#C77CFF"))
        
        data3 = ddply(plot_data, ~ library+P, summarize, min=min(value), max=max(value), mean=median(value))
        
        scaling_label = paste(scaling, "scaling", sep = " ")
        if (variant == "strong"){
          size_label = paste("N = ", size, sep = " ")
        }
        else {
          size_label = paste("N = ", size, " * sqrt(P)", sep = " ")
        }
        
        
        p <- ggplot(mapping=aes(x=P, y=mean, ymin=min, ymax=max, fill=library, color=library, shape=library)) +
          geom_ribbon(data=data3[data3$library != "CARMA [21] ",], alpha=0.3, show.legend=TRUE)+
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
          theme_bw(27)
          # annotate("text", x = annotx, y = annoty, label = annotl, size=textSize/3) +
          # annotate("segment", x = annotPointX2[1], xend = annotPointX1[1],
          #          y = annotPointY2[1], yend = annotPointY1[1]) +
          # annotate("segment", x = annotPointX2[2], xend = annotPointX1[2],
          #          y = annotPointY2[2], yend = annotPointY1[2]) +
          # annotate("segment", x = annotPointX2[3], xend = annotPointX1[3],
          #          y = annotPointY2[3], yend = annotPointY1[3]) +
          # annotate("segment", x = annotPointX2[4], xend = annotPointX1[4],
          #          y = annotPointY2[4], yend = annotPointY1[4]) 
        print(p)
        
        dev.off()
      }
    }
  }
}
