library(dplyr)
library(reshape2)
library(ggplot2)
library(plyr)
library(scales)

source("commvol_models.R")
source("plot_settings.R")
exp_filename = "/../benchmarks.csv"
source("SPCL_Stats.R")
source("plot_settings.R")

rawData <- read.csv(file=paste(getwd(), exp_filename, sep = ""), sep=",", stringsAsFactors=FALSE, header=TRUE)
rawData[rawData$N_base == "-" & rawData$type == "strong",]$N_base <- rawData[rawData$N_base == "-" & rawData$type == "strong",]$N
rawData[rawData$N_base == "-" & rawData$type == "weak",]$N_base <- rawData[rawData$N_base == "-" & rawData$type == "weak",]$N / sqrt(rawData[rawData$N_base == "-" & rawData$type == "weak",]$P)
rawData[str_cmp(rawData$library, "capital"),]$library = "candmc"
if (nrow(rawData[str_cmp("psychol", rawData$library),]) > 0) {
  rawData[str_cmp("psychol", rawData$library),]$library = "conflux"
}

rawData <- rawData[rawData$P > 8,]

rawData$case = paste(rawData$algorithm, rawData$N_base,sep ="_")

# take only 128 blocks for slate and mkl
rawData2 <- rawData[rawData$library != "slate" | rawData$blocksize == 128,]
rawData2 <- rawData2[rawData2$library != "mkl" | rawData2$blocksize == 128,]

rawData <- rawData2
# min_df <- as.data.frame(rawData %>% group_by(algorithm, library, N, P, unit, grid) %>% summarise_each(list(min), value))
# min_df[str_detect(min_df$grid, regex("\\dx\\dx4")) | str_detect(min_df$grid, regex("\\dx\\dx8")),]$library <- 
#   paste(min_df[str_detect(min_df$grid, regex("\\dx\\dx4")) | str_detect(min_df$grid, regex("\\dx\\dx8")),]$library, "3D", sep = "_")


min_df <- as.data.frame(rawData %>% group_by(algorithm, library, N, P, unit) %>% summarise_each(list(min), value))


df_comm <- min_df[min_df$unit == "bytes",]

algos = c("cholesky", "lu")
detach("package:dplyr",unload = TRUE)
for (alg in algos) {
  df <- df_comm[df_comm$algorithm == alg,]
  
  df$totMB = df$value / 1e6
  df <- df[c("library", "N", "P", "totMB")]
  df <- reshape(df, idvar = c("N", "P"), timevar = "library", direction = "wide")
  df$datasrc <- "measured"    #lets add a column to seperate model and meassurements
  # df <- rename(df, c("totMB.mkl"="MKL", "totMB.candmc"="CANDMC", "totMB.candmc_3D"="CANDMC 3D","totMB.slate"="SLATE", "totMB.conflux"="COnfLUX", "totMB.conflux_3D"="COnfLUX 3D")) #rename the columns to algorithms
  df <- rename(df, c("totMB.mkl"="MKL", "totMB.candmc"="CAPITAL", "totMB.slate"="SLATE", "totMB.conflux"="COnfCHOX")) #rename the columns to algorithms
  
  
  extrap_range <- 2^(4:19)
  nP <- rep(extrap_range, each=length(unique(df$N)))
  nN <- rep(unique(df$N), length(extrap_range))
  Nf <- data.frame(N=nN, P=nP)
  Nf$datasrc <- "modelled"
  #Nf$MKL <- model_2D(Nf$N, Nf$P)
  Nf$MKL <- model_2D_chol(Nf$N, Nf$P)
  #Nf$CANDMC <- model_candmc(Nf$N, Nf$P)
  Nf$CAPITAL <- model_capital(Nf$N, Nf$P)
  #Nf$COnfLUX <- model_conflux(Nf$N, Nf$P)
  Nf$COnfCHOX <- model_psychol(Nf$N, Nf$P)
  
  df <- rbind.fill(df, Nf)  #this overrides the old df with the new extended one
  
  # debug only, to examine correctness
  # tmp <- reshape(df, 
  #               idvar = c("N", "P"), 
  #               timevar = "implementation", direction = "wide")
  a = 1
  
  df <- melt(df, id = c('P', 'N', 'datasrc'), variable.name = "algo", value.name = "comm_vol")
  df$effP <- calc_effP(df$P, df$algo)
  
  data <- df[df$N == 16384,]
  
  pdf(file= paste("../fixed_N_", alg, ".pdf", sep = ''),  width = w, height = w*aspRatio, onefile=FALSE)
  
  # add important machine sizes
  machines=data.frame(name=c("Daint", "Summit",  "Sunway TaihuLight"), nodes=c(1813, 4608, 40960))
  machines$N <- 16384 
  #machines$Conflux <- model_conflux(machines$N, machines$nodes)
  machines$Conflux <- model_psychol(machines$N, machines$nodes)
  #machines$LibSci <- model_2D(machines$N, machines$nodes)
  machines$LibSci <- model_2D_chol(machines$N, machines$nodes)
  machines$speedup = machines$LibSci / machines$Conflux
  print(machines)
  
  p <- ggplot(data=data, aes(x=P, y=comm_vol/effP, shape=as.factor(algo), color=as.factor(algo))) +
    geom_line(data=data[data$datasrc=="modelled",]) +
    geom_jitter(data=data[data$datasrc=="measured",], size=3, height=0.04) +
    scale_x_log10("Number of available Nodes", labels=scales::comma) +
    scale_y_log10("Communication Volume [MB/Proc]") +
    guides(fill = guide_legend(override.aes = list(linetype = 0)),
           color = guide_legend(override.aes = list(linetype = 0))) +
    # we can't go negative on a log axis :( 
    #annotate("rect", xmin=0, xmax=20000, ymin=50, ymax=15000) +
    geom_vline(data=machines, aes(xintercept=nodes), linetype="dashed") +
    theme_bw(20) +
    theme(legend.position = c(0.9, 0.8), legend.title = element_blank())
  print(p)
  dev.off()
}
