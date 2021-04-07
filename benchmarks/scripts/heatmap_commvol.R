
library(plyr)
library(dplyr)
library(reshape2)
library(ggplot2)
# library(dplyr)

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

rawData$case = paste(rawData$algorithm, rawData$N_base,sep ="_")
comm_data <- rawData[rawData$unit == "bytes", ]
min_df <- as.data.frame(comm_data[comm_data$library != "slate",] %>% group_by(algorithm, library, N, P, unit) %>% summarise_each(list(min), value))
min_df_slate <- as.data.frame(comm_data[comm_data$library == "slate",] %>% group_by(algorithm, library, N, P, unit) %>% summarise_each(list(max), value))
min_df <- rbind(min_df, min_df_slate)
# min_df <- as.data.frame(rawData %>% group_by(algorithm, library, N, P, unit, grid) %>% summarise_each(list(min), value))
# min_df[str_detect(min_df$grid, regex("\\dx\\dx4")) | str_detect(min_df$grid, regex("\\dx\\dx8")),]$library <- 
#   paste(min_df[str_detect(min_df$grid, regex("\\dx\\dx4")) | str_detect(min_df$grid, regex("\\dx\\dx8")),]$library, "3D", sep = "_")




df_comm <- min_df[min_df$unit == "bytes",]

algos = c("lu", "cholesky")


for (alg in algos) {
  df <- df_comm[df_comm$algorithm == alg,]
    
  df$totMB = df$value / 1e6
  df <- df[c("library", "N", "P", "totMB")]
  df <- reshape(df, idvar = c("N", "P"), timevar = "library", direction = "wide")
  df$datasrc <- "measured"    #lets add a column to seperate model and meassurements
  #df <- rename(df, c("totMB.mkl"="MKL", "totMB.candmc"="CANDMC", "totMB.candmc_3D"="CANDMC 3D","totMB.slate"="SLATE", "totMB.conflux"="COnfLUX", "totMB.COnfLUX_3D"="COnfLUX 3D")) #rename the columns to algorithms
  
  df <- plyr::rename(df, c("totMB.mkl"="MKL", "totMB.candmc"="CANDMC", "totMB.slate"="SLATE", "totMB.conflux"="COnfLUX")) #rename the columns to algorithms
    
  
  
  df$best_other = pmin(df$MKL, df$CANDMC, df$SLATE, na.rm=T)
  df$COnfLUXRatio = df$best_other / df$COnfLUX
  df <- df[!is.na(df$COnfLUXRatio),]
  df$best_other_name <- "M"
  df$best_other_name[df$SLATE < df$MKL | ((is.na(df$MKL))&(!is.na(df$SLATE)) )] <- "S"
  df$best_other_name[((df$CANDMC < df$MKL) & (df$CANDMC < df$SLATE)) | (is.na(df$SLATE) & is.na(df$MKL))] <- "C"
  
  
  
  p_range <- 2^seq(4,18)
  p_range <- append(p_range, unique(df$P))
  p_range <- append(p_range, 512)
  p_range <- unique(p_range)
  p_range <- sort(p_range)
  
  n_range <- 2^(12:20)
  n_range <- append(n_range, unique(df$N))
  n_range <- unique(n_range)
  n_range <- sort(n_range)
  
  Nf <- expand.grid(p_range, n_range)
  colnames(Nf) <- c("P", "N")
  Nf$datasrc <- "modelled"
  Nf$MKL <- model_2D(Nf$N, Nf$P)
  Nf$SLATE <- model_2D(Nf$N, Nf$P)
  Nf$CANDMC <- model_candmc(Nf$N, Nf$P)
  Nf$COnfLUX <- model_conflux(Nf$N, Nf$P)
  
  Nf$best_other = pmin(Nf$MKL, Nf$CANDMC, Nf$SLATE, na.rm = T)
  Nf$best_other_name <- " "
  Nf$best_other_name[Nf$SLATE < Nf$MKL] <- "S"
  Nf$best_other_name[(Nf$CANDMC < Nf$MKL) & (Nf$CANDMC < Nf$SLATE)] <- "C"
  
  Nf$COnfLUXRatio = Nf$best_other / Nf$COnfLUX
  
  # for (p in unique(Nf$P)) {
  #   for (n in unique(Nf$N)) {
  #     print(n)
  #     if ( length(df[(df$N==n) & (df$P == p),]$P) == 1) {
  #       Nf[(Nf$P == p) & (Nf$N == n), ]$datasrc <- "measured"
  #       Nf[(Nf$P == p) & (Nf$N == n), ]$MKL <- df[(df$N==n) & (df$P == p),]$MKL
  #       Nf[(Nf$P == p) & (Nf$N == n), ]$COnfLUX <- df[(df$N==n) & (df$P == p),]$COnfLUX
  #       Nf[(Nf$P == p) & (Nf$N == n), ]$CANDMC <- df[(df$N==n) & (df$P == p),]$CANDMC
  #       Nf[(Nf$P == p) & (Nf$N == n), ]$SLATE <- df[(df$N==n) & (df$P == p),]$SLATE
  #       Nf[(Nf$P == p) & (Nf$N == n), ]$COnfLUXRatio <- df[(df$N==n) & (df$P == p),]$COnfLUXRatio
  #       Nf[(Nf$P == p) & (Nf$N == n), ]$best_other <- df[(df$N==n) & (df$P == p),]$best_other
  #       Nf[(Nf$P == p) & (Nf$N == n), ]$best_other_name <- df[(df$N==n) & (df$P == p),]$best_other_name
  #     }
  #   }
  # }
  
  
  # if we have both, measured and model data for a P,N combination, throw away the model
  #df = df %>% group_by(N,P) %>% arrange(datasrc) %>% top_n(1, datasrc) %>% as.data.frame()
  
  data = Nf #Nf[Nf$P > 60,]
  
  pdf(file= paste("../heatmap_labeled_commvol_", alg , ".pdf", sep = ''),  width = w, height = w*aspRatio)
  print(data)
  print(paste("../heatmap_labeled_commvol_", alg , ".pdf", sep = ''))
  p = ggplot(data=data, aes(x=as.factor(P), y=as.factor(N), 
                        fill=COnfLUXRatio, 
                        label=paste(round(COnfLUXRatio,1), 
                                    best_other_name, sep="\n") )) +
    geom_tile() +
    geom_text() +
   # geom_text(fontface="bold") +
   # geom_point(aes(shape=as.factor(datasrc)), size=3) +
    scale_x_discrete("Available Nodes") +
    scale_y_discrete("Matrix Size [N]") +
    scale_fill_gradient("", low = "orange", high = "green") +
    theme_bw(20) + theme(legend.position = "none") +
    theme(axis.text.x = element_text(angle = 90))
  print(p)
  dev.off()
  
  pdf(file= paste("../heatmap_commvol_", alg , ".pdf", sep = ''),  width = w, height = w*aspRatio)
  p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N), 
                        fill=COnfLUXRatio, 
                        label=paste(best_other_name))) +
    geom_tile() +
    geom_text(fontface="bold") +
    #geom_point(aes(shape=as.factor(datasrc)), size=3) +
    scale_x_discrete("Available Nodes") +
    scale_y_discrete("Matrix Size [N]") +
    scale_fill_gradient("", low = "orange", high = "green") +
    theme_bw(20) + theme(legend.position = "none") +
    theme(axis.text.x = element_text(angle = 90))
  print(p)
  dev.off()
}