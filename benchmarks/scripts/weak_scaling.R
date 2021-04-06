library(dplyr)
library(plyr)
library(reshape2)
library(ggplot2)
#library(forcats)


source("commvol_models.R")
source("plot_settings.R")

df = read.csv("../daint/weak_2.csv", header=TRUE,stringsAsFactors = FALSE)
df$totMB = df$aggr_bytes / 1e6
df <- df[c("implementation", "chosen_N", "chosen_P", "totMB")]
df <- reshape(df, idvar = c("chosen_N", "chosen_P"), timevar = "implementation", direction = "wide")
df$datasrc <- "measured"    #lets add a column to seperate model and meassurements
df <- rename(df, c("chosen_N"="N", "chosen_P"="P", "totMB.mkl"="LibSci", "totMB.candmc"="CANDMC", "totMB.slate"="SLATE", "totMB.conflux"="Conflux")) #rename the columns to algorithms
df <- melt(df, id.vars = c("N", "P", "datasrc"), variable.name = "algo", value.name = "commvol") 

extrap_range <- seq(4, 2^17, 16)
modelP <- rep(extrap_range)
modelN <- modelP^(1/3) * 3200
modeldf <- data.frame(P=modelP, N=modelN)
modeldf$LibSci <- model_2D(modeldf$N, modeldf$P)
modeldf$CANDMC <- model_candmc(modeldf$N, modeldf$P)
modeldf$Conflux <- model_conflux(modeldf$N, modeldf$P)
modeldf$datasrc = "model"

modeldf <- melt(modeldf, id.vars = c("N", "P", "datasrc"), variable.name = "algo", value.name = "commvol") 

modeldf <- modeldf[modeldf$P > 7,]
df <- df[df$P > 8,]
machines=data.frame(name=c("Daint", "Summit",  "Sunway TaihuLight"), nodes=c(1813, 4608, 40960))
df$algo <- factor(df$algo, levels = c("LibSci",  "CANDMC","SLATE", "Conflux"))
modeldf$algo <- factor(modeldf$algo, levels = c("LibSci","CANDMC", "SLATE",  "Conflux"))

pdf(file= paste("weak_scaling.pdf", sep = ''),  width = w, height = w*aspRatio, onefile=FALSE)
ggplot() +
  geom_line(data=modeldf, aes(x=P, y=commvol/P, color=algo, shape=algo )) +
  geom_jitter(data=df, aes(x=P, y=commvol/P, color=algo, shape=algo), size=3, height=0.04) +
  scale_colour_manual(values = c("#7aae00", "#c77aff", "#f98b84", "#00bdc2")) +
  scale_shape_manual(values = c(17,3,19,15)) +
  scale_x_log10("Number of available nodes", breaks =c(100, 1000, 10000, 100000) , labels = function(x) format(x, scientific = FALSE)) + #limits =c(100, 2^17)
  #scale_y_log10("Communication Volume [MB/Proc]") +
  scale_y_continuous("Communication Volume [MB/Proc]", limits =c(0, 800)) +
  guides(fill = guide_legend(override.aes = list(linetype = 0)),
         color = guide_legend(override.aes = list(linetype = 0))) +
  geom_vline(data=machines, aes(xintercept=nodes), linetype="dashed") +
  theme_bw(20) +
  theme(legend.position = "none")#c(0.4, 0.6), legend.title = element_blank())
dev.off()

