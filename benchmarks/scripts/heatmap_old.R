library(plyr)
library(reshape2)
library(ggplot2)
#library(dplyr)

source("plot_settings.R")
source("commvol_models.R")

df = read.csv("../old_comm_vol.csv", header=TRUE,stringsAsFactors = FALSE)
df$totMB = df$aggr_bytes / 1e6

df <- df[c("implementation", "N", "P", "totMB")]
df <- reshape(df, idvar = c("N", "P"), timevar = "implementation", direction = "wide")
df$datasrc <- "meassured"    #lets add a column to seperate model and meassurements
df <- rename(df, c("totMB.mkl"="MKL", "totMB.candmc"="CANDMC", "totMB.slate"="SLATE", "totMB.conflux"="Conflux")) #rename the columns to algorithms
df <- df[!is.na(df$MKL),]


df$best_other = pmin(df$MKL, df$CANDMC, df$SLATE, na.rm=T)
df$ConfluxRatio = df$best_other / df$Conflux
df <- df[!is.na(df$ConfluxRatio),]
df$best_other_name <- "M"
df$best_other_name[df$SLATE < df$MKL | ((is.na(df$MKL))&(!is.na(df$SLATE)) )] <- "S"
df$best_other_name[((df$CANDMC < df$MKL) & (df$CANDMC < df$SLATE)) | (is.na(df$SLATE) & is.na(df$MKL))] <- "C"



p_range <- 2^seq(4,18)
# p_range <- append(p_range, unique(df$P))
p_range <- append(p_range, 512)
p_range <- unique(p_range)
p_range <- sort(p_range)

n_range <- 2^(12:18)
# n_range <- append(n_range, unique(df$N))
n_range <- unique(n_range)
n_range <- sort(n_range)

Nf <- expand.grid(p_range, n_range)
colnames(Nf) <- c("P", "N")
Nf$datasrc <- "modelled"
Nf$MKL <- model_2D(Nf$N, Nf$P)
Nf$SLATE <- model_2D(Nf$N, Nf$P)
Nf$CANDMC <- model_candmc(Nf$N, Nf$P)
Nf$Conflux <- model_conflux(Nf$N, Nf$P)

Nf$best_other = pmin(Nf$MKL, Nf$CANDMC, Nf$SLATE, na.rm = T)
Nf$best_other_name <- " "
Nf$best_other_name[Nf$SLATE < Nf$MKL] <- "S"
Nf$best_other_name[(Nf$CANDMC < Nf$MKL) & (Nf$CANDMC < Nf$SLATE)] <- "C"

Nf$ConfluxRatio = Nf$best_other / Nf$Conflux

for (p in unique(Nf$P)) {
  for (n in unique(Nf$N)) {
    if ( length(df[(df$N==n) & (df$P == p),]$P) == 1) {
      Nf[(Nf$P == p) & (Nf$N == n), ]$datasrc <- "measured"
      Nf[(Nf$P == p) & (Nf$N == n), ]$MKL <- df[(df$N==n) & (df$P == p),]$MKL
      Nf[(Nf$P == p) & (Nf$N == n), ]$Conflux <- df[(df$N==n) & (df$P == p),]$Conflux
      Nf[(Nf$P == p) & (Nf$N == n), ]$CANDMC <- df[(df$N==n) & (df$P == p),]$CANDMC
      Nf[(Nf$P == p) & (Nf$N == n), ]$SLATE <- df[(df$N==n) & (df$P == p),]$SLATE
      Nf[(Nf$P == p) & (Nf$N == n), ]$ConfluxRatio <- df[(df$N==n) & (df$P == p),]$ConfluxRatio
      Nf[(Nf$P == p) & (Nf$N == n), ]$best_other <- df[(df$N==n) & (df$P == p),]$best_other
      Nf[(Nf$P == p) & (Nf$N == n), ]$best_other_name <- df[(df$N==n) & (df$P == p),]$best_other_name
    }
  }
}


# if we have both, measured and model data for a P,N combination, throw away the model
#df = df %>% group_by(N,P) %>% arrange(datasrc) %>% top_n(1, datasrc) %>% as.data.frame()
#data = Nf
data = Nf[Nf$P > 60,]

pdf(file= paste("heatmap_labelled.pdf", sep = ''),  width = w, height = w*aspRatio)
p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N), 
                      fill=ConfluxRatio, 
                      label=paste(round(ConfluxRatio,1), 
                                  best_other_name, sep="\n") )) +
  geom_tile() +
  geom_text() +
 # geom_text(fontface="bold") +
 # geom_point(aes(shape=as.factor(datasrc)), size=3) +
  scale_x_discrete("Number of nodes") +
  scale_y_discrete("Matrix size [N]") +
  scale_fill_gradient("", low = "orange", high = "green") +
  theme_bw(20) + theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle = 90))
print(p)
dev.off()

pdf(file= paste("heatmap.pdf", sep = ''),  width = w, height = w*aspRatio)
p <- ggplot(data=data, aes(x=as.factor(P), y=as.factor(N), 
                      fill=ConfluxRatio, 
                      label=paste(best_other_name))) +
  geom_tile() +
  geom_text(fontface="bold") +
  #geom_point(aes(shape=as.factor(datasrc)), size=3) +
  scale_x_discrete("Number of nodes") +
  scale_y_discrete("Matrix size [N]") +
  scale_fill_gradient("", low = "orange", high = "green") +
  theme_bw(20) + theme(legend.position = "none") +
  theme(axis.text.x = element_text(angle = 90))
print(p)
dev.off()

