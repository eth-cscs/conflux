#SPCL_Stats.R
library(stringr)
library(dplyr)

#-----------helper functions-------------------------#
find_optimal_blocks <- function(rawData) {
  rawData[rawData$blocksize == "", ]$blocksize = 1
  time_data <- rawData[rawData$unit == "time",]
  time_data <- time_data[complete.cases(time_data),]
  comm_data <- rawData[rawData$unit == "bytes",]
  comm_data <- comm_data[complete.cases(comm_data),]
  fastest_blocks <- as.data.frame(time_data %>% group_by(algorithm, library, N, P) %>% summarise_each(list(min), value))
  comm_min_blocks <- as.data.frame(comm_data %>% group_by(algorithm, library, N, P) %>% summarise_each(list(min), value))
  rows_to_remove = c()
  
  for (row in 1:nrow(time_data)) {
    cur_blocksize = time_data[row, "blocksize"]
    cur_p = time_data[row, "P"]
    cur_n = time_data[row, "N"]
    cur_lib = time_data[row, "library"]
    cur_val = time_data[row, "value"]
    cur_alg = time_data[row, "algorithm"]
    all_values = time_data[time_data$algorithm == cur_alg & time_data$library == cur_lib & time_data$N == cur_n & time_data$P == cur_p & time_data$blocksize == cur_blocksize,]$value
    best_block_value = fastest_blocks[fastest_blocks$algorithm == cur_alg & fastest_blocks$library == cur_lib & fastest_blocks$N == cur_n &fastest_blocks$P == cur_p, ]$value
    
    smallest_com_vol = comm_min_blocks[comm_min_blocks$algorithm == cur_alg & comm_min_blocks$library == cur_lib & comm_min_blocks$N == cur_n & comm_min_blocks$P == cur_p, ]$value
    if (!identical(smallest_com_vol, numeric(0))){
      time_data[row, "V"] = smallest_com_vol / cur_p * 1e-6
    }
    if (min(all_values) != best_block_value | cur_val > 1.3 * best_block_value){
      rows_to_remove <- c(rows_to_remove, row)
    }
  }
  filtered_data = time_data[-rows_to_remove, ]
}




find_statistics <- function(filtered_data) {
  time_data <- filtered_data[filtered_data$unit == "time",]
  
  relevant_cols <- c("algorithm", "library", "N", "case", "P", "flops")
  time_data <- time_data[relevant_cols]
  
  peak_flops <- as.data.frame(time_data %>% group_by(algorithm, library, N, P) %>% summarise_each(list(max), flops))
  peak_flops$metric = "peak"
  mean_flops <- as.data.frame(time_data %>% group_by(algorithm, library, N, P) %>% summarise_each(list(mean), flops))
  mean_flops$metric = "mean"
  final_data <- rbind(peak_flops, mean_flops)
}




str_cmp <- function(str1, str2){
  str_detect(str1, regex(str2, ignore_case = TRUE))
}


filterData <- function(rawData, importantCols,mShape, scaling, p, pEnd, algorithm, flops, flopsEnd, V, Vend, commModelRatio,t, tEnd){
  rawData[] <- lapply(rawData, function(x) if(is.factor(x)) as.character(x) else x)
  if(missing(mShape)) {
    mShape = ''
    mShapeEnd = 'z'
  } else {
    mShapeEnd = mShape
  }
  if(missing(scaling)) {
    scaling = ''
    scalingEnd = 'z'
  } else {
    scalingEnd = scaling
  }
  if(missing(algorithm)) {
    algorithm = ''
    algorithmEnd = 'z'
  } else {
    algorithmEnd = algorithm
  }
  if(missing(p)) {
    p = 0
  }
  if(missing(pEnd)) {
    pEnd = 999999
  }
  if(missing(flops)) {
    flops = 0
  } 
  if(missing(flopsEnd)) {
    flopsEnd = 99999
  }
  if(missing(V)) {
    V = 0
  }
  if(missing(Vend)) {
    Vend = 999999
  }
  if(missing(t)) {
    t = 0
  }
  if(missing(tEnd)) {
    tEnd = 999999
  }
  filtered = rawData[importantCols][rawData$mShape >= mShape & rawData$mShape <= mShapeEnd
                         & rawData$scaling >= scaling & rawData$scaling <= scalingEnd
                         & rawData$p >= p & rawData$p <= pEnd
                         & rawData$V >= V & rawData$V <= Vend
                         & rawData$time >= t & rawData$time <= tEnd
                         & rawData$algorithm >= algorithm & rawData$algorithm <= algorithmEnd
                         & rawData$flops >= flops & rawData$flops <= flopsEnd,]
  filtered[complete.cases(filtered),]
}

#-----------end of helper functions------------------#

gm_mean = function(x, na.rm=TRUE){
  exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
}

# Summarizes data. 
#
# Args:
#   data: a data frame.
#   measurevar: the name of a column that contains the variable to be summariezed
#   groupvars: a vector containing names of columns that contain grouping variables
#   na.rm: a boolean that indicates whether to ignore NA's
#   conf.interval: the percent range of the confidence interval (default is 95%)
#
# Returns:
#   element count, mean, median, min, max, 
#   standard deviation, standard error of the mean,
#   confidence interval (default 95%) using student-T distribution, and conf. int. using normal distribution.
#
# invocation example: summary_collectives <- CalculateDataSummary(data_collectives, measurevar="Time",
#                                             groupvars=c("PEs"), conf.interval=.99)

summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                                 conf.interval=.95, quantile.interval=.95, .drop=TRUE) {
  library(plyr)
  
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  
  
  nonNormalDistrData.CI <- function(data, probs, high=TRUE){
    n <- length(data)
    z.val <- -qnorm((1.0-probs)/2.0)

#     print(paste("n: ",n))
#     print(paste("p: ",probs))
#     print(paste("z: ",z.val))
#     print(paste("s: ",sqrt(n)))
    if(high){
      rank <- 1 + (( n + z.val  * sqrt(n) ) / 2)
  #     print(paste("rank: ",rank))
      rank <- ceiling( rank )
    } else {
      rank <- (( n - z.val  * sqrt(n) ) / 2)
   #    print(paste("rank: ",rank))
      rank <- floor(  rank )
    }  
    data.sorted <- sort(data)
    
    result <- data.sorted[rank]
#     print(data.sorted)
#     print(paste("rank: ",rank))
#     print(result)
#     print(paste("result: ",result))
    return(result)
  }

  
  # This does the summary. For each group's data frame, return a vector with
  # N, mean, and sd
  qPerc <- (1 - quantile.interval)/2
  datac <- ddply(data, groupvars, .drop=.drop,
                 .fun = function(xx, col) {
                   c(NumMeasures    = length2(xx[[col]], na.rm=na.rm),
                     min  = min    (xx[[col]], na.rm=na.rm),
                     max  = max    (xx[[col]], na.rm=na.rm),
                     mean = mean   (xx[[col]], na.rm=na.rm),
                     median = median(xx[[col]], na.rm=na.rm),
                     StandardDev   = sd     (xx[[col]], na.rm=na.rm),
                     Quantile.low = quantile(xx[[col]], probs=qPerc, names=FALSE ),
                     Quantile.high = quantile(xx[[col]], probs=1 - qPerc, names=FALSE ),
                     #interval for non normal distributions (up and lower bounds)
                     #CI.NNorm.low = nonNormalDistrData.CI(xx[[col]], probs=(1 - conf.interval)/2, high=FALSE),
                     #CI.NNorm.high = nonNormalDistrData.CI(xx[[col]], probs=(1 - conf.interval)/2, high=TRUE)
                     cil = nonNormalDistrData.CI(xx[[col]], probs=conf.interval, high=FALSE),
                     cih = nonNormalDistrData.CI(xx[[col]], probs=conf.interval, high=TRUE)
                   )
                 },
                 measurevar, .inform=TRUE
  )
  
  datac$StandardErr <- datac$StandardDev / sqrt(datac$NumMeasures)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval: 
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMultT <- qt(conf.interval/2 + .5, datac$NumMeasures-1)
  datac$CI.Norm.StudT <- datac$StandardErr * ciMultT
  #normal distr
  ciMultN <- qnorm(conf.interval/2 + .5)
  datac$CI.Norm.Norm <- datac$StandardErr * ciMultN
  
  return(datac)
}

#############################################################################################
#
#############################################################################################
CalculateConfidenceIntervalNormalDistr <- function(sampleMean, standardDeviation, SampleSize, levelOfConfidence ){
  a <- 1 - levelOfConfidence
  a <- a / 2
  a <- 1 - a
  error <- qnorm(a)*standardDeviation/sqrt(SampleSize)
  res <- c( low=sampleMean - error, up=sampleMean + error)
  return(res)
}

CalculateConfidenceIntervalStudentTDistr <- function(sampleMean, standardDeviation, SampleSize, levelOfConfidence ){
  a <- 1 - levelOfConfidence
  a <- a / 2
  a <- 1 - a
  error <- qt(a,df=SampleSize-1)*standardDeviation/sqrt(SampleSize)
  res <- c( low=sampleMean - error, up=sampleMean + error)
  return(res)
}

GetOutliersLimits <- function(x, na.rm = TRUE, quantile.probs = c(.05, .95), ...) {
  qnt <- quantile(x, probs= quantile.probs , na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  res <- c(qnt[1] - H,qnt[2] + H)
  return(res)
}

RemoveOutliers <- function(x, na.rm = TRUE, quantile.probs = c(.05, .95), ...) {
  qnt <- quantile(x, probs= quantile.probs , na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

# To test normality of a dataset
TestDataNormality <- function(x,...){
  res <- shapiro.test(x)
  return(res)
}

# Z-trasform
#http://www.inside-r.org/packages/cran/GeneNet/docs/hotelling.transform
#Example
# # load GeneNet library
# library("GeneNet")
# 
# # small example data set 
# r <- c(-0.26074194, 0.47251437, 0.23957283,-0.02187209,-0.07699437,
#        -0.03809433,-0.06010493, 0.01334491,-0.42383367,-0.25513041)
# 
# # transformed data
# z1 <- z.transform(r)
# z2 <- hotelling.transform(r,7)
# z1
# z2