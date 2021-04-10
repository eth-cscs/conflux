### Communication volume models for different LU algorithms

el_size = 8
scaling_factor = 1e6
c=1


calc_effP <- function(P, algo) {
  return(P)
}
  
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# !!! all models return aggregated, summed over all P, communication volume in MB !!! #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

model_2D <- function(N, P) {
  return((N^2 * sqrt(calc_effP(P, "MKL"))  + N^2) / scaling_factor * el_size)
}

model_candmc <- function(N, P) {
  return((5*N^2 / sqrt(P*P^(1/3)) + 3*P^(1/3) * N^2/P + N^2*log2(P*P^(1/3)) / (2*log2(P)*sqrt(P*P^(1/3))) + 3*N^2/(log2(P)*sqrt(P*P^(1/3))))/ scaling_factor * el_size * P)
}

model_conflux <- function(N,P) {
  return(model_conflux_2(N,P))
}


# different test models
model_conflux_1 <- function(N,P) {
  return( ( (N + P^(1/3))* ( 2*N*P^(4/3) - N + P^2 - P^(4/3) - 3*P^(5/3) + P^(7/3) + P^(4/3) * log2(P) / 3  ) ) / P / scaling_factor * el_size  )
}

model_conflux_2_nv <- function(N,P) {
  return( ( (N + P^(1/3))* ( 2*N*P^(4/3) - N + P^2 - P^(4/3) - 3*P^(5/3) + P^(4/3) * log2(P) / 3  ) ) / P / scaling_factor * el_size  )
}

model_conflux_3 <- function(N,P) {
  return( ( (N + P^(1/3))* ( 2*N*P^(4/3) - N + P^2 - P^(4/3) - 3*P^(5/3) + P^(4/3) * log2(P) / 3  ) ) / P )
}

model_conflux_2 <- Vectorize(model_conflux_2_nv)


# --------------------- MODELS ADAPTED FOR CHOLESKY DECOMPOSITION ---- #

model_psychol_nv <- function(N,P) {
  return( 0.8* ( (N + P^(1/3))* ( 2*N*P^(4/3) - N + P^2 - P^(4/3) - 3*P^(5/3) + P^(4/3) * log2(P) / 3  ) ) / P / scaling_factor * el_size  )
}

model_psychol_2 <- Vectorize(model_psychol_nv)

model_psychol <- function(N,P) {
  return(model_psychol_2(N,P))
}


model_2D_chol <- function(N, P) {
  return( 0.8* (N^2 * sqrt(calc_effP(P, "MKL"))  + N^2) / scaling_factor * el_size)
}

model_capital <- function(N, P) {
  return(1.3*(5*N^2 / sqrt(P*P^(1/3)) + 3*P^(1/3) * N^2/P + N^2*log2(P*P^(1/3)) / (2*log2(P)*sqrt(P*P^(1/3))) + 3*N^2/(log2(P)*sqrt(P*P^(1/3))))/ scaling_factor * el_size * P)
}



model_all_nv <- function(N,P,algo) {
  if ((algo == "mkl") || (algo=="slate")) return (model_2D(N,P))
  else if (algo == "candmc") return (model_candmc(N,P))
  else if (algo == "conflux") return (model_conflux(N,P))
  else warn(paste("Algorithm", algo, "unknown!"))
}
