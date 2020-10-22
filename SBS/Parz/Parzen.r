euDist <- function(u, v){
  sqrt( sum((u - v)^2) )
}

Dist <- function(z, xl, metr = euDist ){
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  
  dist <- matrix(NA, l, 4)
  
  for (i in 1:l) {
    dist[i, ] <- c(metr(xl[i, 1:n], z), xl[i,1], xl[i,2], xl[i, 3])
  }

  return(dist)
}

Kernel <- function(r){
  
   #return((1/(sqrt(2*pi))) * exp((-1/2) * r * r))
   
    
    if (abs(r) <= 1) {
        #return(1 / 2)
        #return(1 - abs(r))
        #return( (3 * (1 - r^2) ) / 4)
        return( (15 * (1 - r^2) ^ 2 ) / 16)
      }
      
    return(0)
    
}

Parz <- function(z, xl, h, kern = Kernel){
    #xl <- Dist( z, xl )
  
    l <- dim(xl)[1]
    n <- dim(xl)[2] 
    
    classes <- xl[1:l, n ]
    
    counters <- table(classes)
    names(counters) <- c("setosa", "versicolor", "virginica")
    counters[1:length(counters)] <- 0  
  
    for (i in 1:l) {
        counters[xl[i, 4]] <- counters[xl[i, 4]] + kern(xl[i, 1] / h)
    }

    
    if (max(counters) != 0 ) {
      return(names(which.max(counters)))
    }
    return(4)
}

LOO <- function(xl, alg = Parz){
    l <- dim(xl)[1]
    h <- seq(0.1, 2, 0.1)
    MLOO <- matrix(0, length(h), 1)
  
    for (i in 1:l) {
        point <- c(xl[i, 1:2])
        
        dist <- Dist(point, xl[-i,])
        
        for (j in 1:length(h) ) {
            if (alg(point, dist, h[j]) != xl[i, 3]) {
                MLOO[j][1] <-  MLOO[j][1] + 1 / l
            }  
          
        }
        
    }

    return(MLOO)
  
}


z <- c(4.6, 2.0)
xl <- iris[, 3:5]
class <- Parz(z, xl, 0.1)


loo <- LOO(xl)

min <- which.min(loo)
plot(loo, xlab = "h", ylab = "LOO", main = "LOO для парзеновского окна с квартическим ядром", type = "o", xaxt = "n")
axis(1, at = 1:20, labels = seq(0.1, 2, 0.1))
points(min[1], loo[min[1]], pch = 21, bg = "red", cex = 1)
text(min[1]  , loo[min[1]] + 0.01, "(0.4; 0.04)" ,cex = 0.8, col = "red")


