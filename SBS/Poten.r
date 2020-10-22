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
  
  return((1/(sqrt(2*pi))) * exp((-1/2) * r * r))


  #if (abs(r) <= 1) {
    #return(1 / 2)
    #return(1 - abs(r))
    #return( (3 * (1 - r^2) ) / 4)
    #return( (15 * (1 - r^2) ^ 2 ) / 16)
  #}
  
  #return(0)
  
}

Parz <- function(z, xl, h, y, kern = Kernel){
  #xl <- Dist( z, xl )
  
  l <- dim(xl)[1]
  n <- dim(xl)[2] 
  
  classes <- xl[1:l, n  ]
  
  counters <- table(classes)
  names(counters) <- c("setosa", "versicolor", "virginica")
  counters[1:length(counters)] <- 0
  
  for (i in 1:l) {
    counters[xl[i, n ]] <- counters[xl[i, n ]] + y[i] * kern(xl[i, 1] / h)
  }
  
  if (max(counters) != 0 ) {
    return(names(which.max(counters)))
  }
  return(4)
}

Potenc <- function(xl, miss, alg = Parz) {
    l <- dim(xl)[1]
    #y <- matrix(0, l, 1)
    y <- rep(0, l)
    h <- 1
    err <- miss + 1
    
    
    
    dist <- matrix(NA, l, l)
    for (i in 1:l) {
      
      for (j in 1:l) {
        
        dist[i, j] <- euDist(xl[i, 1:2], xl[j, 1:2])
        
      }
      
    }
    
    while (err > miss) {
      
      err <- 0
      for (i in 1:l) {
        dist2 <- cbind(dist[,i], iris[, 3:5])
        dist2 <- dist2[-i,]
        point <- c(xl[i, 1:2])
        
        
        if (alg(point, dist2, h, y) != xl[i, 3] && y[i] < 6) {
          y[i] <- y[i] + 1
          #err <- err + 1
        }
        
      }
      
      for (i in 1:l) {
        dist2 <- cbind(dist[,i], iris[, 3:5])
        dist2 <- dist2[-i,]
        point <- c(xl[i, 1:2])
        
        
        if (alg(point, dist2, h, y) != xl[i, 3]) {
          err <- err + 1
        }
        
      }

      print(err)
      
    }
    
    return(y)
}

xl <- iris[, 3:5]

z <- Potenc(xl, 10)
colors <- c( "setosa" = "purple","versicolor" = "pink2", "virginica" = "yellow2", "white" )
plot( iris[, 3:4], pch = 21, xlim = c(1, 7), ylim = c(-1,3),main = "Разброс потенциалов", bg = colors[iris$Species], cex = 1.3 )

for (i in 1:150) {
   if (z[i] != 0) {
     points(iris[i,3], iris[i,4], pch = 21, col = colors[iris[i,5]], cex = 3 + z[i])
   }
 
}







