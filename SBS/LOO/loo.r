euDist <- function(u, v){
  sqrt( sum((u - v)^2) )
}

sortByDist <- function(z, xl, metr = euDist ){
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  
  dist <- matrix(NA, l, 2)
  print(xl[1, 1:n])
  for (i in 1:l) {
    dist[i, ] <- c(i, metr(xl[i, 1:n], z))
  }
  
  ordXl <- xl[order(dist[, 2]), ]
  return(ordXl)
}

KNN <- function(z, xl, k){
  n <- dim(xl)[2] - 1
  
  classes <- xl[1:k, n + 1]
  counts <- table(classes)
  
  class <- names(which.max(counts))
  
  return(class)
}

LOO <- function(xl, alg = KNN ){
  MLOO <- matrix(0, dim(xl)[1] , 1)
  l <- dim(xl)[1]
  
  for (i in 1:l) {
    point <- c(xl[i,1:2])
    
    new_iris <- xl[-i,]
    ordXl <- sortByDist( point, new_iris )
    
    for (k in 1:l  ) {
      if ( alg(point,  ordXl, k ) != xl[i, 3] ) {
        MLOO[k][1] <-  MLOO[k][1] + 1 / l
      }
    }
  }
  return(MLOO)
}


xl <- iris[,3:5]
miss <- LOO(xl)


#min <- order(miss)
min <- which.min(miss)

#par(mfrow = c(1,2))

plot(miss, xlab = "k", ylab = "LOO", main = "Работа LOO для полной выборки (KNN)", type = "l")
points(min[1], miss[min[1]], pch = 21, bg = "red",cex = 0.6)
text(min[1] + 4, miss[min[1]] + 0.035, " (6, 0.334) " ,cex = 0.8, col = "red")



#colors <- c( "setosa" = "purple","versicolor" = "pink2", "virginica" = "yellow2")
#plot( iris[, 3:4], pch = 21,
#      bg = colors[iris$Species], col = "black",
#      asp = 1, ylim = c(-1, 3) )

