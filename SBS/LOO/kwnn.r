euDist <- function(u, v){
  sqrt( sum((u - v)^2) )
}

sortByDist <- function(z, xl, metr = euDist ){
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  
  dist <- matrix(NA, l, 2)
  
  for (i in 1:l) {
    dist[i, ] <- c(i, metr(xl[i, 1:n], z))
  }
  
  ordXl <- xl[order(dist[, 2]), ]
  return(ordXl)
}

KWNN <- function(z, xl, k, q ){
  
  n <- dim(xl)[2] 
  
  #xl <- sortByDist(z, xl )
  
  classes <- xl[1:k, n ]
  counters <- table(classes)
  counters[1:length(counters)] <- 0
  
  
  for (i in 1:k) {
      counters[classes[i]] <- counters[classes[i]] + q^i
  }
  print(counters)
  
  return(names(which.max(counters)))
  
}

LOO <- function(xl, alg = KWNN ){
  l <- dim(xl)[1]
  
  r1 <- 0.02
  r2 <- 0.98
  
  MLOO <- matrix(0, l - 1 , r2/r1 )
  
  
      for (i in 1:l) {
      print(i)
      
      point <- c(xl[i,1:2])
  
      
      new_iris <- xl[-i,]
  
      ordXl <- sortByDist( point, new_iris )
      
      for (k in 1:(l - 1)) {
          j <- 1
          for (q in seq(r1, r2, r1 ) ) {
              
              if ( alg(point,  ordXl, k, q ) != xl[i, 3] ) {
                MLOO[k, j] <- MLOO[k, j] + 1/l
              }
            
              j <- j + 1
              
          }
        
      }
    
  }
  return(MLOO)
}



xl <- iris[, 3:5]
z <- c(4.6, 2.0)
colors <- c( "setosa" = "purple","versicolor" = "pink2", "virginica" = "yellow2", "black" )
plot( iris[, 3:4], pch = 21, xlim = c(1, 7), ylim = c(-1,3),main = "Карта классификации ирисов Фишера для 6NN", bg = colors[iris$Species], col = colors[4], cex = 1.3 )



miss <- LOO(xl)
class <- KWNN(z, xl, 30, 0.96)


points(z[1], z[2], pch = 22, bg = colors[class], cex = 1.3)

q <- which.min(miss) %/% (dim(xl)[1] - 1) + 1
k <- which.min(miss) %% (dim(xl)[1] - 1)

print(k)
print(q)

print(miss[k,q])

heatmap.2(miss,dendrogram = 'none', Rowv = FALSE, Colv = FALSE,trace ='none', main = "Тепловая карта для LOO для KWNN", xlab = "q", ylab = "k")





