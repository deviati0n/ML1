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

op <- function(z, xl){
    
    l <- dim(xl)[1]
    n <- dim(xl)[2] - 1
    
    dist <- matrix(NA, l, 1)
    
    for (i in 1:l) {
        dist[i] <- euDist(xl[i, 1:n], z)
    }
    
    dist <- cbind(xl, dist)
    min <- which.min(dist[, 4])
    return(dist[min, 3])
    

}

NN <- function(z, xl){
    ordXl <- sortByDist( z, xl )
    n <- dim(ordXl)[2] - 1

    class <- ordXl[1, n + 1]
    return(class)
    
    ordXl <- op(z, xl)
    return(ordXl)
}

colors <- c( "setosa" = "purple","versicolor" = "pink", "virginica" = "yellow" )
plot( iris[, 3:4], pch = 21, bg = colors[iris$Species], col = colors[iris$Species], asp = 1 )

z <- c(4, 1)
xl <- iris[, 3:5]
#class <- NN(z, xl)
class <- op(z, xl)
points(z[1], z[2], pch = 22, bg = colors[class], asp = 1)