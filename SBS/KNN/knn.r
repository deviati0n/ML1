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

KNN <- function(z, xl, k){
    ordXl <- sortByDist( z, xl )
    n <- dim(ordXl)[2] - 1

    classes <- ordXl[1:k, n + 1]
    counts <- table(classes)
    
    class <- names(which.max(counts))
    return(class)
}


z <- c(4.6, 2.0)
xl <- iris[, 3:5]




