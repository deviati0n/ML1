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

# Параметры: выборка и метод, который используют для классификации
LOO <- function(xl, etalon, alg = KNN ){
  
  # Инициализация вектора для LOO
  MLOO <- matrix(0, dim(xl)[1] - 1, 1)
  l <- dim(xl)[1]
  
  for (i in 1:l){
    # Берем точку из выборки и переопределяем выборку
    point <- c(xl[i,1:2])

    # Сортируем выборку
    ordXl <- sortByDist( point, etalon )
    
    for (k in 1:(l - 1)){
      # Сравнение работы алогоритма и класса исключенной точки
      if( alg(point, ordXl, k ) != xl[i, 3] ){
        
        # Если алгоритм ошибся, то штрафуем его
        MLOO[k][1] <-  MLOO[k][1] + (1/l)
      }
    }
  }
  # Выбираем оптимальный k
  min <- which.min(MLOO)
  print(MLOO)
  return(min)
  
} 

marginF <- function(point, xl, k){
  n <- dim(xl)[2]
  ordXl <- sortByDist(point[1:2], xl)
  
  classes <- ordXl[1:k, n]
  counts <- table(classes)
  
  r1 <- counts[point$Species]
  
  r2 <- sum(counts[names(counts) != point$Species])
  
  return(r1 - r2)
  
}

stolpF <- function(xl, delta, miss, algK = KNN, algM = marginF ) {
  k <- 7
  marg <- matrix(0, dim(xl)[1], 1)
  
  for (i in 1:dim(xl)[1]) {
    
    marg[i] <- marginF(xl[i,], xl[-i,], k)
    
  }
  
  xd <- xl[which(marg > 0),]
  marg <- marg[which(marg > 0)]
  
  omega <- data.frame()
  omegaT <- c()
  
  for (i in levels(xd[, 3])) {
    class <- which(xd[, 3] == i)
    omega <- rbind(omega, xd[class[which.max(marg[class])],])
    omegaT <- c(omegaT, class[which.max(marg[class])])
    
  }
  xd <- xd[-omegaT,] 
  l <- dim(xd)[1]
  
  while (dim(xl)[1] > dim(omega)[1]) {
    
    k <- LOO(xd, omega)
    print(k)
    
    rownames(xd) <- c(1:l)
    
    marg <- matrix(0, dim(xd)[1], 1)
    err <- 0
    
    for (i in 1:dim(xd)[1]) {
      
      marg[i] <- marginF(xd[i,], omega, k)
      
      point <- c(xd[i,1], xd[i,2])
      
      if (algK(point, omega, k) != xd[i, 3]) {
        err <- err + 1 
      }
      
    } # for
    
    print(paste("miss: ", err))

    E <- xd[which(marg < 0),]
    marg <- marg[which(marg < 0)]
    
    if (dim(E)[1] < miss ) {
      break
    }
    
    
    # ïåðåäåëàé, ñëàäêàÿ
    omega <- rbind(omega, E[which.min(marg),])
    omegaT <- as.numeric(rownames(E[which.min(marg),]))
    
    xd <- xd[-omegaT, ]
    
    print(omega)
    
    l <- dim(xd)[1]
    
    
  }# while
  
  return(omega)
  
}

classMap <- function(xl){
  xd <- iris[,3:5]
  colors <- c( "setosa" = "purple","versicolor" = "pink2", "virginica" = "yellow2", "black" )
  plot( xl[, 1:2], pch = 21, xlim = c(1, 7), ylim = c(-1,3),main = "...", bg = colors[xl[,3]], col = colors[4], cex = 1.3 )
  i <- 0.8
  while (i <= 7.3) {
    j <- -1.1
    while (j <= 3.2) {
      z <- c(i, j)
      class <- KNN(z, xl, 1)
      points(z[1], z[2], pch = 1, col = colors[class], asp = 1)
      j <- j + 0.1
    }
    i <- i + 0.1
  }
  
}

stolpCount <- function(){
  xl <- iris[, 3:5]
  st <- stolpF(xl, -0.01, 1)
  
  time(st)
  
  #classMap(st)
  
  #colors <- c( "setosa" = "purple","versicolor" = "pink2", "virginica" = "yellow2", "black" )
  #plot( iris[, 3:4], pch = 21, xlim = c(1, 7), ylim = c(-1,3),
  #      main = "Карта классификации ирисов фишера",
  #      col = colors[iris$Species], cex = 1.3 )
  #
  #for (i in 1:dim(st)[1]) {
  #  points(st[i,1:2], pch = 21,  bg = colors[st[i,3]], col = colors[st[i,3]], cex = 1.4)
  #  
  #}
  
}


marginCount <- function(){
  xl <- iris[, 3:5]
  
  margin <- matrix(0, dim(xl)[1], 1)
  
  for (i in 1:dim(xl)[1]) {
    point <- xl[i,]
    margin[i] <- marginF(point, xl[-i,], 7)
    
  }
  
  margin <- margin[order(margin)]
  
  plot(margin, type = "l")
  points(rep(0, 150), type = "l")
  
}

time <- function(st){
  xd <- iris[,3:5] 
  start <- Sys.time()
  for(i in 1:dim(xd)[1]) {
    
    xl <- xd[-i,]
    point <- c(xd[i,1:2])
    KNN(point, st, 1)
    
  }
  
  end <- Sys.time()
  
  print(end - start)
  
}

stolpCount()
