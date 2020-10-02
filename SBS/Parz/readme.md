## Метод парзеновского окна ##

В данном методе мы будем присваивать вес каждому объекту на основе **расстояния** от элемента, который мы классифицируем, до элемента из выборки. Классифицируемый объект будет иметь некоторую окрестность с радиусом **h**. *С помощью (авылаоывл)будем находить отношение между расстоянием до. Для каждого объекта из выборки будем считать абвыавы*


``` r
Parz <- function(z, xl, h, kern = Kernel){
  dist <- Dist( z, xl )

  l <- dim(dist)[1]
  n <- dim(dist)[2] - 1
  
  classes <- xl[1:l, n ]
  
  
  counters <- table(classes)
  counters[1:length(counters)] <- 0  

  for (i in 1:l) {
      counters[dist[i, 4]] <- counters[dist[i, 4]] + kern(dist[i,1] / h)

  }
  
  if (max(counters) != 0 ) {
    return(names(which.max(counters)))
  }
  return(4)
}
```
<img src = "https://user-images.githubusercontent.com/71149650/94863884-d226a500-0443-11eb-8ec6-e2029c14cb32.png" width = "550"/>
