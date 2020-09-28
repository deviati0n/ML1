## Метод k взвешенных ближайших соседей (KNN) ##

## Реализация метода k взвешенных ближайших соседей (KNN) ##
``` r
  ## Параметрами функции являются входной объект, выборка, кол-во проверяемых соседей и значение q для весовой функции (q^i)  
  KWNN <- function(z, xl, k, q ){
    
    ## Сортировка выборки по входному объекту
    ordXl <- sortByDist( z, xl )
    n <- dim(ordXl)[2] 
    
    ## Выбор k ближайших объектов
    classes <- ordXl[1:k, n ]
    counters <- matrix(0, 3, 1)
    
    ## Вычисление вклада i-го соседа при классификации объекта z
    for (i in 1:k) {
      counters[classes[i]] <- counters[classes[i]] + q^i
    }

    names(counters) <- c("setosa", "versicolor", "virginica")
    
    ## Функция возвращает имя класса
    return(names(which.max(counters)))
    
  }
```
