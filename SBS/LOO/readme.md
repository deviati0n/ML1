### Реализация скользящего контроля LOO для KNN ###
``` r
# Параметры: выборка и метод, который используют для классификации
LOO <- function(xl, alg = KNN ){

    # Инициализация вектора для LOO
    MLOO <- matrix(0, dim(xl)[1] - 1, 1)
    l <- dim(xl)[1]
    
    for (i in 1:l){
        # Берем точку из выборки и переопределяем выборку
        point <- c(xl[i,1:2])
        newXl <- xl[-i,]
        
        # Сортируем выборку
        ordXl <- sortByDist( point, newXl )
        
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
    return(min)
}
```
#### Работа LOO для выборки из 30 случайных элементов ####
<img src="https://user-images.githubusercontent.com/71149650/94165933-6cf71080-fe93-11ea-8914-6bd689c58458.png" alt="LOO для выборки из 30 случайных элементов" />

#### Работа LOO для полной выборки и карта классификаци для 6NN ####
<img src="https://user-images.githubusercontent.com/71149650/94172577-56ed4e00-fe9b-11ea-8ef1-082e1b5da603.jpg" alt="LOO для полной выборки" />
