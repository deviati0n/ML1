<h1 align = "center"> ML1 </h1>

## Метрические алгоритмы классификации

**Метрические методы обучения** - методы, основаны на анализе сходства объектов. То есть, схожим объектам - схожий ответ. 

Самым простым примером метрического метода обучения есть **метод ближайшего соседа**. С названия видно, что требуется среди всех объектов найти того, с кем будет минимальное расстояние и определить входной объект к классу, к которому этот ближайший сосед относится. 

### Реализация метода ближайшего соседа ###
``` r
  ## Параметрами функции являются входной объект и выборка
  NN <- function(z, xl){ 
    
    ## Сортировка выборки по входному объекту
    ordXl <- sortByDist( z, xl )
    n <- dim(ordXl)[2] - 1
    
    ## Выбор самого ближайшего класса
    class <- ordXl[1, n + 1]
    return (class)
  }
```
<img src="https://user-images.githubusercontent.com/71149650/94142517-154aac00-fe77-11ea-8fdf-1196cb69e5d8.png" alt="1NN" width="550"/>

### Реализация метода k ближайших соседей ###
``` r
## Параметрами функции являются входной объект,выборка и кол-во соседей для проверки
KNN <- function(z, xl, k){
    ## Сортировка выборки по входному объекту
    ordXl <- sortByDist( z, xl )
    n <- dim(ordXl)[2] - 1

    ## Выбор k ближайших классов
    classes <- ordXl[1:k, n + 1]
    counts <- table(classes)
    
    ## Поиск класса с максимальным кол-вом вхождений
    class <- names(which.max(counts))

    return (class)
}
```
#### Карта классификации ирисов Фишера для 4NN ####
<img src="https://user-images.githubusercontent.com/71149650/94147378-e126b980-fe7d-11ea-9cf1-3db0bccaebf9.png" alt="Карта классифкации для 4NN" width="550"/>

### Реализация скользящего контроля LOO для KNN ###
``` r
# Параметры: выборка и метод, который используют для классификации
LOO <- function(xl, alg = KNN ){
    # Инициализация вектора для LOO
    MLOO <- matrix(0, dim(xl)[1] - 1, 1)
    l <- dim(xl)[1]

    for (i in 1:l){
        point <- c(xl[i,1:2])
        new_iris <- xl[-i,]
        ordXl <- sortByDist( point, new_iris )
      
        for (k in 1:(l - 1)){
            if( alg(point, ordXl, k ) != xl[i, 3] ){
                MLOO[k][1] <-  MLOO[k][1] + (1/l)
            }
        }
    }
    return(MLOO)
}
```
#### Работа LOO для выборки из 30 случайных элементов ####
<img src="https://user-images.githubusercontent.com/71149650/94165933-6cf71080-fe93-11ea-8914-6bd689c58458.png" alt="LOO для выборки из 30 случайных элементов" />

#### Работа LOO для полной выборки и карта классификаци для 6NN ####
<img src="https://user-images.githubusercontent.com/71149650/94172577-56ed4e00-fe9b-11ea-8ef1-082e1b5da603.jpg" alt="LOO для полной выборки" />

