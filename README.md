<h1 align = "center"> ML1 </h1>

## Метрические алгоритмы классификации

**Метрические методы обучения** - методы, основаны на анализе сходства объектов. То есть, схожим объектам - схожий ответ. 

Самым простым примером метрического метода обучения есть **метод ближайшего соседа**. С названия видно, что требуется среди всех объектов найти того, с кем будет минимальное расстояние. При этом расстояние можно определить евклидовой метрикой. Далее следует определить входной объект к классу, к которому этот ближайший сосед относится. 

### Реализация метода ближайшего соседа ###
``` r
  ## Вызов функции 1NN 
  
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
## Вызов функции KNN 
  
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