<h1 align = "center"> ML1 </h1>

# Постановка задачи #

Существует множество объектов **X** , для которых дано множество ответов **Y**. Между этими двумя множествами существует некая зависимость - **целевая зависимость**. Каждую пару объект-ответ будем называть **прецедентом** (sample). А матрицу таких зависимостей - **обучающая выборка** (training sample). 

Перед нами стоит цель построить алгоритм **a : X -> Y**, который способен классифицировать произвольный объект из множества **X**. 

# Метрические алгоритмы классификации #

**Метрические методы обучения** - методы, основаны на **анализе сходства объектов**. Близкие объекты лежат в одном классе.

Ниже представлены примеры метрических методов.

+ [1NN](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B5%D0%B3%D0%BE-%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B0-1nn)
+ [KNN](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-k-%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B8%D1%85-%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B5%D0%B9-knn)
+ [KWNN](https://github.com/deviati0n/ML1/tree/master/SBS/KWNN)
+ [Метод парзеновского окна](https://github.com/deviati0n/ML1/tree/master/SBS/Parz)
+ [LOO](https://github.com/deviati0n/ML1/tree/master/SBS/LOO)

## Метод ближайшего соседа (1NN) ##
Самым простым примером метрического метода обучения есть **метод ближайшего соседа**. С названия видно, что требуется среди всех объектов найти того, с кем будет минимальное расстояние до элемента, который классифицируется, и определить входной объект к классу, к которому этот ближайший сосед относится. 

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
<img src="https://user-images.githubusercontent.com/71149650/94456760-52d37000-01bc-11eb-82e0-c286f933c238.png" alt="1NN" width="550"/>

**Достоинства**
+ Простота реализации

**Недостатки**
+ Нет параметров
+ Требуется хранить полную выборку
+ Качество алгоритма сильно зависит от выбранной метрики

## Метод k ближайших соседей (KNN) ##
Метод KNN среди всех объектов ищет k самых ближайших соседей и определяет входной объект к классу, у которого максимальное вхождение среди k соседей. 

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
<img src="https://user-images.githubusercontent.com/71149650/94458268-6384e580-01be-11eb-8eb7-48b76bade767.png" alt="Карта классифкации для 4NN" width="550"/>

**Достоинства**
+ Простота реализации
+ При правильном подборе k будет хорошее качество классификации

**Недостатки**
+ Малое кол-во параметров
+ Требуется хранить полную выборку
+ Качество алгоритма сильно зависит от выбранной метрики



