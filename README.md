<h1 align = "center"> ML1 </h1>

# Постановка задачи #

Существует множество объектов **X** , для которых дано множество ответов **Y**. Между этими двумя множествами существует некая зависимость - **целевая зависимость**. Каждую пару объект-ответ будем называть **прецедентом** (sample). А матрицу таких зависимостей - **обучающая выборка** (training sample). 

Перед нами стоит цель построить алгоритм **a : X -> Y**, который способен классифицировать произвольный объект из множества **X**. 

+ [Метрические алгоритмы классификации](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5-%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC%D1%8B-%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8)
  + [1NN](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B5%D0%B3%D0%BE-%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B0-1nn)
  + [KNN](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-k-%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B8%D1%85-%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B5%D0%B9-knn)
  + [KWNN](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-k-%D0%B2%D0%B7%D0%B2%D0%B5%D1%88%D0%B5%D0%BD%D0%BD%D1%8B%D1%85-%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B8%D1%85-%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B5%D0%B9-kwnn)
  + [Парзеновское окно](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-%D0%BF%D0%B0%D1%80%D0%B7%D0%B5%D0%BD%D0%BE%D0%B2%D1%81%D0%BA%D0%BE%D0%B3%D0%BE-%D0%BE%D0%BA%D0%BD%D0%B0)
  + [Потенциальные функции](https://github.com/deviati0n/ML1#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-%D0%BF%D0%BE%D1%82%D0%B5%D0%BD%D1%86%D0%B8%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D1%85-%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D0%B9)
  + [Отбор эталонных объектов](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BE%D1%82%D0%B1%D0%BE%D1%80-%D1%8D%D1%82%D0%B0%D0%BB%D0%BE%D0%BD%D0%BD%D1%8B%D1%85-%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%D0%BE%D0%B2)
  + [LOO](https://github.com/deviati0n/ML1/blob/master/README.md#%D1%81%D0%BA%D0%BE%D0%BB%D1%8C%D0%B7%D1%8F%D1%89%D0%B8%D0%B9-%D0%BA%D0%BE%D0%BD%D1%82%D1%80%D0%BE%D0%BB%D1%8C-loo)
+ [Байесовские методы классификации](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%B1%D0%B0%D0%B9%D0%B5%D1%81%D0%BE%D0%B2%D1%81%D0%BA%D0%B8%D0%B5-%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D1%8B-%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8)
  + [Линии уровня нормального распределения](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BB%D0%B8%D0%BD%D0%B8%D0%B8-%D1%83%D1%80%D0%BE%D0%B2%D0%BD%D1%8F-%D0%BD%D0%BE%D1%80%D0%BC%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B3%D0%BE-%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F)
  + [Наивный нормальный байесовский классификатор](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BD%D0%B0%D0%B8%D0%B2%D0%BD%D1%8B%D0%B9-%D0%BD%D0%BE%D1%80%D0%BC%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9-%D0%B1%D0%B0%D0%B9%D0%B5%D1%81%D0%BE%D0%B2%D1%81%D0%BA%D0%B8%D0%B9-%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%82%D0%BE%D1%80)
  + [Подстановочный алгоритм (plug-in)](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BF%D0%BE%D0%B4%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BE%D1%87%D0%BD%D1%8B%D0%B9-%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC-plug-in)
  + [Линейный дискриминант Фишера - ЛДФ](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BB%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D1%8B%D0%B9-%D0%B4%D0%B8%D1%81%D0%BA%D1%80%D0%B8%D0%BC%D0%B8%D0%BD%D0%B0%D0%BD%D1%82-%D1%84%D0%B8%D1%88%D0%B5%D1%80%D0%B0)
  
+ [Линейные алгоритмы классификации](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BB%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D1%8B%D0%B5-%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC%D1%8B-%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8)
  + [Метод стохастического градиента](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-%D1%81%D1%82%D0%BE%D1%85%D0%B0%D1%81%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B3%D0%BE-%D0%B3%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%B0)
    + [ADALINE](https://github.com/deviati0n/ML1/blob/master/README.md#adaline)
    + [Правило Хэбба](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BF%D1%80%D0%B0%D0%B2%D0%B8%D0%BB%D0%BE-%D1%85%D1%8D%D0%B1%D0%B1%D0%B0)
    + [Логистическая регрессия](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BB%D0%BE%D0%B3%D0%B8%D1%81%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F-%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F)
  + [Метод опорных векторов](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-%D0%BE%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D1%85-%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BE%D0%B2-svm)

# Метрические алгоритмы классификации #

**Метрические методы обучения** - методы, основаны на **анализе сходства объектов**. Близкие объекты лежат в одном классе.

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

[Оглавление](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BF%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8)

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

[Оглавление](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BF%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8)

## Метод k взвешенных ближайших соседей (KWNN) ##

Данный метод схож с методом k ближайших соседей (KNN), добавляется лишь весовая функция, которая будет оценивать степень важности i-го соседа для классификации объекта z. При реализации метода KWNN было выбрана функция q^i. 

### Реализация метода k взвешенных ближайших соседей (KWNN) ###
``` r
  ## Параметрами функции являются входной объект, выборка, кол-во проверяемых соседей и значение q для весовой функции (q^i)  
  KWNN <- function(z, xl, k, q ){
    
    ## Сортировка выборки по входному объекту
    ordXl <- sortByDist( z, xl )
    n <- dim(ordXl)[2] 
    
    ## Выбор k ближайших объектов
    classes <- ordXl[1:k, n ]
    counters <- table(classes)
    counters[1:length(counters)] <- 0 
    
    ## Вычисление вклада i-го соседа при классификации объекта z
    for (i in 1:k) {
      counters[classes[i]] <- counters[classes[i]] + q^i
    }
    
    ## Функция возвращает имя класса
    return(names(which.max(counters)))
    
  }
```

<img src = "https://user-images.githubusercontent.com/71149650/94804503-9663ef00-03f3-11eb-8698-37b74d83a4d3.png" width = "550"/>

**Достоинства**
+ Простота реализации
+ При правильном подборе k и q будет хорошее качество классификации

**Недостатки**
+ Качество алгоритма сильно зависит от выбранной метрики.
+ Требуется хранить полную выборку

### Сравнение KNN и KWNN ###

Рассмотрим искуственную выборку с двумя классами и одной входной точкой, которую необходимо проклассифицировать. Интуитивно понятно, что точка будет принадлежать второму классу. 
Рассмотрим работу двух методов. На первом графике представлена работа **KNN** (k = 5) и видно, что метод классифицирует точку к первому классу. На втором графике работа **KWNN** (k = 5, q = 0.7) и относит точку ко второму классу. Таки образом, можно увидеть, что на некоторых выборках **KWNN** будет работать точнее, чем **KNN**.

<img src = "https://user-images.githubusercontent.com/71149650/95547893-a50e5f80-0a0c-11eb-93d0-3451380d80f4.png" />

[Оглавление](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BF%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8)

## Метод парзеновского окна ##

В данном методе мы будем присваивать вес каждому объекту на основе **расстояния** от элемента, который мы классифицируем, до элемента из выборки. Классифицируемый объект будет иметь некоторую окрестность с радиусом **h**. Для каждого объекта и выборки считается отношение расстояние до входного объекта и радиуса окна. Далее вызывается **ядерная функция**, которая вычисляет вес опираясь на значение отношения. 


**Существует 5 ядер:** 
+ Прямоугольное ядро: <a href="https://www.codecogs.com/eqnedit.php?latex=P(r)&space;=&space;\frac{1}{2}[|r|&space;\leq&space;1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(r)&space;=&space;\frac{1}{2}[|r|&space;\leq&space;1]" title="P(r) = \frac{1}{2}[|r| \leq 1]" /></a>
+ Треугольное ядро: <a href="https://www.codecogs.com/eqnedit.php?latex=T(r)&space;=&space;(1&space;-&space;|r|)[|r|&space;\leq&space;1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(r)&space;=&space;(1&space;-&space;|r|)[|r|&space;\leq&space;1]" title="T(r) = (1 - |r|)[|r| \leq 1]" /></a>
+ Ядро Епанечникова: <a href="https://www.codecogs.com/eqnedit.php?latex=T(r)&space;=&space;\frac{3}{4}(1-r^{2})[|r|&space;\leq&space;1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(r)&space;=&space;\frac{3}{4}(1-r^{2})[|r|&space;\leq&space;1]" title="T(r) = \frac{3}{4}(1-r^{2})[|r| \leq 1]" /></a>
+ Квартическое ядро: <a href="https://www.codecogs.com/eqnedit.php?latex=Q(r)&space;=&space;\frac{15}{16}(1-r^{2})^{2}[|r|&space;\leq&space;1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(r)&space;=&space;\frac{15}{16}(1-r^{2})^{2}[|r|&space;\leq&space;1]" title="Q(r) = \frac{15}{16}(1-r^{2})^{2}[|r| \leq 1]" /></a>
+ Гауссовское ядро: <a href="https://www.codecogs.com/eqnedit.php?latex=G(r)&space;=&space;(2\pi&space;)^{(-\frac{1}{2})e^{(-\frac{1}{2}r^{2})}&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G(r)&space;=&space;(2\pi&space;)^{(-\frac{1}{2})e^{(-\frac{1}{2}r^{2})}&space;}" title="G(r) = (2\pi )^{(-\frac{1}{2})e^{(-\frac{1}{2}r^{2})} }" /></a>


Явным недостатком первых четырех ядер в том, что они не могут классифицировать точки, которые не попали в окно входного объекта. Гауссовское ядро решает эту проблему.

### Реализация метода парзеновского окна ###
``` r
# Параметрами функции являются входной объект, выборка, радиус окна и функция ядра
Parz <- function(z, xl, h, kern = Kernel){
    # Нахождение расстояние от входного объекта до каждого элемента выборки
    dist <- Dist( z, xl )

    l <- dim(dist)[1]
    n <- dim(dist)[2] - 1

    # Формируй таблицу классов и обнуляем значения
    classes <- xl[1:l, n ]  
    counters <- table(classes)
    counters[1:length(counters)] <- 0  

    for (i in 1:l) {
        # Вычисление вклада i-го элемента выборки 
        counters[dist[i, 4]] <- counters[dist[i, 4]] + kern(dist[i,1] / h)
    }

    # Проверяем, что хоть один класс попал в окно
    if (max(counters) != 0 ) {

      # Возвращаем имя класса, у которого максимальное кол-во "голосов"
      return(names(which.max(counters)))

    }
    
    # Если никакой класс не попал в окно, возвращаем другой класс
    return(4)  
}
```

### Прямоугольное ядро ###
<img src = "https://user-images.githubusercontent.com/71149650/95464618-d6424d80-0982-11eb-9d0f-6a462bfd685d.png" />

### Треугольное ядро ###
<img src = "https://user-images.githubusercontent.com/71149650/95464919-30dba980-0983-11eb-99f3-f7232841a65d.png" />


### Ядро Епанечникова ###
<img src = "https://user-images.githubusercontent.com/71149650/95465185-79936280-0983-11eb-8877-bac88a4e5a7c.png" />


### Квартическое ядро ###
<img src = "https://user-images.githubusercontent.com/71149650/95465365-b7908680-0983-11eb-873a-c4a2408f6d4a.png" />

### Гауссовское ядро ###
<img src="https://user-images.githubusercontent.com/71149650/95022762-e3f58b80-0681-11eb-93f6-f3a3bdf0865c.png" />

**Достоинства**
+ Простота реализации
+ Не требуется сортировка выборки
+ При правильном подборе h будет хорошее качество классификации

**Недостатки**
+ Если входной объект не попадет в окно с радиусом h, то его не возможно будет проклассифицировать (не используя гауссовское ядро).
+ Требуется хранить полную выборку

[Оглавление](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BF%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8)

## Метод потенциальных функций ##

Метод схож с парзеновским окном, однако теперь у каждой точки выборки будет еще свой потенциал. Этот потенциал будет учитываться при подсчете веса точки выборки (умножать на значение ядерной функции). 


``` r
# Параметрами функции являются выборка, максимальное кол-во допустимых ошибок и алгоритм классификации
Potenc <- function(xl, miss, alg = Parz) {
    l <- dim(xl)[1]
    y <- matrix(0, l, 1)
    h <- 0.4
    err <- miss + 1
    
    # Задаем матрицу расстояний
    dist <- matrix(NA, l, l)
    for (i in 1:l) {
      
      for (j in 1:l) {
        
        dist[i, j] <- euDist(xl[i, 1:2], xl[j, 1:2])
        
      }
      
    }
    
    # Цикл для подбора потенциалов
    while (err > miss) {
      
      err <- 0
      for (i in 1:l) {
      
        # Добавляем к матрице выборки столбец расстояний 
        dist2 <- cbind(dist[,i], iris[, 3:5])
        dist2 <- dist2[-i,]
        point <- c(xl[i, 1:2])
        
        # Если алгоритм ошибся - усиливаем силу потенциала точки выборки
        if (alg(point, dist2, h, y) != xl[i, 3] && y[i] < 6) {
          y[i] <- y[i] + 1
        }
        
      }
      
      for (i in 1:l) {
      
        # Добавляем к матрице выборки столбец расстояний 
        dist2 <- cbind(dist[,i], iris[, 3:5])
        dist2 <- dist2[-i,]
        point <- c(xl[i, 1:2])
        
        # Если алгоритм ошибся - штрафуем его
        if (alg(point, dist2, h, y) != xl[i, 3]) {
          err <- err + 1
        }
        
      }

      print(err)
      
    }
    
    # Возвращаем вектор потенциалов
    return(y)
}
```

### Прямоугольное ядро ###
<img src = "https://user-images.githubusercontent.com/71149650/96218179-e6fb5080-0f8c-11eb-8828-8d97058ab95f.png" />

### Ядро Епанечникова ###
<img src = "https://user-images.githubusercontent.com/71149650/96218188-ecf13180-0f8c-11eb-8e7d-a770cf4c052b.png" />

### Квартическое ядро ###
<img src = "https://user-images.githubusercontent.com/71149650/96151403-0dc97080-0f14-11eb-895c-20dd39f31aa8.png" />

**Достоинства**
+ Простота реализации
+ Не требуется сортировка выборки
+ При правильном подборе h и y будет хорошее качество классификации
+ Большое кол-во параметров, которые можно настраивать 

**Недостатки**
+ Если входной объект не попадет в окно с радиусом h, то его не возможно будет проклассифицировать (не используя гауссовское ядро).
+ Требуется хранить полную выборку

[Оглавление](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BF%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8)

## Отбор эталонных объектов ##

Для начала требуется определить, что такое отступ. **Отступ** - это величина, которая показывает степень типичности элемента определенного класса. Отступ считается, основываясь на любом из метрических алгоритмов. В примере использовался **KNN**. 

### Реализация функции отступа ###
``` r
# Параметры: элемент выборки, выборка и значение k для KNN
marginF <- function(point, xl, k){

  n <- dim(xl)[2]
  ordXl <- sortByDist(point[1:2], xl)

  classes <- ordXl[1:k, n]
  counts <- table(classes)
  
  # Вес класса входного объекта
  r1 <- counts[point$Species]
  
  # Сумма весов классов, которые не равны классу входного объекта
  r2 <- sum(counts[names(counts) != point$Species])
  
  # Возвращаем разницу между двумя величинами
  return(r1 - r2)
  
}
```

Данная функция позволяет разбить множество элементов выборки на 5 классов: эталонные, неинформативные, пограничные, ошибочные, шумовые. К примеру, **шумовые** объекты выборки будут иметь большое **отрицательное значение отступа**. 

### График отступов для объектов относительно KNN ###

<img src="https://user-images.githubusercontent.com/71149650/96899557-ba1dd080-1499-11eb-93ab-2ecaa4d23d75.png" width = "550" />

### Выборка ирисов фишера без шумовых объектов ###

<img src="https://user-images.githubusercontent.com/71149650/96902300-06b6db00-149d-11eb-8432-40b27d7d83c6.png"  width = "550"/>

**Алгоритм STOLP** разбивает обучающие объеты выборки на три категории: шумовые, неинформативные и эталонные. Таким образом, алгоритм производит сжатие выборки.

### Реализация алгоритма STOLP ###
``` r
# Параметры: выборка, порог фильтрации выборки, допустимая доля ошибок, метрический алгоритм и функция отступа
stolpF <- function(xl, delta, miss, algK = KNN, algM = marginF ) {
  k <- 7
  marg <- matrix(0, dim(xl)[1], 1)
  
  # Считаем отступы для каждого элемента выборки
  for (i in 1:dim(xl)[1]) {
  
    marg[i] <- marginF(xl[i,], xl[-i,], k)
    
  }
    
  # Выбрасываем с выборки шумовые объекты 
  xd <- xl[which(marg > delta),]
  marg <- marg[which(marg > delta)]
  
  omega <- data.frame()
  omegaT <- c()
  
  # Выделяем один эталонные объект для каждого класса
  for (i in levels(xd[, 3])) {
    class <- which(xd[, 3] == i)
    omega <- rbind(omega, xd[class[which.max(marg[class])],])
    omegaT <- c(omegaT, class[which.max(marg[class])])
    
  }
  # Удаляем с выборки эталоннеы объекты
  xd <- xd[-omegaT,] 
  l <- dim(xd)[1]
  
  
  while (dim(xl)[1] > dim(omega)[1]) {
    rownames(xd) <- c(1:dim(xd)[1])
    
    # Пресчитываем k
    k <- LOO(xd, omega)
    
    marg <- matrix(0, dim(xd)[1], 1)
    err <- 0
    
    # Пересчитываем отступы, в качестве обуч. выборки - эталонные объекты 
    for (i in 1:dim(xd)[1]) {
      
      marg[i] <- marginF(xd[i,], omega, k)
      
      point <- c(xd[i,1], xd[i,2])
      
      if (algK(point, omega, k) != xd[i, 3]) {
        err <- err + 1 
      }
      
    } # for
    
    # Определяем объекты, на которых алгоритм ошибается   
    E <- xd[which(marg < 0),]
    marg <- marg[which(marg < 0)]
    
    if (dim(E)[1] < miss ) {
      break
    }

    # Добавляем в эталонные объект с минимальным отступом
    omega <- rbind(omega, E[which.min(marg),])
    omegaT <- as.numeric(rownames(E[which.min(marg),]))
   
    xd <- xd[-omegaT, ]
    
    print(omega)   
    
  }# while
  
  return(omega)
  
}
```

### Эталонные объекты выборки ирисов фишера ###

<img src="https://user-images.githubusercontent.com/71149650/96968310-82566d80-1519-11eb-9621-5e9371749672.png" />

### Карта классификации ирисов фишера по эталонным объектам ###

<img src = "https://user-images.githubusercontent.com/71149650/96970068-cf3b4380-151b-11eb-90b7-fbf9a11437f5.png" width = "550" />

<table>
    <tr>
        <td>Метрический метод</td>
        <td>Параметры</td>
        <td>Значение LOO</td>
        <td>Время работы / сек </td>
    </tr>
    <tr>
        <td>KNN</td>
        <td> k = 6 </td>
        <td>0.034</td>
        <td>10.088</td>
    </tr>
    <tr>
        <td>KNN + STOLP</td>
        <td>k = 1</td>
        <td>0.034</td>
        <td>0.308</td>
    </tr>
 </table>
 
[Оглавление](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BF%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8)
 

## Скользящий контроль LOO ##

Данная процедура помогает найти оптимальное значение параметров **k** (KNN, KWNN) и **q** (весовая функция для KWNN). Для каждого объекта из выборки проверяется, правильно ли он классифицируется по своим k ближайшим соседям (k = [1; l], q = (0; 1)).

### Реализация скользящего контроля LOO для [KNN](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-k-%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B8%D1%85-%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B5%D0%B9-knn) ###
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

<img src="https://user-images.githubusercontent.com/71149650/94484221-bf626500-01e4-11eb-87f9-24c663d2970b.png" alt="LOO для полной выборки" />

### Реализация скользящего контроля LOO для [KWNN](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-k-%D0%B2%D0%B7%D0%B2%D0%B5%D1%88%D0%B5%D0%BD%D0%BD%D1%8B%D1%85-%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B8%D1%85-%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B5%D0%B9-kwnn) ###
В данной функции идет поиск оптимального **k** и **q** одновременно.

``` r
# Параметры: выборка и метод, который используют для классификации
LOO <- function(xl, alg = KWNN ){
    l <- dim(xl)[1]

    # Определяем рамки для поиска значения весовой функции
    r1 <- 0.02
    r2 <- 0.98
  
    # Инициализация матрицы для LOO
    MLOO <- matrix(0, l - 1, r2 / r1 )

    for (i in 1:l) {
    
        # Берем точку из выборки и переопределяем выборку 
        point <- c(xl[i,1:2])
        new_iris <- xl[-i,]
        
        # Сортируем выборку
        ordXl <- sortByDist( point, new_iris )
        
        # Цикл по k
        for (k in 1:(l - 1)) {
            j <- 1
              
            # Цикл по q
            for (q in seq(r1, r2, r1 ) ) {
            
                # Сравнение работы алогоритма и класса исключенной точки
                if ( alg(point,  ordXl, k, q ) != xl[i, 3] ) {
                    
                    # Если алгоритм ошибся, то штрафуем его
                    MLOO[k,j] <-  MLOO[k,j] + 1/l
                      
                }
                    
                j <- j + 1 
                   
            }

        }
    
    }
    
    # Находим минимальное значение в матрице(столбец и строку минимального значения)
    q <- which.min(MLOO) %/% (dim(xl)[1] - 1) + 1
    k <- which.min(MLOO) %% (dim(xl)[1] - 1)
   
    # Возвращаем найденные оптимальные параметры 
    return(c(q, k))
}  
```

<img src="https://user-images.githubusercontent.com/71149650/95449907-1945f600-096e-11eb-9310-ea6a8523d001.png" width = "800"/>

### Реализация скользящего контроля LOO для [парзеновское окно](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-%D0%BF%D0%B0%D1%80%D0%B7%D0%B5%D0%BD%D0%BE%D0%B2%D1%81%D0%BA%D0%BE%D0%B3%D0%BE-%D0%BE%D0%BA%D0%BD%D0%B0) ###
В данной функции идет поиск оптимального **h** - радиус окна.

``` r
# Параметры: выборка и метод, который используют для классификации
LOO <- function(xl, alg = Parz){
    l <- dim(xl)[1]
    
    # Инциализация вектора для перебора радиуса окна и вектора для LOO
    h <- seq(0.1, 2, 0.1)
    MLOO <- matrix(0, length(h), 1)
  
    for (i in 1:l) {
    
        # Берем точку из выборки и переопределяем выборку с расстояниями
        point <- c(xl[i, 1:2])        
        dist <- Dist(point, xl[-i,])
        
        for (j in 1:length(h) ) {
        
            # Сравнение работы алогоритма и класса исключенной точки
            if (alg(point, dist, h[j]) != xl[i, 3]) {
            
                # Если алгоритм ошибся, то штрафуем его
                MLOO[j][1] <-  MLOO[j][1] + 1 / l
            }  
          
        }
        
    }
    
    # Выбираем оптимальный h и возваращаем его
    min <- which.min(MLOO)
    return(min)  
}
```
Все графики LOO для парзеновского окна находятся в разделе данного [метода](https://github.com/deviati0n/ML1/blob/master/README.md#%D1%80%D0%B5%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F-%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D0%B0-%D0%BF%D0%B0%D1%80%D0%B7%D0%B5%D0%BD%D0%BE%D0%B2%D1%81%D0%BA%D0%BE%D0%B3%D0%BE-%D0%BE%D0%BA%D0%BD%D0%B0)


## Сравнительная таблица ##

<table>
    <tr>
        <td>Метрический метод</td>
        <td>Параметры</td>
        <td>Значение LOO</td>
    </tr>
    <tr>
        <td>1NN</td>
        <td> - </td>
        <td>0.046</td>
    </tr>
    <tr>
        <td>KNN</td>
        <td>k = 6</td>
        <td>0.034</td>
    </tr>
    <tr>
        <td>KWNN</td>
        <td>k = 30, q = 0.96</td>
        <td>0.034</td>
    </tr>
     <tr>
        <td rowspan = "5">Парзеновское окно</td>
        <td>Прямоугольное ядро, h = 0.4</td>
        <td>0.04</td>
    </tr>
    <tr>
        <td>Треугольное ядро, h = 0.4</td>
        <td>0.04</td>
    </tr>
    <tr>
        <td>Ядро Епанечникова, h = 0.4</td>
        <td>0.04</td>
    </tr>
    <tr>
        <td>Квартическое ядро, h = 0.4</td>
        <td>0.04</td>
    </tr>
    <tr>
        <td>Гауссовское ядро, h = 0.1</td>
        <td>0.04</td>
    </tr>
  <td rowspan = "5">Потенциальные функции</td>
        <td>Прямоугольное ядро, h = 0.4</td>
        <td>0.067</td>
    </tr>
    <tr>
        <td>Треугольное ядро, h = 0.4</td>
        <td>0.07</td>
    </tr>
    <tr>
        <td>Ядро Епанечникова, h = 0.4</td>
        <td>0.07</td>
    </tr>
    <tr>
        <td>Квартическое ядро, h = 0.4</td>
        <td>0.07</td>
    </tr>
    <tr>
        <td>Гауссовское ядро, h = 0.1</td>
        <td>0.04</td>
    </tr>
    <tr>
        <td>KNN  + STOLP</td>
        <td>k = 1</td>
        <td>0.034</td>
    </tr>
    
</table>

Анализируя данные в таблице, прийдем к выводу, что самым опимальным алгоритмом для ирисов фишер является **KNN** с параметром k = 6.

[Оглавление](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BF%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8)

# Байесовские методы классификации #

## Линии уровня нормального распределения ##
**Линией уровня** функции двух переменных называется линия (множество точек) на координатной плоскости, в которых функция принимает одинаковые значения. Перед нами стоит задача построить линии уровня нормального распределения, когда даны *ковариационная матрица нормального распределения* и *мат ожидание*. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{N}(x;&space;\mu;&space;\Sigma&space)&space;=&space;\frac{1}{\sqrt{(2\pi&space;)^{n}&space;\left&space;|&space;\Sigma&space;\right&space;|}}&space;\exp&space;\left(&space;-\frac{1}{2}(x-\mu&space;)^{T}\Sigma&space;^{-1}(x-\mu&space;)&space;\right&space;),&space;x&space;\in&space;\mathbb{R}&space;^&space;{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{N}(x;&space;\mu&space;,&space;\Sigma&space;)&space;=&space;\frac{1}{\sqrt{(2\pi&space;)^{n}&space;\left&space;|&space;\Sigma&space;\right&space;|}}&space;\exp&space;\left(&space;-\frac{1}{2}(x-\mu&space;)^{T}\Sigma&space;^{-1}(x-\mu&space;)&space;\right&space;),&space;x&space;\in&space;\mathbb{R}&space;^&space;{n}" title="\mathcal{N}(x; \mu , \Sigma ) = \frac{1}{\sqrt{(2\pi )^{n} \left | \Sigma \right |}} \exp \left( -\frac{1}{2}(x-\mu )^{T}\Sigma ^{-1}(x-\mu ) \right ), x \in \mathbb{R} ^ {n}" /></a> - **n-мерное нормальное (гауссовское) распределение** с мат ожиданием (центром) <a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;\in&space;\mathbb{R}&space;^&space;{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;\in&space;\mathbb{R}&space;^&space;{n}" title="\mu \in \mathbb{R} ^ {n}" /></a> и ковариационной матрицей <a href="https://www.codecogs.com/eqnedit.php?latex=\Sigma&space;\in&space;\mathbb{R}&space;^&space;{n&space;\times&space;n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Sigma&space;\in&space;\mathbb{R}&space;^&space;{n&space;\times&space;n}" title="\Sigma \in \mathbb{R} ^ {n \times n}" /></a>.

**Рассмотрим особые случаи:**

1) Если матрица ковариации **пропорциональна единичной**, то все компоненты нормального распределения являются независимыми друг от друга и имеют одинаковую дисперсию. Линии уровня образуют **окружности**.

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;=(0,&space;0),&space;\Sigma&space;=&space;\begin{pmatrix}&space;1&space;&&space;0&space;\\&space;0&space;&&space;1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;=(0,&space;0),&space;\Sigma&space;=&space;\begin{pmatrix}&space;1&space;&&space;0&space;\\&space;0&space;&&space;1&space;\end{pmatrix}" title="\mu =(0, 0), \Sigma = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}" /></a>

<img src="https://user-images.githubusercontent.com/71149650/97670904-4894e200-1a98-11eb-91c7-a22642836927.png" width="550"/>

2) **Диагональная** матрица ковариации соответствует независимым компонентам, но с различными дисперсиями. В этом случае линии уровня - **эллипсы с осями, параллельными координатным осям**.

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;=(0,&space;0),&space;\Sigma&space;=&space;\begin{pmatrix}&space;3&space;&&space;0&space;\\&space;0&space;&&space;1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;=(0,&space;0),&space;\Sigma&space;=&space;\begin{pmatrix}&space;3&space;&&space;0&space;\\&space;0&space;&&space;1&space;\end{pmatrix}" title="\mu =(0, 0), \Sigma = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}" /></a>

<img src="https://user-images.githubusercontent.com/71149650/97671019-80038e80-1a98-11eb-9701-666706bf9751.png" width="550"/>

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;=(0,&space;0),&space;\Sigma&space;=&space;\begin{pmatrix}&space;1&space;&&space;0&space;\\&space;0&space;&&space;3&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;=(0,&space;0),&space;\Sigma&space;=&space;\begin{pmatrix}&space;1&space;&&space;0&space;\\&space;0&space;&&space;3&space;\end{pmatrix}" title="\mu =(0, 0), \Sigma = \begin{pmatrix} 1 & 0 \\ 0 & 3 \end{pmatrix}" /></a>

<img src="https://user-images.githubusercontent.com/71149650/97671044-90b40480-1a98-11eb-9f7e-067fec78c3c4.png" width="550"/>                                                                                                                            

3) Если признаки коррелированы, то матрица ковариации **не диагональна**. В этом случае линии уровня - **эллипсы, оси которых повернуты**.

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;=(0,&space;0),&space;\Sigma&space;=&space;\begin{pmatrix}&space;4&space;&&space;2&space;\\&space;2&space;&&space;2&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;=(0,&space;0),&space;\Sigma&space;=&space;\begin{pmatrix}&space;4&space;&&space;2&space;\\&space;2&space;&&space;2&space;\end{pmatrix}" title="\mu =(0, 0), \Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 2 \end{pmatrix}" /></a>

<img src="https://user-images.githubusercontent.com/71149650/97671130-b93bfe80-1a98-11eb-8d79-7459bd432cb0.png" width="550"/>  

[Оглавление](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BF%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8)


## Наивный нормальный байесовский классификатор ##
Данный метод основывается на том, что объекты описываются **n** статистически независимыми признаками. Это предположение существенно облегчает задачу, так как оценить **n** одномерных плотностей легче, чем одну **n**-мерную плотность.
 
**Оценка априорных вероятностей класса** имеет вид:
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{P}_{y}&space;=&space;\frac{\left&space;|&space;X_{y}^{l}\right&space;|}{l}&space;\qquad&space;y&space;\in&space;Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{P}_{y}&space;=&space;\frac{\left&space;|&space;X_{y}^{l}\right&space;|}{l}&space;\qquad&space;y&space;\in&space;Y" title="\hat{P}_{y} = \frac{\left | X_{y}^{l}\right |}{l} \qquad y \in Y" /></a> 

**Эмпирическая оценка плотности распределения** имеет вид:

<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{{p}_{yj}}(\xi&space;)&space;=&space;\frac{1}{\sigma&space;_{yj}\sqrt{2\pi}}&space;\exp\left&space;(&space;-\frac{(\xi&space;-\mu&space;_{yj})^{2}}{2\sigma&space;_{yj}^{2}}&space;\right&space;),&space;\qquad&space;y&space;\in&space;Y,&space;\qquad&space;j&space;=&space;\overline{1,n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{{p}_{yj}}(\xi&space;)&space;=&space;\frac{1}{\sigma&space;_{yj}\sqrt{2\pi}}&space;\exp\left&space;(&space;-\frac{(\xi&space;-\mu&space;_{yj})^{2}}{2\sigma&space;_{yj}^{2}}&space;\right&space;),&space;\qquad&space;y&space;\in&space;Y,&space;\qquad&space;j&space;=&space;\overline{1,n}" title="\widehat{{p}_{yj}}(\xi ) = \frac{1}{\sigma _{yj}\sqrt{2\pi}} \exp\left ( -\frac{(\xi -\mu _{yj})^{2}}{2\sigma _{yj}^{2}} \right ), \qquad y \in Y, \qquad j = \overline{1,n}" /></a>

Подставим описанные выше оценки и получим **НБК**:

<a href="https://www.codecogs.com/eqnedit.php?latex=a(x)&space;=&space;\arg&space;\max_{y\in&space;Y}&space;\left&space;(&space;\ln&space;\lambda_{y}&space;\hat{P}_{y}&space;&plus;&space;\sum_{j&space;=&space;1}^{n}&space;\ln&space;\hat{p}_{yj}&space;(\xi&space;_{j})\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a(x)&space;=&space;\arg&space;\max_{y\in&space;Y}&space;\left&space;(&space;\ln&space;\lambda_{y}&space;\hat{P}_{y}&space;&plus;&space;\sum_{j&space;=&space;1}^{n}&space;\ln&space;\hat{p}_{yj}&space;(\xi&space;_{j})\right&space;)" title="a(x) = \arg \max_{y\in Y} \left ( \ln \lambda_{y} \hat{P}_{y} + \sum_{j = 1}^{n} \ln \hat{p}_{yj} (\xi _{j})\right )" /></a>



### Реализация алгоритма НБК ###
``` r

# Оценка mu и sigma
for (i in 1:m) {
  for (j in 1:n) {
    temp <- xl[xl[,3] == classes[i],][,j] 
    
    mu[i, j] <- mean(temp)
    sigma[i, j] <- sqrt(var(temp))

  }
}


# Параметры: входная точка, априорные вероятности классов, матрица ковариации, мат ожидание и величина потери для каждого класса
naiveBC <- function(x, Py, sigma, mu, l = c(1, 1, 1)){
  
  Ver <- rep(0, dim(Py))
  
  for (i in 1:dim(Py)) {
      sum <- 0
      
      for (j in 1:length(x)) {

        # Считаем вторую часть формулы (сумму)
        sum <- sum + log(plotn(x[j], mu[i, j], sigma[i, j]))

      }
      
    # Находим плотность классов в заданной точке
    Ver[i] <- log(l[i] * Py[i]) + sum
  }
  
  # Возвращаем класс с максимальной апостериорной вероятностью 
  return( c(which.max(Ver), max(Ver)) )
  
}
```

### Работа НБК на ирисах Фишера с различными "лямбда" ###


| ![](https://user-images.githubusercontent.com/71149650/99665352-96d74880-2a7a-11eb-9764-f2d76cd446bf.png) | ![](https://user-images.githubusercontent.com/71149650/99666659-511b7f80-2a7c-11eb-9edd-ac0ee444d372.png) |
| - | - |
| λ = (1, 1, 1) | λ = (39, 1, 1) |
| ![](https://user-images.githubusercontent.com/71149650/99666774-7e682d80-2a7c-11eb-990d-47253cbe2939.png) | ![](https://user-images.githubusercontent.com/71149650/99666870-9f308300-2a7c-11eb-97c2-fbd384dde914.png) |
| λ = (1, 43, 1) | λ = (1, 1, 51) |



## Подстановочный алгоритм (plug-in) ##

**Plug-in** - это еще один вариант байесовского алгоритма классификации. В данном методе требуется восстанавливать параметры нормального распределения для каждого класса и подставлять восстановленные плотности в формулу байесовского классификатора. 

Параметры нормального распределения оцениваются согласно формулам:

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;_{y}&space;=&space;\frac{1}{l_{y}}\sum_{x_{i}:y_{i}=y}x_{i}&space;\qquad&space;\Sigma_{y}&space;=&space;\frac{1}{l_{y}&space;-&space;1}\sum_{x_{i}:y_{i}=y}(x_{i}&space;-&space;\mu&space;_{y})(x_{i}&space;-&space;\mu&space;_{y})^{T}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;_{y}&space;=&space;\frac{1}{l_{y}}\sum_{x_{i}:y_{i}=y}x_{i}&space;\qquad&space;\Sigma_{y}&space;=&space;\frac{1}{l_{y}&space;-&space;1}\sum_{x_{i}:y_{i}=y}(x_{i}&space;-&space;\mu&space;_{y})(x_{i}&space;-&space;\mu&space;_{y})^{T}" title="\mu _{y} = \frac{1}{l_{y}}\sum_{x_{i}:y_{i}=y}x_{i} \qquad \Sigma_{y} = \frac{1}{l_{y} - 1}\sum_{x_{i}:y_{i}=y}(x_{i} - \mu _{y})(x_{i} - \mu _{y})^{T}" /></a>



``` r

Plugin <- function(x, Py, mu1, sigma1, mu2, sigma2, l = c(3, 1)){
  
  Ver <- rep(0, 2)
  mu <- rbind(mu1, mu2)
  
  for (i in 1:2) {
    
    Ver[i] <- Py[i] * l[i]
    
    if (i == 1) {
      sigma <- sigma1
    }
    else{
      sigma <- sigma2
    }
    
    Ver[i] <- Ver[i] * (-(t(x - mu[i,]) %*% solve(sigma) %*% (x - mu[i, ])) / 2 - log(det(sigma))/2 )
    
  }
  
  return(which.max(Ver))
}

```

### Пример работы подстановочного алгоритма ###

| ![](https://user-images.githubusercontent.com/71149650/99770153-76f66200-2b18-11eb-9c9f-20a3a1d37aaa.png) | ![](https://user-images.githubusercontent.com/71149650/99770106-634afb80-2b18-11eb-81a9-9af870475ee0.png) |
| - | - |
| λ = (3, 1) | λ = (1, 3) |

 | <img src = "https://user-images.githubusercontent.com/71149650/99770043-50382b80-2b18-11eb-9aec-2ad2bd2cc924.png" width = "700"/> |
 | - |
 | λ = (1, 1) |


### Возможные варианты разделяющей кривой ###

+ **Эллипс**

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;_{1}&space;=&space;(0,&space;2)&space;\qquad&space;\Sigma_{1}&space;=&space;\begin{pmatrix}&space;1&space;&&space;0&space;\\&space;0&space;&&space;8&space;\end{pmatrix}&space;\qquad&space;\mu&space;_{2}&space;=&space;(3,&space;2)&space;\qquad&space;\Sigma_{2}&space;=&space;\begin{pmatrix}&space;0.1&space;&&space;0&space;\\&space;0&space;&&space;0.1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;_{1}&space;=&space;(0,&space;2)&space;\qquad&space;\Sigma_{1}&space;=&space;\begin{pmatrix}&space;1&space;&&space;0&space;\\&space;0&space;&&space;8&space;\end{pmatrix}&space;\qquad&space;\mu&space;_{2}&space;=&space;(3,&space;2)&space;\qquad&space;\Sigma_{2}&space;=&space;\begin{pmatrix}&space;0.1&space;&&space;0&space;\\&space;0&space;&&space;0.1&space;\end{pmatrix}" title="\mu _{1} = (0, 2) \qquad \Sigma_{1} = \begin{pmatrix} 1 & 0 \\ 0 & 8 \end{pmatrix} \qquad \mu _{2} = (3, 2) \qquad \Sigma_{2} = \begin{pmatrix} 0.1 & 0 \\ 0 & 0.1 \end{pmatrix}" /></a>

<img src="https://user-images.githubusercontent.com/71149650/98244594-bbaec480-1f80-11eb-836d-49885a604aa0.png" width="550"/>  

+ **Парабола** 

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;_{1}&space;=&space;(10,&space;0)&space;\qquad&space;\Sigma_{1}&space;=&space;\begin{pmatrix}&space;10&space;&&space;0&space;\\&space;0&space;&&space;1&space;\end{pmatrix}&space;\qquad&space;\mu&space;_{2}&space;=&space;(3,&space;2)&space;\qquad&space;\Sigma_{2}&space;=&space;\begin{pmatrix}&space;1&space;&&space;0&space;\\&space;0&space;&&space;1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;_{1}&space;=&space;(10,&space;0)&space;\qquad&space;\Sigma_{1}&space;=&space;\begin{pmatrix}&space;10&space;&&space;0&space;\\&space;0&space;&&space;1&space;\end{pmatrix}&space;\qquad&space;\mu&space;_{2}&space;=&space;(3,&space;2)&space;\qquad&space;\Sigma_{2}&space;=&space;\begin{pmatrix}&space;1&space;&&space;0&space;\\&space;0&space;&&space;1&space;\end{pmatrix}" title="\mu _{1} = (10, 0) \qquad \Sigma_{1} = \begin{pmatrix} 10 & 0 \\ 0 & 1 \end{pmatrix} \qquad \mu _{2} = (3, 2) \qquad \Sigma_{2} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}" /></a>

<img src="https://user-images.githubusercontent.com/71149650/98244660-d719cf80-1f80-11eb-8e0c-35b532cee121.png" width="550"/>  

+ **Гипербола** 

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;_{1}&space;=&space;(1,&space;2)&space;\qquad&space;\Sigma_{1}&space;=&space;\begin{pmatrix}&space;3&space;&&space;0&space;\\&space;0&space;&&space;0.3&space;\end{pmatrix}&space;\qquad&space;\mu&space;_{2}&space;=&space;(5,&space;1)&space;\qquad&space;\Sigma_{2}&space;=&space;\begin{pmatrix}&space;1&space;&&space;0&space;\\&space;0&space;&&space;8&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;_{1}&space;=&space;(1,&space;2)&space;\qquad&space;\Sigma_{1}&space;=&space;\begin{pmatrix}&space;3&space;&&space;0&space;\\&space;0&space;&&space;0.3&space;\end{pmatrix}&space;\qquad&space;\mu&space;_{2}&space;=&space;(5,&space;1)&space;\qquad&space;\Sigma_{2}&space;=&space;\begin{pmatrix}&space;1&space;&&space;0&space;\\&space;0&space;&&space;8&space;\end{pmatrix}" title="\mu _{1} = (1, 2) \qquad \Sigma_{1} = \begin{pmatrix} 3 & 0 \\ 0 & 0.3 \end{pmatrix} \qquad \mu _{2} = (5, 1) \qquad \Sigma_{2} = \begin{pmatrix} 1 & 0 \\ 0 & 8 \end{pmatrix}" /></a>

<img src="https://user-images.githubusercontent.com/71149650/98244981-53acae00-1f81-11eb-8c2b-a0db240c4a85.png" width="550"/>  

## Линейный дискриминант Фишера ###

В основе данного метода лежит гипотеза, что матрицы ковариаций классов **равны**. Данная эвристика позволяет повысить устойчивость ковариационной матрицы и упростить алгоритим обучения. 

Принятая гипотеза позволяет вычислить некоторую "среднюю" матрицу ковариации, используя всю выборку:

<a href="https://www.codecogs.com/eqnedit.php?latex=\Sigma&space;=&space;\frac{1}{l&space;-&space;|Y|}&space;\sum_{i=1}^{l}(x_{i}&space;-&space;\hat{\mu}&space;_{y_{i}})(x_{i}&space;-&space;\hat{\mu}&space;_{y_{i}})^{T}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Sigma&space;=&space;\frac{1}{l&space;-&space;|Y|}&space;\sum_{i=1}^{l}(x_{i}&space;-&space;\hat{\mu}&space;_{y_{i}})(x_{i}&space;-&space;\hat{\mu}&space;_{y_{i}})^{T}" title="\Sigma = \frac{1}{l - |Y|} \sum_{i=1}^{l}(x_{i} - \hat{\mu} _{y_{i}})(x_{i} - \hat{\mu} _{y_{i}})^{T}" /></a>
 
 где <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat{\mu}_{y_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{\mu}_{y_{i}}" title="\hat{\mu}_{y_{i}}" /></a> - векторы средних значение классов, *l* - кол-во объектов в обучающей выборке, |*Y*| - кол-во классов в обучающей выборке. 
 
 
 Для двухклассовой задачи ЛДФ **разделяющая гиперплоскость** имеет вид: <a href="https://www.codecogs.com/eqnedit.php?latex=\vec{w}&space;\cdot&space;\vec{x}&space;=&space;c" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\vec{w}&space;\cdot&space;\vec{x}&space;=&space;c" title="\vec{w} \cdot \vec{x} = c" /></a> 
 
 где  <a href="https://www.codecogs.com/eqnedit.php?latex=\overrightarrow{w}&space;=&space;{\Sigma}^{-1}&space;(\mu_{1}&space;-&space;\mu_{2})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\overrightarrow{w}&space;=&space;{\Sigma}^{-1}&space;(\mu_{1}&space;-&space;\mu_{2})" title="\overrightarrow{w} = {\Sigma}^{-1} (\mu_{1} - \mu_{2})" /></a>,
 
 <a href="https://www.codecogs.com/eqnedit.php?latex=c&space;=&space;\frac{1}{2}(\mu_{1}&space;&plus;&space;\mu_{2})&space;\cdot&space;\vec{w}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c&space;=&space;\frac{1}{2}(\mu_{1}&space;&plus;&space;\mu_{2})&space;\cdot&space;\vec{w}" title="c = \frac{1}{2}(\mu_{1} + \mu_{2}) \cdot \vec{w}" /></a>
 
### Программная реализация ЛДФ ###

``` r
LDF <- function(Py, x, objects1, objects2, mu1, mu2, l = c(1, 1)){
  
  mu <- rbind(mu1, mu2) 
    
  Ver <- rep(0, 2)


  sigma <- ldfSIGMA(objects1, objects2, mu1, mu2)
  
  for (i in 1:2) {
    
    Ver[i] <- Py[i] * l[i]
    Ver[i] <- Ver[i] * ( t(x) %*% solve(sigma) %*% mu[i, ] + (-1/2) * t(mu[i,]) %*% solve(sigma) %*% mu[i,] + log2(pi) )
    
  }

  return(which.max(Ver))
  
}
...

# Значения для разделяющей прямой
sigma <- ldfSIGMA(objectsOfFirstClass, objectsOfSecondClass, mu1, mu2)
w <- solve(sigma) %*% t(mu1 - mu2)
c <- ((mu1 + mu2) / 2) %*% w

```
### Пример работы ЛДФ ###

| ![](https://user-images.githubusercontent.com/71149650/99768880-9096aa00-2b16-11eb-8c20-c13a8e1c699d.png) | ![](https://user-images.githubusercontent.com/71149650/99768817-7b218000-2b16-11eb-9b13-b5a95c2815cd.png) |
| - | - |
| λ = (5, 1) | λ = (1, 3) |

 | <img src = "https://user-images.githubusercontent.com/71149650/99767248-ca19e600-2b13-11eb-846c-577e0afca38a.png" width = "700"/> |
 | - |
 | λ = (1, 1) |
 
 ### ЛДФ и Plug-in ###
 
  Если длина выборки будет меньше чем размерность пространства, то *sigma* для каждого класса в **Plug-in** будут вырождены, а **ЛДФ** решает эту проблему. Еще одним плюсом **ЛДФ** есть то, что в нем не надо считатать квадраты, а следовательно будет потрачено меньше ресурсов. В то же время **Plug-in** более универсален, ведь именно за счет тех самых квадратов, он может огибать точки классов, которые **ЛДФ** мог бы просто откинуть. 
 
 Для примера, два класса имеют одинаковые параметры. Видно, что **Plug-in** не смог отрисовать карту классификации, т.к. *sigma* для каждого класса - вырождена. 
 | ![](https://user-images.githubusercontent.com/71149650/99775944-2fc09f00-2b21-11eb-9745-ca342976478d.png) | ![](https://user-images.githubusercontent.com/71149650/99776019-46ff8c80-2b21-11eb-8212-1ff0aea19454.png) |
| - | - |


# Линейные алгоритмы классификации #
Линейные модели широко используются в машинном обучении благодаря их относительной простоте, в некоторых случаях хорошей интерпретируемости и наличию глубоко проработанных численных методов.

Рассмотрим задачу классификации с двумя классами, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Y&space;=&space;\left&space;\{&space;-1,&space;&plus;1&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y&space;=&space;\left&space;\{&space;-1,&space;&plus;1&space;\right&space;\}" title="Y = \left \{ -1, +1 \right \}" /></a>.

Линейный классификатор имеет вид:

<a href="https://www.codecogs.com/eqnedit.php?latex=a(x)&space;=\mathrm{sign}&space;\left&space;(&space;\left&space;\langle&space;w,&space;x&space;\right&space;\rangle&space;-&space;w_{0}&space;\right&space;)&space;=&space;\mathrm{sign}&space;\left&space;(&space;\sum_{j=1}^{n}&space;w_{j}&space;f_{j}(x)&space;-&space;w_{0}&space;\right&space;)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?a(x)&space;=\mathrm{sign}&space;\left&space;(&space;\left&space;\langle&space;w,&space;x&space;\right&space;\rangle&space;-&space;w_{0}&space;\right&space;)&space;=&space;\mathrm{sign}&space;\left&space;(&space;\sum_{j=1}^{n}&space;w_{j}&space;f_{j}(x)&space;-&space;w_{0}&space;\right&space;)." title="a(x) =\mathrm{sign} \left ( \left \langle w, x \right \rangle - w_{0} \right ) = \mathrm{sign} \left ( \sum_{j=1}^{n} w_{j} f_{j}(x) - w_{0} \right )." /></a>

Величина <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;M_{i}(w)&space;=&space;y_{i}f(x_{i},&space;w)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;M_{i}(w)&space;=&space;y_{i}f(x_{i},&space;w)" title="M_{i}(w) = y_{i}f(x_{i}, w)" /></a> - **отступ** объекта <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x_{i}" title="x_{i}" /></a>, относительно алгоримтма линейной классификации.  Если величина отрицательная, то алгортм допускает ошибку на этом объекте.

## Метод стохастического градиента ##

**Градиентные методы** - это широкий класс оптимизационных алгоритмов, которые используются не только в машинном обучении. В данном случае метод стохастического градиента будет использоваться для нахождения синаптических весов **w**, при которых будет достигаться минимум апроксимированного эмпирического риска:

<img src = "https://user-images.githubusercontent.com/71149650/100413373-38b4f180-3088-11eb-9096-75fd371bcebc.png" />

Для этого понадобится задать некоторое начальное приближение для ***w***, затем запустить итерационный процесс, где на каждом шаге вектор ***w*** будет изменяться в направлении наиболее быстрого убывания функционала ***Q*** - направление антиградиента. 

<img src = "https://user-images.githubusercontent.com/71149650/100415208-ad8a2a80-308c-11eb-9ef5-5f372251703d.png" width = "300" />

Для того, чтобы улучшить качество градиентных методов использовались такие эвристики:
+ Нормализация данных.

<a href="https://www.codecogs.com/eqnedit.php?latex=x^{j}&space;=&space;\frac{x^{j}&space;-&space;x^{j}_{min}}{x^{j}_{max}&space;-&space;x^{j}_{min}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{j}&space;=&space;\frac{x^{j}&space;-&space;x^{j}_{min}}{x^{j}_{max}&space;-&space;x^{j}_{min}}" title="x^{j} = \frac{x^{j} - x^{j}_{min}}{x^{j}_{max} - x^{j}_{min}}" /></a>

+ Выбор объектов выборки в случайном порядке, чтобы итерационный процесс не зациклился или не разошелся. 

Ниже представлены классические частные случаи.

### ADALINE ###

Рассмотрим случай, когда функция потерь квадратична:

<img src = "https://user-images.githubusercontent.com/71149650/100416878-9baa8680-3090-11eb-95c2-0057a6ea688e.png" width = "175" />

Тогда правило обновления весов на каждом шаге будем иметь вид:

<a href="https://www.codecogs.com/eqnedit.php?latex=w&space;:=&space;w&space;=-&space;\eta&space;(&space;\left&space;\langle&space;w,&space;x_{i}&space;\right&space;\rangle&space;-&space;y_{i})x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w&space;:=&space;w&space;=-&space;\eta&space;(&space;\left&space;\langle&space;w,&space;x_{i}&space;\right&space;\rangle&space;-&space;y_{i})x_{i}" title="w := w =- \eta ( \left \langle w, x_{i} \right \rangle - y_{i})x_{i}" /></a>


#### Программная реализация ####

``` r
ADALINE <- function(xl, eta = 1, lambda = 1/100) {

  count <- 0
  w <- c(1/2, -1/2, 1/2)
  
  #Инициализация начальной оценки функционала Q
  Q <- 0
  for (i in 1:dim(xl)[1]) {
    skPr <- sum(w * xl[i,1:(dim(xl)[2] - 1)])
    mar <- skPr *  xl[i,dim(xl)[2]]
    
    Q <- Q + LossFunc(mar)
    
  }
  
  ERR <- Q
  
  # тело цикла
  while (TRUE) { 
    count <- count + 1 
    
    # отступы для всех объектов выборки
    mars <- array(0, dim(xl)[1])
    for (i in 1:dim(xl)[1]) {
      
      skPr <- sum(w * xl[i,1:(dim(xl)[2] - 1) ])
      mars[i] <- skPr *  xl[i,dim(xl)[2]]
      
    }
    
    errI <- which(mars <= 0)

    if (length(errI) == 0 ) {
      break
    }
    
    else{
      
      #случайно выбираем объект с отрицательным отступом 
      randomI <- sample(errI, 1)
      
      skPr <- sum(w * xl[randomI,1:(dim(xl)[2] - 1) ])
      mar <- skPr *  xl[randomI, dim(xl)[2]]
      
      #ошибка
      err <- LossFunc(mar)
      
      eta <- 1 / count
      
      #обновление вектора весов
      w <- w - eta * (skPr - xl[randomI,dim(xl)[2]]) * xl[randomI,1:(dim(xl)[2] - 1) ]
      
      prevQ <- Q
      
      #оцениваем значение функционала
      Q <- (1 - lambda) * prevQ + lambda * err
      ERR <- rbind(ERR, Q)
      
      #критерии для остановки
      if ( abs(Q - prevQ) < 0.00001 ) {
        break
      }
      
      if (count > 1000) {        
        break        
      }
      
    }
    
  } 
  
  return(w)
}
```

<img src = "https://user-images.githubusercontent.com/71149650/100421392-5a6aa480-3099-11eb-8ccc-04845cd22db5.png" />

### Правило Хэбба ### 

При классификации <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a(x_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a(x_{i})" title="a(x_{i})" /></a> объекта <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x_{i}" title="x_{i}" /></a> возможны три случая:

+ Если ответ <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a(x_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a(x_{i})" title="a(x_{i})" /></a> совпадает с истинным ответом, то вектор весов изменять не надо

+ Если <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a(x_{i})&space;=&space;-1,&space;y_{i}&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a(x_{i})&space;=&space;-1,&space;y_{i}&space;=&space;1" title="a(x_{i}) = -1, y_{i} = 1" /></a>, то вектор весов увеличивается <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;w&space;=&space;w&space;&plus;&space;\eta&space;x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;w&space;=&space;w&space;&plus;&space;\eta&space;x_{i}" title="w = w + \eta x_{i}" /></a>

+ Если <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a(x_{i})&space;=&space;1,&space;y_{i}&space;=&space;-1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a(x_{i})&space;=&space;1,&space;y_{i}&space;=&space;-1" title="a(x_{i}) = 1, y_{i} = -1" /></a>, то вектор весов уменьшается <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;w&space;=&space;w&space;-&space;\eta&space;x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;w&space;=&space;w&space;-&space;\eta&space;x_{i}" title="w = w - \eta x_{i}" /></a>

Эти три случая объединяются в ***правило Хэбба***:

<a href="https://www.codecogs.com/eqnedit.php?latex=if&space;\quad&space;\left&space;\langle&space;w,x_{i}&space;\right&space;\rangle&space;y_{i}&space;<&space;0&space;\quad&space;then&space;\quad&space;w&space;=&space;w&space;&plus;&space;\eta&space;x_{i}y_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?if&space;\quad&space;\left&space;\langle&space;w,x_{i}&space;\right&space;\rangle&space;y_{i}&space;<&space;0&space;\quad&space;then&space;\quad&space;w&space;=&space;w&space;&plus;&space;\eta&space;x_{i}y_{i}" title="if \quad \left \langle w,x_{i} \right \rangle y_{i} < 0 \quad then \quad w = w + \eta x_{i}y_{i}" /></a>

При этом функция потерь имеет вид:

<img src = "https://user-images.githubusercontent.com/71149650/100417750-a534ee00-3092-11eb-9cb9-371bf65fde13.png" width = "175" />

#### Программная реализация ####
``` r
HEBB <- function(xl, eta = 1, lambda = 1/100) {
  count <- 0
  w <- c(1/2, -1/2, 1/2)
  
  #Инициализация начальной оценки функционала Q
  Q <- 0
  for (i in 1:dim(xl)[1]) {
    
    skPr <- sum(w * xl[i,1:(dim(xl)[2] - 1)])
    mar <- skPr *  xl[i,dim(xl)[2]]
    
    Q <- Q + LossFunc(mar)
    
  }
  
  ERR <- Q
  
  # тело цикла 
  while (TRUE) {
   
    count <- count + 1 
     
    # отступы для всех объектов выборки
    mars <- array(0, dim(xl)[1])
    for (i in 1:dim(xl)[1]) {
      
      skPr <- sum(w * xl[i,1:(dim(xl)[2] - 1) ])
      mars[i] <- skPr *  xl[i,dim(xl)[2]]
      
    }
    
    errI <- which(mars <= 0)
    
    if (length(errI) == 0 ) {
      break
    }
    
    else{
      
      #случайно выбираем объект с отрицательным отступом
      randomI <- sample(errI, 1)
      
      skPr <- sum(w * xl[randomI,1:(dim(xl)[2] - 1) ])
      mar <- skPr *  xl[randomI, dim(xl)[2]]
      

      #ошибка
      err <- LossFunc(mar)
      
      eta <- 1 / count
      
      #обновление вектора весов 
      w <- w + eta * xl[randomI,1:(dim(xl)[2] - 1) ] * xl[randomI,dim(xl)[2]]
      
      prevQ <- Q
      
      #оцениваем значение функционала 
      Q <- (1 - lambda) * prevQ + lambda * err
      ERR <- rbind(ERR, Q)
      
      #критерии для остановки
      if ( abs(Q - prevQ) < 0.00001 ) {
        break
      }
      
      if (count > 10000) {
        
        break
        
      }
      
    }
    
  } 
   
  return(w)
  
}

```
<img src = "https://user-images.githubusercontent.com/71149650/100421550-a3225d80-3099-11eb-90a0-0cc040907330.png" />


#### Сравнение ADALINE и пр. Хэбба ####

<img src = "https://user-images.githubusercontent.com/71149650/100420902-5c803380-3098-11eb-99dd-c770c7b5e262.png" />

### Логистическая регрессия ###

В логистической регресси для того, чтобы настроить вектор весов, решается задача минимизации эмпирического риска с функцией потерь специального вида:

<img src = "https://user-images.githubusercontent.com/71149650/101124285-c607c100-3607-11eb-9e6d-13d84e5450f2.png" width = "200" />

Правило обновления весов имеет вид:

<a href="https://www.codecogs.com/eqnedit.php?latex=w&space;=&space;w&space;&plus;&space;\eta&space;x_{i}y_{i}\sigma&space;(&space;-&space;\left&space;\langle&space;w,&space;x_{i}&space;\right&space;\rangle&space;y_{i})," target="_blank"><img src="https://latex.codecogs.com/gif.latex?w&space;=&space;w&space;&plus;&space;\eta&space;x_{i}y_{i}\sigma&space;(&space;-&space;\left&space;\langle&space;w,&space;x_{i}&space;\right&space;\rangle&space;y_{i})," title="w = w + \eta x_{i}y_{i}\sigma ( - \left \langle w, x_{i} \right \rangle y_{i})," /></a>
где <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\sigma&space;-" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\sigma&space;-" title="\sigma -" /></a> сигмоидная функция <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\sigma(z)&space;=&space;\frac{1}{1&space;&plus;&space;\exp(-z)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\sigma(z)&space;=&space;\frac{1}{1&space;&plus;&space;\exp(-z)}" title="\sigma(z) = \frac{1}{1 + \exp(-z)}" /></a>.

#### Программная реализация ####

``` r
LogReg <- function(xl, eta = 1, lambda = 1/100) {
  count <- 0
  w <- c(1/2, -1/2, 1/2)
  
  #Инициализация начальной оценки функционала Q
  Q <- 0
  for (i in 1:dim(xl)[1]) {
    
    skPr <- sum(w * xl[i,1:(dim(xl)[2] - 1)])
    mar <- skPr *  xl[i,dim(xl)[2]]
    
    Q <- Q + LossFunc(mar)
    
  }
  
  ERR <- Q
  
  # тело цикла 
  while (TRUE) {
    count <- count + 1 
    
    mars <- array(0, dim(xl)[1])
    
    # отступы для всех объектов выборки
    for (i in 1:dim(xl)[1]) {
      
      skPr <- sum(w * xl[i,1:(dim(xl)[2] - 1) ])
      mars[i] <- skPr *  xl[i,dim(xl)[2]]
      
    }
    
    errI <- which(mars <= 0)
    
    if (length(errI) == 0 ) {
      break
    }
    
    else{
      
      randomI <- sample(errI, 1)
      
      skPr <- sum(w * xl[randomI,1:(dim(xl)[2] - 1) ])
      mar <- skPr *  xl[randomI, dim(xl)[2]]
      
      
      #ошибка
      err <- LossFunc(mar)
      
      eta <- 1 / count
      
      #обновление вектора весов 
      w <- w + eta * xl[randomI,1:(dim(xl)[2] - 1) ] * xl[randomI,dim(xl)[2]] * SigmFunc(-mar)
      
      prevQ <- Q
      
      #оцениваем значение функционала 
      Q <- (1 - lambda) * prevQ + lambda * err
      ERR <- rbind(ERR, Q)
      
      #критерии для остановки
      if ( abs(Q - prevQ) < 0.00001 ) {
        break
      }
      
      if (count > 10000) {
        
        break
        
      }
      
    }
    
  } 
  
  return(w)  
}
```

<img src = "https://user-images.githubusercontent.com/71149650/101130523-aa56e780-3614-11eb-8075-1f1938f17535.png" />

**Апостериорную вероятность** принадлежности произвольного объект классу можно вычислить следующим образом:

<a href="https://www.codecogs.com/eqnedit.php?latex=\textup{P}(y|x)&space;=&space;\sigma&space;(\left&space;\langle&space;w,&space;x&space;\right&space;\rangle&space;y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{P}(y|x)&space;=&space;\sigma&space;(\left&space;\langle&space;w,&space;x&space;\right&space;\rangle&space;y)" title="\textup{P}(y|x) = \sigma (\left \langle w, x \right \rangle y)" /></a>

<img src = "https://user-images.githubusercontent.com/71149650/101129117-0e2be100-3612-11eb-994f-3e93a18a4a09.png" width = "700" />

#### Сравнение ADALINE, пр. Хэбба и логистической регрессии ####
<img src = "https://user-images.githubusercontent.com/71149650/101130706-03268000-3615-11eb-887b-7d31f6b5f5ef.png" width = "1100" />

## Метод опорных векторов (SVM) ##
 Метод опорных объектов в настоящее время считается одним из самых лучших методом классификации. Данный метод основывается на построении оптимальной разделяющей гиперплоскости. 
  
  ### Линейно разделимая выборка ###
  
  Если выборка **линейно разделима** и функционал числа ошибок принимает нулевое значение, тогда разделяющая гиперплоскоть не единственная. Для того, чтобы разделяющая гиперплоскость была оптимальной, она должна максимально далеко стоять от ближайших к ней точек обоих классов.

Для этого ширина полосы (разделитель классов) должна быть максимальной. Тогда получим задачу квадратичного программирования:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{cases}&space;\left&space;\langle&space;w,w&space;\right&space;\rangle&space;\rightarrow&space;\min;&space;\\&space;y_{i}(\left&space;\langle&space;w,&space;x_{i}&space;\right&space;\rangle&space;-&space;w_{0})&space;\geqslant&space;1,&space;&&space;i&space;=&space;\overline{1,l}.&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{cases}&space;\left&space;\langle&space;w,w&space;\right&space;\rangle&space;\rightarrow&space;\min;&space;\\&space;y_{i}(\left&space;\langle&space;w,&space;x_{i}&space;\right&space;\rangle&space;-&space;w_{0})&space;\geqslant&space;1,&space;&&space;i&space;=&space;\overline{1,l}.&space;\end{cases}" title="\begin{cases} \left \langle w,w \right \rangle \rightarrow \min; \\ y_{i}(\left \langle w, x_{i} \right \rangle - w_{0}) \geqslant 1, & i = \overline{1,l}. \end{cases}" /></a>
  
  Однако на практике линейно разделимы классы встречаются редко. 
  
  ### Линейно неразделимая выборка ###
  
   + Если выборка **линейно неразделима**, необходимо  позволить алгоритму допусакать ошибки на обучающих объектах, но при этом постараемся, чтобы их было меньше.  Для этого введем дополнительные переменные, которые будут характеризовать величину ошибки на объектах выборки. Получим обобщенную задачу:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{cases}&space;\frac{1}{2}\left&space;\langle&space;w,w&space;\right&space;\rangle&space;&plus;&space;C\sum_{i=1}^{l}\xi_{i}&space;\rightarrow&space;\underset{w,&space;w_{0},&space;\xi}{\min};&space;\\&space;y_{i}(\left&space;\langle&space;w,&space;x_{i}&space;\right&space;\rangle&space;-&space;w_{0})&space;\geqslant&space;1&space;-&space;\xi_{i},&space;&&space;i&space;=&space;\overline{1,l};&space;\\&space;\xi_{i}&space;\geqslant&space;0&space;,&space;&&space;i&space;=&space;\overline{1,l}.&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{cases}&space;\frac{1}{2}\left&space;\langle&space;w,w&space;\right&space;\rangle&space;&plus;&space;C\sum_{i=1}^{l}\xi_{i}&space;\rightarrow&space;\underset{w,&space;w_{0},&space;\xi}{\min};&space;\\&space;y_{i}(\left&space;\langle&space;w,&space;x_{i}&space;\right&space;\rangle&space;-&space;w_{0})&space;\geqslant&space;1&space;-&space;\xi_{i},&space;&&space;i&space;=&space;\overline{1,l};&space;\\&space;\xi_{i}&space;\geqslant&space;0&space;,&space;&&space;i&space;=&space;\overline{1,l}.&space;\end{cases}" title="\begin{cases} \frac{1}{2}\left \langle w,w \right \rangle + C\sum_{i=1}^{l}\xi_{i} \rightarrow \underset{w, w_{0}, \xi}{\min}; \\ y_{i}(\left \langle w, x_{i} \right \rangle - w_{0}) \geqslant 1 - \xi_{i}, & i = \overline{1,l}; \\ \xi_{i} \geqslant 0 , & i = \overline{1,l}. \end{cases}" /></a>
  
  *C* - (гиперпараметр) положительная константа, которая позволяет находить компромисс между максимизацией ширины разделяющей полосы и минимизацией суммарной ошибки. Ее обычно выбирают по критерию скользящего контроля.
  
  Далее решаем двойственную задачу поиска седловой точки функции Лагранжа, которая эквивалента обобщенной задаче.
  <img src="https://user-images.githubusercontent.com/71149650/99395547-a6775580-28f1-11eb-80ab-a0917bbf2b78.png" width="350"/>
  
 В итоге **алгоритм классификации** имеет вид: 
 
 <a href="https://www.codecogs.com/eqnedit.php?latex=a(x)&space;=\mathrm{sign}&space;\left&space;(&space;\sum_{i=1}^{l}&space;\lambda_{i}&space;y_{i}&space;\left&space;\langle&space;x_{i},&space;x&space;\right&space;\rangle&space;-&space;w_{0}&space;\right&space;)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?a(x)&space;=\mathrm{sign}&space;\left&space;(&space;\sum_{i=1}^{l}&space;\lambda_{i}&space;y_{i}&space;\left&space;\langle&space;x_{i},&space;x&space;\right&space;\rangle&space;-&space;w_{0}&space;\right&space;)." title="a(x) =\mathrm{sign} \left ( \sum_{i=1}^{l} \lambda_{i} y_{i} \left \langle x_{i}, x \right \rangle - w_{0} \right )." /></a>
 
 + Также еще одним способом решения проблемы линейно неразделимой выборки явлется переход от исходного пространтства **X** новому **H** с помощью некоторого преобразования.
 
 Введем понятие **ядра**. Функция **K** - **ядро**, если её можно представить <a href="https://www.codecogs.com/eqnedit.php?latex=K(x,&space;{x}')&space;=&space;\left&space;\langle&space;\psi(x),&space;\psi({x}')&space;\right&space;\rangle" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K(x,&space;{x}')&space;=&space;\left&space;\langle&space;\psi(x),&space;\psi({x}')&space;\right&space;\rangle" title="K(x, {x}') = \left \langle \psi(x), \psi({x}') \right \rangle" /></a> при некотором отображении <a href="https://www.codecogs.com/eqnedit.php?latex=\psi(x):&space;X&space;\rightarrow&space;H" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\psi(x):&space;X&space;\rightarrow&space;H" title="\psi(x): X \rightarrow H" /></a>, **H** - пространство со скалярным произведением. 
 
 И заменить  скалярное произведение на ядро. Далее перенумеруем объекты так, чтобы первые *h* объектов были опорными и получим:
 
 <a href="https://www.codecogs.com/eqnedit.php?latex=a(x)&space;=\mathrm{sign}&space;\left&space;(&space;\sum_{i=1}^{h}&space;\lambda_{i}&space;y_{i}&space;K(x_{i},&space;x)&space;-&space;w_{0}&space;\right&space;)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?a(x)&space;=\mathrm{sign}&space;\left&space;(&space;\sum_{i=1}^{h}&space;\lambda_{i}&space;y_{i}&space;K(x_{i},&space;x)&space;-&space;w_{0}&space;\right&space;)." title="a(x) =\mathrm{sign} \left ( \sum_{i=1}^{h} \lambda_{i} y_{i} K(x_{i}, x) - w_{0} \right )." /></a>
 
**Преимущества SVM**

+ Задача квадратичного программирования имеет единственное решение.
+ Автоматически определяется сложность суперпозиции — число нейронов первого слоя, равное числу опорных векторов.
+ Максимизация зазора между классами улучшает обобщающую способность.

**Недостатки SVM**

+ Неустойчивость к шуму в исходных данных. Объекты-выбросы являются опорными и существенно влияют на результат обучения
+ До сих пор не разработаны общие методы подбора ядер под конкретную задачу.
+ Подбор параметра *C* требует многократного решения задачи
