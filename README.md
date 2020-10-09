<h1 align = "center"> ML1 </h1>

# Постановка задачи #

Существует множество объектов **X** , для которых дано множество ответов **Y**. Между этими двумя множествами существует некая зависимость - **целевая зависимость**. Каждую пару объект-ответ будем называть **прецедентом** (sample). А матрицу таких зависимостей - **обучающая выборка** (training sample). 

Перед нами стоит цель построить алгоритм **a : X -> Y**, который способен классифицировать произвольный объект из множества **X**. 

# Метрические алгоритмы классификации #

**Метрические методы обучения** - методы, основаны на **анализе сходства объектов**. Близкие объекты лежат в одном классе.

Ниже представлены примеры метрических методов.

+ [1NN](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B5%D0%B3%D0%BE-%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B0-1nn)
+ [KNN](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-k-%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B8%D1%85-%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B5%D0%B9-knn)
+ [KWNN](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-k-%D0%B2%D0%B7%D0%B2%D0%B5%D1%88%D0%B5%D0%BD%D0%BD%D1%8B%D1%85-%D0%B1%D0%BB%D0%B8%D0%B6%D0%B0%D0%B9%D1%88%D0%B8%D1%85-%D1%81%D0%BE%D1%81%D0%B5%D0%B4%D0%B5%D0%B9-kwnn)
+ [Парзеновское окно](https://github.com/deviati0n/ML1/blob/master/README.md#%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-%D0%BF%D0%B0%D1%80%D0%B7%D0%B5%D0%BD%D0%BE%D0%B2%D1%81%D0%BA%D0%BE%D0%B3%D0%BE-%D0%BE%D0%BA%D0%BD%D0%B0)
+ [LOO](https://github.com/deviati0n/ML1/blob/master/README.md#%D1%81%D0%BA%D0%BE%D0%BB%D1%8C%D0%B7%D1%8F%D1%89%D0%B8%D0%B9-%D0%BA%D0%BE%D0%BD%D1%82%D1%80%D0%BE%D0%BB%D1%8C-loo)

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



## Скользящий контроль LOO ##

Данная процедура помогает найти оптимальное значение параметров **k** (KNN, KWNN) и **q** (весовая функция для KWNN). Для каждого объекта из выборки проверяется, правильно ли он классифицируется по своим k ближайшим соседям (k = [1; l], q = (0; 1)).

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

<img src="https://user-images.githubusercontent.com/71149650/94484221-bf626500-01e4-11eb-87f9-24c663d2970b.png" alt="LOO для полной выборки" />

### Реализация скользящего контроля LOO для KWNN ###
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

<img src="https://user-images.githubusercontent.com/71149650/95449907-1945f600-096e-11eb-9310-ea6a8523d001.png" />

### Реализация скользящего контроля LOO для парзеновского окна ###
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
</table>

Анализируя данные в таблице, прийдем к выводу, что самым опимальным алгоритмом для ирисов фишер является **KNN** с параметром k = 6.



