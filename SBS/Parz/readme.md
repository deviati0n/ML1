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
<img src = "https://user-images.githubusercontent.com/71149650/94997426-aaeee580-05b3-11eb-82eb-f0276621e019.png" width = "1000"/>

### Треугольное ядро ###
<img src = "https://user-images.githubusercontent.com/71149650/94997729-adead580-05b5-11eb-9cce-84e08d22e8be.png" width = "1000"/>



