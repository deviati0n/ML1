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
+ Подбор параметра *C* требует многократного решения задачи.
