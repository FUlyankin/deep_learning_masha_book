# 7. Незаметный бэкпроп

Маша собрала нейросеть:

	
\begin{equation*}
y =   \max \left( 0;  X \cdot  \begin{pmatrix} 1 & -1 \\ 0.5 & 0 \end{pmatrix} \right) \cdot \begin{pmatrix} 0.5 \\ 1 \end{pmatrix} 
\end{equation*}

__а)__ Первый слой нашей нейросетки --- линейный. По какой формуле делается forward pass? Сделайте его для матрицы 

$$
X =\begin{pmatrix} 1 & 2 \\ -1 & 2 \end{pmatrix}.
$$

__б)__ Найдите для первого слоя производную выхода по входу. При обратном движении по нейросетке, в первый слой пришёл накопленный градиент

$$
d = \begin{pmatrix} -0.5 & 0 \\ 0 & 0 \end{pmatrix}.
$$

Каким будет новое накопленное значение градиента, которое выплюнет из себя линейный слой? По какой формуле делается backward pass? 

__в)__ Второй слой нейросетки --- функция активации, $ReLU.$ По какой формуле делается forward pass? Сделайте его для матрицы 

$$
H_1 = \begin{pmatrix} 2 & -0.5 \\ 0 & 1 \end{pmatrix}.
$$

__г)__ Найдите для второго слоя производную выхода по входу. При обратном движении по нейросетке во второй слой пришёл накопленный градиент 

$$
d = \begin{pmatrix} -0.5 & -1 \\ 0 & 0 \end{pmatrix}.
$$

Каким будет новое накопленное значение градиента, которое выплюнет из себя $ReLU$?  По какой формуле делается backward pass? 

__д)__ 
Третий слой нейросетки --- линейный.  По какой формуле делается forward pass? Сделайте его для матрицы 

$$
O_1 =\begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}.
$$

__е)__ Найдите для третьего слоя производную выхода по входу. При обратном движении по нейросетке, в третий слой пришёл накопленный градиент $d = (-1, 0)^T$. Каким будет новое накопленное значение градиента, которое выплюнет из себя линейный слой? 

__ё)__ Мы решаем задачу Регрессии. В качестве функции ошибки мы используем 

$$
MSE = \frac{1}{2n} \sum (\hat y_i - y_i)^2.
$$

Пусть для рассматриваемых наблюдений реальные значения  $y_1 = 2, y_2 = 1$. Найдите значение $MSE$.


__ж)__  Чему равна производная $MSE$ по прогнозу? Каким будет накопленное значение градиента, которое $MSE$ выплюнет из себя в предыдущий слой нейросетки? 


__з)__ Пусть скорость обучения $\gamma = 1$.  Сделайте для весов нейросети шаг градиентного спуска.


__и)__ Посидела Маша, посидела, и поняла, что неправильно она всё делает. В реальности перед ней не задача регрессии, а задача классификации. Маша применила к выходу из нейросетки сигмоиду. Как будет для неё выглядеть forward pass? 

__к)__  В качестве функции потерь Маша использует $logloss.$ Как для этой функции потерь выглядит forward pass? Сделайте его. 

__л)__ Найдите для $logloss$ производную прогнозов по входу в сигмоиду. Как будет выглядеть backward pass, если $y_1 = 0, y_2 = 1?$ Как поменяется оставшаяся часть алгоритма обратного распространения ошибки? 

```{dropdown} Решение
Весь путь по нейросети от начала к концу, то есть forward pass будет выглядеть следующим образом: 

\begin{equation*}
    \begin{aligned} 
    & H_1 = X \cdot W_1 = \begin{pmatrix} 1 & 2 \\ -1 & 2 \end{pmatrix} \cdot \begin{pmatrix} 1 & -1 \\ 0.5 & 0 \end{pmatrix} =  \begin{pmatrix} 2 & -0.5 \\ 0 & 1 \end{pmatrix} \\
    & O_1 = ReLU(H_1) = \begin{pmatrix} \max(0,2) & \max(0,-0.5) \\ \max(0,0) & \max(0,1) \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix} \\
    & \hat{y} = O_1 \cdot W_2 = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix} \cdot \begin{pmatrix} 0.5 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} \\ 
    & MSE = \frac{1}{4} \cdot ((1 - 2)^2 + (1 - 1)^2) = 0.25
    \end{aligned}
\end{equation*}

Все необходимые для обратного прохода производные выглядят как

\begin{equation*}
    \begin{aligned} 
    & \frac{\partial MSE}{\partial \hat y} = \begin{pmatrix} \hat y_1 - y_1  \\ \hat y_2 - y_2 \end{pmatrix} = \begin{pmatrix} -1 \\ 0 \end{pmatrix} \\ 
    & \frac{\partial\hat y}{\partial O_1} = W_2^T = (0.5, 1) \qquad  \frac{\partial\hat y}{\partial W_2} = O_1^T = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix} \\
    & \frac{\partial O_1}{\partial H_1} = [H_{ij} > 0] = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \\
    & \frac{\partial H_1}{\partial X} = W_1^T = \begin{pmatrix} 1 & 0.5 \\ -1 & 0 \end{pmatrix} \qquad  \frac{\partial H_1}{\partial W_1} = X^T = \begin{pmatrix} 1 & -1 \\ 2 & 2 \end{pmatrix}
    \end{aligned}
\end{equation*}

Когда мы считаем производную $MSE,$ мы ищем её по каждому прогнозу. В случае производной для $ReLU$ запись $[H_{ij} > 0]$ означает, что на месте $ij$ стоит $1$, если элемент больше нуля и ноль иначе.  Делаем шаг обратного распространения ощибки

\begin{equation*}
    \begin{aligned} 
    & d = \begin{pmatrix} -1 \\ 0 \end{pmatrix} \\ 
    & \frac{\partial MSE}{\partial W_2} =  O_1^T \cdot d = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix} \cdot \begin{pmatrix} -1 \\ 0 \end{pmatrix} = \begin{pmatrix} -2 \\ 0 \end{pmatrix}\\
    & d = d \cdot W_2^T \odot [H_{ij} > 0] = \begin{pmatrix} -1 \\ 0 \end{pmatrix}  \cdot (0.5, 1) \odot \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} -0.5 & 0 \\ 0 & 0 \end{pmatrix} \\
    & \frac{\partial MSE}{\partial W_1} =  X^T \cdot d = \begin{pmatrix} 1 & -1 \\ 2 & 2 \end{pmatrix} \cdot \begin{pmatrix} -0.5 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} -0.5 & 0 \\ -1 & 0 \end{pmatrix} \\
    \end{aligned}
\end{equation*}

Делаем шаг градиентного спуска

\begin{equation*}
    \begin{aligned} 
      & W_1 = \begin{pmatrix} 1 & -1 \\ 0.5 & 0 \end{pmatrix} - \gamma \cdot \begin{pmatrix} -0.5 & -1 \\ 0 & 0 \end{pmatrix}\\
      & W_2 = \begin{pmatrix} 0.5 \\ 1 \end{pmatrix}  - \gamma \cdot \begin{pmatrix} - 2 \\ 0 \end{pmatrix}.
    \end{aligned}
\end{equation*}

Меняем MSE на logloss и добавляем сигмоиду. Производная для сигмоиды выглядит как 

\begin{equation*}
    \begin{aligned} 
    & logloss = y_i \cdot \ln \hat p_i + (1 - y_i) \cdot (1 - \hat p_i) \\ 
    & \frac{\partial logloss}{\partial \hat p_i} = \frac{y_i}{\hat p_i} - \frac{1 - y_i}{1 - \hat p_i}.
    \end{aligned}
\end{equation*}

Так как в бинарной классификации $y_i$ принимает значения $\{0,1\},$ производная равна либо первому либо второму слагаемому. Получаем вычисления 

\begin{equation*}
    \begin{aligned} 
    & O_2 = \sigma(\hat y) = \begin{pmatrix} 0.73 \\ 0.73 \end{pmatrix} \qquad  \frac{\partial O_2}{\partial \hat y} = O_2 \cdot (1 - O_2) \approx \begin{pmatrix} 0.2 \\ 0.2 \end{pmatrix} \\ 
    & \frac{\partial logloss}{\partial \hat p} \approx \begin{pmatrix} 3.7 \\ 1.4 \end{pmatrix} \qquad \frac{\partial logloss}{\partial \hat y} \approx \begin{pmatrix} 3.7 \\ 1.4 \end{pmatrix} \odot \begin{pmatrix} 0.2 \\ 0.2 \end{pmatrix} = \begin{pmatrix} 0.74 \\ 0.28 \end{pmatrix}
    \end{aligned}
\end{equation*}

Дальше алгоритм делается ровно также, только в качестве стартового $d$ используется $logloss'_{\hat y},$ а не $MSE'_{\hat y}.$

```
