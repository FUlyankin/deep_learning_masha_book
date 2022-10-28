# 4. Линейный слой

Маша знает, что главный слой в нейронных сетях --- линейный. В матричном виде его можно записать как $Z = XW.$ 

Маша хочет использовать этот слой внутри нейросети. Предполагается, что после прямого шага наши вычисления будут использованы в другой части нейросети. В конечном итоге, по выходу из нейросети мы вычислим какую-то функцию потерь $L$. 

Чтобы обучить нейросеть, Маше понадобятся производные $\frac{\partial L}{\partial X}$ и $\frac{\partial L}{\partial W}.$  Аккуратно найдите их и запишите в матричном виде[^url_note]. Предполагается, что 

$$
X = \begin{pmatrix} x_{11} & x_{12} \\  x_{21} & x_{22}  \end{pmatrix} \qquad W  = \begin{pmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23}  \end{pmatrix} 
$$ 

$$
Z = XW = \begin{pmatrix} z_{11} & z_{12} & z_{13}\\  z_{21} & z_{22} & z_{23}  \end{pmatrix} = \begin{pmatrix} x_{11}w_{11} + x_{12}w_{21} & x_{11}w_{12} + x_{12}w_{22} & x_{11}w_{13} + x_{12}w_{23} \\ x_{21}w_{11} + x_{22}w_{21} & x_{21}w_{12} + x_{22}w_{22} & x_{21}w_{13} + x_{22}w_{23} \end{pmatrix}
$$

```{dropdown} Решение
При обратном распространении ошибки мы предполагаем, что производная $\frac{\partial L}{\partial Z}$ у нас уже есть. Так как $Z$ --- это матрица размера $2 \times 3,$ эта производная будет выглядеть как 

$$
\frac{\partial L}{\partial Z} = \begin{pmatrix} \frac{\partial L}{\partial z_{11}}  & \frac{\partial L}{\partial z_{12}} & \frac{\partial L}{\partial z_{13}} \\ \frac{\partial L}{\partial z_{21}}  & \frac{\partial L}{\partial z_{22}} & \frac{\partial L}{\partial z_{23}}  \end{pmatrix}.
$$

По цепному правилу мы можем использовать $\frac{\partial L}{\partial Z}$ для поиска интересующих нас градиентов 

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \cdot \frac{\partial Z}{\partial X} \qquad  \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Z} \cdot \frac{\partial Z}{\partial W}.
$$

Нужно, чтобы у матриц совпали размерности. Производные $\frac{\partial Z}{\partial X}$ и $\frac{\partial Z}{\partial W}$ --- это матрицы Якоби нашего линейного слоя. Пусть $W$ это параметры, а $X$ аргумент функции. Функция $f(X) = XW$ бьёт из пространства матриц $X_{[2 \times 2]}$ в пространство матриц $Z_{[2 \times 3]}$. Нам надо взять производную от каждого элемента матрицы $Z$ по каждому элементу из матрицы $X$. Всего получится $24$ производных. По правилам из матана мы должны будем записать их в виде четырёхмерной матрицы\footnote{Про это можно более подробно почитать в разделе про матричные производные.}. Это жутко неудобно. 

К счастью, многие производные будут нулевыми. Поэтому мы можем схитрить, сначала найти $\frac{\partial L}{\partial X},$ 

$$
X = \begin{pmatrix} x_{11} & x_{12} \\  x_{21} & x_{22}  \end{pmatrix} \quad \Rightarrow \quad  \frac{\partial L}{\partial X} = \begin{pmatrix} \frac{\partial L}{\partial x_{11}}  & \frac{\partial L}{\partial x{12}} \\ \frac{\partial L}{\partial x_{21}}  & \frac{\partial L}{\partial x_{22}}   \end{pmatrix},
$$

а затем написать удобные формулы в общем виде. Найдём $\frac{\partial L}{\partial x_{11}}$ с помощью цепного правила 

$$
\frac{\partial L}{\partial x_{11}} = \sum_{i=1}^n \sum_{j=1}^d \frac{\partial L}{\partial z_{ij}} \cdot \frac{\partial z_{ij}}{\partial x_{11}} = \langle \frac{\partial L}{\partial Z} , \frac{\partial Z}{\partial x_{11}} \rangle.
$$

Работать с суммами неудобно. Мы помним, что $\frac{\partial L}{\partial Z}$ и $\frac{\partial Z}{\partial x_{11}}$ --- матрицы из производных. Поэтому сумму можно записать в виде скалярного произведения матриц. Мы должны в нём умножить элементы матриц друг на друга, а затем сложить.  Давайте найдём производную матрицы $Z$ по $x_{11}$

$$
Z = XW = \begin{pmatrix} x_{11}w_{11} + x_{12}w_{21} & x_{11}w_{12} + x_{12}w_{22} & x_{11}w_{13} + x_{12}w_{23} \\ x_{21}w_{11} + x_{22}w_{21} & x_{21}w_{12} + x_{22}w_{22} & x_{21}w_{13} + x_{22}w_{23} \end{pmatrix}.
$$

Переменная $x_{11}$ фигурирует только в первой строке

$$
\frac{\partial Z}{\partial x_{11}} = \begin{pmatrix} w_{11} & w_{12}  & w_{13} \\ 0 & 0 & 0 \end{pmatrix}.
$$

Выходит, что 

\begin{multline*}
\frac{\partial L}{\partial x_{11}}  =  \left\langle \begin{pmatrix} \frac{\partial L}{\partial z_{11}}  & \frac{\partial L}{\partial z_{12}} & \frac{\partial L}{\partial z_{13}} \\ \frac{\partial L}{\partial z_{21}}  & \frac{\partial L}{\partial z_{22}} & \frac{\partial L}{\partial z_{23}}  \end{pmatrix} , \begin{pmatrix} w_{11} & w_{12}  & w_{13} \\ 0 & 0 & 0 \end{pmatrix} \right\rangle = \\ = \frac{\partial L}{\partial z_{11}} \cdot w_{11} + \frac{\partial L}{\partial z_{12}} \cdot w_{12} +  \frac{\partial L}{\partial z_{13}} \cdot w_{13}.
\end{multline*}

По аналогии мы можем найти оставшиеся три производные. Например,

\begin{multline*}
\frac{\partial L}{\partial x_{21}}  =  \left\langle \begin{pmatrix} \frac{\partial L}{\partial z_{11}}  & \frac{\partial L}{\partial z_{12}} & \frac{\partial L}{\partial z_{13}} \\ \frac{\partial L}{\partial z_{21}}  & \frac{\partial L}{\partial z_{22}} & \frac{\partial L}{\partial z_{23}}  \end{pmatrix} , \begin{pmatrix} 0 & 0 & 0 \\ w_{11} & w_{12}  & w_{13}  \end{pmatrix} \right\rangle = \\ =  \frac{\partial L}{\partial z_{21}} \cdot w_{11} + \frac{\partial L}{\partial z_{22}} \cdot w_{12} +  \frac{\partial L}{\partial z_{23}} \cdot w_{13}.
\end{multline*}

Попробуем выписать $\frac{\partial L}{\partial X}$ через $\frac{\partial L}{\partial Z}$ и $W$

\begin{multline*}
\frac{\partial L}{\partial X} = \begin{pmatrix} \frac{\partial L}{\partial x_{11}}  & \frac{\partial L}{\partial x{12}} \\ \frac{\partial L}{\partial x_{21}}  & \frac{\partial L}{\partial x_{22}}   \end{pmatrix} = \\ = \begin{pmatrix} \frac{\partial L}{\partial z_{11}} \cdot w_{11} + \frac{\partial L}{\partial z_{12}} \cdot w_{12} +  \frac{\partial L}{\partial z_{13}} \cdot w_{13}  &  \frac{\partial L}{\partial z_{11}} \cdot w_{21} + \frac{\partial L}{\partial z_{12}} \cdot w_{22} +  \frac{\partial L}{\partial z_{13}} \cdot w_{23} \\ \frac{\partial L}{\partial z_{21}} \cdot w_{11} + \frac{\partial L}{\partial z_{22}} \cdot w_{12} +  \frac{\partial L}{\partial z_{23}} \cdot w_{13}  &   \frac{\partial L}{\partial z_{21}} \cdot w_{21} + \frac{\partial L}{\partial z_{22}} \cdot w_{22} +  \frac{\partial L}{\partial z_{23}} \cdot w_{23}\end{pmatrix} = \\ = \begin{pmatrix} \frac{\partial L}{\partial z_{11}}  & \frac{\partial L}{\partial z_{12}} & \frac{\partial L}{\partial z_{13}} \\ \frac{\partial L}{\partial z_{21}}  & \frac{\partial L}{\partial z_{22}} & \frac{\partial L}{\partial z_{23}}  \end{pmatrix} \cdot \begin{pmatrix} w_{11} & w_{21} \\ w_{12} & w_{22} \\ w_{13} & w_{23} \end{pmatrix} = \frac{\partial L}{\partial Z} W^T
\end{multline*}

Нам повезло! Наша хитрость увенчалась успехом, и нам удалось записать нашу формулу в виде произведения двух матриц без вычисления четырёхмерных якобианов.

**Провернём ровно такой же фокус с поиском производной $\frac{\partial L}{\partial W}.$**

$$
W = \begin{pmatrix} w_{11} & w_{12} & w_{13} \\  w_{21} & w_{22} & w_{23}  \end{pmatrix} \quad \Rightarrow \quad  \frac{\partial L}{\partial W} = \begin{pmatrix} \frac{\partial L}{\partial w_{11}}  & \frac{\partial L}{\partial w_{12}} & \frac{\partial L}{\partial w_{13}} \\ \frac{\partial L}{\partial w_{21}}  & \frac{\partial L}{\partial w_{22}}  & \frac{\partial L}{\partial w_{23}}  \end{pmatrix}.
$$


По аналогии с предыдущей производной 

$$
\frac{\partial L}{\partial w_{kl}} = \sum_{i=1}^n \sum_{j=1}^d \frac{\partial L}{\partial z_{ij}} \cdot \frac{\partial z_{ij}}{\partial w_{kl}} = \langle \frac{\partial L}{\partial Z} , \frac{\partial Z}{\partial w_{kl}} \rangle.
$$

По матрице 

$$
Z = XW = \begin{pmatrix} x_{11}w_{11} + x_{12}w_{21} & x_{11}w_{12} + x_{12}w_{22} & x_{11}w_{13} + x_{12}w_{23} \\ x_{21}w_{11} + x_{22}w_{21} & x_{21}w_{12} + x_{22}w_{22} & x_{21}w_{13} + x_{22}w_{23} \end{pmatrix}
$$

мы можем найти все требуемые производные

\begin{equation*}
    \begin{aligned}
        \frac{\partial Z}{\partial w_{11}} = \begin{pmatrix} x_{11} & 0 & 0 \\ x_{21} & 0 & 0 \end{pmatrix} & \quad \frac{\partial Z}{\partial w_{12}} = \begin{pmatrix} 0 & x_{11} & 0 \\  0 & x_{21} & 0 \end{pmatrix} & \quad \frac{\partial Z}{\partial w_{13}} = \begin{pmatrix} 0  & 0 & x_{11}\\  0  & 0 & x_{21}\end{pmatrix}  \\
        \frac{\partial Z}{\partial w_{21}} = \begin{pmatrix} x_{12} & 0 & 0 \\ x_{22} & 0 & 0 \end{pmatrix}  & \quad  \frac{\partial Z}{\partial w_{22}} = \begin{pmatrix} 0 & x_{12} & 0 \\  0 & x_{22} & 0 \end{pmatrix}  & \quad \frac{\partial Z}{\partial w_{23}} = \begin{pmatrix} 0  & 0 & x_{12}\\  0  & 0 & x_{22}\end{pmatrix}.
    \end{aligned}
\end{equation*}


Чтобы найти $\frac{\partial L}{\partial w_{kl}} $ нам надо посчитать между матрицами $\frac{\partial L}{\partial Z}$ и $\frac{\partial Z}{\partial w_{kl}}$ скалярное произведение. Например,

$$
\frac{\partial L}{\partial w_{21}} =  \left\langle  \begin{pmatrix} \frac{\partial L}{\partial z_{11}}  & \frac{\partial L}{\partial z_{12}} & \frac{\partial L}{\partial z_{13}} \\ \frac{\partial L}{\partial z_{21}}  & \frac{\partial L}{\partial z_{22}}  & \frac{\partial L}{\partial z_{23}}  \end{pmatrix} , \begin{pmatrix} x_{12} & 0 & 0 \\ x_{22} & 0 & 0 \end{pmatrix} \right \rangle = \frac{\partial L}{\partial z_{11}} \cdot x_{12} +   \frac{\partial L}{\partial z_{11}} \cdot x_{22}.
$$

Получается, что всю матрицу $\frac{\partial L}{\partial W}$ целиком можно найти как 

\begin{multline*}
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial W} = \begin{pmatrix} \frac{\partial L}{\partial w_{11}}  & \frac{\partial L}{\partial w_{12}} & \frac{\partial L}{\partial w_{13}} \\ \frac{\partial L}{\partial w_{21}}  & \frac{\partial L}{\partial w_{22}}  & \frac{\partial L}{\partial w_{23}}  \end{pmatrix} = \\ = \begin{pmatrix}  \frac{\partial L}{\partial z_{11}} \cdot x_{11} + \frac{\partial L}{\partial z_{21}} \cdot x_{21} &  \frac{\partial L}{\partial z_{12}} \cdot x_{11} + \frac{\partial L}{\partial z_{22}} \cdot x_{21} & \frac{\partial L}{\partial z_{13}} \cdot x_{11} + \frac{\partial L}{\partial z_{23}} \cdot x_{21} \\ \frac{\partial L}{\partial z_{11}} \cdot x_{12} + \frac{\partial L}{\partial z_{21}} \cdot x_{22} &  \frac{\partial L}{\partial z_{12}} \cdot x_{12} + \frac{\partial L}{\partial z_{22}} \cdot x_{22} & \frac{\partial L}{\partial z_{13}} \cdot x_{12} + \frac{\partial L}{\partial z_{23}} \cdot x_{22} \end{pmatrix}  = \\ =  \begin{pmatrix} x_{11} & x_{21} \\ x_{12} & x_{22} \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial L}{\partial z_{11}}  & \frac{\partial L}{\partial z_{12}} & \frac{\partial L}{\partial z_{13}} \\ \frac{\partial L}{\partial z_{21}}  & \frac{\partial L}{\partial z_{22}} & \frac{\partial L}{\partial z_{23}}  \end{pmatrix} = X^T \frac{\partial L}{\partial Z}.
\end{multline*}

Таким образом, для линейного слоя, мы всегда можем посчитать производные как 

$$
\frac{\partial L}{\partial W}  = X^T \frac{\partial L}{\partial Z} \qquad \frac{\partial L}{\partial X}  = \frac{\partial L}{\partial Z} W^T.
$$

```

::::{important}
Во всех следующих задачках под $\frac{\partial Z}{\partial W}$ будем всегда подразумевать $X^T,$ а под  $\frac{\partial Z}{\partial X}$ будем иметь в виду $W^T$.
::::


[^url_note]: Идея задачи взята [отсюда](https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html)

