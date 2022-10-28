# 6. Метод максимального правдоподобия 

Упражняемся в матричном методе максимального правдоподобия.  Допустим, что выборка размера $n$ пришла к нам из многомерного нормального распределения с неизвестными вектором средних $\mu$ и ковариационной матрицей $\Sigma$. 

В этом задании нужно найти оценки максимального правдоподобия для $\hat \mu$ и $\hat \Sigma$.  Обратите внимание, что выборкой здесь будет не $x_1, \ldots, x_n$, а 

\begin{equation*}
	\begin{pmatrix}
		x_{11}, \ldots, x_{n1} \\
		\ldots  \\ 
		x_{n1}, \ldots, x_{nm}
	\end{pmatrix}
\end{equation*}


```{dropdown} Решение
Плотность распределения для $m-$мерного вектора $y$ будет выглядеть как 

$$
f(x \mid \mu, \Sigma) = \frac{1}{(\sqrt{2 \pi})^m \cdot \sqrt{\det \Sigma}} \cdot \exp \left( - \frac{1}{2} \cdot (x - \mu)^T \Sigma^{-1} (x - \mu) \right).
$$


В силу того, что все наблюдения независимы, функция правдоподобия для выборки объёма $n$ примет вид: 
	
$$
L(x \mid \mu, \Sigma) = \frac{1}{(\sqrt{2 \pi})^{m \cdot n} \cdot \sqrt{\det \Sigma}^n} \cdot \exp \left( - \frac{1}{2} \cdot \sum_{i = 1}^n (x_i - \mu)^T \Sigma^{-1} (x_i - \mu) \right).
$$
	
Прологарифмировав правдоподобие, получим
	
$$
\ln L(x \mid \mu, \Sigma) = - \frac{m \cdot n}{2} \ln 2 \pi - \frac{n}{2} \ln \det \Sigma - \frac{1}{2} \sum_{i=1}^n (x_i - \mu)^T \Sigma^{-1} (x_i - \mu) 
$$
	
Нам нужно найти максимум этой функции по $\mu$  и $\Sigma$.  Начнём с $\mu$.  Аргумент $\Sigma$ будем считать константой. Обозначим такую функцию за $f(\mu)$. Эта функция бьёт с множества векторов в множество скаляров.  Значит дифференциал этой функции можно записать в виде: 
	
$$
df(\mu) = \nabla f^T d \mu. 
$$
	
Найдём этот дифференциал. Не будем забывать, что дифференциал от константы нулевой, а также что дифференциал суммы равен сумме дифференциалов
	
\begin{multline*}
	d f(\mu) = -\frac{1}{2} \cdot d \sum_{i=1}^n (x_i - \mu)^T \Sigma^{-1} (x_i - \mu) = -\frac{1}{2} \cdot \sum_{i=1}^n d[(x_i - \mu)^T \Sigma^{-1} (x_i - \mu)] = \\ = -\frac{1}{2} \cdot \sum_{i=1}^n d[(x_i - \mu)^T] \Sigma^{-1} (x_i - \mu) + (x_i - \mu)^T \Sigma^{-1} d[(x_i - \mu)] = \\ =   \frac{1}{2} \cdot \sum_{i=1}^n d\mu^T \Sigma^{-1} (x_i - \mu) + (x_i - \mu)^T \Sigma^{-1} d\mu.
\end{multline*}
	
Первое слагаемое под суммой имеет размерность $1 \times m \cdot m \times m \cdot m \times 1$. Это константа. Если мы протранспонируем константу, ничего не изменится. Обратим внимание, что матрица $\Sigma$ симметричная и при транспонировании не меняется. Сделаем этот трюк 
	
\begin{multline*}
	 \frac{1}{2} \cdot \sum_{i=1}^n d\mu^T \Sigma^{-1} (x_i - \mu) + (x_i - \mu)^T \Sigma^{-1} d\mu = \\ = \frac{1}{2} \cdot \sum_{i=1}^n  (x_i - \mu)^T \Sigma^{-1}  d\mu  + (x_i - \mu)^T \Sigma^{-1} d\mu = \\ = \frac{1}{2} \cdot \sum_{i=1}^n  [(x_i - \mu)^T \Sigma^{-1}  + (x_i - \mu)^T \Sigma^{-1} ] d\mu =  \left[  \cdot \sum_{i=1}^n  (x_i - \mu)^T \Sigma^{-1} \right] d\mu 
\end{multline*}	
	
Получается, что $f'(\mu) = \sum_{i=1}^n   \Sigma^{-1} (x_i - \mu)$.  Приравняв производную к нулю и домножив обе части уравнения слева на $\Sigma$, получим оптимальное значению $\mu$: 
	
\begin{equation*}
	\begin{aligned}
	&\sum_{i=1}^n   \Sigma^{-1} (x_i - \hat \mu) = 0 \\
	&\sum_{i=1}^n   (x_i - \hat \mu) = 0 \\
	&\sum_{i=1}^n   x_i =  n \cdot \hat \mu \Rightarrow \hat \mu = \bar x.
	\end{aligned} 
\end{equation*}
		
Не будем забывать, что в записях выше $x$ и $\mu$ были векторами-столбцами размерности $m \times 1$. В итоговом ответе они также являются векторами-столбцами такой размерности. 

Займёмся оценкой для $\Sigma.$ Аргумент $\mu$ будем считать константой. Обозначим такую функцию за $f(\Sigma)$

$$
f(\Sigma) = - \frac{n}{2} \ln \det \Sigma - \frac{1}{2} \sum_{i=1}^n (x_i - \mu)^T \Sigma^{-1} (x_i - \mu).
$$

Эта функция бьёт с множества матриц в множество скаляров. Значит дифференциал этой функции можно записать в виде: 

$$
d f(\Sigma) = tr (\nabla f^T dx).
$$

Начнём с первого слагаемого. Для него нам понадобится вспомнить как выглядит дифференциал для определителя

$$
- \frac{n}{2} \frac{1}{\det \Sigma}  d[\det \Sigma]  = - \frac{n}{2} \frac{1}{\det \Sigma}  tr( \det \Sigma \cdot \Sigma^{-T} d \Sigma) =   -  tr( \frac{n}{2} \cdot \Sigma^{-1} d \Sigma).
$$

Теперь поработаем со вторым слагаемым. В нём нас интересует дифференциал обратной матрицы

$$
- \frac{1}{2} \sum_{i=1}^n (x_i - \mu)^T d[\Sigma^{-1}] (x_i - \mu) = \frac{1}{2} \sum_{i=1}^n (x_i - \mu)^T \Sigma^{-1} \cdot  d \Sigma \cdot \Sigma^{-1} (x_i - \mu). 
$$
	
Под знаком суммы размерность каждого слагаемого $1 \times m \cdot m \times m \cdot m \times m \cdot m \times m \cdot m \times 1$. Это константа. Если мы возьмём от неё след, ничего не изменится. Взяв след, переставим внутри множители

$$
\frac{1}{2} \sum_{i=1}^n (x_i - \mu)^T \Sigma^{-1} \cdot  d \Sigma \cdot \Sigma^{-1} (x_i - \mu)  = \frac{1}{2} \sum_{i=1}^n tr( \Sigma^{-1} (x_i - \mu) \cdot (x_i - \mu)^T \Sigma^{-1} \cdot  d \Sigma). 
$$

Сумма следов -- след суммы. Объединяем наши слагаемые в месте. В первом множитель $n$ подменяем на сумму 

$$
d f(\Sigma) = tr \left( \left [ - \frac{1}{2} \sum_{i = 1}^n \Sigma^{-1} + \Sigma^{-1} (x_i - \mu) \cdot (x_i - \mu)^T \Sigma^{-1} \right] d \Sigma \right)
$$

Забираем себе из-под знака дифференциала производную. Под знаком суммы после транспонирования ничего не поменяется. Приравниваем производную к нулю, домножим справа каждое слагаемое на $\Sigma$. На четвёртой строчке домножим слева на $\Sigma$: 

\begin{equation*}
	\begin{aligned} 
		 & \frac{1}{2} \sum_{i = 1}^n - \Sigma^{-1} + \Sigma^{-1} (x_i - \mu) \cdot (x_i - \mu)^T \Sigma^{-1}  = 0 \\
		 & - n \cdot \Sigma^{-1} + \sum_{i = 1}^n  \Sigma^{-1} (x_i - \mu) \cdot (x_i - \mu)^T \Sigma^{-1}  = 0 \\
		 & - n + \Sigma^{-1} \sum_{i = 1}^n  (x_i - \mu) \cdot (x_i - \mu)^T = 0 \\
		 & - n \Sigma+ \sum_{i = 1}^n  (x_i - \mu) \cdot (x_i - \mu)^T = 0 \\
		 &  \Sigma  = \frac{1}{n} \sum_{i = 1}^n  (x_i - \mu) \cdot (x_i - \mu)^T 
	\end{aligned}
\end{equation*}
	
До оценок остался один шаг. Вспоминаем оценку для $\mu$, подставляем её в уравнение и получаем, что 

$$
\hat \Sigma = \frac{1}{n} \sum_{i = 1}^n  (x_i - \bar x) \cdot (x_i - \bar x)^T.
$$
	
Не забываем, что $x_i$ и $\bar x$ -- вектора размерности $m \times 1$. 

```



	
