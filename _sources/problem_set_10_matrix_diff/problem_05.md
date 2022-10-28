# 5. Производная сложной функции


Наконец, научимся считать градиенты для сложных функций.
Допустим, даны функции

$$
f: \mathbb{R}^n \to \mathbb{R}^m$ \qquad g: \mathbb{R}^m \to \mathbb{R}.
$$

Тогда градиент их композиции можно вычислить как

$$
\nabla_x g \left( f(x) \right)
=
\mathfrak{J}_{f}^T (x)
\nabla_z \left. g(z) \right|_{z = f(x)},
$$


где $\mathfrak{J}_f (x) = \left( \frac{\partial f_i(x)}{\partial x_j}  \right)_{i, j = 1}^{m, n}$ -- матрица Якоби для функции $f$.

Если $m = 1$ и функция $g(z)$ имеет всего один аргумент, то формула упрощается:

$$
\nabla_x g \left( f(x) \right)
=
g'(f(x))
\nabla_x f(x).
$$


Вычислите градиент логистической функции потерь для линейной модели по параметрам этой модели:

$$
\nabla_w
\log \left(
    1
    +
    \exp(-y \langle w, x \rangle)
\right).
$$

Здесь $y$ принимает значения $1$ и $-1$. Иногда классы удобно обозначать так, иногда удобнее говорить, что $y$ принимает значения $1$ и $0$. От одной записи к другой можно перейти с помощью замены переменной. 


```{dropdown} Решение
Воспользуемся правилом взятия производной сложной функции и производной скалярного произведения

\begin{multline*}
	\nabla_w \log \left(1 + \exp(-y \langle w, x \rangle) \right) = \\ = \frac{1}{1 + \exp(-y \langle w, x \rangle)} \nabla_w \left(1 + \exp(-y \langle w, x \rangle) \right) = \\ = \frac{1}{1 + \exp(-y \langle w, x \rangle)} exp(-y \langle w, x \rangle) \nabla_w \left(-y \langle w, x \rangle \right) = \\ = -\frac{1}{1 + \exp(-y \langle w, x \rangle)} \exp(-y \langle w, x \rangle) y x = \\ = \left\{ \sigma(z) = \frac{1}{1 + \exp(-z)} \right\} = \\ = - \sigma(-y \langle w, x \rangle) y x
\end{multline*}
```



