# 7. Матричный лагранжиан

Найдите симметричную матрицу $X$, наиболее близкую к матрице $A$ по норме Фробениуса, $\sum_{i,j} (x_{ij} - a_{ij})^2$. Тут мы просто из каждого элемента $X$ вычитаем стоящий на том же месте элемент $A$ и смотрим на сумму квадратов таких разностей. То есть решите задачку условной матричной минимизации 
	
\begin{equation*}
	\begin{cases}
		& ||X - A||^2 \to \min_{X}  \\
		& X^T = X
	\end{cases}
\end{equation*}
	

```{dropdown} Подсказка
Надо будет выписать лагранжиан.  А ещё пригодится тот факт, что $\sum_{i,j} (x_{ij} - a_{ij})^2 = ||X-A||^2 =  \text{tr}((X-A)^T (X-A))$.
```

```{dropdown} Решение
Выписываем лагранжиан

\begin{multline*}
\mathscr{L} = \sum_{i,j} (x_{ij} - a_{ij})^2 + \sum_{ij} \lambda_{ij} (x_{ij} - x_{ji}) = \\ = \text{tr}((X-A)^T (X-A)) + \text{tr}(\Lambda^T (X - X^T)) = \\ = \text{tr}(X^TX) - 2 \text{tr}(X^TA) + \text{tr}(A^TA) + \text{tr}(\Lambda^T (X - X^T))
\end{multline*}

Найдём все необходимые нам дифференциалы

\begin{equation*}
    \begin{aligned} 
    & d[\text{tr}(X^TX)] = \text{tr}(d(X^TX))  = \text{tr}(X^T dX) + \text{tr}(dX^T X) = \text{tr}(2 X^T dX) \\
    & d[\text{tr}(X^TA)] = \text{tr}(A^T dX) \\
    & d[\text{tr}(\Lambda^TX)] = \text{tr}(\Lambda^T dX) \\
    & d[\text{tr}(\Lambda^TX^T)] = \text{tr}(\Lambda dX) \\
    \end{aligned} 
\end{equation*}

Выписываем в явном виде транспонированную производную по $X$ 

$$
(\nabla_X \mathscr{L})^T = 2X^T - 2A^T + \Lambda^T - \Lambda = 0
$$

Нужно избавиться от $\Lambda$, давайте транспонируем уравнение

$$
\nabla_X \mathscr{L} = 2X - 2A + \Lambda - \Lambda^T = 0,
$$

а после прибавим его к исходному, тогда лишние части исчезнут 

$$
4X - 2A^T - 2A = 0 \qquad X = \frac{1}{2}(A + A^T).
$$
```
