# 7. Матричный лагранжиан

Найдите симметричную матрицу $X$ наиболее близкую к матрице $A$ по норме Фробениуса, $\sum_{i,j} (x_{ij} - a_{ij})^2$. Тут мы просто из каждого элемента вычитаем каждый и смотрим на сумму квадратов таких разностей. То есть решите задачку условной матричной минимизации 
	
\begin{equation*}
	\begin{cases}
		& ||X - A||^2 \to \min_{A}  \\
		& X^T = X
	\end{cases}
\end{equation*}
	

```{dropdown} Подсказка
Надо будет выписать Лагранджиан.  А ещё пригодится тот факт, что $\sum_{i,j} (x_{ij} - a_{ij})^2 = ||X-A||^2 =  tr((X-A)^T (X-A))$.
```

```{dropdown} Решение
Выписываем лагранджиан

\begin{multline*}
\mathscr{L} = \sum_{i,j} (x_{ij} - a_{ij})^2 + \sum_{ij} \lambda_{ij} (x_{ij} - x_{ji}) = \\ = tr((X-A)^T (X-A)) + tr(\Lambda^T (X - X^T)) = \\ = tr(X^TX) - 2 tr(X^TA) + tr(A^TA) + tr(\Lambda^T (X - X^T))
\end{multline*}

Найдём все необходимые нам дифференциалы

\begin{equation*}
    \begin{aligned} 
    & d[tr(X^TX)] = tr(d(X^TX))  = tr(X^T dX) + tr(dX^T X) = tr(2 X^T dX) \\
    & d[tr(X^TA)] = tr(A^T dX) \\
    & d[tr(\Lambda^TX)] = tr(\Lambda^T dX) \\
    & d[tr(\Lambda^TX^T)] = tr(\Lambda dX) \\
    \end{aligned} 
\end{equation*}

Выписываем в яном виде производную по $X$ 

$$
\frac{\partial \mathscr{L}}{\partial X} = 2X^T - 2A^T + \Lambda^T - \Lambda = 0
$$

Нужно избавиться от $\Lambda$, давайте транспонируем уравнение

$$
\frac{\partial \mathscr{L}}{\partial X} = 2X - 2A + \Lambda - \Lambda^T = 0,
$$

а после прибавим его к исходному, тогда лишние части исчезнут 

$$
4X - 2A^T - 2A = 0 \qquad X = \frac{1}{2}(A + A^T).
$$
```
