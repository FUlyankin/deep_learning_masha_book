# 7. Матричный лагранжиан

Найдите симметричную матрицу $X$, наиболее близкую к матрице $A$ по норме Фробениуса, $\sum_{i,j} (x_{ij} - a_{ij})^2$. Тут мы просто из каждого элемента вычитаем соответствующий и смотрим на сумму квадратов таких разностей. То есть решите задачку условной матричной минимизации 
	
\begin{equation*}
	\begin{cases}
		& ||X - A||^2 \to \min_{X}  \\
		& X^T = X
	\end{cases}
\end{equation*}
	

```{dropdown} Подсказка
Надо будет выписать лагранжиан.  А ещё пригодится тот факт, что $\sum_{i,j} (x_{ij} - a_{ij})^2 = ||X-A||^2 =  \operatorname{tr}((X-A)^T (X-A))$.
```

```{dropdown} Решение
Выписываем лагранжиан

\begin{multline*}
\mathscr{L} = \sum_{i,j} (x_{ij} - a_{ij})^2 + \sum_{ij} \lambda_{ij} (x_{ij} - x_{ji}) = \\ = \operatorname{tr}((X-A)^T (X-A)) + \operatorname{tr}(\Lambda^T (X - X^T)) = \\ = \operatorname{tr}(X^TX) - 2 \operatorname{tr}(X^TA) + \operatorname{tr}(A^TA) + \operatorname{tr}(\Lambda^T (X - X^T))
\end{multline*}

Найдём все необходимые нам дифференциалы

\begin{equation*}
    \begin{aligned} 
    & d[\operatorname{tr}(X^TX)] = \operatorname{tr}(d(X^TX))  = \operatorname{tr}(X^T dX) + \operatorname{tr}(dX^T X) = \operatorname{tr}(2 X^T dX) \\
    & d[\operatorname{tr}(X^TA)] = \operatorname{tr}(A^T dX) \\
    & d[\operatorname{tr}(\Lambda^TX)] = \operatorname{tr}(\Lambda^T dX) \\
    & d[\operatorname{tr}(\Lambda^TX^T)] = \operatorname{tr}(\Lambda dX) \\
    \end{aligned} 
\end{equation*}

Выписываем в явном виде производную по $X$ 

$$
\left(\frac{\partial \mathscr{L}}{\partial X}\right)^T = 2X^T - 2A^T + \Lambda^T - \Lambda = 0
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
