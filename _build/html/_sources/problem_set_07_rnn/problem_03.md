# 3. Взрыв градиентов

::::{important}
Тут будет упражнение про взрыв/затухание градиентов рекуррентных архитектурах и способы лечить его. От LSTM до инициализации ортогональными матрицами с неумирающими активациями.
::::


<!-- Производная по весам $W$ в рекуррентной сети считается как

$$
\frac{\partial L_t}{\partial W} = \frac{\partial L_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial h_t} \cdot \sum_{i=0}^t \left( \prod_{i=k+1}^t  \frac{\partial h_i}{\partial h_{i-1}} \right) \cdot  \frac{\partial h_k}{\partial W}.
$$



__а)__  Такая сложная производная может приводить к затуханию и взрыву градиентов. Объясните, в чём заключаются эти проблемы.

```{dropdown} Решение


```

__б)__ Предложите способ побороть проблему взрыва градиентов.


```{dropdown} Решение

[^proof_note]

```

__в)__ Предложите способ побороть проблему затухания градиентов.


```{dropdown} Решение





		\begin{equation*} 
			\begin{aligned}
				h_t =& f_h(b_h + W \cdot h_{t-1} + V \cdot x_t) = f_h(pr_t)\\
				\hat{y}_t =& f_y(b_y + U \cdot h_t)
			\end{aligned}
		\end{equation*} 
		
		\par \mbox{} \par
		
		\[\frac{\partial L_t}{\partial W}  \propto \sum_{k=0}^t \left( \alert{\prod_{i=k+1}^t  \frac{\partial h_i}{\partial h_{i-1}}} \right) \cdot  \frac{\partial h_k}{\partial W} \]
		
		\par \mbox{} \par
		
	\only<2>{
		\[
			\frac{\partial h_i}{\partial h_{i-1}} = \frac{\partial h_i}{\partial inp_i} \cdot \frac{\partial inp_i}{\partial h_{i-1}} = diag(f'_h(pr_t)) \cdot \alert{?}
		\]
	}		

		\par \mbox{} \par
		
	\only<3>{
		\[
			\frac{\partial h_i}{\partial h_{i-1}} = \frac{\partial h_i}{\partial inp_i} \cdot \frac{\partial inp_i}{\partial h_{i-1}} = diag(f'_h(pr_t)) \cdot \alert{W}
		\]



\begin{frame}{Аккуратная инициализация (2015)}
		
		\[\frac{\partial L_t}{\partial W}  \propto \sum_{k=0}^t \left( \alert{\prod_{i=k+1}^t  \frac{\partial h_i}{\partial h_{i-1}}} \right) \cdot  \frac{\partial h_k}{\partial W} \]
				
		\[
		\frac{\partial h_i}{\partial h_{i-1}} = \frac{\partial h_i}{\partial inp_i} \cdot \frac{\partial inp_i}{\partial h_{i-1}} = diag(f'_h(pr_t)) \cdot W
		\]
	
	\begin{wideitemize} 
		\item Функция активации $f_h(pr_t)$ не должна способствовать затуханию градиентов
		
		\item Если $W$ ортогональная матрица, тогда $W^T W = I, \quad W^{-1} = W^T$ и  произведение $\prod_i W_i$ не будет взрываться и затухать % (все собственные числа либо 0 либо 1)
	
		\item Инициализируем $W$ ортогональной матрицей
	\end{wideitemize} 
	
	\vfill 
	\footnotesize 
	\color{blue} \url{https://arxiv.org/abs/1504.00941} 
\end{frame}


```

[^proof_note]: On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063)
 -->
