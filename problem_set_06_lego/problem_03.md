# 3. Weight Decay

В случае $l_2$-регуляризации, к базовой функции потерь добавляют дополнительное слагаемое и вместо функционала

$$
L(w) = \frac{1}{n} \cdot \sum\limits_{i=1}^{n} L(y_i, x_i, w)
$$

оптимизируют функционал 

$$
Q_\lambda(w) = L(w) + \frac{1}{2}\lambda \cdot ||w||^2_2.
$$

Будем считать, что регуляризатор наложен на все веса нейронной сети. Обычно регуляризатор добавляют к функции потерь, чтобы избежать переобучения. Градиентный спуск можно переписать с учётом регуляризатора немного в другом виде. Такой вид называется **weight decay.** 

В пакетах для обучения нейронных сеток у оптимизаторов обычно есть такой параметр. Давайте проделаем это переписывание для нескольких разных градиентных спусков. 

__а)__ Выпишите шаг momentum-SGD для такой модели. Выразите получившийся шаг в виде 

$$
w_t = g(\lambda) \cdot w_{t-1} - \eta_t \cdot h(\nabla_w L(w_{t-1}))
$$


```{dropdown} Решение
Один шаг momentum-SGD выглядит как

$$
\begin{cases}
	g_t = \nabla_w L(w_{t-1}) + \lambda \cdot w_{t-1} \\
	m_t = \mu \cdot m_{t-1} + g_t\\
	w_t = w_{t-1} - \eta_t \cdot m_t \\
\end{cases}
$$

Подставим первую строку во вторую, а вторую в третью

\begin{multline*}
w_t = w_{t-1} - \eta_t \cdot (\mu \cdot m_{t-1} + \nabla_w L(w_{t-1}) + \lambda \cdot w_{t-1}) = \\ =\underbrace{(1-\eta_t\cdot \lambda)}_{<1}\cdot w_{t-1} - \eta_t\cdot(\mu\cdot m_{t-1} + \nabla_w L(w_{t-1}))
\end{multline*}

Получается, что, когда мы добавляем к модели $l_2$ регуляризацию, мы на каждом шаге сначала умножаем старые веса на $(1 - \eta_t \cdot \lambda)$, то есть немного сжимаем их к нулю, а потом делаем шаг по градиенту без регуляризатора. Параметр $\lambda$ в оптимизаторах называется weight decay. Обычно при обучении нейронных сетей вместо регуляризации используют его. Но в накопленном импульсе $m_{t-1}$ уже сидят прошлые слагаемые $\lambda \cdot w$, поэтому для momentum-SGD $l_2$-регуляризация и честное затухание весов совпадают только при $\mu = 0$. 

```

__б)__ Выпишите шаг Adam для такой модели. Выразите получившийся шаг в виде 

$$
w_t = g(\lambda) \cdot w_{t-1} - \eta_t \cdot h(\nabla_w L(w_{t-1}))
$$

```{dropdown} Решение
Один шаг Adam выглядит как 

$$
\begin{cases}
	g_t = \nabla_w L(w_{t-1}) + \lambda \cdot w_{t-1} \\
	m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
	v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2\\
	\hat{m}_t = \frac{1}{1-\beta_1^t} \cdot m_t \\
	\hat{v}_t = \frac{1}{1-\beta_2^t}v_t \\
	w_t = w_{t-1} -\eta_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
\end{cases}
$$

Сделаем все подстановки и получим 

\begin{multline*}
w_t = w_{t-1} - \eta_t \cdot \frac{m_t}{1-\beta_1^t} \cdot \frac{1}{\sqrt{\hat{v}_t} + \varepsilon} = \\ = w_{t-1} - \eta_t \cdot \frac{\beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t}{1-\beta_1^t} \cdot \frac{1}{\sqrt{\hat{v}_t} + \varepsilon} = \\ = w_{t-1} - \eta_t \cdot \frac{\beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot (\nabla_w L(w_{t-1}) + \lambda \cdot w_{t-1})}{1-\beta_1^t} \cdot \frac{1}{\sqrt{\hat{v}_t} + \varepsilon} =\\=w_{t-1}\cdot \left(\underbrace{1}_{\text{вектор единиц}} - \frac{\eta_t \cdot \lambda\cdot (1-\beta_1)}{1-\beta_1^t} \cdot \underbrace{\frac{1}{\sqrt{\hat{v}_t}+\varepsilon}}_{(*)}\right) -\dots
\end{multline*}

$(*) \Rightarrow$ регуляризация работает по-разному — разные веса будут по-разному затухать.
Штраф делится на $\sqrt{\hat{v}_t}$, поэтому веса с большими градиентами затухают слабее, а веса с маленькими — сильнее. Регуляризация получается не той, которую мы задумывали.


Выпишем уравнения для AdamW[^statia]

$$
\begin{cases}
g_t = \nabla_w L(w_{t-1}) \\
m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2\\
\hat{m}_t = \frac{1}{1-\beta_1^t} \cdot m_t \\
\hat{v}_t = \frac{1}{1-\beta_2^t}v_t \\
w_t = (1-\eta_t\cdot\lambda)\cdot w_{t-1} -\eta_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
\end{cases}
$$

В дефолтном Adam мы учитываем weight decay в $g_t$,  а в модификации AdamW — в самом шаге обновления весов $w_t$, отдельно от градиента и от адаптивного масштаба $\sqrt{\hat{v}_t}$, поэтому все веса затухают одинаково.

```


[^statia]: [DECOUPLED WEIGHT DECAY REGULARIZATION](https://arxiv.org/pdf/1711.05101.pdf)














