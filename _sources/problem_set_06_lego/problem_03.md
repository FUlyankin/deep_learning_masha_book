# 3. Weight Decay

В случае $l_2$-регуляризации, к базовой функции потерь добавляют дополнительное слагаемое и вместо функционала

\[
L(w) = \frac{1}{n} \cdot \sum\limits_{i=1}^{n} \nabla_w L(y_i, x_i, w)
\]

оптимизируют функционал 

\[
Q_\lambda(w) = L(w) + \frac{1}{2}\lambda \cdot ||w||^2_2.
\]

Будем считать, что регуляризатор наложен на все веса нейронной сети. Обычно регуляризатор добавляют к функции потерь, чтобы избежать переобучения. Градиентный спуск можно переписать с учётом регуляризатора немного в другом виде. Такой вид называется **weight decay.** 

В пакетах для обучения нейронных сеток у оптимизаторов обычно есть такой параметр. Давайте проделаем это переписывание для нескольких разных градиентных спусков. 

__а)__ Выпишите шаг momentum-SGD для такой модели. Выразите получившийся шаг в виде 

\[
w_t = g(\lambda) \cdot w_{t-1} - \eta_t \cdot h(\nabla_w L(w_{t-1}))
\]


```{dropdown} Решение
Один шаг momentum-SGD выглядит как

\[
\begin{cases}
	g_t = \nabla_w Q(w_{t-1}) + \lambda \cdot w_{t-1} \\
	m_t = \mu \cdot m_{t-1} + g_t\\
	w_t = w_{t-1} - \eta_t \cdot m_t \\
\end{cases}
\]

Подставим первую строку во вторую, а вторую в третью

\begin{multline*}
w_t = w_{t-1} - \eta_t \cdot (\mu \cdot m_{t-1} + \nabla_w L(w_{t-1}) + \lambda \cdot w_{t-1}) = \\ =\alert{\underbrace{(1-\eta_t\cdot \lambda)}_{<1}}\cdot w_{t-1} - \eta_t\cdot(\mu\cdot m_{t-1} + \nabla_w L(w_{t-1}))
\end{multline*}

Получается, что когда мы добавляем к модели $l_2$ регуляризацию, мы делаем каждый шаг градиентного спуска по старому градиенту без регуляризатора, но из новых весов. Мы сдвигаем старые веса на какую-то константу и движемся из неё. Этот параметр в оптимизиторах называется weight decay. Обычно при обучении нейронных сетей вместо регуляризации используют его. 

```

__б)__ Выпишите шаг Adam для такой модели. Выразите получившийся шаг в виде 

\[
w_t = g(\lambda) \cdot w_{t-1} - \eta_t \cdot h(\nabla_w L(w_{t-1}))
\]

```{dropdown} Решение
Один шаг Adam выглядит как 

\[
\begin{cases}
	g_t = \nabla_w Q(w_{t-1}) + \lambda \cdot w_{t-1} \\
	m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
	v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2\\
	\hat{m}_t = \frac{1}{1-\beta_1^t} \cdot m_t \\
	\hat{v}_t = \frac{1}{1-\beta_2^t}v_t \\
	w_t = w_{t-1} -\eta_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
\end{cases}
\]

Сделаем все подстановки и получим 

\[
	\Rightarrow w_t = w_{t-1} - \eta_t \cdot \frac{m_t}{1-\beta_1^t} \cdot \frac{1}{\sqrt{\hat{v}_t} + \varepsilon} = \\ = w_t = w_{t-1} - \eta_t \cdot \frac{\beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t}{1-\beta_1^t} \cdot \frac{1}{\sqrt{\hat{v}_t} + \varepsilon} = \\ =w_t = w_{t-1} - \eta_t \cdot \frac{\beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot (\nabla_w Q(w_{t-1}) + \lambda \cdot w_{t-1})}{1-\beta_1^t} \cdot \frac{1}{\sqrt{\hat{v}_t} + \varepsilon} =\\=w_{t-1}\cdot \left(\underbrace{1}_{\text{вектор единиц}} - \frac{\eta_t \cdot \lambda\cdot (1-\beta_1)}{1-\beta_1^t} \cdot \underbrace{\frac{1}{\sqrt{\hat{v}_t}+\varepsilon}}_{(*)}\right) -\dots
\]

$(*) \Rightarrow$ регуляризация работает по-разному — разные веса будут по-разному затухать
В случае с Adam мы слишком хорошо оптимизируемся $\Rightarrow$ обязательно переобучаемся


Выпишем уравнения для AdamW


\begin{cases}
g_t = \nabla_w Q(w_{t-1}) \\
m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2\\
\hat{m}_t = \frac{1}{1-\beta_1^t} \cdot m_t \\
\hat{v}_t = \frac{1}{1-\beta_2^t}v_t \\
w_t = (1-\eta_t\cdot\lambda)\cdot w_{t-1} -\eta_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
\end{cases}

В дефолтном Adam мы учитываем weight decay в $g_t$,  а в модификации AdamW — в 

```


[^statia]: [DECOUPLED WEIGHT DECAY REGULARIZATION](https://arxiv.org/pdf/1711.05101.pdf)














