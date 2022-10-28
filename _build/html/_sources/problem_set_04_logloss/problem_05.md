# 5. ReLU и её друзья

Функция $f(t) = ReLU(t) = \max(t, 0)$ называется ReLU.


__а)__ Как выглядит производная ReLU? ыпишите формулы для forward pass и backward pass через слой с ReLU. Возможен ли в случае ReLU параличь нейросети? Если да, как его избежать? 

```{dropdown} Решение

```

__б)__ Как у ReLU дела с центированием относительно нуля?

```{dropdown} Решение

```

Функция $f(t) = \begin{cases} t, t \ge 0 \\ \alpha \cdot t, t < 0  \end{cases},$ называется Leaky ReLU.


__в)__ Чем такая функция активации лучше, чем ReLU? 

```{dropdown} Решение


```

Функция $f(t) = \begin{cases} t, t \ge 0 \\ \alpha \cdot (e^t - 1), t < 0  \end{cases},$ называется ELU.


__г)__ Как выглядит шаг обратного распространения ошибки через ELU? 

```{dropdown} Решение

```

__д)__ Чем такая функция активации лучше, чем ReLU? 

```{dropdown} Решение

```

Функция $f(t) = \begin{cases} t, t \ge 0 \\ \alpha \cdot (e^t - 1), t < 0  \end{cases},$ называется SELU (Scaled Exponential Linear Units activation)

__е)__ Как думаете, зачем нужен параметр $\lambda$ и почему эта функция называется нормализованной (scaled)? 

```{dropdown} Решение

```

__ж)__ В 2017 году учёные из Google Google Brain хитрым автоматическим поиском на основе RNN нашли функцию активации Swish, работающую лучше ReLU

сюда ченить про их взаимосвязь + ченить про Mish

```{dropdown} Решение

```

А что на практике? 




