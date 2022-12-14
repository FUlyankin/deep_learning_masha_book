# 9. Скользящее среднее

Временные ряды иногда сглаживают с помощью скользящего среднего. Этот процесс можно рассматривать как одномерную свёртку. Как выглядит ядро такой свёртки? Какой физический смысл стоит за размером такой свёртки и дополнением нулями? Опишите как она работает. 


```{dropdown} Решение
Разберем скользящее среднее на примере. Пусть у нас есть временной ряд, состоящий из 6 значений: $(x_1, x_2, x_3, x_4, x_5, x_6)$.

Мы хотим сгладить наши данные окном размера $k$, то есть применяем одномерную свертку: $(\frac{1}{k}, \frac{1}{k}, \ldots, \frac{1}{k})$. Возьмём $k = 3$, тогда ядро свёртки примет вид $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3}).$ Будем считать якорем свёртки центральный элемент. 

Применим свёртку к нашему ряду без дополнения нулями, получим 

$$
(\frac{x_1 + x_2 + x_3}{3}, \frac{x_2 + x_3 + x_4}{3}, \frac{x_3 + x_4 + x_5}{3}, \frac{x_4 + x_5 + x_6}{3}).
$$

Дополнение нулями может помочь нам сохранить концы нашего ряда

$$
(0, x_1, x_2, x_3, x_4, x_5, x_6, 0).
$$

Пробуем применить свёртку 

$$
(\frac{0 + x_1 + x_2}{3}, \frac{x_1 + x_2 + x_3}{3}, \frac{x_2 + x_3 + x_4}{3}, \frac{x_3 + x_4 + x_5}{3}, \frac{x_4 + x_5 + x_6}{3}, \frac{x_5 + x_6 + 0}{3}).
$$

В общем случае функция для скользящего среднего, записанная в терминах $1D$ свёртки, будет выглядеть как 

	def moving_average(x, k):
    	return np.convolve(x, np.ones(k), 'valid') / k
```
