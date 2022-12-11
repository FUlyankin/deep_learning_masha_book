# 8. Стрельба по ногам

Однажды на следующий день после жёсткой вечеринки в середине недели Миша принёс Маше несколько разных архитектур. Они выглядят довольно странно. Помогите Маше разобраться, что именно Миша сделал в них неправильно.[^shad]

__а)__ Миша решает задачу регрессии. Он предсказывает цены на недвижимость. На вход в нейросеть подаётся $39$ регрессоров. 

	model = Sequential()
	model.add(InputLayer([39])
	model.add(BatchNormalization())
	model.add(Dense(128, kernel_initializer=keras.initializers.zeros()))
	model.add(Dense(1))
	model.compile(optimizer='sgd', loss='mean_squared_error')


```{dropdown} Решение
- Инициализация нулями -- зло, веса будут обновляться на одинаковые величины, а мы хотим сделать нейроны непохожими друг на друга.
- В качестве базового оптимизатора хорошо бы выбрать Adam, это бесплатно и ускоряет сходимость сетки.
- В нормализации по батчам в самом начале нет ничего плохого, она работает как StandartScaler.
- Переход от 128 нейронов к 1 слишком резкий, хочется идти более плавно, хотябы по степени двойки.
- Возможно, что у линейных слоёв нет функции активации, обычно ReLU стоит как её дефолтное значение, но лучше это лишний раз проверить.

```

__б)__ Миша решает задачу классификации картинок размера $28 \times 28$ на $10$ классов. Каждое изображение принадлежит только к одному классу.

	model = Sequential()
	model.add(InputLayer([28, 28, 1]))
	model.add(Conv2D(filters=512, kernel_size=(10, 10)))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Activation('softmax'))
	model.add(Dropout(0.1))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	model.add(Dropout(0.1))
	model.compile(optimizer='rmsprop', loss='mean_squared_error')


```{dropdown} Решение
- Первый слой из 512 фильтров довольно бесполезен. Мы поначалу ищем простые паттерны и для их поиска должно хватать меньшего чесли фильтров. Есть смысл делать более большое число фильтров ближе к концу свёрточной сетки.
- У свёрток слишком большой размер. Обычно их пытаются делать максимально маленькими для экономии параметров. Поле обзора наращивают с помощью пулинга. 
- Софтмакс в середине в качестве функции активации -- очень неудачный выбор. Он ведёт себя как мягкий arg max и приводит к срабатыванию только одного нейрона из всего слоя. 
- Дропаут после предсказания вероятностей уничножит нейросеть. Часть выходов будет зануляться, вероятности не будут давать в сумме единицу и кросс-энтропия будет улетать в бесконечность.

```

__в)__ Решается задача классификация картинок размера  $100 \times 100$ на $10$ классов. Каждое изображение принадлежит только к одному классу.

	model = Sequential()
	model.add(InputLayer([100, 100, 3]))
	for filters in [32, 64, 128, 256]:
		model.add(Conv2D(filters, kernel_size=(5, 5)))
		model.add(Conv2D(filters, kernel_size=(1, 1)))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	model.compile(optimizer='adam', loss='sparse_categorical_accuracy')

```{dropdown} Решение
Эта сетка не соберётся, потому что у свёрток нет дополнения (padding). У картинки не хватит пикселей и мы получим на выходе ошибку. 

```


__г)__ Миша решает задачу классификации RGB-изображения  размера $100 \times 100$ на $10$ классов. Каждое изображение принадлежит только к одному классу.

	model = Sequential() 
    model.add(InputLayer([100, 100, 3]))
        
    model.add(Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="glorot_uniform")) 
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
        
    model.add(Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="glorot_uniform")) 
    model.add(Activation('relu'))
        
    model.add(Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer="glorot_uniform"))
    model.add(Conv2D(filters=32, kernel_size=(1, 1), kernel_initializer="glorot_uniform"))
    model.add(MaxPool2D(pool_size=(10, 10)))
        
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(rate=1))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.add(Dropout(rate=0.5))


```{dropdown} Решение

Проблемы:

- 512 фильтров размера 3x3 в первом слое -- очень много. Как правило, для квадрата 3x3 пикселя кодируются 32-64 фильтров.
- Уменьшение числа фильтров вглубь по сети -- не очень эффективно, сеть "теряет capacity" -- последующие слои могут не уместить всю полезную информацию из предыдущих. Лучше начать с малого и увеличивать.
- Пулинг 10x10 -- огромный размер, изображение станет меньше размера фильтра. Да и в целом это неэффективно.
- Неправильная инициализация для релу, нужна инициализация Хе.
- Свёртка 1x1 -- вообще она имеет смысл, но в коде перед ней нет нелинейности, поэтому это лишнее линейное преобразование.
- Дропаут на вероятности в самом конце - сломает функцию потерь и будет мешать обучаться хоть чему-то.
- Дропаут равный единице в середине убьёт сетку.
- Нет активации между линейными слоями (обычно она в дефолтных параметрах, но тем не менее).
- Не ошибка, но можно несколько оптимизировать вычисления, если переставить ReLU после Max Pooling, а результат будет эквивалентный.
- На последнем слое должен быть softmax, так как мы работаем с 10 классами.
```


[^shad]: задача составлена по мотивам [семинара из ШАД.](https://github.com/yandexdataschool/Practical_DL/tree/master/week03_convnets/other_frameworks)

