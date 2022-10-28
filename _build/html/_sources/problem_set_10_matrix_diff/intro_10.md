# Листочек 10: матричное дифФфФфФференцирование

<img src="../images/problem_set_10/img_tree.png" alt="triangle" height="70px" align="center">

```{epigraph}
It’s going down, I’m yelling timber, You better move, you better dance.

-- Джек и бобовый стебель (1890)
```

Эта часть виньетки необязательна для изучения. В ней мы подробно поговорим про матричные производные. С их помощью удобно заниматься оптимизацией, в том числе алгоритмом обратного распространения ошибки. 

Зачем нам нужно научиться искать матричные производные? **Машинное обучение -- это сплошная оптимизация.** В нём мы постоянно вынуждены искать минимум какой-нибудь функции потерь. Матричные производные довольно сильно в этом помогают.

Часть задач я взял из [задачинка Бориса Демешева.](https://github.com/bdemeshev/mlearn\_pro/blob/master/mlearn\_pro.pdf) Часть задач я взял из [теоретического дз ФКН.](https://github.com/esokolov/ml-course-hse/blob/master/2022-fall/homeworks-theory/homework-theory-02-derivatives.pdf) Часть задач я взял из семинара ШАД. Часть задач я придумал и добавил в книгу и в [конспекты ФКН.](https://github.com/esokolov/ml-course-hse/blob/master/2022-fall/seminars/sem03-vector-diff.pdf)  Можете почитать его. Там всё то же самое и даже больше. 


Также можно почитать книгу [The Matrix Cookbook]( https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) и воспользоваться [матричным калькулятором.](http://www.matrixcalculus.org/) 