# 8. Минималочка

Шестилетняя сестрёнка ворвалась в квартиру Маши и разрисовала ей все обои.

```{figure} ../images/problem_set_01/img08_oboi.png
---
width: 95%
name: oboi
---
```

Маша по жизни оптимистка. Поэтому она увидела не дополнительные траты на ремонт, а четыре задачи классификации. И теперь в её голове вопрос, сколько минимально нейронов нужно, чтобы эти задачи решить? 

```{dropdown} Решение

__1)__ Нам надо выделить треугольник. Всё, что внутри будет относится к первому классу. Получается, что на первом слое надо три нейрона. Каждый из них настроим так, что если мы попадаем внутрь треугольника, он выдаёт $1$. Тогда на втором слое будет достаточно одного нейрона, который удостоверится, что все три результата с первого слоя оказались равны $1$. Вернитесь к предыдущей задаче, сходите на сайт с демкой и постройте оптимальную нейросетку.

<img src="../images/problem_set_01/img08_triangle.png" alt="triangle" width="95%" align="center">

__2)__ Первый слой должен построить нам три линии. Это три нейрона. Второй слой должен принять решение в какой из полос мы оказались. Будем считать, что если мы попали направо, нейрон выдаёт единицу. Если мы попали налево, ноль. В качестве функции активации используем единичную ступеньку. 

<img src="../images/problem_set_01/img08_krest.png" alt="krest" width="95%" align="center">

Вопрос в том, хватит ли нам на втором слое одного нейрона для того, чтобы обработать все четыре возможные ситуации. Нам нужно, чтобы выполнялись следующие условия

\begin{equation*}
    \begin{cases} 
    & f(1 \cdot w_1 + 1 \cdot w_2 + 1 \cdot w_3) = 1 \\ 
    & f(1 \cdot w_1 + 0 \cdot w_2 + 0 \cdot w_3) = 0 \\
    & f(1 \cdot w_1 + 1 \cdot w_2 + 01 \cdot w_3) = 0 \\
    & f(0 \cdot w_1 + 0 \cdot w_2 + 0 \cdot w_3) = 0 \\
    \end{cases}.
\end{equation*}

Для того, чтобы со вторым уравнением всё было хорошо, возьмём $w_1 = 1.$ Тогда вес $w_2$ надо взять отрицательным, а $w_3$ положительным, например, $w_2 = -2$, а $w_3 = 4$. Тогда один нейрон на внешнем слое решит нашу задачу. Выходит, что всего надо задействовать 4 нейрона. 

__3)__ Оценим число нейронов сверху. Перед нами две $XoR$ задачи, которые лежат рядом с друг-другом. Для решения каждой надо $3$ нейрона. Чтобы объединить получившиеся решения нужен ещё один нейрон. Получается трёхслойная сетка с $7$ нейронами.

<img src="../images/problem_set_01/img08_hard.png" alt="triangle" width="70%" align="center">

Если мы попробуем подойти к задаче также, как в предыдущем пункте, на втором слое мы получим несовместимую систему из уравнений. То есть третьего слоя точно не избежать. 

Можно первым слоем построить $3$ линии, вторым решить задачу из предыдущего пункта, а на третьем добавить информацию о том, выше горизонтальной линии мы оказались или ниже. Тогда мы потратим $6$ нейронов. Нейросетка получится неполносвязной. 

__4)__ Думайте, рассуждайте, а автор умывает руки.

```