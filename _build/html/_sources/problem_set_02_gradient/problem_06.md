# 6. Adam и его формула*

Маша обучает модель с одним параметром $w$. Она решила упростить Adam и использовать для пересчёта формулы

\begin{equation*}
    \begin{aligned}
        &h_t = \beta_1 \cdot h_{t-1} + (1 - \beta_1) \cdot g_t \\
        &v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \\
        &w_t = w_{t-1} - \frac{\eta_t}{\sqrt{v_t + \varepsilon}} \cdot h_t.
    \end{aligned}
\end{equation*} 

Докажите ей, что в таком случае оценка градиента окажется смещена.


```{dropdown} Решение
Мы хотим, чтобы первые два уравнения давали нам несмещённые оценки для первого и второго момента $\mathbb{E}(h_t) = \mathbb{E}(g_t)$ и $\mathbb{E}(v_t) = \mathbb{E}(g^2_t).$ Давайте убедимся, что формулы Маши дают смещение. Для начала избавимся от рекуррентности в формуле 

\begin{equation*} 
    \begin{aligned} 
        & h_0 = 0 \\
        & h_1 = \beta_1 \cdot h_0 + (1 - \beta_1) g_1 = (1 - \beta_1) g_1 \\ 
        & h_2 = \beta_1 \cdot h_1 + (1 - \beta_1) g_2 = \beta_1 (1 - \beta_1) g_1 + (1 - \beta_1) g_2 \\
        & h_3 = \beta_1 \cdot h_2 + (1 - \beta_1) g_3 = \beta^2_1 (1 - \beta_1) g_1 +  \beta_1 (1 - \beta_1) g_2 + (1 - \beta_1) g_3  \\
        & \ldots \\
        & h_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i
    \end{aligned} 
\end{equation*}

Теперь найдём математическое ожидание 

\begin{multline*}
    \mathbb{E}(h_t) = \mathbb{E}\left( (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right) = \mathbb{E}(g_i) \cdot (1 - \beta_1) \cdot \sum_{i=1}^t \beta_1^{t-i} = \\ = \mathbb{E}(g_i) \cdot (1 - \beta_1) \cdot \frac{1 - \beta_1^t}{1 - \beta_1} = (1 - \beta_1^t) \cdot \mathbb{E}(g_i).
\end{multline*}
    
Получается, что нашу формулы пересчёта надо дополнительно поделить на $1 - \beta_1^t.$ По аналогии можно провести рассуждения для второго уравнения. 


```
