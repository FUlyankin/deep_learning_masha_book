# 3. Из формулы в картинку

У Маши есть два рекуррентных нейрона. Помогите ей изобразить их в виде вычислительных графов.

__а)__ Однонаправленный
    
\begin{equation*} 
	\begin{aligned}
	h_t =& f_h(b_h + W \cdot h_{t-1} + V \cdot x_t)\\
	y_t =& f_y(b_y + U \cdot h_t)
	\end{aligned}
\end{equation*} 


__б)__ Двунаправленный


\begin{equation*} 
	\begin{aligned}
	h_t =& f_h(b_h + W \cdot h_{t-1} + V \cdot x_t)\\
	s_t =& f_s(b_s + W' \cdot s_{t+1} + V' \cdot x_t)\\
	o_t =& b_y + U \cdot h_t + U' \cdot s_t \\
	y_t =& f_y(o_t)
	\end{aligned}
\end{equation*} 
