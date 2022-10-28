# 8. Стрельба по ногам


Миша принёс Маше несколько разных архитектур. Они выглядят довольно странно. Помогите Маше разобраться, что именно Миша сделал неправильно. 


% \begin{enumerate} 

% \item Решается задача регрессии, предсказываются цены на недвижимость. На вход в сеть подаётся $39$ регрессоров. 

% \begin{alltt}
% model = Sequential() \\
% model.add(InputLayer([39]) \\
% model.add(BatchNormalization()) \\
% model.add(Dense(128, kernel\_initializer=keras.initializers.zeros())) \\
% model.add(Dense(1)) \\
% model.compile(optimizer='sgd', loss='mean\_squared\_error')
% \end{alltt}

% \item Решается задача классификация картинок размера $28 \times 28$ на $10$ классов.

% \begin{alltt}
% model = Sequential() \\
% model.add(InputLayer([28, 28, 1])) \\
% model.add(Conv2D(filters=512, kernel\_size=(3, 3))) \\
% model.add(Activation('relu')) \\
% model.add(MaxPool2D(pool_size=(2, 2))) \\
% model.add(Flatten()) \\
% model.add(Dense(100)) \\
% model.add(Activation('softmax')) \\
% model.add(Dropout(0.1)) \\
% model.add(Dense(10)) \\
% model.add(Activation('softmax')) \\
% model.add(Dropout(0.1)) \\
% model.compile(optimizer='rmsprop', loss='mean\_squared\_error')
% \end{alltt}


% \item Решается задача классификация картинок размера  $100 \times 100$ на $10$ классов. 
% \begin{alltt}
% model = Sequential() \\
% model.add(InputLayer([100, 100, 3])) \\

% for filters in [32, 64, 128, 256]:
%     model.add(Conv2D(filters, kernel\_size=(5, 5)))
%     model.add(Conv2D(filters, kernel\_size=(1, 1)))
%     model.add(MaxPooling2D(pool\_size=(3, 3)))
%     model.add(Activation('relu'))
%     model.add(BatchNormalization())
    
% model.add(Flatten())
% model.add(Dense(100, activation='relu'))
% model.add(Dropout(0.5))
% model.add(Dense(10, activation='softmax'))

% model.compile(optimizer='adam', loss='sparse\_categorical\_accuracy')
% \end{alltt}

% \end{enumerate} 
\end{problem}

