# Тятя! Тятя! Нейросети заменили продавца!


```{epigraph}
Я попала в сети, которые ты метил, я самая счастливая на всей планете.

-- Юлианна Караулова
```

## Abstract

В этой книге собрана коллекция ручных задачек о нейросетях. Вместе с Машей можно попробовать по маленьким шажкам, с ручкой и бумажкой, раскрыть у себя в теле несколько чакр и глубже понять модели глубокого обучения.


## Введение

Однажды Маша услышала про какой-то Машин лёрнинг. Маша сразу же смекнула, что именно она --- та самая Маша, кому этот лёрнинг должен принадлежать. Ещё она смекнула, что если хочет владеть лёрнингом по праву, ни одна живая душа не должна сомневаться в том, что она шарит. Поэтому она постоянно изучает что-то новое. 

Её друг, Миша, захотел стать адептом Машиного лёрнинга, и спросил её, как можно за вечер зашарить алгоритм обратного распространения ошибки. Тогда Маша открыла свою коллекцию учебников по глубокому обучению. В каких-то из них авторы писали, что ей никогда не придётся реализовывать алгоритм обратного распространения ошибки, а значит и смысла тратить время на его формулировку нет[^mynote1]. В каких-то она находила слишком сложную математику, с которой за один вечер точно не разберёшься[^mynote2]. В каких-то алгоритм описывался понятно, но оставалось много недосказанностей[^mynote3]. 

Маша решила, что для вечерних разборок нужно что-то более инфантильное. Тогда она поскребла по лёрнингу и собрала коллекцию ручных задачек. Новые адепты Машиного лёрнинга могут прорешать её и раскрыть свои диплернинговые чакры.


## Структура книги

Книга состоит из нескольких листочков с задачками. В каждом листочке раскрывается какая-то отдельная тема, связанная с нейросетями. Первые три листочка --- **это база.** В первом мы поймём, что нейросеть --- всего лишь функция. Во втором разберёмся, как работает градиентный спуск. В третьем, аккуратно, руками, на простых примерах выведем бэкпроп. 

Изначально я хотел ограничиться этим тремя листочками. Однако мой юношеский максимализм не даёт мне это сделать. Я прочитал курс по нейросеткам и написал под него [ещё кучу листочков.](https://github.com/FUlyankin/deep_learning_tf) 

Поэтому вторая часть этой книги состоит из ещё $N$ листочков, в которых информация подаётся чуть хаотичнее, чем в первых трёх. А также там могу быть пробелы в решениях. 

Четвёртый листочек расскажет про функции активации, функции потерь и то, как можно умелым движением рук адаптировать нейронки под разные задачи. Пятый листочек погрузит нас в свёрточные сети. Шестой листочек расскажет про специфические слои и красивые инженерные решения. Седьмой листочек погрузит в рекуррентные сети. В восьмом листочке речь пойдёт про векторные представления и автокодировщики. Девятый листочек расскажет про трансформеры. Десятый листочек будет немного в стороне от нейронок и научит нас брать матричные производные. 

**P.S.** По ссылке лежит курс по нейросетям на tensorflow, но он переедет на pytorch. Я поставил [не на ту лошадь и проиграл.](https://t.me/gonzo_ML/1026) 

## Про пулл-реквесты

Автор -- невнимательный балбес. В книге могут быть ошибки. Если вы нашли такую и хотите исправить, вы всегда можете сделать пулл-реквест. Автор будет только рад. 

Если для какой-то задачки не хватает решения и вы хотите дописать его, также делайте пулл-реквест. Если вам кажется, что в книжке не хватает чего-то клёвого, тоже смело пишите автору.  


## Об авторе

У автора особо нет регалий, но он преподаёт разные штуки связанные с ML и много им занимается на практике. Преподаёт, в том числе, в Вышке на ФКН. А заниматеся на практике, в том числе, в Яндексе. 

Подписывайтесь [на мой лайфстайл канал в телеге,](https://t.me/ppilif_chanel) ищите свежие стримы с лекций [на youtube,](https://www.youtube.com/channel/UCO9ZmLCnIh669tYaPRsg4_w/playlists) а все материалы [на гитхабе.](https://github.com/FUlyankin)Связаться со мной можно, написав в телеграм [@ppilif,](https://t.me/ppilif) либо на почту filfonul@gmail.com

Если вы модный издатель и хотите меня опубликовать и заплатить мне кучу денег, то я от вас тащусь, пишите мне скорее. 


## О читателе

Ой, а что это за сладкий пирожочек зашёл почитать про нейросети? 💜 💜 💜

Если бы кто-нибудь когда-нибудь решил бы сделать нейросеть на основе самых умных, начитанных и смекалистых людей, тебя несомненно надо было бы взять в обучающую выборку! 


## О логотипе

Это книга про нейросети. Поэтому логотип к ней был сгенерирован с помощью нейросети [DALL-E 2.](https://openai.com/dall-e-2/) Вы можете заметить его в левом верхнем углу. Угадайте, что на нём изображено. Скажите это вслух. 


## Оглавление


```{tableofcontents}
```

[^mynote1]: Франсуа Шолле, Глубокое обучение на Python.
[^mynote2]: Goodfellow I., Bengio Y., Courville A. Deep learning. – MIT press, 2016.
[^mynote3]: Николенко С., Кадурин А., Архангельская Е. Глубокое обучение. Погружение в мир нейронных сетей - Санкт-Петербург, 2018.

