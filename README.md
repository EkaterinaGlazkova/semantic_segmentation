# README #

### Основная информация ###

Имя студента: Глазкова Екатерина Васильевна

Группа: БПМИ-152 

Тема проекта: Семантическая сегментация изображений для автоматической разметки аэрофотоснимков

### Актуальность решаемой задачи ###

Семантическая сегментация изображения – разбиение изображения на сегменты (группы пикселей) и определение типа, к которому относится каждый сегмент. Цель сегментации – изменение изображения для последующего упрощения его анализирования. 

В рамках проекта рассматривается семантическая сегментация определенной группы изображений – аэрофотоснимков. Каждому из сегментов присваивается один из шести классов: здание, низкая растительность, дерево, автомобиль, водоем, непроницаемая поверхность. Решение этой задачи актуально при создании карт. 

Актуальность для студента: изучить методы работы с изображениями и методы машинного обучения и применить их при решении реальной задачи. При получении хорошего результата - поучаствовать в контесте по данной теме - ISPRS 2D Semantic Labeling Contest.

### Технологические решения ###

Для обучения используются аэрофотоснимки городской и загородной местности сделанные несколькими способами (цветные RGB изображения, карты высот, ИК-снимки), а так же уже размеченные снимки. При сегментации желателен учет различных видов фотоснимков. Например, ИК-снимки позволяют точнее определять растительность, а карты высот – здания. 

Первый этап сегментации изображения – пересегментация – выделение небольших однородных сегментов («суперпикселей»), которые точно принадлежат одному классу. Рассмотренные методы пересегментации: Meanshift, Efficient Graph-Based, Turbopixel и SLIC(Simple Linear Iterative Clustering). Для реализации был выбран метод SLIC. 

Одной из главных задач при реализации алгоритмов машинного обучения является выделение признаков. Здесь могут учитываться как самые простые признаки: цвет пикселя в различных цветовых моделях, высота объекта, определенная с помощью карты высот, ИК-излучение, - так и более сложные, например, основанные на анализе окружающих пикселей для выделения текстуры и поворота объекта. 

В ходе проекта будут рассмотрены методы классификации: Random Forest, SSVM, ANN-CNN-FCN; и методы обработки изображений: CRF/FCRF, CNN/FCN. 

Используемые языки программирования: C++ - для вычислений, подсчета точности работы алгоритма, Python – для реализации алгоритмов машинного обучения. 

Используемые библиотеки: OpenCV.


### Реализованная функциональность и описание (на 25.03) ###


Все реализации и модели представлены в двух форматах - файл jupyter-notebook и файл python. Первый тип предоставляет возможность также посмотреть результаты.

Реализации проекта:

**Реализация первая.**

Обучение Random Forest на пикселях с использованием 9 признаков. (Файл RFC_on_pixels.ipynb, результаты там же)

Входные параметры:

    List LABELED_PICS - номера изображений, для которых известно ground truth и которые будут входить в обучающую выборку 
    List TEST_PICS - номера изображений для тестирования
    MAIN_PATH - путь к папке ISPRS_semantic_labeling_Vaihingen 
    TREES_NUM, JOBS_NUM - параметры RandomForestClassifier, количество деревьев и количество потоков при работе

Выходные данные:

    Локально, в текущую директорию сохраняется файл с результатом классификации. На стандартный поток выводятся два изображения: полученное в результате классификации и ground truth (эталонное разбиение). Также выводится значение метрики f1 для полученного результата.

Функциональность и описание работы модуля: 

    Для каждого пикселя определяется значения цвета в 3-х цветовых моделях (RGB, HSV, LAB) и метка разбиения из ground truth. Далее по полученным значениям признаков(всего 9) и полученным меткам происходит обучение Random Forest Classifier. После чего для каждого из тестируемых изображений происходит классификация и вывод полученных результатов. 
    Обучение Random Forest на пикселях с использованием 9 признаков: по три признака из каждой цветовой модели RGB, HSV, LAB. Были выбраны именно эти признаки, так как они одни из самых простых для получения, однако вполне неплохо позволяют оценивать пиксели. 

Точность реализации

    При обучении на двух картинках(34 и 37, город Vaihingen) удалось добиться следующей точности (f1 score):

        1 снимок - 0.66238863135663095
        7 снимок - 0.63931156074982709
        15 снимок - 0.62523955073254323

Причины плохой точности и возможности улучшения реализации: 

1. Маленькая выборка для обучения
2. Неудачный подбор выборки для обучения. На выбранных снимках по большей части только дома, деревья и дорога, можно было бы выбрать снимки, например, с водоемами.
3. Небольшой набор признаков.

**Реализация вторая.** 

Модуль 1. SLIC - разбиение RGB изображения на суперпиксели. (SLIC.ipynb)

Входные параметры:

    List SEGM_NUM_LIST - количество сегментов разбиения
    List IND_LIST - номера изображений для разбиения
    MAIN_PATH - путь к папке ISPRS_semantic_labeling_Vaihingen
    MAX_ITER - параметр алгоритма SLIC, максимальное количество итераций алгоритма кластеризации k-means
		
Изменение параметров происходит путем изменения констант в программе.

Выходные данные:
			
    Локально, в папку MAIN_PATH/Outputs/SLIC, сохраняются по два файла для каждого снимка и каждого количества сегментов: маска, в которой все пиксели, входящие в один суперпиксель помечены одним цветом; исходное изображение с отмеченными границами разбиения.

Функциональность модуля: 

    разбиение исходного изображения на суперпиксели для последующей сегментации.

		

Модуль 2. Оценка качества разбиения алгоритмом SLIC. (Segmentation_accuracy_loss.ipynb)

Входные параметры:

    LABEL_NUM - количество типов объектов при разбиении
    List SEGM_NUM_LIST - количество сегментов разбиения
    List IND_LIST - номера изображений, которые были разбиты на суперпиксели
    MAIN_PATH - путь к папке ISPRS_semantic_labeling_Vaihingen 

Выходные данные:

    Ошибка разбиения в пикселях и процентах.

Функциональность и описание работы модуля: 
    
    Программа присваивает каждому сегменту разбиения одну метку - ту, к которой относится наибольшее количество пикселей, попавших в сегмент. За каждый пиксель, относящийся к одной метке, но лежащий в сегменте с другой присвоенной меткой начисляется штраф. Вывод программы - количество таких пикселей(штраф) и из отношение к общему числу пикселей изображения.


Модуль 3. Получение признаков для сегментов, обучение и тестирование. (SLIC_RFC.ipynb)
		
Входные параметры:

    PICS_FEATURES_NUM = 10 - количество признаков одного пикселя
    LABEL_NUM - количество типов объектов при разбиении
    List SEGM_NUM_LIST - количество сегментов разбиения
    List IND_LIST - номера изображений, на которых будет происходить обучение
    List TEST_PICS - номера изображений для тестирования
    MAIN_PATH - путь к папке ISPRS_semantic_labeling_Vaihingen 
    TREES_NUM, JOBS_NUM - параметры RandomForestClassifier, количество деревьев и количество потоков при работе

Выходные данные:

    Локально, в папку MAIN_PATH/Outputs/RES, сохраняется файл с результатом классификации. На стандартный поток выводятся два изображения: полученное в результате классификации и ground truth (эталонное разбиение). Также выводится значение метрики f1 для полученного результата.

Функциональность и описание работы модуля: 
			
    Программа с помощью алгоритма поиска в глубину (DFS) обходит все сегменты, подсчитывает среднее значение и дисперсию каждого из 10 признаков (по три от RGB, HSV, LAB и DSM) для каждого сегмента. Также определяется метка каждого сегмента(по наибольшему количеству пикселей с одной меткой, метки пикселей нам известны из ground truth). Далее по полученным значениям признаков(всего 20) и полученным меткам происходит обучение Random Forest Classifier. После чего для каждого из тестируемых изображений происходит классификация и вывод полученных результатов. 

Точность реализации

    При обучении на 4 снимках (5, 23, 26, 37, город Vaihingen) с разбиением на 20000 суперпикселей удалось добиться следующей точности (f1 score):

        1 снимок - 0.532005131647
        7 снимок - 0.572288879787
        15 снимок - 0.669156583785


### План работы ###

Ноябрь

* реализация вычисления точностных характеристик по размеченному изображению

* изучение 4 основных методов пересегментации 

Декабрь

* реализация оценки качества пересегментации

* реализация построения графа соседства на суперпикселях

* выбор признаков

* изучение Random Forest 

Январь

* Рабочая реализация Random Forest(RF) обучения на суперпикселях 

Февраль – июнь

* Подробное изучение моделей и реализация и обучение выбранных моделей: Conditional Random Field (CRF), FCRF, CNN, FCN