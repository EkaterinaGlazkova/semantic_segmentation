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