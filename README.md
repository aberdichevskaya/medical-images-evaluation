## Оценка качества разметки медицинских данных
Решение задачи "Разработка инструмента оценки качества работы алгоритмов разметки медицинских изображений" хакатона "Лидеры Цифровой Трансформации" команды trying to pretend.

### Структура проекта
- Папка model: код решения и пример использования.
  - `evaluator.py` модуль с обученной моделью.
  - `example.ipynb`: пример использования модуля `evaluator.py` для оценки разметки датасета.
  - `clean-solution.ipynb` финальное обучение модели и сохранение файлов.
  - `solution.ipynb` код и описание методов, применённых в процессе поиска лучшей модели и подбора гиперпараметров, содержит также не вошедшие в финальную версию, но потенциально полезные идеи.
  - папка metrics: реализация метрик для сравнения двух изображений.
- Папка application: клиент с графическим интерфейсом.

### Основные технологии
- Python 3 + библиотеки numpy, pandas, scikit-learn
- PyQt

### Руководство по использованию библиотеки
Создание объекта модели: `model = Evaluator()`.
Обучение модели: `model.fit(predicted, expert)`, где `predicted` -- оцениваемая разметка, `expert` -- правильная разметка, ожидаемые тип данных: `np.ndarray`.
Получение значения метрик для всех объектов выборки: `model.evaluate()`.

## О решении
Для поиска различий между автоматической и экспертной разметками применяются метрики:
- Standard surface distance
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?AvD=\frac{1}{Y}\sum_{y\in%20Y}D_X(y)" /> 
</p>

- Symmetric surface distance
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?AvSD=\frac{1}{|X|+|Y|}\left(\sum_{x\in%20X}D_Y(x)+\sum_{y\in%20Y}D_X(y)\right)=\frac{|Y|AvD_Y(X,%20Y)+|X|AvD(Y,%20X)}{|X|+|Y|}" /> 
</p>

- Volume overlap error
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?VOE=100\times\left(1-\frac{|X\cap%20Y|}{|X|+|Y|}\right)" /> 
</p>

- Relative volume difference
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?RVD%20=%20100\times\frac{|X|-|Y|}{|Y|}" /> 
</p>

- Dice coefficient
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?DICE%20=%20\frac{2|X\cap%20Y|}{|X|+|Y|}" /> 
</p>

- Метрика Хаусдорфа

и их модификации.


Алгоритм оценки разметки:
1. Составление признаков из расстояний между разметкой эксперта и оцениваемой разметкой, заполнение возникших из-за возможного деления на 0 признаков `-1` или `0` в случае отсутствия разметки.
2. Применение StandardScaler
3. Получение оценки как взвешенного среднего по результату работы случайного леса, SVC с rbf ядром и логистической регрессии.

### Литература
- SergioVera, DeboraGil и др. “Medial structure generation for registration of anatomical structures”, Skeletonization, Chapter 11 (2017)
- Ramprasaath R. Selvaraju и др. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization”, IEEE (2017).
- R. Padilla, S. L. Netto и E. A. B. da Silva, "A Survey on Performance Metrics for Object-Detection Algorithms", IWSSIP (2020)
Dinu Dragan, и Dragan Vojo Ivetic, “Region Marking Software Tool for Medical Images”, eTELEMED (2012)
- Реализация метрик https://github.com/deepmind/surface-distance
