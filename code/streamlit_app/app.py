# импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import streamlit as st

# импортируем классы ETNA
from etna.datasets import TSDataset
from etna.models import (CatBoostPerSegmentModel, 
                         CatBoostMultiSegmentModel)
from etna.transforms import (
    LogTransform,
    TimeSeriesImputerTransform,
    LinearTrendTransform,
    LagTransform,
    DateFlagsTransform,
    FourierTransform,
    SegmentEncoderTransform,
    MeanTransform
)
from etna.metrics import SMAPE
from etna.pipeline import Pipeline
from etna.analysis import (plot_forecast,
                           plot_backtest)

# отключаем вывод ненужного предупреждения
st.set_option("deprecation.showPyplotGlobalUse", False)

# задаем название приложения
st.title("Прогнозирование потребления электроэнергии по 4 сегментам")

# задаем заголовок раздела
st.header("Исходные данные")

# задаем поясняющий текст
st.write(
    "Данные должны быть в расплавленном, длинном формате. "
    "Столбец с метками времени должен называться **timestamp**. "
    "Столбец с сегментами должен называться **segment**. "
    "Столбец с зависимой переменной должен называться **target**.")

# поле загрузки данных
data_file = st.file_uploader("Загрузите ваш CSV-файл")

# загружаем данные или останавливаем приложение
if data_file is not None:   
    data_df = pd.read_csv(data_file)
else:
    st.stop()
    
# выводим первые 10 наблюдений
st.dataframe(data_df.head(10))

# переводим данные в формат ETNA
df = TSDataset.to_dataset(data_df)
ts = TSDataset(df, freq="D")

# задаем заголовок раздела
st.header("Визуализация рядов")

# радиокнопка - визуализировать ряды или нет
visualize = st.radio(
    "Визуализировать ряды?",
    ("Нет", "Да"))

# если не нужно визуализировать, ничего не делаем,
# а если нужно, строим график
if visualize == "Нет":
    pass
else:
    st.pyplot(ts.plot())

# задаем заголовок раздела
st.header("Горизонт прогнозирования")

# ввод горизонта прогнозирования
HORIZON = st.number_input(
    "Задайте горизонт прогнозирования", 
    min_value=1, value=14)

# разбиваем набор на обучающую и тестовую выборку
train_ts, test_ts = ts.train_test_split(
    test_size=HORIZON)

# взглянем временные рамки выборок
st.write(f"временные рамки обучающей выборки: "
         f"{train_ts.index[0].strftime('%Y-%m-%d')} - "
         f"{train_ts.index[-1].strftime('%Y-%m-%d')}")
st.write(f"временные рамки тестовой выборки: "
         f"{test_ts.index[0].strftime('%Y-%m-%d')} - "
         f"{test_ts.index[-1].strftime('%Y-%m-%d')}")

# задаем заголовок раздела
st.header("Преобразования зависимой переменной")

# задаем список классов, ответственных за преобразования
tf_classes_options = ["LogTransform", 
                      "TimeSeriesImputerTransform", 
                      "LinearTrendTransform"]

# создаем в боковой панели поле множественного выбора 
# классов, ответственных за преобразования, на выходе - 
# список выбранных классов, ответственных за преобразования
tf_classes_lst = st.sidebar.multiselect(
    "Список классов, создающих преобразования", tf_classes_options)

# пустой словарь для классов, ответственных за преобразования
tf_classes_dict = {}

# проверяем наличие соответствующего класса в списке выбранных
# классов, ответственных за преобразования, и обновляем словарь
if "LogTransform" in tf_classes_lst: 
    log = LogTransform(in_column="target")
    
    tf_classes_dict.update({"LogTransform": log})
    
if "TimeSeriesImputerTransform" in tf_classes_lst:
    
    imputer_title = (
        "<p style='font-family:Arial; color:Black; font-size: 18px;'" + 
        ">Выберите настройки для TimeSeriesImputerTransform</p>")
    st.markdown(imputer_title, unsafe_allow_html=True)
    
    # поле выбора - стратегия импутации пропусков
    strategy = st.selectbox(
    "Стратегия импутации пропусков",
    ["constant", "mean", "running_mean", 
     "seasonal", "forward_fill"])
    
    # два числовых ввода - ширина скользящего окна и сезонность
    window = st.number_input(
        "Введите ширину скользящего окна", min_value=-1)
    seasonality = st.number_input(
        "Введите длину сезонности", min_value=1)
       
    imputer = TimeSeriesImputerTransform(
        in_column="target", 
        strategy=strategy,
        window=window,
        seasonality=seasonality)
    
    tf_classes_dict.update({"TimeSeriesImputerTransform": imputer})
    
if "LinearTrendTransform" in tf_classes_lst: 
    detrend = LinearTrendTransform(in_column="target")
    
    tf_classes_dict.update({"LinearTrendTransform": detrend})

# формируем итоговый список значений из словаря, в который 
# положили выбранные классы, ответственные за преобразования
final_tf_classes_lst = list(tf_classes_dict.values())

# задаем заголовок раздела
st.header("Конструирование признаков")

# задаем список классов, ответственных за создание признаков
fe_classes_options = ["LagTransform", "MeanTransform", 
                      "DateFlagsTransform", 
                      "SegmentEncoderTransform"]

# создаем в боковой панели поле множественного выбора классов, 
# ответственных за создание признаков, на выходе - список 
# выбранных классов, ответственных за создание признаков
fe_classes_lst = st.sidebar.multiselect(
    'Список классов, создающих признаки', fe_classes_options)

# пустой словарь для классов, ответственных за создание признаков
fe_classes_dict = {}

# проверяем наличие соответствующего класса в списке выбранных
# классов, ответственных за создание признаков, и обновляем словарь
if "LagTransform" in fe_classes_lst:   
    lags_title = (
        "<p style='font-family:Arial; color:Black; font-size: 18px;'" + 
        ">Выберите настройки для LagTransform</p>")
    st.markdown(lags_title, unsafe_allow_html=True)
    
    # три числовых ввода - нижняя граница порядка лага, 
    # верхняя граница лага, шаг прироста порядка лага
    lower_limit = st.number_input("Нижняя граница порядка лага", 
                                  min_value=HORIZON)
    upper_limit = st.number_input("Верхняя граница порядка лага", 
                                  min_value=2 * HORIZON)
    increment = st.number_input("Шаг прироста порядка лага", 
                                min_value=int(np.sqrt(HORIZON)))   
    lags = LagTransform(in_column="target", 
                        lags=list(range(lower_limit, upper_limit, increment)), 
                        out_column="target_lag")
    
    fe_classes_dict.update({"LagTransform": lags})
    
if "MeanTransform" in fe_classes_lst:   
    means_title = (
        "<p style='font-family:Arial; color:Black; font-size: 18px;'" + 
        ">Выберите настройки для MeanTransform</p>")
    st.markdown(means_title, unsafe_allow_html=True)
    
    # поле числового ввода - количество скользящих окон
    means_number = st.number_input("Введите количество окон", min_value=1)
    # слайдер - настройка ширины конкретного скользящего окна 
    numbers = [st.slider(f"Введите ширину {i+1}-го окна", 
                         min_value=HORIZON, 
                         max_value=3 * HORIZON)
               for i in range(means_number)] 
    
    for number in numbers:
        fe_classes_dict.update({f"MeanTransform{number}": MeanTransform(
            in_column="target",
            window=number, 
            out_column=f"target_mean{number}")})

if "DateFlagsTransform" in fe_classes_lst:
    dateflags_title = (
        "<p style='font-family:Arial; color:Black; font-size: 18px;'" + 
        ">Выберите настройки для DateFlagsTransform</p>")
    st.markdown(dateflags_title, unsafe_allow_html=True)
    
    # флаги - календарные признаки
    day_number_in_week = st.checkbox(
        "Порядковый номер дня в неделе", 
        value=False)   
    day_number_in_month = st.checkbox(
        "Порядковый номер дня в месяце", 
        value=True)
    week_number_in_month = st.checkbox(
        "Порядковый номер недели в месяце", 
        value=False)  
    month_number_in_year = st.checkbox(
        "Порядковый номер месяца в году", 
        value=True)
    season_number = st.checkbox(
        "Порядковый номер сезона в году", 
        value=False)
    is_weekend = st.checkbox(
        "Индикатор выходного дня", 
        value=False)
    
    dateflags = DateFlagsTransform(
        day_number_in_week=day_number_in_week,
        day_number_in_month=day_number_in_month,
        week_number_in_month=week_number_in_month,
        month_number_in_year=month_number_in_year,
        season_number=season_number,
        is_weekend=is_weekend,
        out_column="date_flag")
    
    fe_classes_dict.update({"DateFlagsTransform": dateflags})

if "SegmentEncoderTransform" in fe_classes_lst:
    seg = SegmentEncoderTransform()   
    fe_classes_dict.update({"SegmentEncoderTransform": seg})

# формируем итоговый список значений из словаря, в который 
# положили выбранные классы, ответственные за создание признаков    
final_fe_classes_lst = list(fe_classes_dict.values())

# объединяем список классов, выполняющих преобразования, 
# и список классов, создающих признаки
final_classes_lst = final_tf_classes_lst + final_fe_classes_lst

# формируем список классов, создающих признаки по умолчанию,
# когда эти классы не заданы нами вручную
default_lags = LagTransform(
    in_column="target", 
    lags=list(range(HORIZON, 3 * HORIZON, HORIZON)), 
    out_column="target_lag")

default_dateflags = DateFlagsTransform(
    day_number_in_week=True,
    week_number_in_month=True, 
    month_number_in_year=True,
        out_column="date_flag")

default_classes_lst = [default_lags, default_dateflags]

# если классы, создающие признаки, не заданы вручную,
# формируем список классов по умолчанию, в противном
# случае используем объединенный список, созданный
# выше
if len(final_fe_classes_lst) == 0:
    transforms = default_classes_lst
else:
    transforms = final_classes_lst

# задаем заголовок раздела
st.header("Обучение базовой модели Catboost")

# поля числового ввода - значения основных
# гиперпараметров модели CatBoost
iterations = st.number_input(
    "Введите количество деревьев",
    min_value=1, max_value=2000, value=1000)
learning_rate = st.number_input(
    "Введите темп обучения", 
    min_value=0.001, max_value=1.0, value=0.03)
depth = st.number_input(
    "Введите максимальную глубину деревьев", 
    min_value=1, max_value=16, value=6)

# задаем выбор типа модели CatBoost с помощью поля
# одиночного выбора в боковой панели
catboost_model_type = st.sidebar.selectbox(
    "Какую модель CatBoost обучить?",
    ["PerSegment", "MultiSegment"])

if catboost_model_type == "PerSegment":
    # создаем модель CatBoostPerSegmentModel
    model = CatBoostPerSegmentModel(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth)
else:
    # создаем модель CatBoostPerSegmentModel
    model = CatBoostMultiSegmentModel(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth)
    
# радиокнопка - строить модель или нет
run_model = st.radio(
    "Обучить базовую модель?",
    ("Нет", "Да"))

# либо останавливаем приложение, либо строим модель
if run_model == "Нет":
    st.stop()
else:
    pass

# создаем конвейер
pipeline = Pipeline(model=model, 
                    transforms=transforms, 
                    horizon=HORIZON)
# обучаем конвейер
pipeline.fit(train_ts)
# получаем прогнозы
forecast_ts = pipeline.forecast()

# создаем экземпляр класса SMAPE
smape = SMAPE()
# вычисляем метрики по сегментам
smape_values = smape(y_true=test_ts, 
                     y_pred=forecast_ts)
# кладем метрики по сегментам в датафрейм 
# и даем имя столбцу с ними
smape_values = pd.DataFrame({"Прогнозы": smape_values})
# вычисляем среднее значение SMAPE
smape_mean = smape_values["Прогнозы"].mean()

# задаем заголовок раздела
st.header("Оценка качества прогнозов базовой модели - SMAPE")
# печатаем метрики по сегментам и среднее значение метрики
st.write(smape_values)
st.write("Среднее значение:", smape_mean)

# задаем заголовок раздела
st.header("Визуализация прогнозов базовой модели")

# слайдер - количество последних наблюдений в обучающей выборке
n_train_samples = st.slider(
    "N последних наблюдений в обучающей выборке", 
    min_value=3 * HORIZON)

# строим график прогнозов
st.pyplot(
    plot_forecast(
        forecast_ts=forecast_ts,
        test_ts=test_ts,
        train_ts=train_ts,
        n_train_samples=n_train_samples,
    )
)

# задаем заголовок раздела
st.header("Перекрестная проверка модели Catboost")

# поля одиночного выбора и числового ввода
# - настройки перекрестной проверки
mode = st.selectbox(
    "Стратегия перекрестной проверки",
    ["expand", "constant"])

n_folds = st.number_input(
    "Введите количество блоков перекрестной проверки", 
    min_value=1, max_value=24, value=3)

# радиокнопка - запускать перекрестную проверку или нет
run_cv = st.radio(
    "Запустить перекрестную проверку?",
    ("Нет", "Да"))

# либо останавливаем приложение, либо 
# запускаем перекрестную проверку
if run_cv == "Нет":
    st.stop()
else:
    pass

# находим метрики моделей по сегментам 
# по итогам перекрестной проверки
metrics_cv, forecast_cv, _ = pipeline.backtest(
    mode=mode, 
    n_folds=n_folds,
    ts=ts, 
    metrics=[smape], 
    aggregate_metrics=True)

# вычисляем среднее значение метрики по сегментам
cv_mean_smape = metrics_cv['SMAPE'].mean()

# задаем заголовок раздела
st.header("Оценка качества прогнозов по итогам " 
          "перекрестной проверки - SMAPE")

# печатаем метрики по сегментам и среднее значение метрики
st.write(metrics_cv)
st.write("Среднее значение:", cv_mean_smape)

# задаем заголовок раздела
st.header("Визуализация прогнозов по итогам перекрестной проверки")

# визуализируем результаты перекрестной проверки
st.pyplot(
    plot_backtest(forecast_cv, ts, history_len=0)
)

