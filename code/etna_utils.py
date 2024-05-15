import numpy as np
import pandas as pd
import re

from typing import List

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from etna.pipeline import Pipeline
from etna.analysis import plot_forecast

from etna.models.sklearn import (SklearnPerSegmentModel, 
                                 SklearnMultiSegmentModel)

from etna.models.base import (BaseAdapter, 
                              NonPredictionIntervalContextIgnorantAbstractModel)

from etna.models.mixins import (PerSegmentModelMixin,
                                MultiSegmentModelMixin,
                                NonPredictionIntervalContextIgnorantModelMixin)

from etna.datasets.tsdataset import TSDataset

from sklearn.ensemble import (BaggingRegressor,
                              RandomForestRegressor,
                              GradientBoostingRegressor,
                              HistGradientBoostingRegressor)

# функция преобразования в формат TSDataset
def transform_etna_format(df, date_column, freq='D', melt_df=False):
    """
    Преобразовывает набор данных с временными 
    рядами в объект TSDataset.
    
    Параметры
    ----------
    df: pandas.DataFrame
        Набор данных с временным рядом.
    date_column: string
        Cтолбец с датами.
    freq: string, по умолчанию 'D'
        Частота временного ряда.
    melt_df: bool, по умолчанию False
        Имеет ли набор данных длинный формат 
        (long format).
    """
    df = df.rename(columns={date_column: 'timestamp'})   
    if not melt_df:
        df = df.melt(id_vars='timestamp', 
                     var_name='segment', 
                     value_name='target')
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq=freq)
    return ts


# пишем функцию построения модели ETNA и оценки ее качества
def train_and_evaluate_model(ts, 
                             model,
                             transforms,
                             horizon,
                             metrics,
                             print_metrics=False,
                             print_plots=False,
                             n_train_samples=None):
    """
    Обучает модель, вычисляет прогнозы для 
    тестовой выборки и строит график прогнозов.  
    
    Параметры
    ----------
    ts: pandas.DataFrame
        Временной ряд.
    model: instance of class etna
        Экземпляр класса библиотеки etna.
    transforms: list
        Список преобразований.
    horizon: int
        Горизонт прогнозирования.
    metrics: instance of class etna.metrics.SMAPE/
        MAE/R2/MAPE/MedAE/MSLE   
        Метрика качества.
    print_metrics: bool, по умолчанию False
        Печать метрик.
    print_plots: bool, по умолчанию False
        Печать графиков прогнозов.
    n_train_sample: int
        n последних наблюдений обучающей выборки 
        на графике прогнозов.
    """
    if not print_plots and n_train_samples is not None:
        raise ValueError(
            "Параметр n_train_samples задается при print_plots=True")
        
    # разбиваем набор на обучающую и тестовую выборки 
    # с учетом временной структуры, размер тестовой 
    # выборки задаем равным горизонту
    train_ts, test_ts = ts.train_test_split(test_size=horizon)
    # создаем конвейер
    pipe = Pipeline(model=model,
                    transforms=transforms,
                    horizon=horizon)
    # обучаем конвейер
    pipe.fit(train_ts)
    # получаем прогнозы
    forecast_ts = pipe.forecast()
    # оцениваем качество прогнозов по сегментам
    segment_metrics = metrics(test_ts, forecast_ts)
    segment_metrics = pd.Series(segment_metrics)
    
    if print_metrics:
        print(segment_metrics.to_string())
        print("")
        # оцениваем качество прогнозов в среднем
        print(f"Усредненная метрика:"
              f"{sum(segment_metrics) / len(segment_metrics)}")
    
    if print_plots:
        # визуализируем прогнозы, здесь n_train_samples
        # - n последних наблюдений в обучающей выборке
        plot_forecast(forecast_ts, test_ts, 
                      train_ts, n_train_samples=n_train_samples)


# пишем функцию, выполняющую подбор гиперпараметра - набора 
# преобразований/признаков с помощью перекрестной проверки
def etna_cv_optimize(ts, model, horizon, transfrms, n_folds, mode, metrics, 
                     refit=True, n_train_samples=10):
    """
    Выполняет подбор гиперпараметра - набора преобразований/
    признаков с помощью перекрестной проверки.  
    
    Параметры
    ----------
    ts: pandas.DataFrame
        Мультииндексный pandas.DataFrame в формате ETNA 
        (объект TSDataset), содержащий один или 
        несколько временных рядов.
    model: instance of class etna.models
        Модель прогнозирования.
    horizon: int
        Горизонт прогнозирования.
    transfrms: list
        Список преобразований/признаков.
    n_folds, int
        Количество тестовых выборок перекрестной проверки.
    mode: str
        Тип перекрестной проверки.
    metrics: instance of class etna.metrics
        Метрика качества.
    refit: bool
        Нужно ли строить наилучшую модель на всей обучающей выборке.
    n_train_sample: int
        n последних наблюдений обучающей выборки 
        на графике прогнозов.
    """
    # разбиваем набор на обучающую и тестовую выборки 
    # с учетом временной структуры, размер тестовой 
    # выборки задаем равным горизонту
    train_ts, test_ts = ts.train_test_split(test_size=horizon)
    
    # инициализируем наилучшее значение метрики 
    # положительной бесконечностью
    best_score = np.inf
    
    # с помощью цикла for
    for trans in transfrms:
        # создаем конвейер с моделью и списком преобразований
        pipe = Pipeline(
            model=model,
            transforms=trans,
            horizon=horizon)
        
        # находим метрики моделей по сегменту/сегментам 
        # по итогам перекрестной проверки
        df_metrics, _, _ = pipe.backtest(mode=mode, 
                                         n_folds=n_folds,
                                         ts=train_ts, 
                                         metrics=[metrics], 
                                         aggregate_metrics=False,
                                         joblib_params=dict(verbose=0))
        # вычисляем значение метрики, усредненное по тестовым выборкам
        metrics_mean = df_metrics[metrics.__class__.__name__].mean()
        # вычисляем стандартное отклонение метрики
        metrics_std = df_metrics[metrics.__class__.__name__].std()
        print(f"trans:\n{trans}")
        print(f"{metrics.__class__.__name__}_mean: {metrics_mean}")
        print(f"{metrics.__class__.__name__}_std: {metrics_std}\n")
    
        # если получаем максимальное усредненное значение, сохраняем
        # его и наилучший набор преобразований/признаков
        if metrics_mean < best_score:
            best_score = metrics_mean
            best_parameters = {'trans': trans}
            
    # печатаем наилучший набор преобразований/признаков 
    # и наилучшее значение метрики по итогам
    # перекрестной проверки
    print(f"Наилучший набор преобразований/признаков:\n{best_parameters}\n")
    print(f"Лучшее значение {metrics.__class__.__name__} cv: {best_score:.4f}\n")
    
    if refit:
        # создаем конвейер с наилучшим набором преобразований/признаков
        pipe = Pipeline(model=model,
                        transforms=best_parameters.get('trans'),
                        horizon=horizon)        
        # обучаем конвейер на всей обучающей выборке
        pipe.fit(train_ts)
        # получаем прогнозы
        forecast_ts = pipe.forecast()        
        # оцениваем качество прогнозов
        print(metrics(y_true=test_ts, y_pred=forecast_ts))        
        # визуализируем прогнозы
        plot_forecast(forecast_ts, test_ts, 
                      train_ts, n_train_samples=n_train_samples)
        

# пишем функцию, выполняющую подбор гиперпараметра - набора 
# преобразований/признаков с помощью перекрестной проверки
# (в рамках пошагового прохода по конвейеру)
def etna_staged_cv_optimize(ts, model, horizon, init_transfrms, 
                            transfrms, n_folds, mode, metrics, 
                            refit=True, n_train_samples=10):
    """
    Выполняет подбор гиперпараметра - набора преобразований/
    признаков с помощью перекрестной проверки (пошаговый
    проход конвейера).  
    
    Параметры
    ----------
    ts: pandas.DataFrame
        Мультииндексный pandas.DataFrame в формате ETNA 
        (объект TSDataset), содержащий один или 
        несколько временных рядов.
    model: instance of class etna.models
        Модель прогнозирования.
    horizon: int
        Горизонт прогнозирования.
    init_transfrms: list
        Список исходных преобразований.
    transfrms: list
        Список преобразований/признаков.
    n_folds, int
        Количество тестовых выборок перекрестной проверки.
    mode: str
        Тип перекрестной проверки.
    metrics: instance of class etna.metrics
        Метрика качества.
    plot_backtest_results: bool, по умолчанию True
    refit: bool
        Нужно ли строить наилучшую модель 
        на всей обучающей выборке.
    n_train_sample: int, по умолчанию 10
        n последних наблюдений обучающей выборки 
        на графике прогнозов.
    """
    # разбиваем набор на обучающую и тестовую выборки 
    # с учетом временной структуры, размер тестовой 
    # выборки задаем равным горизонту
    train_ts, test_ts = ts.train_test_split(test_size=horizon)
    
    # инициализируем наилучшее значение метрики 
    # положительной бесконечностью
    best_score = np.inf

    # задаем пустые списки
    el_lst = list()
    transfrms_lst = list()
    
    # получаем список списков
    for el in transfrms:
        el_lst.append(el)
        transfrms_lst.append(el_lst.copy())
    
    # добавляем в начало каждого списка списка список 
    # со стартовыми преобразованиями
    for i in range(len(transfrms_lst)):
        transfrms_lst[i] = init_transfrms + transfrms_lst[i]
    
    # с помощью цикла for
    for trans in transfrms_lst:
        # создаем конвейер с моделью и списком преобразований
        pipe = Pipeline(
            model=model,
            transforms=trans,
            horizon=horizon)
        
        # находим метрики моделей по сегменту/сегментам 
        # по итогам перекрестной проверки
        df_metrics, _, _ = pipe.backtest(mode=mode, 
                                         n_folds=n_folds,
                                         ts=train_ts, 
                                         metrics=[metrics], 
                                         aggregate_metrics=False,
                                         joblib_params=dict(verbose=0))
        # вычисляем значение метрики, усредненное по тестовым выборкам
        metrics_mean = df_metrics[metrics.__class__.__name__].mean()
        # вычисляем стандартное отклонение метрики
        metrics_std = df_metrics[metrics.__class__.__name__].std()
        print(f"trans:\n{trans}")
        print(f"{metrics.__class__.__name__}_mean: {metrics_mean}")
        print(f"{metrics.__class__.__name__}_std: {metrics_std}\n")
    
        # если получаем максимальное усредненное значение, сохраняем
        # его и наилучший набор преобразований/признаков
        if metrics_mean < best_score:
            best_score = metrics_mean
            best_parameters = {'trans': trans}
            
    # печатаем наилучший набор преобразований/признаков 
    # и наилучшее значение метрики по итогам
    # перекрестной проверки
    print(f"Наилучший набор преобразований/признаков:\n{best_parameters}\n")
    print(f"Лучшее значение {metrics.__class__.__name__} cv: {best_score:.4f}\n")
    
    if refit:
        # создаем конвейер с наилучшим набором преобразований/признаков
        pipe = Pipeline(model=model,
                        transforms=best_parameters.get('trans'),
                        horizon=horizon)        
        # обучаем конвейер на всей обучающей выборке
        pipe.fit(train_ts)
        # получаем прогнозы
        forecast_ts = pipe.forecast()        
        # оцениваем качество прогнозов
        print(metrics(y_true=test_ts, y_pred=forecast_ts))        
        # визуализируем прогнозы
        plot_forecast(forecast_ts, test_ts, 
                      train_ts, n_train_samples=n_train_samples)


# пишем ядро - внутренний класс _LGBMAdapter,
# внутри - класс LGBMRegressor
class _LGBMAdapter(BaseAdapter):
    def __init__(
        self,
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        **kwargs
    ):
        self.model = LGBMRegressor(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            **kwargs
        )
        self._categorical = None

    def fit(self, df: pd.DataFrame, regressors: List[str]):
        df = df.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        features = df.drop(columns=['timestamp', 'target'])
        self._categorical = features.select_dtypes(
            include=['category']).columns.to_list()
        target = df['target']
        self.model.fit(X=features, y=target, 
                       categorical_feature=self._categorical)
        return self

    def predict(self, df: pd.DataFrame):
        features = df.drop(columns=['timestamp', 'target'])
        pred = self.model.predict(features)
        return pred
    
    def get_model(self) -> LGBMRegressor:
        return self.model


class LGBMPerSegmentModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel):
    def __init__(
        self,
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        **kwargs
    ):
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        super().__init__(
            base_model=_LGBMAdapter(
                boosting_type=boosting_type,
                num_leaves=num_leaves,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                **kwargs
            )
        )


class LGBMMultiSegmentModel(
    MultiSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel):
    def __init__(
        self,
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        **kwargs
    ):
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        super().__init__(
            base_model=_LGBMAdapter(
                boosting_type=boosting_type,
                num_leaves=num_leaves,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                **kwargs
            )
        )

        
# пишем ядро - внутренний класс _XGBAdapter,
# внутри - класс XGBRegressor
class _XGBAdapter(BaseAdapter):
    def __init__(
        self,
        booster='gbtree',
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        **kwargs
    ):
        self.model = XGBRegressor(
            booster=booster,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            **kwargs
        )

    def fit(self, df: pd.DataFrame, regressors: List[str]):
        df = df.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        features = df.drop(columns=['timestamp', 'target'])
        cat_feat = features.select_dtypes(
            include=['category']).columns.to_list()
        for feat in cat_feat:
            features[feat] = features[feat].cat.codes
        target = df['target']
        self.model.fit(X=features, y=target)
        return self

    def predict(self, df: pd.DataFrame):
        features = df.drop(columns=['timestamp', 'target'])
        cat_feat = features.select_dtypes(
            include=['category']).columns.to_list()
        for feat in cat_feat:
            features[feat] = features[feat].cat.codes
        pred = self.model.predict(features)
        return pred
    
    def get_model(self) -> XGBRegressor:
        return self.model

        
# пишем класс XGBPerSegmentModel, который строит 
# отдельную модель XGBoost для каждого сегмента
class XGBPerSegmentModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel):
    def __init__(
        self,
        booster='gbtree',
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        **kwargs
    ):
        self.booster = booster
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        super().__init__(
            base_model=_XGBAdapter(
                booster=booster,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                **kwargs
            )
        )
        

# пишем класс XGBMultiSegmentModel, который строит 
# одну модель XGBoost для всех сегментов
class XGBMultiSegmentModel(
    MultiSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel):
    def __init__(
        self,
        booster='gbtree',
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        **kwargs
    ):
        self.booster = booster
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        super().__init__(
            base_model=_XGBAdapter(
                booster=booster,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                **kwargs
            )
        )
        
        
# пишем классы GBPerSegmentModel и GBMultiSegmentModel
# для применения класса GradientBoostingRegressor в ETNA
class GBPerSegmentModel(SklearnPerSegmentModel):
    """
    Класс :py:class:`sklearn.ensemble.GradientBoostingRegressor` 
    для сегмента.
    """

    def __init__(self, 
                 learning_rate: float = 0.01, 
                 n_estimators: int = 100,
                 max_depth: int = 20,
                 **kwargs):
        """
        Создает экземпляр класса GradientBoostingRegressor 
        с заданными параметрами.
        
        Параметры
        ----------
        learning_rate:
            Темп обучения.
        n_estimators:
            Количество деревьев.
        max_depth:
            Максимальная глубина деревьев.
        """
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.kwargs = kwargs
        super().__init__(regressor=GradientBoostingRegressor(
            learning_rate=self.learning_rate, 
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            **self.kwargs))

        
class GBMultiSegmentModel(SklearnMultiSegmentModel):
    """
    Класс :py:class:`sklearn.ensemble.GradientBoostingRegressor` 
    для всех сегментов.
    """

    def __init__(self, 
                 learning_rate: float = 0.01, 
                 n_estimators: int = 100,
                 max_depth: int = 20,
                 **kwargs):
        """
        Создает экземпляр класса GradientBoostingRegressor
        с заданными параметрами.
        
        Параметры
        ----------
        learning_rate:
            Темп обучения.
        n_estimators:
            Количество деревьев.
        max_depth:
            Максимальная глубина деревьев.   
        """
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.kwargs = kwargs
        super().__init__(regressor=GradientBoostingRegressor(
            learning_rate=self.learning_rate, 
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            **self.kwargs))
        

# пишем классы HistGBPerSegmentModel и HistGBMultiSegmentModel
# для применения класса HistGradientBoostingRegressor в ETNA
class HistGBPerSegmentModel(SklearnPerSegmentModel):
    """
    Класс :py:class:`sklearn.ensemble.HistGradientBoostingRegressor` 
    для сегмента.
    """

    def __init__(self, 
                 learning_rate: float = 0.05, 
                 max_iter: int = 100,
                 max_depth: int = 20,
                 **kwargs):
        """
        Создает экземпляр класса HistGradientBoostingRegressor 
        с заданными параметрами.
        
        Параметры
        ----------
        learning_rate:
            Темп обучения.
        max_iter:
            Максимальное количество деревьев.
        max_depth:
            Максимальная глубина деревьев.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.kwargs = kwargs
        super().__init__(regressor=HistGradientBoostingRegressor(
            learning_rate=self.learning_rate, 
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            **self.kwargs))


class HistGBMultiSegmentModel(SklearnMultiSegmentModel):
    """
    Класс :py:class:`sklearn.ensemble.HistGradientBoostingRegressor` 
    для всех сегментов.
    """

    def __init__(self, 
                 learning_rate: float = 0.05, 
                 max_iter: int = 100,
                 max_depth: int = 20,
                 **kwargs):
        """
        Создает экземпляр класса HistGradientBoostingRegressor
        с заданными параметрами.
        
        Параметры
        ----------
        learning_rate:
            Темп обучения.
        max_iter:
            Максимальное количество деревьев.
        max_depth:
            Максимальная глубина деревьев.   
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.kwargs = kwargs
        super().__init__(regressor=HistGradientBoostingRegressor(
            learning_rate=self.learning_rate, 
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            **self.kwargs))
        
        
# пишем классы RFPerSegmentModel и RFMultiSegmentModel для
# применения класса RandomForestRegressor в ETNA
class RFPerSegmentModel(SklearnPerSegmentModel):
    """
    Класс :py:class:`sklearn.ensemble.RandomForestRegressor` 
    для сегмента.
    """

    def __init__(self, 
                 n_estimators: int = 100, 
                 max_depth: int = 20,
                 **kwargs):
        """
        Создает экземпляр класса RandomForestRegressor 
        с заданными параметрами.
        
        Параметры
        ----------
        n_estimators:
            Количество деревьев случайного леса.
        max_depth:
            Максимальная глубина деревьев
            случайного леса.   
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.kwargs = kwargs
        super().__init__(regressor=RandomForestRegressor(
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth,
            **self.kwargs))

        
class RFMultiSegmentModel(SklearnMultiSegmentModel):
    """
    Класс :py:class:`sklearn.ensemble.RandomForestRegressor` 
    для всех сегментов.
    """

    def __init__(self, 
                 n_estimators: int = 100, 
                 max_depth: int = 20,
                 **kwargs):
        """
        Создает экземпляр класса RandomForestRegressor
        с заданными параметрами.
        
        Параметры
        ----------
        n_estimators:
            Количество деревьев случайного леса.
        max_depth:
            Максимальная глубина деревьев
            случайного леса.   
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.kwargs = kwargs
        super().__init__(regressor=RandomForestRegressor(
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth,
            **self.kwargs))
        
        
# пишем классы BRPerSegmentModel и BRMultiSegmentModel
# для применения класса BaggingRegressor в ETNA
class BRPerSegmentModel(SklearnPerSegmentModel):
    """
    Класс :py:class:`sklearn.ensemble.BaggingRegressor` 
    для сегмента.
    """

    def __init__(self, 
                 n_estimators: int = 100, 
                 max_samples: float = 1.0,
                 max_features: float = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = False,
                 **kwargs):
        """
        Создает экземпляр класса BaggingRegressor 
        с заданными параметрами.
        
        Параметры
        ----------
        n_estimators:
            Количество базовых моделей.
        max_samples:
            Количество наблюдений, используемых 
            для обучения каждой базовой модели.  
        max_features:    
            Количество признаков, используемых 
            для обучения каждой базовой модели.
        bootstrap:    
            Нужен ли отбор наблюдений с возвращением.
        bootstrap_features:    
            Нужен ли отбор признаков с возвращением.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.kwargs = kwargs
        super().__init__(regressor=BaggingRegressor(
            n_estimators=self.n_estimators, 
            max_samples=self.max_samples,
            max_features = max_features,
            bootstrap = bootstrap,
            bootstrap_features = bootstrap_features,
            **self.kwargs))

        
class BRMultiSegmentModel(SklearnMultiSegmentModel):
    """
    Класс :py:class:`sklearn.ensemble.BaggingRegressor` 
    для всех сегментов.
    """

    def __init__(self, 
                 n_estimators: int = 100, 
                 max_samples: float = 1.0,
                 max_features: float = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = False,
                 **kwargs):
        """
        Создает экземпляр класса BaggingRegressor
        с заданными параметрами.
        
        Параметры
        ----------
        n_estimators:
            Количество базовых моделей.
        max_samples:
            Количество наблюдений, используемых 
            для обучения каждой базовой модели.  
        max_features:    
            Количество признаков, используемых 
            для обучения каждой базовой модели.
        bootstrap:    
            Нужен ли отбор наблюдений с возвращением.
        bootstrap_features:    
            Нужен ли отбор признаков с возвращением.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.kwargs = kwargs
        super().__init__(regressor=BaggingRegressor(
            n_estimators=self.n_estimators, 
            max_samples=self.max_samples,
            max_features = max_features,
            bootstrap = bootstrap,
            bootstrap_features = bootstrap_features,
            **self.kwargs))