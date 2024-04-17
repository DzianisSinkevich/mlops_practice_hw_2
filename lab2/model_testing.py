from sklearn.model_selection import StratifiedKFold  # при кросс-валидации разбиваем данные в пропорции целевой метки
from sklearn.metrics import f1_score  # f1-мера от Scikit-learn
from sklearn.model_selection import cross_validate  # функция кросс-валидации от Scikit-learn
from sklearn.metrics import classification_report  # функция scikit-learn которая считает много метрик классификации

from sklearn.base import BaseEstimator, TransformerMixin  # для создания собственных преобразователей / трансформеров данных

import numpy as np  # библиотека Numpy для операций линейной алгебры и прочего
import warnings
import pickle

import pandas as pd  # Библиотека Pandas для работы с табличными данными

warnings.filterwarnings('ignore')

print("<<< Start model testing >>>")


def read_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print("File " + file_path + " readed successfully.")
        return df
    except IOError:
        print("Error uccured while readed file '" + file_path + "'.")


def load_pipeline():
    try:
        pipeline = pickle.load(open('pipeline/pipeline.pkl', 'rb'))
        print("Pipeline pipeline/pipeline.pkl loaded successfully.")
        return pipeline
    except IOError:
        print("Error uccured while loaded pipeline/pipeline.pkl.")


def calculate_metric(model_pipe, x, y, metric=f1_score):
    """Расчет метрики.
    Параметры:
    ===========
    model_pipe: модель или pipeline
    X: признаки
    y: истинные значения
    metric: метрика (f1 - по умолчанию)
    """
    y_model = model_pipe.predict(x)
    return metric(y, y_model, average=('weighted'))


def cross_validation(x, y, model, scoring, cv_rule):
    """Расчет метрик на кросс-валидации.
    Параметры:
    ===========
    model: модель или pipeline
    x: признаки
    y: истинные значения
    scoring: словарь метрик
    cv_rule: правило кросс-валидации
    """
    scores = cross_validate(model, x, y, scoring=scoring, cv=cv_rule)
    DF_score = pd.DataFrame(scores)
    print('Cross validation error:')
    print(DF_score.mean()[2:])


def main(test_df_path):
    df = read_file(test_df_path)

    x_test, y_test = df.drop(['cut'], axis=1), df['cut']
    pipeline = load_pipeline()

    print("\nModel's f1: ")
    print(f"f1 on test data: {calculate_metric(pipeline, x_test, y_test):.4f}")

    print("\nModel's scores: ")
    print(classification_report(y_test, pipeline.predict(x_test), target_names={'low', 'hight'}))

    scoring_clf = {'ACC': 'accuracy',
                   'F1': 'f1',
                   'Precision': 'precision',
                   'Recall': 'recall'}

    cross_validation(x_test, y_test,
                     pipeline,
                     scoring_clf,
                     StratifiedKFold(n_splits=5, shuffle=True, random_state=42))


class QuantileReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.quantiles = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include='number'):
            low_quantile = X[col].quantile(self.threshold)
            high_quantile = X[col].quantile(1 - self.threshold)
            self.quantiles[col] = (low_quantile, high_quantile)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X.select_dtypes(include='number'):
            low_quantile, high_quantile = self.quantiles[col]
            rare_mask = ((X[col] < low_quantile) | (X[col] > high_quantile))
            if rare_mask.any():
                rare_values = X_copy.loc[rare_mask, col]
                replace_value = np.mean([low_quantile, high_quantile])
                if rare_values.mean() > replace_value:
                    X_copy.loc[rare_mask, col] = high_quantile
                else:
                    X_copy.loc[rare_mask, col] = low_quantile
        return X_copy


class RareGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05, other_value='Other'):
        self.threshold = threshold
        self.other_value = other_value
        self.freq_dict = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object']):
            freq = X[col].value_counts(normalize=True)
            self.freq_dict[col] = freq[freq >= self.threshold].index.tolist()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in X.select_dtypes(include=['object']):
            X_copy[col] = X_copy[col].apply(lambda x: x if x in self.freq_dict[col] else self.other_value)
        return X_copy


main('test/df_test_0.csv')
print("<<< Finish model testing >>>")
