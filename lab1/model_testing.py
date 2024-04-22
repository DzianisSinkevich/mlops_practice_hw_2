from sklearn.metrics import r2_score  # коэффициент детерминации  от Scikit-learn
from sklearn.metrics import mean_squared_error as mse  # метрика MSE от Scikit-learn
import warnings
from random import randint
import sys
import pickle

import pandas as pd  # Библиотека Pandas для работы с табличными данными

warnings.filterwarnings('ignore')

df_count = int(sys.argv[1])
test_df_path = "test/df_test_" + str(randint(0, df_count - 1)) + ".csv"

print("<<< Start model testing >>>")


def read_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df['day_mean_temp'] = pd.to_numeric(df['day_mean_temp'])
        df = df.fillna(0)
        print("File " + file_path + " readed successfully.")
        return df
    except IOError:
        print("Error uccured while readed file '" + file_path + "'.")


def load_model():
    try:
        model = pickle.load(open('model/model.pkl', 'rb'))
        print("Model model/model.pkl loaded successfully.")
        return model
    except IOError:
        print("Error uccured while loaded model/model.pkl.")



def calculate_metric(model_pipe, x, y, metric=r2_score, **kwargs):
    """Расчет метрики.
    Параметры:
    ===========
    model_pipe: модель или pipeline
    X: признаки
    y: истинные значения
    metric: метрика (r2 - по умолчанию)
    """
    y_model = model_pipe.predict(x)
    return metric(y, y_model, **kwargs)


def main(test_df_path):
    # разбиваем на тестовую и валидационную
    dft = read_file(test_df_path)
    x_test = dft.drop(['month'], axis=1)
    y_test = dft['month']

    model = load_model()

    print("\nModel results:")
    print(f"r2 of model on test data: {calculate_metric(model, x_test, y_test):.4f}")
    print(f"mse of model on test data: {calculate_metric(model, x_test, y_test, mse):.4f}")
    print(f"rmse of model on test data: {calculate_metric(model, x_test, y_test, mse, squared=False):.4f}")


main(test_df_path)
print("<<< Finish model testing >>>")
