from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # Импортируем стандартизацию от scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor  # Линейная регрессия с градиентным спуском от scikit-learn
import warnings
from random import randint
import pickle
import sys
import os
import pandas as pd  # Библиотека Pandas для работы с табличными данными

warnings.filterwarnings('ignore')

print("<<< Start preparation >>>")
df_count = int(sys.argv[1])
train_df_path = "train/df_train_" + str(randint(0, df_count - 1)) + ".csv"


def read_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df['day_mean_temp'] = pd.to_numeric(df['day_mean_temp'])
        df = df.fillna(0)
        print("File " + file_path + " readed successfully.")
        return df
    except IOError:
        print("Error uccured while readed file '" + file_path + "'.")


def save_model(model):
    try:
        if not os.path.isdir('model'):
            os.mkdir('model')
        pickle.dump(model, open('model/model.pkl', 'wb'))
        print("Model model/model.pkl saved successfully.")
    except IOError:
        print("Error uccured while saved model/model.pkl.")


def preparation(train_df_path):
    df = read_file(train_df_path)
    num_pipe_day_mean_temp = Pipeline([('scaler', StandardScaler())])
    num_day_mean_temp = ['day_mean_temp']

    preprocessor = ColumnTransformer(transformers=[('num_day_mean_temp', num_pipe_day_mean_temp, num_day_mean_temp)])

    # df = read_file(file_path)
    # не забываем удалить целевую переменную цену из признаков
    x_train = df.drop(['month'], axis=1)
    y_train = df['month']

    # Сначала обучаем на тренировочных данных
    x_train_prep = preprocessor.fit_transform(x_train)
    model = SGDRegressor(random_state=42)
    model.fit(x_train_prep, y_train)
    save_model(model)


preparation(train_df_path)

print("<<< Finish preparation >>>\n")
