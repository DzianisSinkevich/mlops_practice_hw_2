from sklearn.preprocessing import StandardScaler  # Импортируем стандартизацию от scikit-learn
from sklearn.preprocessing import OneHotEncoder  # Импортируем One-Hot Encoding от scikit-learn
import pandas as pd  # Библиотека Pandas для работы с табличными данными
import warnings
import sys

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

print("<<< Start preprocessing >>>")
df_count = int(sys.argv[1])
num_columns = ['day_mean_temp']
cat_columns = ['month']


def read_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df['day_mean_temp'] = pd.to_numeric(df['day_mean_temp'])
        df = df.fillna(0)
        print("File " + file_path + " readed successfully.")
        return df
    except IOError:
        print("Error uccured while readed file '" + file_path + "'.")


def save_file(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        print("File " + file_path + " created successfully.")
    except IOError:
        print("Error uccured while creating file " + file_path + " .")


def df_prerpocessing(file_path):
    df = read_file(file_path)

    # Предобработка числовых значений. Стандартиация
    scale = StandardScaler()
    scale.fit(df[num_columns])

    scaled = scale.transform(df[num_columns])
    df_standard = pd.DataFrame(scaled, columns=num_columns)

    file_path_standard = file_path[:-4] + "_standard.csv"
    save_file(df_standard, file_path_standard)

    # Предобработка категориального признака. One-hot кодировиние
    # Создание Объекта OneHotEncoder() и его "обучение" .fit
    ohe = OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False)
    ohe.fit(df[cat_columns])

    # Применяем трансформацию .transform и сохраняем результат в Dataframe
    ohe_feat = ohe.transform(df[cat_columns])
    df_ohe = pd.DataFrame(ohe_feat, columns=ohe.get_feature_names_out()).astype(int)
    df_ohe

    file_path_ohe = file_path[:-4] + "_ohe.csv"
    save_file(df_ohe, file_path_ohe)


for i in range(df_count):
    file_path = "train/df_train_" + str(i) + ".csv"
    print("<< Start preprocessing '" + file_path + "' >>")
    df_prerpocessing(file_path)
    print("<< Finish preprocessing '" + file_path + "' >>\n")

print("<<< Finish preprocessing >>>\n")
