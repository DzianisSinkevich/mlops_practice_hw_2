import os
from random import randint
import pandas as pd
import sys

print("<<< Start creation >>>")

df_count = int(sys.argv[1])


def delete_files(dir_path):
    try:
        files = os.listdir(dir_path)
        for file in files:
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


dir_path = 'test'
if os.path.isdir(dir_path):
    delete_files(dir_path)
dir_path = 'train'
if os.path.isdir(dir_path):
    delete_files(dir_path)


def save_file(df, dir_path, file_name):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    try:
        file_path = os.path.join(dir_path, file_name)
        df.to_csv(file_path, index=False)
        print("File " + file_path + " created successfully.")
    except IOError:
        print("Error uccured while creating file " + file_path + " .")


def outlier_gen(df):
    for i in range(randint(3, 10)):
        a = randint(1, 6)
        mask = df['month'] == a
        max_val = df[mask]['day_mean_temp'].max(axis=0)
        outlier = randint(7, 10)
        dft = pd.DataFrame([{'month': a, 'day_mean_temp': max_val+outlier}])
        df = pd.concat([df, dft], ignore_index=True)
    # print(df)
    return df


def create_df(dir_path, file_name):
    months_mean_val = {1: [0, 5], 2: [7, 11], 3: [11, 17], 4: [16, 22],
                       5: [23, 28], 6: [28, 35]}

    value_list = []
    for i in range(len(months_mean_val)):
        value_list.append([randint(months_mean_val[i+1][0],
                                   months_mean_val[i+1][1]) for j in range(30)])

    data = {'month': [0], 'day_mean_temp': [0]}
    df = pd.DataFrame(data)

    for i in range(len(value_list)):
        for j in range(30):
            df.loc[len(df.index)] = [i+1, value_list[i][j]]

    df = df.drop(0)

    df = outlier_gen(df)
    save_file(df, dir_path, file_name)


for i in range(df_count):
    create_df('train', 'df_train_' + str(i) + '.csv')
    create_df('test', 'df_test_' + str(i) + '.csv')

print("<<< Finish creation >>>\n")
