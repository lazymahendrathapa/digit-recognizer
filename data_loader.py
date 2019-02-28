import pandas as pd
import numpy as np


def load_train_dataset(label, path="data/train.csv"):

    train_df = pd.read_csv(path)
    pos_data = train_df[train_df.label == label].copy()
    neg_data = train_df[train_df.label != label].copy()

    pos_data.label = 1
    neg_data.label = 0

    train_df = pd.concat([pos_data, neg_data])

    train_df = train_df.sample(frac=1).reset_index(drop=True)

    x = np.array(train_df.loc[:, train_df.columns != 'label'].values)
    y = np.array(train_df['label'].values)

    val_length = int(len(y)*0.2)

    x_train = x[:val_length]/255
    y_train = y[:val_length]

    x_val = x[val_length:]/255
    y_val = y[val_length:]

    return x_train, y_train, x_val, y_val


def load_test_dataset(path="data/test.csv"):

    test_df = pd.read_csv(path)

    x_test = np.array(test_df.values)/255

    return x_test
