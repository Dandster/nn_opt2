import numpy as np
import sklearn
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras


def str_to_int_list(values):
    try:
        og_values = values  # to track errors
        values = ''.join(values.split())
        splits = values.split(sep=",")
        return remove_duplicates([int(s) for s in splits])
    except ValueError:
        print("Reading config unsuccessful, check the formatting of: " + og_values)
        exit(0)


def str_to_str_list(values):
    try:
        og_values = values  # to track errors
        values = ''.join(values.split())
        return remove_duplicates(values.split(sep=","))
    except ValueError:
        print("Reading config unsuccessful, check the formatting of: " + og_values)
        exit(0)


def remove_duplicates(input_list):
    return list(dict.fromkeys(input_list))


def do_scaling(x):
    scaler = StandardScaler()
    return scaler.fit_transform(x)


def do_one_hot(y):
    encode = OneHotEncoder()
    return encode.fit_transform(y[:, np.newaxis]).toarray()


def split_nparray_to_3(array, r1, r2, r3):
    if r1 + r2 + r3 != 1.0:
        print("Dataset split (train:validation:test) ratios must sum to 1.0!")
        exit(0)
    else:
        try:
            train, validation, test = np.split(array, [int(r1 * len(array)), int((r1+r2) * len(array))])
            return train, validation, test
        except ValueError:
            print("Splitting dataset did not result in an equal division")
            exit(0)


def print_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    print(cm)
