import tensorflow as tf
from keras import layers
from sklearn.metrics import classification_report
from tensorflow import keras

import numpy as np
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def mnist_digits():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    X = np.concatenate((X_train, X_test))
    X = tf.reshape(X, [70000, 28, 28])

    Y = np.concatenate((Y_train, Y_test))

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #
    # encode = OneHotEncoder()
    # Y = encode.fit_transform(Y[:, np.newaxis]).toarray()

    np.savetxt('Y:/PythonProjekty/Datasets/MNIST_digits28x28/x.csv', X, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/MNIST_digits28x28/y.csv', Y, delimiter=',')


def cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    X = np.concatenate((x_train, x_test))

    X = tf.reshape(X, [60000, 32*32*3])

    Y = np.concatenate((y_train, y_test))

    Y = tf.reshape(Y, [60000, 1])

    np.savetxt('Y:/PythonProjekty/Datasets/cifar10/x.csv', X, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/cifar10/y.csv', Y, delimiter=',')


def titanic():
    # Preparing titanic dataset, 2 classes (died or survived), 8 features, 627 training pairs, 264 testing pairs
    dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
    dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

    concatenated_df = pd.concat([dftrain, dfeval], axis=0, ignore_index=True)

    y = concatenated_df.pop('survived')

    cat_features = ["sex", "class", "deck", "embark_town", "alone"]

    encoded_x = pd.get_dummies(concatenated_df, columns=cat_features)

    np_x = encoded_x.to_numpy()
    np_y = y.to_numpy()

    print(np_x.shape)
    print(np_y.shape)

    # Save numpy array to file
    # np.savetxt('Y:/PythonProjekty/Datasets/titanic/x.csv', np_x, delimiter=',')
    # np.savetxt('Y:/PythonProjekty/Datasets/titanic/y.csv', np_y, delimiter=',')

titanic()