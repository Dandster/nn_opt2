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

    #X = np.mean(X, axis=3)

    X = tf.reshape(X, [60000, 32*32*3])

    Y = np.concatenate((y_train, y_test))

    Y = tf.reshape(Y, [60000, 1])

    np.savetxt('Y:/PythonProjekty/Datasets/cifar10/x.csv', X, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/cifar10/y.csv', Y, delimiter=',')


def penguins():
    pass


cifar10()
