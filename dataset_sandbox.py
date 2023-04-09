import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def mnist_digits():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    X = np.concatenate((X_train, X_test))
    X = tf.reshape(X, [70000, 784])

    Y = np.concatenate((Y_train, Y_test))

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #
    # encode = OneHotEncoder()
    # Y = encode.fit_transform(Y[:, np.newaxis]).toarray()

    np.savetxt('Y:/PythonProjekty/Datasets/MNIST_digits/x.csv', X, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/MNIST_digits/y.csv', Y, delimiter=',')
