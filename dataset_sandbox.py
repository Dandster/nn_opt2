import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def mnist_digits():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    X_train = tf.reshape(X_train, [60000, 784])
    X_test = tf.reshape(X_test, [10000, 784])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    encode = OneHotEncoder()
    Y_train = encode.fit_transform(Y_train[:, np.newaxis]).toarray()
    Y_test = encode.fit_transform(Y_test[:, np.newaxis]).toarray()

    np.savetxt('Datasets/MNIST_digits/x_train.csv', X_train, delimiter=',')
    np.savetxt('Datasets/MNIST_digits/y_train.csv', Y_train, delimiter=',')
    np.savetxt('Datasets/MNIST_digits/x_test.csv', X_test, delimiter=',')
    np.savetxt('Datasets/MNIST_digits/y_test.csv', Y_test, delimiter=',')

