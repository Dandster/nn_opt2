import sklearn.datasets as skds
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd


def mnist_digits():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    X = np.concatenate((X_train, X_test))
    X = tf.reshape(X, [70000, 28*28])

    Y = np.concatenate((Y_train, Y_test))

    np.savetxt('Y:/PythonProjekty/Datasets/MNIST_digits/x.csv', X, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/MNIST_digits/y.csv', Y, delimiter=',')


def mnist_fashion():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

    X = np.concatenate((X_train, X_test))

    X = np.split(X, 2)[0]  # split in half to reduce training time

    X = tf.reshape(X, [35000, 28*28])

    Y = np.concatenate((Y_train, Y_test))

    Y = np.split(Y, 2)[0]

    np.savetxt('Y:/PythonProjekty/Datasets/MNIST_fashion/x.csv', X, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/MNIST_fashion/y.csv', Y, delimiter=',')


def cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    X = np.concatenate((x_train, x_test))

    X = tf.reshape(X, [60000, 32*32*3])

    Y = np.concatenate((y_train, y_test))

    Y = tf.reshape(Y, [60000, 1])

    np.savetxt('Y:/PythonProjekty/Datasets/cifar10/x.csv', X, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/cifar10/y.csv', Y, delimiter=',')


def titanic():
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

    np.savetxt('Y:/PythonProjekty/Datasets/titanic/x.csv', np_x, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/titanic/y.csv', np_y, delimiter=',')


def iris():
    x, y = skds.load_iris(return_X_y=True)

    xy = np.c_[x, y]
    np.random.shuffle(xy)

    x = xy[:, :4]
    y = xy[:, 4:]

    np.savetxt('Y:/PythonProjekty/Datasets/iris/x.csv', x, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/iris/y.csv', y, delimiter=',')


def penguins():
    X = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/'
                    'data/palmer_penguins/penguins_processed.csv')

    X = X.to_numpy()
    np.random.shuffle(X)

    y = X[:, :1]
    x = X[:, 1:]

    np.savetxt('Y:/PythonProjekty/Datasets/penguins/x.csv', x, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/penguins/y.csv', y, delimiter=',')


def dnp3():
    training = pd.read_csv("Y:/PythonProjekty/Datasets/DNP3/training.csv")
    testing = pd.read_csv("Y:/PythonProjekty/Datasets/DNP3/testing.csv")

    print(training.head())

    ds = pd.concat([training, testing], axis=0, ignore_index=True)

    y = ds.pop(' Label')
    x = ds

    to_be_removed = ["flow ID", " source IP", " destination IP",
                     " date", ' source port', ' destination port']

    x.drop(columns=to_be_removed, inplace=True)

    to_one_hot = [" firstPacketDIR"]

    x = pd.get_dummies(x, columns=to_one_hot)
    y = pd.get_dummies(y, columns=[' Label'])

    np_x = x.to_numpy()
    np_y = y.to_numpy()

    print(np_x.shape)
    print(np_y.shape)

    np.savetxt('Y:/PythonProjekty/Datasets/DNP3_p/x.csv', np_x, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/DNP3_p/y.csv', np_y, delimiter=',')
