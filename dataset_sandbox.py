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
    X = tf.reshape(X, [70000, 784])

    Y = np.concatenate((Y_train, Y_test))

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #
    # encode = OneHotEncoder()
    # Y = encode.fit_transform(Y[:, np.newaxis]).toarray()

    np.savetxt('Y:/PythonProjekty/Datasets/MNIST_digits/x.csv', X, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/MNIST_digits/y.csv', Y, delimiter=',')


def cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    X = np.concatenate((x_train, x_test))

    X = np.mean(X, axis=3)

    X = tf.reshape(X, [60000, 32*32])

    Y = np.concatenate((y_train, y_test))

    Y = tf.reshape(Y, [60000, 1])

    np.savetxt('Y:/PythonProjekty/Datasets/cifar10_grey/x.csv', X, delimiter=',')
    np.savetxt('Y:/PythonProjekty/Datasets/cifar10_grey/y.csv', Y, delimiter=',')


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
