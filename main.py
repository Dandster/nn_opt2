import tensorflow as tf
from tensorflow import keras
from arch_generator import ArchGen

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

n_output_neurons = 10
selected_loss = "categorical_crossentropy"
output_act_func = "softmax"

hall_of_fame = []

ag = ArchGen(n_output_neurons, output_act_func, selected_loss)
X_train, Y_train, X_test, Y_test = ag.read_dataset()

model_collection = ag.generate_archs()

for model in model_collection:
    model.compile(optimizer='adam',
                  loss=selected_loss,
                  metrics=['accuracy'])

    # reduce the number of epochs for MNIST datasets
    model.fit(x=X_train, y=Y_train, epochs=15)

    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

    print("test accuracy: " + str(test_accuracy))

    hall_of_fame.append((model_collection.index(model), test_accuracy))


# Sort by best performace, print top 3 structures
def sort_by_performance(e):
    return e[1]


hall_of_fame.sort(key=sort_by_performance, reverse=True)
print("number of structures tested: " + str(len(hall_of_fame)))
print(len(model_collection))

for i in hall_of_fame[:3]:
    print("index of structure and its performace: " + str(i))
    print(model_collection[i[0]].summary())
    for j in model_collection[i[0]].layers:
        print((j.name, j.activation))
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

hall_of_fame.clear()



