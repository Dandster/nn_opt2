import tensorflow as tf
from tensorflow import keras
import configparser as config
import tools as t
from arch_generator import ArchGen

import numpy as np
import pandas as pd

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

config = config.ConfigParser()
config.read('conf.ini')
metrics = config['metrics']

n_output_neurons = 10
selected_loss = "categorical_crossentropy"
output_act_func = "softmax"

hall_of_fame = []

ag = ArchGen(n_output_neurons, output_act_func, selected_loss)
x_train, y_train, x_val, y_val, x_test, y_test = ag.read_dataset()

model_collection = ag.generate_archs()

for model in model_collection:
    model.compile(optimizer='adam',
                  loss=selected_loss,
                  metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Recall(class_id=0)])

    # reduce the number of epochs for MNIST datasets
    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=10)

    test_loss, test_accuracy, test_recall = model.evaluate(x_test, y_test, verbose=0)

    #  pridat funkci do tools na zobrazeni metrik, podobnou jak je tato na confusion matrix
    #  toto jeste predelat, at nevolam model.predict vickrat jak blbec!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if eval(metrics['print_confusion_matrix']):
        t.print_confusion_matrix(model, x_test, y_test)

    if eval(metrics['print_classification_report']):
        t.print_classification_report(model, x_test, y_test)

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
        print((j, j.activation))
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

hall_of_fame.clear()



