import configparser as config
import copy
import itertools

import numpy as np

import tools as t
import tensorflow as tf
from tensorflow import keras


class ArchGen:

    def __init__(self, number_of_output_neurons, output_activation_func, selected_loss):
        self.output_neurons = number_of_output_neurons  # based on task type, manually given by user
        self.output_function = output_activation_func  # based on task type
        self.selected_loss = selected_loss  # based on task type MOZNA ODDELAT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.config = config.ConfigParser()
        self.config.read('conf.ini')

    def read_hyperpars(self):

        hyperpars = self.config['hyperpars']

        number_of_layers = t.str_to_int_list(hyperpars['number_of_hidden_layers'])
        number_of_neurons = t.str_to_int_list(hyperpars['number_of_neurons'])
        activation_funcs = t.str_to_str_list(hyperpars['activation_funcs'])

        return number_of_layers, number_of_neurons, activation_funcs

    def read_dataset(self):

        dataset = self.config['dataset']

        x_train = np.loadtxt(dataset['x_train'], delimiter=',')
        y_train = np.loadtxt(dataset['y_train'], delimiter=',')
        x_test = np.loadtxt(dataset['x_test'], delimiter=',')
        y_test = np.loadtxt(dataset['y_test'], delimiter=',')

        return x_train, y_train, x_test, y_test

    def generate_archs(self):
        layers, neurons, activations_funcs = self.read_hyperpars()

        possible_layers = []

        for i in neurons:
            for j in activations_funcs:
                possible_layers.append(keras.layers.Dense(i, activation=j))

        model_collection = []

        layer_number = 0#  needed because every layer name must be unique, tohle mozna jde udelat min debilne

        for i in layers:
            prod = itertools.product(possible_layers, repeat=i)
            for j in prod:
                model = keras.Sequential()
                for layer in j:
                    con = layer.get_config()
                    cloned_layer = type(layer).from_config(con)
                    cloned_layer._name = cloned_layer.name + str(layer_number)
                    layer_number = layer_number + 1
                    model.add(cloned_layer)
                model.add(keras.layers.Dense(self.output_neurons, activation=self.output_function))
                model_collection.append(model)


        return model_collection


