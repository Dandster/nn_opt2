import configparser as config
import itertools

import numpy as np

import tools as t
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

        layer_range = bool(hyperpars['layer_range'])
        number_of_layers = t.str_to_int_list(hyperpars['number_of_hidden_layers'])

        if layer_range:
            number_of_layers = range(1, max(number_of_layers)+1)

        number_of_neurons = t.str_to_int_list(hyperpars['number_of_neurons'])
        activation_funcs = t.str_to_str_list(hyperpars['activation_funcs'])

        return number_of_layers, number_of_neurons, activation_funcs

    def read_dataset(self):

        dataset = self.config['dataset']

        x = np.loadtxt(dataset['x'], delimiter=',')
        y = np.loadtxt(dataset['y'], delimiter=',')

        if bool(dataset['y_do_one_hot']):
            y = t.do_one_hot(y)

        if bool(dataset['x_do_scaling']):
            x = t.do_scaling(x)

        x_train, x_val, x_test = t.split_nparray_to_3(x, r1=float(dataset['train_part']), r2=float(dataset['validation_part']), r3=float(dataset['test_part']))
        y_train, y_val, y_test = t.split_nparray_to_3(y, r1=float(dataset['train_part']), r2=float(dataset['validation_part']), r3=float(dataset['test_part']))

        return x_train, y_train, x_val, y_val, x_test, y_test

    def generate_archs(self):
        layers, neurons, activations_funcs = self.read_hyperpars()

        possible_layers = []

        for i in neurons:
            for j in activations_funcs:
                layer_blueprint = keras.layers.Dense(i, activation=j)
                possible_layers.append(layer_blueprint)
                #  fixnout jmena mozna??????????????????????????????????????????????????????????????????????


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
        # #smazat
        # for i in model_collection:
        #     print(i)
        #     for j in i.layers:
        #         print(j)

        return model_collection


