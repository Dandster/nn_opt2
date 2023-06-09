import configparser as config
import itertools
import numpy as np
import tools as t
from tensorflow import keras


class ArchGen:

    def __init__(self, conf):
        self.config = config.ConfigParser()
        self.config.read(conf)
        self.output_neurons = self.config.getint('learning_settings', 'output_neurons')
        self.output_function = self.config['learning_settings']['output_activation']
        self.selected_loss = self.config['learning_settings']['loss_function']
        self.epochs = self.config.getint('learning_settings', 'epochs')
        self.input_shape = (t.str_to_int_tuple(self.config['learning_settings']['input_shape'])[0])

    def get_learning_params(self):
        return self.selected_loss, self.epochs, self.output_neurons, self.output_function

    def read_hyperpars(self):

        hyperpars = self.config['hyperpars']

        layer_range = eval(hyperpars['layer_range'])
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

        if eval(dataset['y_do_one_hot']):
            y = t.do_one_hot(y)

        if eval(dataset['x_do_scaling']):
            x = t.do_scaling(x)

        x = x.reshape(t.str_to_int_tuple(dataset['x_shape'])[0])

        x_train, x_val, x_test = t.split_nparray_to_3(x, r1=float(dataset['train_part']), r2=float(dataset['validation_part']), r3=float(dataset['test_part']))
        y_train, y_val, y_test = t.split_nparray_to_3(y, r1=float(dataset['train_part']), r2=float(dataset['validation_part']), r3=float(dataset['test_part']))

        return x_train, y_train, x_val, y_val, x_test, y_test

    def generate_archs(self):
        n_layers, n_neurons, activations_funcs = self.read_hyperpars()

        possible_layers = []

        for i in n_neurons:
            for j in activations_funcs:
                layer_blueprint = keras.layers.Dense(i, activation=j)
                possible_layers.append(layer_blueprint)

        model_collection = []

        layer_number = 0

        for i in n_layers:
            prod = itertools.product(possible_layers, repeat=i)
            for j in prod:
                model = keras.Sequential()
                model.add(keras.layers.Input(shape=self.input_shape))
                for layer in j:
                    con = layer.get_config()
                    cloned_layer = type(layer).from_config(con)
                    cloned_layer._name = cloned_layer.name + str(layer_number)
                    layer_number = layer_number + 1
                    model.add(cloned_layer)
                model.add(keras.layers.Dense(self.output_neurons, activation=self.output_function))
                model_collection.append(model)

        return model_collection


