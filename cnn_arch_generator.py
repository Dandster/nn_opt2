import itertools
from arch_generator import ArchGen
import tools as t
from tensorflow import keras
from keras.layers import MaxPooling2D


class CArchGen(ArchGen):
    def read_hyperpars(self):
        hyperpars = self.config['hyperpars']

        layer_range = eval(hyperpars['layer_range'])
        number_of_layers = t.str_to_int_list(hyperpars['number_of_conv_layers'])

        if layer_range:
            number_of_layers = range(1, max(number_of_layers) + 1)

        number_of_filters = t.str_to_int_list(hyperpars['number_of_filters'])
        kernel_size = t.str_to_int_tuple(hyperpars['kernel_size'])
        pooling_size = t.str_to_int_tuple(hyperpars['pooling_size'])

        activation_funcs = t.str_to_str_list(hyperpars['activation_funcs'])

        return number_of_layers, number_of_filters, kernel_size, pooling_size, activation_funcs

    def generate_archs(self):
        n_layers, n_filters, kernel_size, pooling_size, activations_funcs = self.read_hyperpars()

        possible_conv_layers = []
        possible_pool_layers = []

        for i in n_filters:
            for j in activations_funcs:
                for k in kernel_size:
                    layer_blueprint = keras.layers.Conv2D(i, kernel_size=k, activation=j)
                    possible_conv_layers.append(layer_blueprint)

        for i in pooling_size:
            layer_blueprint = keras.layers.MaxPooling2D(pool_size=i)
            possible_pool_layers.append(layer_blueprint)

        model_collection = []

        layer_number = 0

        for i in n_layers:
            prod = itertools.product(possible_conv_layers, repeat=i)
            for j in prod:
                for mp in possible_pool_layers:
                    model = keras.Sequential()
                    model.add(keras.layers.Input(shape=self.input_shape))
                    for layer in j:
                        con = layer.get_config()
                        cloned_layer = type(layer).from_config(con)
                        cloned_layer._name = cloned_layer.name + str(layer_number)
                        layer_number = layer_number + 1
                        model.add(cloned_layer)

                        con = mp.get_config()
                        cloned_layer = type(mp).from_config(con)
                        cloned_layer._name = cloned_layer.name + str(layer_number)
                        layer_number = layer_number + 1
                        model.add(cloned_layer)
                    model.add(keras.layers.Flatten())
                    model.add(keras.layers.Dense(self.output_neurons, activation=self.output_function))
                    model_collection.append(model)

        return model_collection
