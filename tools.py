import os
import keras
import numpy as np
import threading
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras


def train_model(config, model, x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame, verb):
    model.compile(optimizer=config['learning_settings']['optimizer'],
                  loss=config['learning_settings']['loss_function'],
                  metrics='accuracy')

    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=config.getint('learning_settings', 'batch_size'),
              epochs=config.getint('learning_settings', 'epochs'), verbose=verb)

    y_pred_amax, y_test_amax = get_predictions(model, x_test, y_test)

    if eval(config['visualize_results']['generate_confusion_matrix']):
        cm = print_confusion_matrix(y_test_amax, y_pred_amax)
    else:
        cm = 'Confusion matrix was not generated'

    if eval(config['visualize_results']['generate_classification_report']):
        cr = print_classification_report(y_test_amax, y_pred_amax)
    else:
        cr = 'Classification report was not generated'

    acc, pre, rec, f1 = calculate_metrics(y_test_amax, y_pred_amax, config)

    model_results = {'model': model, 'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1,
                     'classification_report': cr, 'confusion_matrix': cm}

    hall_of_fame.append(model_results)


# def train_models(models, config, x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame):
#     if config.getboolean('learning_settings', 'multithreading'):
#         print(f'Training {len(models)} models using multithreading...')
#         threads = []
#         for model in models:
#             t = threading.Thread(target=train_model, args=(config, model, x_train, y_train, x_val,
#                                                            y_val, x_test, y_test, hall_of_fame, 0))
#             threads.append(t)
#             t.start()
#
#         for t in threads:
#             t.join()
#     else:
#         print(f'Training {len(models)} models using single thread...')
#         for model in models:
#             train_model(config, model, x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame, 1)


def train_models(models, config, x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame):

    num_models = len(models)

    if config.getboolean('learning_settings', 'multithreading'):
        num_threads = min(num_models, config.getint('learning_settings', 'number_of_threads'))
        print(f'Training {num_models} models using {num_threads} threads...')
        threads = []

        for i in range(num_threads):
            t = threading.Thread(target=train_models_on_thread, args=(config, models, x_train, y_train, x_val,
                                                                      y_val, x_test, y_test, hall_of_fame, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
    else:
        print(f'Training {num_models} models using a single thread...')
        for model in models:
            train_model(config, model, x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame, 1)


def train_models_on_thread(config, models, x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame, thread_id):
    num_models = len(models)
    num_threads = min(num_models, config.getint('learning_settings', 'number_of_threads'))

    models_per_thread = num_models // num_threads
    start_index = thread_id * models_per_thread
    end_index = start_index + models_per_thread

    if thread_id == num_threads - 1:
        end_index = num_models

    for i in range(start_index, end_index):
        train_model(config, models[i], x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame, 0)


def str_to_int_list(values):
    og_values = values
    try:
        values = ''.join(values.split())
        splits = values.split(sep=",")
        return remove_duplicates([int(s) for s in splits])
    except ValueError:
        print("Reading config unsuccessful, check the formatting of: " + og_values)
        exit(0)


def str_to_int_tuple(values):
    og_values = values
    try:
        values = "".join(values.split())
        splits = values.strip("()").split("),(")
        tuples = [tuple(map(int, pair.split(","))) for pair in splits]
        return tuples
    except ValueError:
        print("Reading config unsuccessful, check the formatting of: " + og_values)
        exit(0)


def str_to_str_list(values):
    og_values = values
    try:
        values = ''.join(values.split())
        return remove_duplicates(values.split(sep=","))
    except ValueError:
        print("Reading config unsuccessful, check the formatting of: " + og_values)
        exit(0)


def remove_duplicates(input_list):
    return list(dict.fromkeys(input_list))


def do_scaling(x):
    scaler = StandardScaler()
    return scaler.fit_transform(x)


def do_one_hot(y):
    encode = OneHotEncoder()
    return encode.fit_transform(y[:, np.newaxis]).toarray()


def split_nparray_to_3(array, r1, r2, r3):
    if r1 + r2 + r3 != 1.0:
        print("Dataset split (train:validation:test) ratios must sum to 1.0!")
        exit(0)
    else:
        try:
            train, validation, test = np.split(array, [int(r1 * len(array)), int((r1 + r2) * len(array))])
            return train, validation, test
        except ValueError:
            print("Splitting dataset did not result in a given division, try different ratios")
            exit(0)


def get_predictions(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    return y_pred, y_test


def print_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    #  print(cm)

    return cm


def print_classification_report(y_test, y_pred):
    cr = classification_report(y_test, y_pred, digits=5)
    #  print(cr)

    return cr


def calculate_metrics(y_test, y_pred, config):
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average=config["sort_by"]["metrics_type"])
    rec = recall_score(y_test, y_pred, average=config["sort_by"]["metrics_type"])
    f1 = f1_score(y_test, y_pred, average=config["sort_by"]["metrics_type"])

    return acc, pre, rec, f1


def get_model_layers_and_activations(model):
    desc = [f"Number of layers: {len(model.layers)}"]

    for i, layer in enumerate(model.layers):
        desc.append(f"Layer {i + 1}: {layer.name} ({layer.activation.__name__}), Number of neurons: {layer.units}")

    desc = "\n".join(desc)
    return desc


def get_cnn_model_info(model):
    desc = ""
    desc += f"Number of layers: {len(model.layers)}\n"
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D):
            desc += f"Layer {i + 1}: Conv2D, {layer.filters} filters, kernel size {layer.kernel_size}, activation function {layer.activation.__name__}\n"
        elif isinstance(layer, MaxPooling2D):
            desc += f"Layer {i + 1}: MaxPooling2D layer, pool size {layer.pool_size}\n"
        elif isinstance(layer, Flatten):
            desc += f"Layer {i + 1}: Flatten layer\n"
        elif isinstance(layer, Dense):
            desc += f"Layer {i + 1}: Dense layer, {layer.units} neurons, activation function {layer.activation.__name__}\n"
    return desc


def sort_hall(list_of_strucs, config):
    metrics = [config.getboolean('sort_by', 'accuracy'), config.getboolean('sort_by', 'precision'),
               config.getboolean('sort_by', 'recall'), config.getboolean('sort_by', 'f1')]

    if metrics.count(True) != 1:
        print("Sorting metrics were not properly specified, so I am sorting by accuracy")
        list_of_strucs.sort(key=lambda x: x.get('accuracy'), reverse=True)
    elif metrics[0]:
        list_of_strucs.sort(key=lambda x: x.get('accuracy'), reverse=True)
    elif metrics[1]:
        list_of_strucs.sort(key=lambda x: x.get('precision'), reverse=True)
    elif metrics[2]:
        list_of_strucs.sort(key=lambda x: x.get('recall'), reverse=True)
    else:
        list_of_strucs.sort(key=lambda x: x.get('f1'), reverse=True)


def format_model_results(model_results):
    formatted_list = []
    for key, value in model_results.items():
        if isinstance(value, keras.Sequential) and isinstance(value.layers[0], keras.layers.Conv2D):
            formatted_list.append(get_cnn_model_info(value))
        elif isinstance(value, keras.Sequential):
            formatted_list.append(get_model_layers_and_activations(value))
        elif isinstance(value, str):
            formatted_list.append(f"{key}: \n{value}")
        elif isinstance(value, (int, float)):
            formatted_list.append(f"{key}: \n{value:.5f}")
        else:
            formatted_list.append(f"{key}: \n{repr(value)}")
    formatted_string = "\n".join(formatted_list)
    return formatted_string


def save_models(models, how_many, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, model in enumerate(models[:how_many]):
        model_path = os.path.join(folder_path, f"model_{i}")
        os.makedirs(model_path, exist_ok=True)
        model['model'].save(os.path.join(model_path, "model"), save_format='h5')

        description = format_model_results(model)

        description_path = os.path.join(model_path, "description.txt")
        with open(description_path, "w") as f:
            f.write(description)


def print_top_struct(config, hall):
    for i, model in enumerate(hall[:config.getint('visualize_results', 'print_top')]):
        print('////////////////////////////////////////////////////////////////////')
        print(f'[{i}]')
        print(format_model_results(model))
        print('////////////////////////////////////////////////////////////////////')
