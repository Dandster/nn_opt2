import numpy as np
import threading
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras


def train_model(config, model, x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame):
    model.compile(optimizer=config['learning_settings']['optimizer'],
                  loss=config['learning_settings']['loss_function'],
                  metrics='accuracy')

    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=config.getint('learning_settings', 'epochs'), verbose=1)

    test_loss, test_accuracy, = model.evaluate(x_test, y_test, verbose=0)

    y_pred_amax, y_test_amax = get_predictions(model, x_test, y_test)

    if eval(config['visualize_results']['print_confusion_matrix']):
        cm = print_confusion_matrix(y_test_amax, y_pred_amax)
    else:
        cm = 'Confusion matrix was not generated'

    if eval(config['visualize_results']['print_classification_report']):
        cr = print_classification_report(y_test_amax, y_pred_amax)
    else:
        cr = 'Classification report was not generated'

    acc, pre, rec, f1 = calculate_metrics(y_test_amax, y_pred_amax)

    print("test accuracy: " + str(test_accuracy))

    model_results = {'model': model, 'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1,
                     'classification_report': cr, 'confusion_matrix': cm}

    hall_of_fame.append(model_results)


def train_models(models, config, x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame):
    if config.getboolean('learning_settings', 'multithreading'):
        threads = []
        for model in models:
            t = threading.Thread(target=train_model, args=(config, model, x_train, y_train, x_val,
                                                           y_val, x_test, y_test, hall_of_fame))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
    else:
        for model in models:
            train_model(config, model, x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame)


def str_to_int_list(values):
    og_values = values
    try:
        values = ''.join(values.split())
        splits = values.split(sep=",")
        return remove_duplicates([int(s) for s in splits])
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
            train, validation, test = np.split(array, [int(r1 * len(array)), int((r1+r2) * len(array))])
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
    print(cm)

    return cm


def print_classification_report(y_test, y_pred):
    cr = classification_report(y_test, y_pred, digits=5)
    print(cr)

    return cr


def calculate_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return acc, pre, rec, f1


def print_model_layers_and_activations(model):
    print(f"Number of layers: {len(model.layers)}")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i+1}: {layer.name} ({layer.activation.__name__}), Number of neurons: {layer.units}")


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
