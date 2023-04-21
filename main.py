import tensorflow as tf
from tensorflow import keras
import configparser as config
import tools as t
from arch_generator import ArchGen


config = config.ConfigParser()
config.read('conf.ini')
visualize_results = config['visualize_results']

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
                  metrics='accuracy')

    # reduce the number of epochs for MNIST datasets
    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=10)

    test_loss, test_accuracy, = model.evaluate(x_test, y_test, verbose=0)

    y_pred_amax, y_test_amax = t.get_predictions(model, x_test, y_test)

    if eval(visualize_results['print_confusion_matrix']):
        cm = t.print_confusion_matrix(y_test_amax, y_pred_amax)
    else:
        cm = 'Confusion matrix was not generated'

    if eval(visualize_results['print_classification_report']):
        cr = t.print_classification_report(y_test_amax, y_pred_amax)
    else:
        cr = 'Classification report was not generated'

    acc, pre, rec, f1 = t.calculate_metrics(y_test_amax, y_pred_amax)

    print("test accuracy: " + str(test_accuracy))
    print(f'sklearn acc= {acc:.4f}')

    model_results = {'model': model, 'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1, 'classification_report': cr, 'confusion_matrix': cm}

    hall_of_fame.append(model_results)


t.sort_hall(hall_of_fame, config)
# hall_of_fame.sort(key=t.sort_by_performance, reverse=True)
print("number of structures tested: " + str(len(hall_of_fame)))
print(len(model_collection))

for i in hall_of_fame[:3]:
    print("Structure performance: \n" + str(i['classification_report']))
    t.print_model_layers_and_activations(i['model'])





