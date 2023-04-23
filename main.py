import time
import configparser as config
import tools as t
from arch_generator import ArchGen

start_time = time.time()

config = config.ConfigParser()
config.read('conf.ini')


hall_of_fame = []

ag = ArchGen()
selected_loss, epochs, output_neurons, output_function = ag.get_learning_params()
x_train, y_train, x_val, y_val, x_test, y_test = ag.read_dataset()

model_collection = ag.generate_archs()

t.train_models(model_collection, config, x_train, y_train, x_val, y_val, x_test, y_test, hall_of_fame)

t.sort_hall(hall_of_fame, config)
# hall_of_fame.sort(key=t.sort_by_performance, reverse=True)
print("number of structures tested: " + str(len(hall_of_fame)))
print(len(model_collection))

for i in hall_of_fame[:3]:
    print("Structure performance: \n" + str(i['classification_report']))
    t.print_model_layers_and_activations(i['model'])


end_time = time.time()
total_time = end_time - start_time

print(f"Program completed in {total_time:.2f} seconds.")


