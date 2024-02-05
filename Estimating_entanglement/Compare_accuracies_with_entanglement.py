import numpy as np

from utils import (estimate_entanglement, plot_entanglements_and_accuracies, load_models_pretrained_on_cifar10,
                   load_dataset, compute_accuracies_on_classes, measure_correlation)


# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
# Load the train set, test set, and the whole dataset. Split all into individual classes (dictionary with k keys)
splitted_train_dataset, splitted_test_dataset, splitted_full_dataset = load_dataset(1500)
# Estimate the entanglement of each of the k class manifolds using 3 different approaches
hnn1_entanglement = estimate_entanglement(splitted_full_dataset, '1hnn', 2)
hnn3_entanglement = estimate_entanglement(splitted_full_dataset, '3hnn', 2)
lsvc_entanglement = estimate_entanglement(splitted_full_dataset, 'lsvc', 2)
# Prepare a list of pretrained models
models = load_models_pretrained_on_cifar10()
# Compute the accuracy of the models on the test set
class_accuracies = compute_accuracies_on_classes(splitted_test_dataset, models)
# Compare the estimates obtained via entanglement with actual class accuracies.
measure_correlation((hnn1_entanglement, hnn3_entanglement, lsvc_entanglement), class_accuracies)
plot_entanglements_and_accuracies((hnn1_entanglement, hnn3_entanglement, lsvc_entanglement),
                                  class_accuracies)
