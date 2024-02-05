import heapq

import numpy as np
from tqdm import tqdm

from utils import calculate_sample_accuracy, find_valuable_data_samples, load_models_pretrained_on_cifar10, load_dataset

seed = 42
# Set random seed for reproducibility
np.random.seed(seed)
# Download CIFAR-10 dataset
splitted_train_dataset, splitted_test_dataset, splitted_full_dataset = load_dataset(500)
valuability = find_valuable_data_samples(splitted_test_dataset, splitted_full_dataset)
# Prepare a list of pretrained models
models = load_models_pretrained_on_cifar10()
dataset_sample_accuracies = {}
for i in range(len(splitted_test_dataset)):
    dataset_sample_accuracies[f'Class {i}'] = []
for model in tqdm(models, desc='Measuring the class accuracies for different models'):
    model.eval()
    # Calculate accuracy on each class subset
    for i in range(10):
        manifold_sample_accuracies = calculate_sample_accuracy(splitted_test_dataset[i], model)
        # class_accuracy = calculate_accuracy(splitted_full_dataset[i], model)
        dataset_sample_accuracies[f'Class {i}'].append(manifold_sample_accuracies)
lowest_valuability_samples = {}
lowest_accuracy_samples = {}
correctness_of_valuability_estimation = {}
number_of_elements_to_extract = [4, 3, 2]
for key in valuability:
    valuability[key] = [sum(col) / len(col) for col in zip(*valuability[key])]
    dataset_sample_accuracies[key] = [sum(col) / len(col) for col in zip(*dataset_sample_accuracies[key])]
    correctness_of_valuability_estimation[key] = []
for n in number_of_elements_to_extract:
    for key in valuability:
        # Extracting indices of 2 elements with the lowest value
        lowest_valuability_samples[key] = [index for index, _ in heapq.nsmallest(n, enumerate(valuability[key]),
                                                                                 key=lambda x: x[1])]
        lowest_accuracy_samples[key] = [index for index, _ in heapq.nsmallest(n,
                                                                              enumerate(dataset_sample_accuracies[key]),
                                                                              key=lambda x: x[1])]
        correctness_of_valuability_estimation[key].append(100 * len(set(lowest_valuability_samples[key]) &
                                                                    set(lowest_accuracy_samples[key])) / n)
print(correctness_of_valuability_estimation)
