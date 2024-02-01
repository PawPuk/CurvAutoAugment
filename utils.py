import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# from cuml.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset


def load_models_pretrained_on_cifar10(n=19):
    """ Pretrained models taken from https://github.com/chenyaofo/pytorch-cifar-models

    :param n: Change to integer lower than 19 to use subset of models
    :return: list of pretrained models
    """
    model_names = ["cifar10_resnet20", "cifar10_resnet32", "cifar10_resnet44", "cifar10_resnet56", "cifar10_vgg11_bn",
                   "cifar10_vgg13_bn", "cifar10_vgg16_bn", "cifar10_vgg19_bn", "cifar10_mobilenetv2_x0_5",
                   "cifar10_mobilenetv2_x0_75", "cifar10_mobilenetv2_x1_0", "cifar10_mobilenetv2_x1_4",
                   "cifar10_shufflenetv2_x0_5", "cifar10_shufflenetv2_x1_0", "cifar10_shufflenetv2_x1_5",
                   "cifar10_shufflenetv2_x2_0", "cifar10_repvgg_a0", "cifar10_repvgg_a1", "cifar10_repvgg_a2"]
    models = [torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)
              for model_name in model_names]
    return random.sample(models, n)

# Function to calculate accuracy on a given dataset
def calculate_accuracy(dataset, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=500, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def calculate_sample_accuracy(dataset, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=500, shuffle=False)
    classification_results = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Compare with true labels and record 0s for incorrect and 1s for correct classifications
            results_batch = (predicted == labels).int().tolist()
            classification_results.extend(results_batch)
    return classification_results


def estimate_entanglement(splitted_full_dataset: Dict[int, Subset], ensemble_count=5):
    number_of_classes = len(splitted_full_dataset.keys())
    entanglement_metrics = {}
    for i in range(number_of_classes):
        entanglement_metrics[f'Class {i}'] = []
    for _ in tqdm(range(ensemble_count), desc='Training and evaluating ensemble of LSVC for entanglement estimation'):
        entanglement_matrix = []
        for _ in range(number_of_classes):
            entanglement_matrix.append([])
        for i in range(number_of_classes):
            for j in range(number_of_classes):
                if i == j:
                    entanglement_matrix[i].append(1)
                elif i > j:
                    entanglement_matrix[i].append(entanglement_matrix[j][i])  # This matrix is symmetric
                else:
                    # Select elements from class i and j
                    full_subset = ConcatDataset((splitted_full_dataset[i], splitted_full_dataset[j]))
                    # Prepare data for LSVC
                    X_full = np.vstack([np.array(image).flatten() for image, _ in full_subset])
                    y_full = np.array([label for _, label in full_subset])
                    # Initialize and train LSVC
                    lsvc_model = SVC(kernel='linear')
                    # lsvc_model = LinearSVC()
                    lsvc_model.fit(X_full, y_full)
                    # Predict and calculate accuracy
                    y_pred_test = lsvc_model.predict(X_full)
                    test_accuracy = accuracy_score(y_full, y_pred_test)
                    entanglement_matrix[i].append(test_accuracy)
        print(entanglement_matrix)
        for i in range(10):
            entanglement_metrics[f'Class {i}'].append(1 - sum(entanglement_matrix[i]) / number_of_classes)
    return entanglement_metrics


def plot_entanglements_and_accuracies(entanglement_metrics, class_accuracies):
    def get_mean_std(data):
        means = []
        stds = []
        for key in data.keys():
            means.append(np.mean(data[key]))
            stds.append(np.std(data[key]))
        return means, stds

    # Calculate means and standard deviations
    means1, stds1 = get_mean_std(entanglement_metrics)
    means2, stds2 = get_mean_std(class_accuracies)
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Subplot 1
    axs[0].errorbar(range(len(means1)), means1, yerr=stds1, fmt='o', color='b', ecolor='r', elinewidth=3, capsize=0)
    axs[0].set_title('Dataset 1')
    axs[0].set_xticks(range(len(means1)))
    axs[0].set_xticklabels(entanglement_metrics.keys(), rotation=45)
    axs[0].set_xlabel('Class')
    axs[0].set_ylabel('Values')
    axs[0].grid(True)
    # Subplot 2
    axs[1].errorbar(range(len(means2)), means2, yerr=stds2, fmt='o', color='g', ecolor='r', elinewidth=3, capsize=0)
    axs[1].set_title('Dataset 2')
    axs[1].set_xticks(range(len(means2)))
    axs[1].set_xticklabels(class_accuracies.keys(), rotation=45)
    axs[1].set_xlabel('Class')
    axs[1].set_ylabel('Values')
    axs[1].grid(True)
    plt.tight_layout()
    plt.savefig('Figures/Entanglement_accuracy_measurements_comparison.pdf')
    plt.show()


def find_valuable_data_samples(splitted_test_dataset, splitted_full_dataset, ensemble_count=2):
    number_of_classes = len(splitted_full_dataset.keys())
    valuability_metric = {}
    for i in range(number_of_classes):
        valuability_metric[f'Class {i}'] = []
    for _ in tqdm(range(ensemble_count), desc='Training and evaluating ensemble of LSVC for valuability estimation'):
        valuability_matrix = []
        for i in range(number_of_classes):
            valuability_matrix.append([])
            for _ in range(number_of_classes):
                valuability_matrix[i].append([])
        for i in range(number_of_classes):
            for j in range(number_of_classes):
                if i == j:
                    # Create two identical lists of 1s for the i == j case
                    # list_of_ones = [1] * len(splitted_full_dataset[i])
                    list_of_ones = [i] * len(splitted_full_dataset[i])
                    valuability_matrix[i][j] = (list_of_ones, list_of_ones.copy())
                elif i > j:
                    valuability_matrix[i][j] = valuability_matrix[j][i][::-1]  # Reverse the tuple
                else:
                    # Select elements from class i and j
                    full_subset = ConcatDataset((splitted_full_dataset[i], splitted_full_dataset[j]))
                    # Prepare data for LSVC
                    X_full = np.vstack([np.array(image).flatten() for image, _ in full_subset])
                    y_full = np.array([label for _, label in full_subset])
                    # Initialize and train LSVC
                    lsvc_model = SVC(kernel='linear')
                    # lsvc_model = LinearSVC()
                    lsvc_model.fit(X_full, y_full)
                    # Estimate the valuability of every data sample in the given binary classification problem
                    X_i = np.vstack([np.array(image).flatten() for image, _ in splitted_test_dataset[i]])
                    y_i = np.array([label for _, label in splitted_test_dataset[i]])
                    X_j = np.vstack([np.array(image).flatten() for image, _ in splitted_test_dataset[j]])
                    y_j = np.array([label for _, label in splitted_test_dataset[j]])
                    # Predict the labels for samples from manifold i
                    y_i_pred = lsvc_model.predict(X_i)
                    # Compare with true labels and create a list indicating correct (1) or incorrect (0) classification
                    correct_classification_i = [1 if pred == true else 0 for pred, true in zip(y_i_pred, y_i)]
                    # Similarly, for samples from manifold j
                    y_j_pred = lsvc_model.predict(X_j)
                    correct_classification_j = [1 if pred == true else 0 for pred, true in zip(y_j_pred, y_j)]
                    valuability_matrix[i][j] = (correct_classification_i, correct_classification_j)
        for i in range(10):
            list_of_lists = [valuability_matrix[i][j][0] for j in range(number_of_classes)]
            summed_columns = [sum(values) for values in zip(*list_of_lists)]
            valuability_metric[f'Class {i}'].append(summed_columns)
    return valuability_metric
