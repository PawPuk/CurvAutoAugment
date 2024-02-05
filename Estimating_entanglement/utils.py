import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# from cuml.svm import LinearSVC
from scipy.stats import pearsonr
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision

from Estimating_entanglement.neural_networks import SingleHyperplaneNet, ThreeHyperplanesNet


def load_dataset(n=-1):
    """

    :param n: number of data samples in train and test set (uses all by default)
    :return:
    """
    # Download CIFAR-10 dataset
    normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize, ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Split the datasets into 10 subsets for each class
    train_class_indices = {i: [] for i in range(10)}
    test_class_indices = {i: [] for i in range(10)}
    splitted_train_dataset, splitted_test_dataset, splitted_full_dataset = {}, {}, {}
    for i, (image, label) in enumerate(train_dataset):
        train_class_indices[label].append(i)
    for i, (image, label) in enumerate(test_dataset):
        test_class_indices[label].append(i)
    for i in range(10):
        splitted_train_dataset[i] = Subset(train_dataset, train_class_indices[i][:n])
        splitted_test_dataset[i] = Subset(test_dataset, test_class_indices[i][:n])
        splitted_full_dataset[i] = ConcatDataset((splitted_train_dataset[i], splitted_test_dataset[i]))
    return splitted_train_dataset, splitted_test_dataset, splitted_full_dataset


def approximate_entanglement_using_lsvc(splitted_full_dataset, i, j):
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
    return round(100*accuracy_score(y_full, y_pred_test), 4)


def train(method, train_loader, j):
    if method == '1hnn':
        model = SingleHyperplaneNet()
    else:
        model = ThreeHyperplanesNet()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    # Define Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # Cosine Annealing Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)
    for _ in range(120):  # loop over the dataset 120 times
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = (labels == j).long()
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                running_loss = 0.0
        # Update learning rate
        scheduler.step()
    return model


def evaluate(model, eval_loader, j):
    # Test the network on the test data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in eval_loader:
            images, labels = data
            labels = (labels == j).long()
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def approximate_entanglement_using_nn(method, splitted_full_dataset, i, j):
    # Select elements from class i and j
    full_subset = ConcatDataset((splitted_full_dataset[i], splitted_full_dataset[j]))
    dataloader = torch.utils.data.DataLoader(full_subset, batch_size=128, shuffle=True, num_workers=2)
    model = train(method, dataloader, j)
    return round(evaluate(model, dataloader, j), 4)


def estimate_entanglement(splitted_full_dataset: Dict[int, Subset], method: str, ensemble_count=5, k=-1):
    if k == -1:
        k = len(splitted_full_dataset.keys())
    entanglement_metrics = {f'Class {i}': [] for i in range(k)}
    for _ in tqdm(range(ensemble_count), desc='Ensemble training and evaluation'):
        entanglement_matrix = [[] for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j:
                    entanglement_matrix[i].append(100)
                elif i > j:
                    entanglement_matrix[i].append(entanglement_matrix[j][i])  # This matrix is symmetric
                else:
                    if method == 'lsvc':
                        entanglement_matrix[i].append(approximate_entanglement_using_lsvc(splitted_full_dataset, i, j))
                    elif method == '1hnn' or method == '3hnn':
                        entanglement_matrix[i].append(approximate_entanglement_using_nn(
                            method, splitted_full_dataset, i, j))
                    else:
                        raise TypeError('You made a typo when setting the method parameter of estimate_entanglement...')
        print(entanglement_matrix)
        for i in range(k):
            entanglement_metrics[f'Class {i}'].append(sum(entanglement_matrix[i]) / k)
    return entanglement_metrics


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


def compute_accuracies_on_classes(splitted_test_dataset, models, k=-1):
    class_accuracies = {}
    if k == -1:
        k = len(splitted_test_dataset)
    for i in range(k):
        class_accuracies[f'Class {i}'] = []
    for model in tqdm(models, desc='Measuring the class accuracies for different models'):
        model.eval()
        # Calculate accuracy on each class subset
        for i in range(k):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            loader = DataLoader(splitted_test_dataset[i], batch_size=500, shuffle=False)
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            class_accuracy = correct / total
            class_accuracies[f'Class {i}'].append(class_accuracy)
    return class_accuracies


def measure_correlation(entanglement_metrics: Tuple, class_accuracies):
    accuracy_mean, accuracy_std = zip(*[(np.mean(values), np.std(values)) for values in class_accuracies.values()])
    for i in range(len(entanglement_metrics)):
        entanglement_mean, entanglement_stds = zip(*[(np.mean(values), np.std(values))
                                                     for values in entanglement_metrics[i].values()])
        correlation, _ = pearsonr(accuracy_mean, entanglement_mean)
        print(f"The correlation between {['SingleHyperplaneNet', 'ThreeHyperplanesNet', 'LSVC'][i]} and class "
              f"accuracies is: {correlation}.")


def plot_entanglements_and_accuracies(entanglement_metrics, class_accuracies):
    def get_mean_std(data):
        means = []
        stds = []
        for key in data.keys():
            means.append(np.mean(data[key]))
            stds.append(np.std(data[key]))
        return means, stds
    n = len(entanglement_metrics)
    # Calculate means and standard deviations
    entanglement_means, entanglement_stds = {}, {}
    for i in range(n):
        entanglement_means[i], entanglement_stds[i] = get_mean_std(entanglement_metrics[i])
    accuracy_means, accuracy_stds = get_mean_std(class_accuracies)
    # Plotting
    fig, axs = plt.subplots(1, n+1, figsize=(12, 6))
    for i in range(n):
        axs[i].errorbar(range(len(entanglement_means[i])), entanglement_means[i], yerr=entanglement_stds[i], fmt='o',
                        color='b', ecolor='r', elinewidth=3, capsize=0)
        axs[i].set_title(f'Dataset {i}')
        axs[i].set_xticks(range(len(entanglement_means[i])))
        axs[i].set_xticklabels(entanglement_metrics[i].keys(), rotation=45)
        axs[i].set_xlabel('Class')
        axs[i].set_ylabel('Values')
        axs[i].grid(True)
    axs[n].errorbar(range(len(accuracy_means)), accuracy_means, yerr=accuracy_stds, fmt='o', color='g', ecolor='r',
                    elinewidth=3, capsize=0)
    axs[n].set_title('Dataset 2')
    axs[n].set_xticks(range(len(accuracy_means)))
    axs[n].set_xticklabels(class_accuracies.keys(), rotation=45)
    axs[n].set_xlabel('Class')
    axs[n].set_ylabel('Values')
    axs[n].grid(True)
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
