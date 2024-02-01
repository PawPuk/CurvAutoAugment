import numpy as np
from torch.utils.data import Subset, ConcatDataset
import torch
import torchvision
from tqdm import tqdm

from utils import (calculate_accuracy, estimate_entanglement, plot_entanglements_and_accuracies,
                   load_models_pretrained_on_cifar10)


seed = 42
# Set random seed for reproducibility
np.random.seed(seed)
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
    splitted_train_dataset[i] = Subset(train_dataset, train_class_indices[i][:500])
    splitted_test_dataset[i] = Subset(test_dataset, test_class_indices[i][:500])
    splitted_full_dataset[i] = ConcatDataset((splitted_train_dataset[i], splitted_test_dataset[i]))
# Estimate the entanglement of each class manifold
entanglement = estimate_entanglement(splitted_full_dataset, 5)
# Prepare a list of pretrained models
models = load_models_pretrained_on_cifar10()
class_accuracies = {}
for i in range(len(splitted_test_dataset)):
    class_accuracies[f'Class {i}'] = []
for model in tqdm(models, desc='Measuring the class accuracies for different models'):
    model.eval()
    # Calculate accuracy on each class subset
    for i in range(10):
        class_accuracy = calculate_accuracy(splitted_test_dataset[i], model)
        # class_accuracy = calculate_accuracy(splitted_full_dataset[i], model)
        class_accuracies[f'Class {i}'].append(class_accuracy)
print(entanglement)
print(class_accuracies)
plot_entanglements_and_accuracies(entanglement, class_accuracies)
