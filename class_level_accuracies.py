import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import initialize_model, load_data_and_normalize, train_model, transform_datasets_to_dataloaders


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def class_level_test(model, device, test_loader):
    model.eval()
    correct = torch.zeros(10, device=device)
    total = torch.zeros(10, device=device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            for label, prediction in zip(target.view_as(pred), pred):
                if label == prediction:
                    correct[label] += 1
                total[label] += 1
    accuracies = (correct / total).cpu().numpy()
    return accuracies


def plot_accuracy_deviations(mean_accuracies, std_accuracies, dataset_name, overfit, subset_size):
    deviations = 1 - mean_accuracies  # Calculate deviations from perfect accuracy
    std_deviations = std_accuracies  # Standard deviations remain the same

    fig, ax = plt.subplots(figsize=(10, 6))
    classes = np.arange(10)
    ax.bar(classes, deviations, yerr=std_deviations, capsize=5, color='skyblue',
           error_kw={'elinewidth':2, 'ecolor':'black'})
    ax.set_xlabel('Class')
    ax.set_ylabel('Deviation from Perfect Accuracy')
    plt.xticks(classes)
    ax.set_ylim(0, max(deviations + std_deviations) * 1.1)  # Adjust the y-axis to better visualize deviations
    plt.tight_layout()
    ax.grid(True, linestyle='--', which='both', alpha=0.7)
    plt.savefig(f'Figures/class_level_{["non", ""][overfit]}overfit_accuracies_on_{subset_size}_{dataset_name}.pdf')
    plt.savefig(f'Figures/class_level_{["non", ""][overfit]}overfit_accuracies_on_{subset_size}_{dataset_name}.png')
    plt.show()


def compute_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(dataloader))[0]  # Returns a batch where the data tensor is the first element
    mean = torch.mean(data)
    std = torch.std(data)
    return mean.item(), std.item()


def main(dataset_name: str, subset_size: int, overfit: bool, runs: int):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_data_and_normalize(dataset_name, subset_size)
    if overfit:
        train_loader = transform_datasets_to_dataloaders(dataset)
        test_loader = train_loader
    else:
        data, targets = dataset.tensors[0], dataset.tensors[1]
        number_of_test_samples = int(subset_size * (10000 / 70000))
        train_dataset = TensorDataset(data[:number_of_test_samples], targets[:number_of_test_samples])
        test_dataset = TensorDataset(data[number_of_test_samples:], targets[number_of_test_samples:])
        print(number_of_test_samples)
        train_loader = transform_datasets_to_dataloaders(train_dataset)
        test_loader = transform_datasets_to_dataloaders(test_dataset)
    accuracies_over_runs = []

    for _ in tqdm(range(runs)):
        model, optimizer = initialize_model()
        train_model(model, train_loader, optimizer, False)
        accuracies = class_level_test(model, device, test_loader)
        accuracies_over_runs.append(accuracies)

    accuracies_over_runs = np.array(accuracies_over_runs)
    mean_accuracies = np.mean(accuracies_over_runs, axis=0)
    std_accuracies = np.std(accuracies_over_runs, axis=0)
    print(mean_accuracies)
    print(std_accuracies)
    plot_accuracy_deviations(mean_accuracies, std_accuracies, dataset_name, overfit, subset_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST and KMNIST.')
    parser.add_argument('--subset_size', default=20000, type=int,
                        help='Specifies the subset of the dataset used for the experiments. Later it will be divided '
                             'into train and testing training and test sets based pm the --train_ratios parameter.')
    parser.add_argument('--overfit', action='store_true', default=False,
                        help='Specifies if the experiment is done on the entire dataset (in which case call the flag) '
                             'or divided into train and test set (in which case do not call the flag)')
    parser.add_argument('--runs', default=100, type=int, help='Specifies how many different models will be '
                                                              'trained to get the average class-level accuracies.')
    args = parser.parse_args()
    main(**vars(args))
