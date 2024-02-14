import copy
import pickle
from statistics import mean, median
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
import torch
import torchvision
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from neural_networks import SimpleNN


PDATA = 8192  # number of elements in the data set
DATA_BLOCK = 1  # Data block to use within the full data set
EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
tdDATASET = torchvision.datasets.FashionMNIST  # the dataset (MNIST, KMNIST, FashionMNIST, CIFAR10)


def load_data():
    dataset = tdDATASET("data", train=True, download=True,
                        transform=torchvision.transforms.ToTensor())
    in_block = lambda n: (DATA_BLOCK - 1) * PDATA <= n < DATA_BLOCK * PDATA
    data_means = torch.mean(torch.cat([a[0] for n, a in enumerate(dataset) if in_block(n)]), dim=0)
    data_vars = torch.sqrt(torch.var(torch.cat([a[0] for n, a in enumerate(dataset) if in_block(n)]), dim=0))
    transf = lambda x: (x - data_means) / (data_vars + EPSILON)
    transform = transforms.Compose([transforms.ToTensor(), transf])

    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    full_dataset = ConcatDataset((train_dataset, test_dataset))
    return train_dataset, test_dataset, full_dataset


def calculate_percentiles(stats, stragglers_stats):
    percentiles = {key: [] for key in stats[0]}  # Initialize dict for percentiles

    for key in stats[0]:
        if key != 'class':
            all_values = [d[key] for d in stats]  # Extract all values for the current key
            # Calculate percentile for each straggler value
            for straggler in stragglers_stats:
                straggler_value = straggler[key]
                percentile = scipy.stats.percentileofscore(all_values, straggler_value, kind='strict')
                percentiles[key].append(percentile)

    return percentiles


def transform_datasets_to_dataloaders(list_of_datasets: List[Dataset], batch_size: int) -> Tuple[DataLoader, ...]:
    return tuple([DataLoader(dataset, batch_size=batch_size, shuffle=False) for dataset in list_of_datasets])


def generate_spiral_data(n_points, noise=0.5):
    """
    Generates a 2D spiral dataset with two classes.
    :param n_points: Number of points per class.
    :param noise: Standard deviation of Gaussian noise added to the data.
    :return: data (features), labels
    """
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (1*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))


def create_spiral_data_loader(data, labels, batch_size=128):
    # Convert the numpy arrays into torch tensors
    data = torch.Tensor(data)
    labels = torch.Tensor(labels).long()  # Labels should be of type Long
    # Create TensorDataset objects
    train_dataset = TensorDataset(data, labels)
    # Create DataLoader objects
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


def load_statistics(filename='statistics.pkl'):
    with open(filename, 'rb') as f:
        statistics = pickle.load(f)
    return statistics


def calculate_statistics(loader, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, targets = [], []
    # Load all data and targets into memory (may require a lot of memory)
    for i, (batch, target) in enumerate(loader):
        data.append(batch.squeeze().to(device))
        targets.append(target.to(device))
        if i == 0:
            break  # This is used to work on the subset (to replicate training scenario or to limit complexity).
    # Use this when processing more than one batch
    """data = torch.stack(data)
    targets = torch.cat(targets)"""
    # Use this when processing a single batch
    data = torch.cat(data, dim=0)
    data_flattened = data.view(data.shape[0], -1)
    targets = torch.cat(targets, dim=0)
    # Pre-compute all pairwise distances (this could be very memory intensive)
    distances = torch.cdist(data_flattened, data_flattened, p=2)
    statistics = []
    print(len(data))
    for i in tqdm(range(len(data)), desc='Calculating statistics for MNIST data samples.'):
        same_class_mask = targets == targets[i]
        diff_class_mask = ~same_class_mask
        same_class_mask[i] = False  # Exclude the current sample
        same_class_distances = distances[i][same_class_mask]
        diff_class_distances = distances[i][diff_class_mask]
        # Calculate statistics
        min_distance_same_class = torch.min(same_class_distances)
        min_distance_diff_class = torch.min(diff_class_distances)
        k_smallest_same_class = torch.topk(same_class_distances, k, largest=False).values.mean()
        k_smallest_diff_class = torch.topk(diff_class_distances, k, largest=False).values.mean()
        avg_distance_same_class = same_class_distances.mean()
        avg_distance_diff_class = diff_class_distances.mean()
        stats = {
            "min_distance_same_class": min_distance_same_class.item(),
            "min_distance_diff_class": min_distance_diff_class.item(),
            "k_smallest_same_class": k_smallest_same_class.item(),
            "k_smallest_diff_class": k_smallest_diff_class.item(),
            "avg_distance_same_class": avg_distance_same_class.item(),
            "avg_distance_diff_class": avg_distance_diff_class.item(),
            "class": targets[i]
        }
        statistics.append(stats)
    with open('statistics.pkl', 'wb') as f:
        pickle.dump(statistics, f)
    return statistics


def plot_gaussian(ax, data, label, color='blue', scatter_points=None):
    mean, std = np.mean(data), np.std(data)
    median = np.median(data)  # Calculate the median
    x = np.linspace(mean - 3*std, mean + 3*std, 1000)
    y = scipy.stats.norm.pdf(x, mean, std)
    ax.plot(x, y, label=label, color=color)
    ax.axvline(median, color='grey', linestyle='--', label='Median')  # Draw a vertical line at the median
    if scatter_points is not None:
        scatter_y = scipy.stats.norm.pdf(scatter_points, mean, std)
        ax.scatter(scatter_points, scatter_y, color=color, s=50, edgecolor='black', zorder=5)


def create_dataloaders_with_straggler_ratio(straggler_data, non_straggler_data, straggler_target, non_straggler_target,
                                            ratio, train_ratio):
    # Randomly shuffle stragglers and non-stragglers
    straggler_perm = torch.randperm(straggler_data.size(0))
    non_straggler_perm = torch.randperm(non_straggler_data.size(0))
    straggler_data, straggler_target = straggler_data[straggler_perm], straggler_target[straggler_perm]
    non_straggler_data, non_straggler_target = non_straggler_data[non_straggler_perm], non_straggler_target[
        non_straggler_perm]
    # Calculate the number of stragglers and non-stragglers for the train/test set based on the ratio
    total_train_stragglers = int(len(straggler_data) * ratio)
    total_test_stragglers = len(straggler_data) - total_train_stragglers
    total_train_non_stragglers = int(round(train_ratio * 70000)) - total_train_stragglers
    total_test_non_stragglers = len(non_straggler_data) - total_train_non_stragglers
    # Create train and test sets
    train_data = torch.cat((straggler_data[:total_train_stragglers], non_straggler_data[:total_train_non_stragglers]),
                           dim=0)
    train_targets = torch.cat(
        (straggler_target[:total_train_stragglers], non_straggler_target[:total_train_non_stragglers]), dim=0)
    full_test_data = torch.cat((straggler_data[-total_test_stragglers:],
                                non_straggler_data[-total_test_non_stragglers:]), dim=0)
    straggler_test_data = straggler_data[-total_test_stragglers:]
    non_straggler_test_data = non_straggler_data[-total_test_non_stragglers:]
    full_test_targets = torch.cat(
        (straggler_target[-total_test_stragglers:], non_straggler_target[-total_test_non_stragglers:]), dim=0)
    straggler_test_targets = straggler_target[-total_test_stragglers:]
    non_straggler_test_targets = non_straggler_target[-total_test_non_stragglers:]
    # Shuffle the datasets and create DataLoaders
    train_permutation = torch.randperm(train_data.size(0))
    train_data, train_targets = train_data[train_permutation], train_targets[train_permutation]
    train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=64, shuffle=True)
    test_loaders = []
    for i in range(3):
        test_permutation = torch.randperm([full_test_data, straggler_test_data, non_straggler_test_data][i].size(0))
        test_data, test_targets = [full_test_data, straggler_test_data, non_straggler_test_data][i][test_permutation], \
            [full_test_targets, straggler_test_targets, non_straggler_test_targets][i][test_permutation]
        test_loaders.append(DataLoader(TensorDataset(test_data, test_targets), batch_size=1000, shuffle=False))
    return train_loader, test_loaders


def interpolate_colors(start_color, end_color, n):
    """
    Generates a list of colors interpolating between start_color and end_color.

    Args:
        start_color (tuple): The RGB tuple for the start color.
        end_color (tuple): The RGB tuple for the end color.
        n (int): The number of colors to generate.

    Returns:
        list of interpolated colors in hex format.
    """
    colors = []
    for i in range(n):
        interpolated_color = [start + (end - start) * i / (n - 1) for start, end in zip(start_color, end_color)]
        colors.append('#' + ''.join(f'{int(c):02x}' for c in interpolated_color))
    return colors


def straggler_ratio_vs_generalisation(straggler_ratios, straggler_data, straggler_target, non_straggler_data,
                                      non_straggler_target, train_ratio):
    settings = ['full', 'stragglers', 'non_stragglers']
    test_accuracies_all_runs = {setting: {ratio: [] for ratio in straggler_ratios} for setting in settings}
    for ratio in straggler_ratios:
        accuracies_for_ratio = [[], [], []]  # Store accuracies for the current ratio across different initializations
        for _ in range(3):  # Train 3 models with different initializations
            train_loader, test_loaders = create_dataloaders_with_straggler_ratio(
                straggler_data, non_straggler_data, straggler_target, non_straggler_target, ratio, train_ratio)
            # Prepare for training
            model = SimpleNN(28 * 28, 2, 40, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            criterion = torch.nn.CrossEntropyLoss()
            num_epochs = 500
            # Train the model
            train_model(model, train_loader, optimizer, criterion, num_epochs, False)
            # Evaluate the model on test sets
            for i in range(3):
                accuracy = test(model, test_loaders[i], False)
                accuracies_for_ratio[i].append(accuracy)
        for i in range(3):
            test_accuracies_all_runs[settings[i]][ratio] = accuracies_for_ratio[i]
    # Compute the average and standard deviation of accuracies for each ratio
    avg_accuracies = {settings[i]: [np.mean(test_accuracies_all_runs[settings[i]][ratio]) for ratio in straggler_ratios]
                      for i in range(3)}
    std_accuracies = {settings[i]: [np.std(test_accuracies_all_runs[settings[i]][ratio]) for ratio in straggler_ratios]
                      for i in range(3)}
    return avg_accuracies, std_accuracies


def plot_statistics(stats, scatter_points=None, i=None, fig=None, axs=None):
    # Extract data for plotting
    min_distances_same = [d['min_distance_same_class'] for d in stats]
    min_distances_diff = [d['min_distance_diff_class'] for d in stats]
    k_smallest_same = [d['k_smallest_same_class'] for d in stats]
    k_smallest_diff = [d['k_smallest_diff_class'] for d in stats]
    avg_distances_same = [d['avg_distance_same_class'] for d in stats]
    avg_distances_diff = [d['avg_distance_diff_class'] for d in stats]
    # Create figure and subplots
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    if i is not None:
        fig.suptitle(f'Gaussian Distributions of Distances with Scatter Points for Class {i}')
    else:
        fig.suptitle('Gaussian Distributions of Distances with Scatter Points')
    # Titles for subfigures
    titles = ['Minimum Distances', 'K-Smallest Distances', 'Average Distances']
    # Data to plot in each subfigure
    data_to_plot = [
        (min_distances_same, min_distances_diff),
        (k_smallest_same, k_smallest_diff),
        (avg_distances_same, avg_distances_diff),
    ]
    # Extract scatter points for each plot, if provided
    scatter_data = None if scatter_points is None else [(
        [d['min_distance_same_class'] for d in scatter_points], [d['min_distance_diff_class'] for d in scatter_points]),
        ([d['k_smallest_same_class'] for d in scatter_points], [d['k_smallest_diff_class'] for d in scatter_points]),
        ([d['avg_distance_same_class'] for d in scatter_points], [d['avg_distance_diff_class'] for d in scatter_points])
    ]
    # Colors for plots
    colors = [('blue', 'red'), ('blue', 'red'), ('blue', 'red')]
    # Plot each set of Gaussian distributions
    for ax, (data_same, data_diff), title, (color_same, color_diff), scatter_data in zip(axs, data_to_plot, titles,
                                                                                         colors, scatter_data or [
                (None, None)] * 3):
        plot_gaussian(ax, data_same, 'Same Class', color=color_same, scatter_points=scatter_data[0])
        plot_gaussian(ax, data_diff, 'Different Class', color=color_diff, scatter_points=scatter_data[1])
        ax.set_title(title)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axs


def extract_top_samples(stats, full_dataset: Union[ConcatDataset, Dataset], percentile=90, n_classes=10):
    """
    Extracts top 8% samples based on six different statistics for each class.

    :param stats: List of dictionaries containing statistics for each sample
    :param full_dataset: Full dataset object
    :param percentile: Percentile to consider for the top samples (default is 92 for top 8%)
    :param n_classes: Number of classes in the dataset
    :return: A dictionary of 6 lists, each containing indices of top samples based on a specific statistic
    """
    # Initialize the result dictionary
    top_samples = {
        "min_distance_same_class": [],
        "min_distance_diff_class": [],
        "k_smallest_same_class": [],
        "k_smallest_diff_class": [],
        "avg_distance_same_class": [],
        "avg_distance_diff_class": []
    }
    # Attempt to aggregate targets from the full_dataset
    try:
        # If full_dataset is a simple dataset with .targets
        if hasattr(full_dataset, 'targets'):
            all_targets = full_dataset.targets
        # If full_dataset is a ConcatDataset
        elif hasattr(full_dataset, 'datasets'):
            all_targets = []
            for dataset in full_dataset.datasets:
                all_targets.extend(dataset.targets)
        else:
            raise AttributeError("Dataset does not have targets or datasets attributes")
    except Exception as e:
        raise ValueError(f"Error processing dataset targets: {str(e)}")
    for key in top_samples.keys():
        # Initialize lists to hold top indices for each class based on the current statistic
        top_indices_per_class = {k: [] for k in range(n_classes)}
        # Separate samples by class
        for class_idx in range(n_classes):
            indices_of_class = [i for i, t in enumerate(all_targets) if t == class_idx]
            stats_of_class = [stats[i][key] for i in indices_of_class]
            # Calculate the threshold for the top percentile for the current class & statistic
            threshold = np.percentile(stats_of_class, percentile)
            # Select samples that are above the threshold
            top_samples_indices = [indices_of_class[i] for i, s in enumerate(stats_of_class) if s >= threshold]
            top_indices_per_class[class_idx].extend(top_samples_indices)
        # Flatten the indices for easy access
        top_samples[key] = [index for class_indices in top_indices_per_class.values() for index in class_indices]
    return top_samples


def calculate_similarity_matrix(top_samples):
    keys = list(top_samples.keys())
    n = len(keys)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Calculate intersection and union
            set_i = set(top_samples[keys[i]])
            set_j = set(top_samples[keys[j]])
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            # Calculate similarity as intersection over union
            similarity_matrix[i, j] = intersection / union if union != 0 else 0
    return keys, similarity_matrix


def plot_similarity_matrix(keys, similarity_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis', xticklabels=keys, yticklabels=keys, square=True,
                cbar_kws={"shrink": .82}, linewidths=.1)
    plt.title('Similarity Between Top Samples by Statistic')
    plt.xlabel('Statistics')
    plt.ylabel('Statistics')
    plt.savefig('Figures/similarity_between_metrics.png')


def create_data_splits(full_dataset, estimated_stragglers):
    splits = {}
    all_indices = set(range(len(full_dataset)))  # Assuming continuous indexing
    for key in estimated_stragglers.keys():
        # Stragglers for the current statistic
        train_indices = set(estimated_stragglers[key])
        # Non-stragglers for the current statistic
        test_indices = all_indices - train_indices
        # Creating Subset objects for train and test splits
        train_split = Subset(full_dataset, list(train_indices))
        test_split = Subset(full_dataset, list(test_indices))
        splits[key] = {'train': train_split, 'test': test_split}
    return splits


def train_model(model, train_loader, optimizer, criterion, epochs, single_batch=True, test_loader=None):
    epoch_radii, error_radii = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if single_batch:
                break
        if single_batch:
            current_radii = model.radii(train_loader)
            if test_loader is not None:
                current_error = 1 - 0.01 * (test(model, test_loader))
            else:
                current_error = 1 - 0.01*(test(model, train_loader))
            if current_error <= 0.5 or True:
                epoch_radii.append((epoch, current_radii))
                error_radii.append((current_error, current_radii))
    return epoch_radii, error_radii


def train_stop_at_inversion(model, train_loader, optimizer, criterion, epochs):
    prev_radii, radii, models = ([[torch.tensor(float('inf'))] for _ in range(10)], [None for _ in range(10)],
                                 [None for _ in range(10)])
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    while None in models and epochs > 0:
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            break
        current_radii = model.radii(train_loader)
        for key in current_radii.keys():
            if models[key] is None and current_radii[key][0].item() > prev_radii[key][0].item() and count > 20:
                models[key] = model
        epochs -= 1
        count += 1
        prev_radii = current_radii
        """if len([x for x in models if x is not None]) > 0:
            break"""
        print(f'At most {epochs} epochs remaining. {len([x for x in models if x is not None])} models found.')
    return models


def test(model, test_loader, single_batch=True):
    model.eval()
    correct, total = 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if single_batch:
                break
    if total == 0:
        return -1
    return 100 * correct / total


def plot_radii(X_type, all_radii, save=False):
    start_color = (0, 0, 0)
    end_color = (192, 192, 192)  # This represents a silver color
    colors = interpolate_colors(start_color, end_color, len(all_radii))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for run_index in range(len(all_radii)):
        radii = all_radii[run_index]
        for i, ax in enumerate(axes.flatten()):
            y = [radii[j][1][i] for j in range(len(radii))]
            X = [radii[j][0] for j in range(len(radii))]
            ax.plot(X, y, marker='o', color=colors[run_index])
            if run_index == 0:
                ax.set_title(f'Class {i} Radii Over {X_type}')
                ax.set_xlabel(X_type)
                ax.set_ylabel('Radius')
                ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f'Figures/radii_over_{X_type}.png')
        plt.savefig(f'Figures/radii_over_{X_type}.pdf')


def plot_spiral_data(data, labels, title='Spiral Data'):

    """
    Plots the 2D spiral data.
    :param data: Features (data points).
    :param labels: Corresponding labels for the data points.
    :param title: Title for the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Spectral, s=25, edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

