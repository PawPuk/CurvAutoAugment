import pickle
from statistics import mean, median
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torchvision
from torch.utils.data import ConcatDataset, Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm


PDATA = 8192  # number of elements in the data set
DATA_BLOCK = 1  # Data block to use within the full data set
EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
tdDATASET = torchvision.datasets.MNIST  # the dataset (MNIST, KMNIST, FashionMNIST, CIFAR10)


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
    if scatter_points is not None:
        scatter_y = scipy.stats.norm.pdf(scatter_points, mean, std)
        ax.scatter(scatter_points, scatter_y, color=color, s=50, edgecolor='black', zorder=5)


def plot_statistics(stats, scatter_points=None):
    # Extract data for plotting
    min_distances_same = [d['min_distance_same_class'] for d in stats]
    min_distances_diff = [d['min_distance_diff_class'] for d in stats]
    k_smallest_same = [d['k_smallest_same_class'] for d in stats]
    k_smallest_diff = [d['k_smallest_diff_class'] for d in stats]
    avg_distances_same = [d['avg_distance_same_class'] for d in stats]
    avg_distances_diff = [d['avg_distance_diff_class'] for d in stats]
    # Create figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
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
        ax.axvline(median, color='grey', linestyle='--', label='Median')  # Draw a vertical line at the median
        ax.set_title(title)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def train_model(model, train_loader, optimizer, criterion, epochs, single_batch=True):
    epoch_radii, error_radii = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in tqdm(range(epochs)):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if single_batch:
                print('adwdasd')
                break
        if single_batch:
            current_radii = model.radii(train_loader)
            current_train_error = 1 - 0.01*(test(model, train_loader))
            if current_train_error <= 0.5 or True:
                epoch_radii.append((epoch, current_radii))
                error_radii.append((current_train_error, current_radii))
    return epoch_radii, error_radii





def train_stop_at_inversion(model, train_loader, optimizer, criterion, epochs):
    prev_radii, radii, models = ([[torch.tensor(float('inf'))] for _ in range(10)], [None for _ in range(10)],
                                 [None for _ in range(10)])
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def test(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            break
    return 100 * correct / total


def plot_radii(X_type, all_radii, save=False):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for run_index in range(len(all_radii)):
        radii = all_radii[run_index]
        for i, ax in enumerate(axes.flatten()):
            y = [radii[j][1][i] for j in range(len(radii))]
            X = [radii[j][0] for j in range(len(radii))]
            ax.plot(X, y, marker='o')
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

