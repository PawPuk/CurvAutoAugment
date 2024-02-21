import copy
import pickle
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from sklearn.model_selection import KFold
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from neural_networks import SimpleNN

PDATA = 8192  # number of elements in the data set
DATA_BLOCK = 1  # Data block to use within the full data set
EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500


def load_data(dataset_name):
    DatasetClass = getattr(datasets, dataset_name)
    # Initial dataset for mean/var calculation
    initial_transform = transforms.Compose([transforms.ToTensor()])
    dataset = DatasetClass(root="./data", train=True, download=True, transform=initial_transform)
    # Calculate means and vars
    tensors = [a[0] for n, a in enumerate(dataset)]
    stacked_tensors = torch.stack(tensors)
    data_means = torch.mean(stacked_tensors, dim=(0, 2, 3))
    data_vars = torch.sqrt(torch.var(stacked_tensors, dim=(0, 2, 3)) + EPSILON)
    # Define transform with normalization
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=data_means, std=data_vars)])
    # Load datasets with the transformation applied
    train_dataset = DatasetClass(root="./data", train=True, download=True, transform=transform)
    test_dataset = DatasetClass(root="./data", train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    return train_dataset, test_dataset, full_dataset


def transform_datasets_to_dataloaders(list_of_datasets: List[Dataset], batch_size: int) -> Tuple[DataLoader, ...]:
    return tuple([DataLoader(dataset, batch_size=batch_size, shuffle=False) for dataset in list_of_datasets])


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


def generate_spiral_data(n_points, noise=0.5):
    """
    Generates a 2D spiral dataset with two classes.
    :param n_points: Number of points per class.
    :param noise: Standard deviation of Gaussian noise added to the data.
    :return: data (features), labels
    """
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (1 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
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
    x = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
    y = scipy.stats.norm.pdf(x, mean, std)
    ax.plot(x, y, label=label, color=color)
    ax.axvline(median, color='grey', linestyle='--', label='Median')  # Draw a vertical line at the median
    if scatter_points is not None:
        scatter_y = scipy.stats.norm.pdf(scatter_points, mean, std)
        ax.scatter(scatter_points, scatter_y, color=color, s=50, edgecolor='black', zorder=5)


def create_dataloaders_with_straggler_ratio(straggler_data, non_straggler_data, straggler_target, non_straggler_target,
                                            split_ratio, reduce_train_ratio, reduce_stragglers=True):
    # Randomly shuffle stragglers and non-stragglers
    straggler_perm = torch.randperm(straggler_data.size(0))
    non_straggler_perm = torch.randperm(non_straggler_data.size(0))
    straggler_data, straggler_target = straggler_data[straggler_perm], straggler_target[straggler_perm]
    non_straggler_data, non_straggler_target = non_straggler_data[non_straggler_perm], non_straggler_target[
        non_straggler_perm]
    # Split data into initial train/test sets based on the split_ratio and make sure that split_ratio is correct
    if not 0 <= split_ratio <= 1:
        raise ValueError('The variable split_ratio must be between 0 and 1.')
    train_size_straggler = int(len(straggler_data) * split_ratio)
    train_size_non_straggler = int(len(non_straggler_data) * split_ratio)

    initial_train_stragglers_data = straggler_data[:train_size_straggler]
    initial_train_stragglers_target = straggler_target[:train_size_straggler]
    initial_test_stragglers_data = straggler_data[train_size_straggler:]
    initial_test_stragglers_target = straggler_target[train_size_straggler:]

    initial_train_non_stragglers_data = non_straggler_data[:train_size_non_straggler]
    initial_train_non_stragglers_target = non_straggler_target[:train_size_non_straggler]
    initial_test_non_stragglers_data = non_straggler_data[train_size_non_straggler:]
    initial_test_non_stragglers_target = non_straggler_target[train_size_non_straggler:]
    # Reduce the number of train samples by reduce_train_ratio
    if not 0 <= reduce_train_ratio <= 1:
        raise ValueError('The variable reduce_train_ratio must be between 0 and 1.')
    if reduce_stragglers:
        reduced_train_size_straggler = int(train_size_straggler * reduce_train_ratio)
        reduced_train_size_non_straggler = train_size_non_straggler
    else:
        reduced_train_size_straggler = train_size_straggler
        reduced_train_size_non_straggler = int(train_size_non_straggler * reduce_train_ratio)

    final_train_data = torch.cat((initial_train_stragglers_data[:reduced_train_size_straggler],
                                  initial_train_non_stragglers_data[:reduced_train_size_non_straggler]), dim=0)
    final_train_targets = torch.cat((initial_train_stragglers_target[:reduced_train_size_straggler],
                                     initial_train_non_stragglers_target[:reduced_train_size_non_straggler]), dim=0)
    # Shuffle the final train dataset
    train_permutation = torch.randperm(final_train_data.size(0))
    train_data, train_targets = final_train_data[train_permutation], final_train_targets[train_permutation]
    train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=len(final_train_data), shuffle=True)
    # Create test loaders
    datasets = [(initial_test_stragglers_data, initial_test_stragglers_target),
                (initial_test_non_stragglers_data, initial_test_non_stragglers_target)]
    full_test_data = torch.cat((datasets[0][0], datasets[1][0]), dim=0)  # Concatenate data
    full_test_targets = torch.cat((datasets[0][1], datasets[1][1]), dim=0)  # Concatenate targets
    # Create test loaders based on the ordered datasets
    test_loaders = []
    for data, target in [(full_test_data, full_test_targets)] + datasets:
        test_loader = DataLoader(TensorDataset(data, target), batch_size=1000, shuffle=False)
        test_loaders.append(test_loader)
    return train_loader, test_loaders


def straggler_ratio_vs_generalisation(reduce_train_ratios, straggler_data, straggler_target, non_straggler_data,
                                      non_straggler_target, split_ratio, reduce_stragglers, dataset_name,
                                      test_accuracies_all_runs):
    generalisation_settings = ['full', 'stragglers', 'non_stragglers']
    for reduce_train_ratio in reduce_train_ratios:
        accuracies_for_ratio = [[], [], []]  # Store accuracies for the current ratio across different initializations
        train_loader, test_loaders = create_dataloaders_with_straggler_ratio(straggler_data, non_straggler_data,
                                                                             straggler_target, non_straggler_target,
                                                                             split_ratio, reduce_train_ratio,
                                                                             reduce_stragglers)
        print('Divided data into train and test split.')
        for _ in tqdm(range(3), desc='Repeating the experiment for different model initialisations'):
            if dataset_name == 'CIFAR10':
                model = SimpleNN(32 * 32 * 3, 8, 20, 1)
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.02)
            else:
                model = SimpleNN(28 * 28, 2, 20, 1)
                optimizer = optim.SGD(model.parameters(), lr=0.1)
            model.to(DEVICE)
            criterion = torch.nn.CrossEntropyLoss()
            # Train the model
            train_model(model, train_loader, optimizer, criterion, False)
            # Evaluate the model on test sets
            for i in range(3):
                accuracy = test(model, test_loaders[i], False)
                accuracies_for_ratio[i].append(accuracy)
        for i in range(3):
            test_accuracies_all_runs[generalisation_settings[i]][reduce_train_ratio].extend(accuracies_for_ratio[i])


def identify_hard_samples(dataset_name, strategy, loader, dataset):
    if dataset_name == 'CIFAR10':
        model = SimpleNN(32*32*3, 8, 20, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.02)
    else:
        model = SimpleNN(28*28, 2, 20, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    stragglers_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    stragglers_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    non_stragglers_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    non_stragglers_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    models = train_stop_at_inversion(model, loader, optimizer, criterion)
    if None in models:  # Check if stragglers for all classes were found. If not repeat the search
        return identify_hard_samples(dataset_name, strategy, loader, dataset)
    stragglers = [None for _ in range(10)]
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        for i in range(10):
            stragglers[i] = ((torch.argmax(models[i](data), dim=1) != target) & (target == i))
            current_non_stragglers = (torch.argmax(models[i](data), dim=1) == target) & (target == i)
            # Concatenate the straggler data and targets
            stragglers_data = torch.cat((stragglers_data, data[stragglers[i]]), dim=0)
            stragglers_target = torch.cat((stragglers_target, target[stragglers[i]]), dim=0)
            # Concatenate the non-straggler data and targets
            non_stragglers_data = torch.cat((non_stragglers_data, data[current_non_stragglers]), dim=0)
            non_stragglers_target = torch.cat((non_stragglers_target, target[current_non_stragglers]), dim=0)

    if strategy == "model":
        stragglers_data, stragglers_target, non_stragglers_data, non_stragglers_target = (
            identify_hard_samples_with_model_accuracy(model, dataset, optimizer, criterion, len(stragglers_data)))
    return stragglers_data, stragglers_target, non_stragglers_data, non_stragglers_target


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


def train_model(model, train_loader, optimizer, criterion, single_batch=True, test_loader=None):
    epoch_radii, error_radii = [], []
    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
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
                current_error = 1 - 0.01 * (test(model, train_loader))
            if current_error <= 0.5 or True:
                epoch_radii.append((epoch, current_radii))
                error_radii.append((current_error, current_radii))
    return epoch_radii, error_radii


def train_stop_at_inversion(model, train_loader, optimizer, criterion) -> List[SimpleNN]:
    prev_radii, radii, models = ([[torch.tensor(float('inf'))] for _ in range(10)], [None for _ in range(10)],
                                 [None for _ in range(10)])
    count = 0
    epochs = copy.copy(EPOCHS)
    while None in models and epochs > 0:
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            break
        current_radii = model.radii(train_loader)
        for key in current_radii.keys():
            if models[key] is None and current_radii[key][0].item() > prev_radii[key][0].item() and count > 20:
                models[key] = model.to(DEVICE)
        epochs -= 1
        count += 1
        prev_radii = current_radii
        """if len([x for x in models if x is not None]) > 0:
            break"""
        print(f'At most {epochs} epochs remaining. {len([x for x in models if x is not None])} models found.')
    return models


def identify_hard_samples_with_model_accuracy(model, dataset, optimizer, criterion, number_of_stragglers):
    # Using KFold cross-validation to train and evaluate model
    kfold = KFold(n_splits=5, shuffle=True)
    results = []
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    for fold, (train_idx, test_idx) in enumerate(kfold.split(indices)):
        # Convert indices to boolean mask for simplicity
        train_mask = torch.zeros(dataset_size, dtype=bool)
        train_mask[train_idx] = True
        # DataLoader for the entire dataset for GD
        full_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_size, shuffle=False)
        # Training phase
        model.train()
        for epoch in range(EPOCHS):
            for data, target in full_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data[train_mask])
                loss = criterion(output, target[train_mask])
                loss.backward()
                optimizer.step()
        # Evaluation phase
        model.eval()
        test_mask = torch.zeros(dataset_size, dtype=bool)
        test_mask[test_idx] = True
        with torch.no_grad():
            for data, target in full_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data[test_mask])
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target[test_mask].view_as(pred)).squeeze()
                confidence = output.max(1)[0].cpu().numpy()
                # Save results: index, confidence, correctness
                results.extend(list(zip(test_idx, confidence, correct.cpu().numpy())))
    # Identify and separate stragglers based on confidence
    results.sort(key=lambda x: x[1])  # Sort by confidence
    stragglers_idx = [x[0] for x in results[:number_of_stragglers]]
    non_stragglers_idx = [x[0] for x in results[number_of_stragglers:]]
    # Extract data and targets for stragglers and non-stragglers
    stragglers_data = torch.stack([dataset[i][0] for i in stragglers_idx], dim=0)
    stragglers_target = torch.tensor([dataset[i][1] for i in stragglers_idx])
    non_stragglers_data = torch.stack([dataset[i][0] for i in non_stragglers_idx], dim=0)
    non_stragglers_target = torch.tensor([dataset[i][1] for i in non_stragglers_idx])
    return stragglers_data, stragglers_target, non_stragglers_data, non_stragglers_target


def test(model, test_loader, single_batch=True):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
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
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(all_radii)))  # Darker to lighter blues
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for run_index in range(len(all_radii)):
        radii = all_radii[run_index]
        for i, ax in enumerate(axes.flatten()):
            y = [radii[j][1][i] for j in range(len(radii))]
            X = [radii[j][0] for j in range(len(radii))]
            ax.plot(X, y, color=colors[run_index])
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
