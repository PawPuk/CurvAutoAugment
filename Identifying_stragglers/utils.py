from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

from neural_networks import SimpleNN

EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500
CRITERION = torch.nn.CrossEntropyLoss()


def introduce_label_noise(dataset: Union[Dataset, TensorDataset], noise_rate: float = 0.0) -> List[int]:
    """ Adds noise to the dataset, and returns the list of indices of the so-creates noisy-labels.

    :param dataset: Dataset or TensorDataset object which labels we want to poison
    :param noise_rate: the ratio of the added label noise. After this, the dataset will contain (100*noise_rate)%
    noisy-labels (assuming all labels were correct prior to calling this function)
    :return: list of indices of the added noisy-labels
    """
    noisy_indices = []  # List to store indices of changed labels
    # Verify if noise_ratio is in appropriate range
    if not 1 > noise_rate >= 0:
        raise ValueError(f'The parameter noise_rate has to be in [0, 1). Value {noise_rate} not allowed.')
    # Extract targets from the dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif isinstance(dataset, TensorDataset):
        targets = dataset.tensors[1]
    else:
        raise TypeError("Dataset provided does not have a recognized structure for applying label noise.")
    # Ensure targets is a tensor for uniform handling
    if isinstance(targets, list):
        targets = torch.tensor(targets)
    # Dynamically compute the number of classes and their indices
    unique_classes = targets.unique().tolist()
    # Apply noise
    for class_label in unique_classes:
        class_indices = torch.where(targets == class_label)[0].tolist()
        num_noisy_labels = int(len(class_indices) * noise_rate)
        noisy_labels_indices = np.random.choice(class_indices, num_noisy_labels, replace=False)
        for idx in noisy_labels_indices:
            original_label = targets[idx].item()
            # Choose a new label different from the original
            new_label_choices = [c for c in unique_classes if c != original_label]
            new_label = np.random.choice(new_label_choices)
            if hasattr(dataset, 'targets'):
                dataset.targets[idx] = new_label
            elif isinstance(dataset, TensorDataset):
                dataset.tensors[1][idx] = new_label
            noisy_indices.append(idx)
    return noisy_indices


def load_data_and_normalize(dataset_name: str, subset_size: int, noise_rate: float = 0.0) -> TensorDataset:
    """ Used to load the data from common datasets available in torchvision, and normalize them. The normalization
    is based on the mean and std of a random subset of the dataset of the size subset_size.

    :param dataset_name: name of the dataset to load. It has to be available in torchvision.datasets
    :param subset_size: used when not working on the full dataset - the results will be less reliable, but the
    complexity will be lowered
    :param noise_rate: used when adding label noise to the dataset. Make sure that noise_ratio is in range of [0, 1)
    :return: random, normalized subset of dataset_name of size subset_size with (noise_rate*subset_size) labels changed
    to introduce label noise
    """
    # Load the train and test datasets based on the 'dataset_name' parameter
    train_dataset = getattr(datasets, dataset_name)(root="./data", train=True, download=True,
                                                    transform=transforms.ToTensor())
    test_dataset = getattr(datasets, dataset_name)(root="./data", train=False, download=True,
                                                   transform=transforms.ToTensor())
    # Add label noise
    introduce_label_noise(train_dataset, noise_rate)
    introduce_label_noise(test_dataset, noise_rate)
    # Concatenate train and test datasets
    full_data = torch.cat([train_dataset.data.unsqueeze(1).float(), test_dataset.data.unsqueeze(1).float()])
    full_targets = torch.cat([train_dataset.targets, test_dataset.targets])
    # Shuffle the combined dataset
    shuffled_indices = torch.randperm(len(full_data))
    full_data, full_targets = full_data[shuffled_indices], full_targets[shuffled_indices]
    # Select a subset based on the 'subset_size' parameter
    subset_data = full_data[:subset_size]
    subset_targets = full_targets[:subset_size]
    # Calculate mean and variance for the subset
    data_means = torch.mean(subset_data, dim=(0, 2, 3)) / 255.0
    data_vars = torch.sqrt(torch.var(subset_data, dim=(0, 2, 3)) / 255.0 ** 2 + EPSILON)
    # Apply the calculated normalization to the subset
    normalize_transform = transforms.Normalize(mean=data_means, std=data_vars)
    normalized_subset_data = normalize_transform(subset_data / 255.0)
    # Create a TensorDataset from the normalized subset. This will make the code significantly faster than passing the
    # normalization transform to the DataLoader (as it's usually done).
    normalized_subset = TensorDataset(normalized_subset_data, subset_targets)
    return normalized_subset


def transform_datasets_to_dataloaders(dataset: TensorDataset) -> DataLoader:
    """ Transforms TensorDataset to DataLoader for bull-batch training. The below implementation makes full-batch
    training faster than it would usually be.

    :param dataset: TensorDataset to be transformed
    :return: DataLoader version of dataset ready for full-batch training
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data, target in loader:
        loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=True)
    return loader


def initialize_model() -> Tuple[SimpleNN, optim.SGD]:
    """ Used to initialize the model and optimizer.

    :return: initialized SimpleNN model and SGD optimizer
    """
    model = SimpleNN(28 * 28, 2, 20, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.to(DEVICE)
    return model, optimizer


def train_model(model: SimpleNN, loader: DataLoader, optimizer: optim.SGD,
                compute_radii: bool = True) -> List[Tuple[int, Dict[int, List[torch.Tensor]]]]:
    """

    :param model: Model to be trained
    :param loader: Loader to be used for training
    :param optimizer: Optimized to be used for training
    :param compute_radii: Flag specifying if the user wants to compute the radii of class manifolds during training
    :return: List of tuples of the form (epoch_index, radii_of_class_manifolds), where the radii are stored in a
    dictionary of the form {class_index: [torch.Tensor]}
    """
    epoch_radii = []
    for epoch in range(EPOCHS):
        model.train()
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = CRITERION(output, target)
            loss.backward()
            optimizer.step()
        # Do not compute the radii for the first 20 epochs, as those can be unstable. The number 20 was taken from
        # https://github.com/marco-gherardi/stragglers
        if compute_radii and epoch > 20:
            current_radii = model.radii(loader)
            epoch_radii.append((epoch, current_radii))
    return epoch_radii


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
        test_loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=False)
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
        for _ in range(3):
            model, optimizer = initialize_model()
            # Train the model
            train_model(model, train_loader, optimizer, False)
            # Evaluate the model on test sets
            for i in range(3):
                accuracy = test(model, test_loaders[i])
                accuracies_for_ratio[i].append(accuracy)
        for i in range(3):
            test_accuracies_all_runs[generalisation_settings[i]][reduce_train_ratio].extend(accuracies_for_ratio[i])


def identify_hard_samples(dataset_name, strategy, loader, dataset, level, noise_ratio):
    model, optimizer = initialize_model()
    stragglers_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    stragglers_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    non_stragglers_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    non_stragglers_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    models = train_stop_at_inversion(model, loader, optimizer)
    if None in models:  # Check if stragglers for all classes were found. If not repeat the search
        print('Have to restart because not all stragglers were found.')
        return identify_hard_samples(dataset_name, strategy, loader, dataset, level, noise_ratio)
    stragglers = [0 for _ in range(10)]
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
    stragglers = [int(tensor.sum().item()) for tensor in stragglers]
    print(f'Found {sum(stragglers)} stragglers.')
    if strategy in ["confidence", "energy"]:
        # Introduce noise and increase the threshold
        indices = introduce_label_noise(dataset, noise_ratio)
        print(f'Added {len(indices)} label noise samples.')
        for index in range(len(stragglers)):
            stragglers[index] = stragglers[index] + int(noise_ratio * len(dataset) / len(stragglers))
        print(f'Hence, now the method should find {sum(stragglers)} stragglers.')
        stragglers_data, stragglers_target, non_stragglers_data, non_stragglers_target = (
            identify_hard_samples_with_model_accuracy(
                indices, dataset_name, dataset, stragglers, strategy, level))
    return stragglers_data, stragglers_target, non_stragglers_data, non_stragglers_target


def train_stop_at_inversion(model, train_loader, optimizer) -> List[Union[SimpleNN, None]]:
    prev_radii, radii, models = ([[torch.tensor(float('inf'))] for _ in range(10)], [None for _ in range(10)],
                                 [None for _ in range(10)])
    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = CRITERION(output, target)
            loss.backward()
            optimizer.step()
            break
        if epoch % 5 == 0:
            current_radii = model.radii(train_loader)
            for key in current_radii.keys():
                if models[key] is None and current_radii[key][0].item() > prev_radii[key][0].item() and epoch > 20:
                    models[key] = model.to(DEVICE)
            prev_radii = current_radii
        # print(f'At most {EPOCHS - epoch} epochs remaining. {len([x for x in models if x is not None])} models found.')
    return models


def calculate_energy(logits, T=1.0):
    # Calculate the energy score based on logits
    return -T * torch.logsumexp(logits / T, dim=1)


def identify_hard_samples_with_model_accuracy(gt_indices, dataset_name, dataset, stragglers, mode,
                                              level='dataset'):
    if mode not in ["confidence", "energy"]:
        raise ValueError(f"The mode parameter must be 'confidence' or 'energy'; {mode} is not allowed.")
    if level not in ['dataset', 'class']:
        raise ValueError(f"The level parameter must be 'dataset' or 'class'; {level} is not allowed.")

    results = []
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    # Extract data and targets for stragglers and non-stragglers
    stragglers_data, stragglers_target, non_stragglers_data, non_stragglers_target = [], [], [], []
    total_stragglers_indices = []
    model, optimizer = initialize_model()
    full_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_size, shuffle=False)
    new_loader = None
    for data, target in full_loader:
        new_loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=False)
    train_model(model, new_loader, optimizer, False)
    model.eval()
    with torch.no_grad():
        for data, target in new_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).squeeze()
            if mode == 'energy':
                energy = calculate_energy(output).cpu().numpy()
                results.extend(list(zip(range(len(dataset)), energy, correct.cpu().numpy())))
            else:
                confidence = output.max(1)[0].cpu().numpy()
                results.extend(list(zip(range(len(dataset)), confidence, correct.cpu().numpy())))
    if level == 'dataset':
        stragglers_indices = select_stragglers_dataset_level(total_stragglers_indices, results, sum(stragglers), mode)
    else:  # level == 'class'
        stragglers_indices = select_stragglers_class_level(results, stragglers, mode, dataset)
    total_stragglers_indices.extend(stragglers_indices)
    for idx in stragglers_indices:
        stragglers_data.append(dataset[idx][0])
        stragglers_target.append(dataset[idx][1])
    non_stragglers_indices = set(range(len(dataset))) - set(total_stragglers_indices)
    for idx in non_stragglers_indices:
        non_stragglers_data.append(dataset[idx][0])
        non_stragglers_target.append(dataset[idx][1])
    if len(gt_indices) > 0:
        accuracy = len(set(total_stragglers_indices).intersection(gt_indices)) / len(gt_indices) * 100
        print(f'Correctly guessed {accuracy}% of label noise '
              f'({len(set(total_stragglers_indices).intersection(gt_indices))} out of {len(gt_indices)}).')
    return torch.stack(stragglers_data), torch.tensor(stragglers_target), torch.stack(
        non_stragglers_data), torch.tensor(non_stragglers_target)


def select_stragglers_dataset_level(previous_straggler_indices, results, num_stragglers, mode):
    results.sort(key=lambda x: x[1], reverse=(mode == 'energy'))
    return [x[0] for x in results[:num_stragglers]]


def select_stragglers_class_level(results, stragglers_per_class, mode, dataset):
    targets = [label for _, label in dataset]
    class_results = {i: [] for i in range(10)}
    for idx, score, correct in results:
        class_label = targets[idx]  # Use the passed targets list/tensor
        class_results[class_label.item() if hasattr(class_label, 'item') else class_label].append((idx, score, correct))
    stragglers_indices = []
    for class_label, class_result in class_results.items():
        class_result.sort(key=lambda x: x[1], reverse=(mode == 'energy'))
        num_stragglers = stragglers_per_class[class_label]
        stragglers_indices.extend([x[0] for x in class_result[:num_stragglers]])
    return stragglers_indices


def test(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            break
    if total == 0:
        return -1
    return 100 * correct / total


def plot_radii(X_type, all_radii, dataset_name, save=False):
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(all_radii)))  # Darker to lighter blues
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for run_index in range(len(all_radii)):
        radii = all_radii[run_index]
        for i, ax in enumerate(axes.flatten()):
            y = [radii[j][1][i] for j in range(len(radii))]
            X = [radii[j][0] for j in range(len(radii))]
            ax.plot(X, y, color=colors[run_index], linewidth=3)
            if run_index == 0:
                ax.set_title(f'Class {i} Radii Over {X_type}')
                ax.set_xlabel(X_type)
                ax.set_ylabel('Radius')
                ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f'Figures/radii_over_{X_type}_on_{dataset_name}.png')
        plt.savefig(f'Figures/radii_over_{X_type}_on_{dataset_name}.pdf')
