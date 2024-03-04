from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.optim import SGD
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
    # Dynamically compute the number of classes in the 'dataset' and their indices
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


def initialize_model() -> Tuple[SimpleNN, SGD]:
    """ Used to initialize the model and optimizer.

    :return: initialized SimpleNN model and SGD optimizer
    """
    model = SimpleNN(28 * 28, 2, 20, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    model.to(DEVICE)
    return model, optimizer


def train_model(model: SimpleNN, loader: DataLoader, optimizer: SGD,
                compute_radii: bool = True) -> List[Tuple[int, Dict[int, torch.Tensor]]]:
    """

    :param model: model to be trained
    :param loader: DataLoader to be used for training
    :param optimizer: optimizer to be used for training
    :param compute_radii: flag specifying if the user wants to compute the radii of class manifolds during training
    :return: list of tuples of the form (epoch_index, radii_of_class_manifolds), where the radii are stored in a
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


def train_stop_at_inversion(model: SimpleNN, loader: DataLoader, optimizer: SGD) -> Dict[int, Union[SimpleNN, None]]:
    """ Train a model and monitor the radii of class manifolds. When an inversion point is identified for a class, save
    the current state of the model to the 'model' list that is returned by this function.

    :param model: this model will be used to find the inversion point
    :param loader: the program will look for stragglers within the data in this loader
    :param optimizer: used for training
    :return: dictionary mapping an index of a class manifold to a model, which can be used to extract stragglers for
    the given class
    """
    prev_radii, models = {class_idx: torch.tensor(float('inf')) for class_idx in range(10)}, {}
    for epoch in range(EPOCHS):
        model.train()
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = CRITERION(output, target)
            loss.backward()
            optimizer.step()
        # To increase sustainability and reduce complexity we check for the inversion point every 5 epochs.
        if epoch % 5 == 0:
            # Compute radii of class manifolds at this epoch
            current_radii = model.radii(loader)
            for key in current_radii.keys():
                # For each class see if the radii didn't increase -> reached inversion point. We only check after epoch
                # 20 for the same reasons as in train_model()
                if key in models.keys() and current_radii[key] > prev_radii[key] and epoch > 20:
                    models[key] = model.to(DEVICE)
            prev_radii = current_radii
    return models


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


def calculate_energy(logits: Tensor, temperature: float = 1.0):
    # Calculate the energy score based on logits
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def identify_hard_samples_with_model_accuracy(gt_indices, dataset, stragglers, strategy, level):
    # TODO: Add Cross-Validation. Make computation of energy and confidence truly class-level (when strategy == 'class')
    if strategy not in ["confidence", "energy"]:
        raise ValueError(f"The mode parameter must be 'confidence' or 'energy'; {strategy} is not allowed.")
    if level not in ['dataset', 'class']:
        raise ValueError(f"The level parameter must be 'dataset' or 'class'; {level} is not allowed.")

    hard_data, hard_target, easy_data, easy_target, results, total_hard_indices = [], [], [], [], [], []
    model, optimizer = initialize_model()
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data, target in loader:  # This is done to increase the speed (works due to full-batch setting)
        loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=False)
    train_model(model, loader, optimizer, False)
    model.eval()
    # Iterate through the data samples in 'loader'; compute and save their confidence/energy (depending on 'strategy')
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).squeeze()
            if strategy == 'energy':
                energy = calculate_energy(output).cpu().numpy()
                results.extend(list(zip(range(len(dataset)), energy, correct.cpu().numpy())))
            else:
                confidence = output.max(1)[0].cpu().numpy()
                results.extend(list(zip(range(len(dataset)), confidence, correct.cpu().numpy())))
    if level == 'dataset':
        stragglers_indices = select_stragglers_dataset_level(total_hard_indices, results, sum(stragglers), strategy)
    else:  # level == 'class'
        stragglers_indices = select_stragglers_class_level(results, stragglers, strategy, dataset)
    total_hard_indices.extend(stragglers_indices)
    for idx in stragglers_indices:
        hard_data.append(dataset[idx][0])
        hard_target.append(dataset[idx][1])
    easy_indices = set(range(len(dataset))) - set(total_hard_indices)
    for idx in easy_indices:
        easy_data.append(dataset[idx][0])
        easy_target.append(dataset[idx][1])
    if len(gt_indices) > 0:
        accuracy = len(set(total_hard_indices).intersection(gt_indices)) / len(gt_indices) * 100
        print(f'Correctly guessed {accuracy}% of label noise '
              f'({len(set(total_hard_indices).intersection(gt_indices))} out of {len(gt_indices)}).')
    return torch.stack(hard_data), torch.tensor(hard_target), torch.stack(easy_data), torch.tensor(easy_target)


def identify_hard_samples(strategy: str, loader: DataLoader, dataset: TensorDataset, level: str,
                          noise_ratio: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """ This function divides 'loader' (or 'dataset', depending on the used 'strategy') into hard and easy samples.

    :param strategy: specifies the strategy used for identifying hard samples; only 'stragglers', 'confidence' and
    'energy' allowed
    :param loader: DataLoader that contains the data to be divided into easy and hard samples (used to find stragglers)
    :param dataset: Dataset that contains the data to be divided into easy and hard samples (used in confidence- and
    energy-based methods)
    :param level: specifies the level at which confidence and energy are computed; only 'dataset' and 'class' allowed
    :param noise_ratio: the ratio of the added label noise; used for confidence- and energy-based methods
    :return: tuple containing the identified hard and easy samples
    """
    model, optimizer = initialize_model()
    # The following are used to store all stragglers and non-stragglers
    hard_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    hard_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    easy_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    easy_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    # Look for inversion point for each class manifold
    models = train_stop_at_inversion(model, loader, optimizer)
    # Check if stragglers for all classes were found. If not repeat the search
    if set(models.keys()) != set(range(10)):
        print('Have to restart because not all stragglers were found.')
        return identify_hard_samples(strategy, loader, dataset, level, noise_ratio)
    # The following is used to know the distribution of stragglers between classes
    stragglers = [torch.tensor(False) for _ in range(10)]
    # Iterate through all data samples in 'loader' and divide them into stragglers/non-stragglers
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        for class_idx in range(10):
            # Find stragglers and non-stragglers for the class manifold
            stragglers[class_idx] = ((torch.argmax(models[class_idx](data), dim=1) != target) & (target == class_idx))
            current_non_stragglers = (torch.argmax(models[class_idx](data), dim=1) == target) & (target == class_idx)
            # Save stragglers and non-stragglers from class 'class_idx' to the outer scope (outside of this for loop)
            hard_data = torch.cat((hard_data, data[stragglers[class_idx]]), dim=0)
            hard_target = torch.cat((hard_target, target[stragglers[class_idx]]), dim=0)
            easy_data = torch.cat((easy_data, data[current_non_stragglers]), dim=0)
            easy_target = torch.cat((easy_target, target[current_non_stragglers]), dim=0)
    # Compute the class-level number of stragglers
    stragglers = [int(tensor.sum().item()) for tensor in stragglers]
    print(f'Found {sum(stragglers)} stragglers.')
    if strategy in ["confidence", "energy"]:
        # Introduce noise and increase the threshold
        indices = introduce_label_noise(dataset, noise_ratio)
        print(f'Poisoned {len(indices)} labels.')
        for class_idx in range(len(stragglers)):
            stragglers[class_idx] = stragglers[class_idx] + int(noise_ratio * len(dataset) / len(stragglers))
        print(f'Hence, now the method should find {sum(stragglers)} stragglers.')
        hard_data, hard_target, easy_data, easy_target = identify_hard_samples_with_model_accuracy(indices, dataset,
                                                                                                   stragglers, strategy,
                                                                                                   level)
    return hard_data, hard_target, easy_data, easy_target


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


def plot_radii(all_radii: List[List[Tuple[int, Dict[int, torch.Tensor]]]], dataset_name: str, save: bool = False):
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(all_radii)))  # Darker to lighter blues
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for run_index in range(len(all_radii)):
        radii = all_radii[run_index]
        for i, ax in enumerate(axes.flatten()):
            y = [radii[j][1][i] for j in range(len(radii))]
            X = [radii[j][0] for j in range(len(radii))]
            ax.plot(X, y, color=colors[run_index], linewidth=3)
            if run_index == 0:
                ax.set_title(f'Class {i} Radii Over Epoch')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Radius')
                ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f'Figures/radii_on_{dataset_name}.png')
        plt.savefig(f'Figures/radii_on_{dataset_name}.pdf')
