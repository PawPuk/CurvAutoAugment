from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

from neural_networks import BasicBlock, SimpleNN, ResNet

EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500
CRITERION = torch.nn.CrossEntropyLoss()


def introduce_label_noise(dataset: Union[Dataset, TensorDataset], noise_rate: float = 0.0) -> List[int]:
    """ Adds noise to the dataset, and returns the list of indices of the so-creates noisy-labels.

    :param dataset: Dataset or TensorDataset object whose labels we want to poison
    :param noise_rate: the ratio of the added label noise. After this, the dataset will contain (100*noise_rate)%
    noisy-labels (assuming all labels were correct prior to calling this function)
    :return: list of indices of the added noisy-labels
    """
    if not 0 <= noise_rate < 1:
        raise ValueError(f'The parameter noise_rate has to be in [0, 1). Value {noise_rate} not allowed.')
    # Extract targets from the dataset
    if hasattr(dataset, 'targets'):
        original_targets = dataset.targets.clone()
    elif isinstance(dataset, TensorDataset):
        original_targets = dataset.tensors[1].clone()
    else:
        raise TypeError("Dataset provided does not have a recognized structure for applying label noise.")

    total_samples = len(original_targets)
    num_noisy_labels = int(total_samples * noise_rate)
    # Randomly select indices to introduce noise
    all_indices = np.arange(total_samples)
    noisy_indices = np.random.choice(all_indices, num_noisy_labels, replace=False)
    # Dynamically compute the number of classes in the 'dataset' and their indices
    unique_classes = original_targets.unique().tolist()
    # Apply noise to the selected indices
    for idx in noisy_indices:
        original_label = original_targets[idx].item()
        # Choose a new label different from the original
        new_label_choices = [c for c in unique_classes if c != original_label]
        new_label = np.random.choice(new_label_choices)
        # Update the dataset with the new label
        if hasattr(dataset, 'targets'):
            dataset.targets[idx] = torch.tensor(new_label, dtype=original_targets.dtype)
        elif isinstance(dataset, TensorDataset):
            dataset.tensors[1][idx] = torch.tensor(new_label, dtype=original_targets.dtype)

    return noisy_indices.tolist()


def load_data_and_normalize(dataset_name: str, subset_size: int) -> TensorDataset:
    """ Used to load the data from common datasets available in torchvision, and normalize them. The normalization
    is based on the mean and std of a random subset of the dataset of the size subset_size.

    :param dataset_name: name of the dataset to load. It has to be available in torchvision.datasets
    :param subset_size: used when not working on the full dataset - the results will be less reliable, but the
    complexity will be lowered
    :return: random, normalized subset of dataset_name of size subset_size with (noise_rate*subset_size) labels changed
    to introduce label noise
    """
    # Load the train and test datasets based on the 'dataset_name' parameter
    train_dataset = getattr(datasets, dataset_name)(root="./data", train=True, download=True,
                                                    transform=transforms.ToTensor())
    test_dataset = getattr(datasets, dataset_name)(root="./data", train=False, download=True,
                                                   transform=transforms.ToTensor())
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
        loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=False)
    return loader


def initialize_model(latent: int = 1, evaluation_network: str = 'SimpleNN') -> Tuple[SimpleNN, SGD]:
    """ Used to initialize the model and optimizer.

    :param latent: the index of the hidden layer used to extract the latent representation for radii computations
    :param evaluation_network: specifies the type of network to initialize
    :return: initialized SimpleNN model and SGD optimizer
    """
    if evaluation_network == 'SimpleNN':
        model = SimpleNN(28 * 28, 2, 20, latent)
        optimizer = SGD(model.parameters(), lr=0.1)
    else:
        model = ResNet(img_channels=1, num_layers=18, block=BasicBlock, num_classes=10)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    model.to(DEVICE)
    return model, optimizer


def train_model(model: Union[SimpleNN, ResNet], loader: DataLoader, optimizer: Union[SGD, Adam],
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
            current_radii = model.radii(loader, set())
            epoch_radii.append((epoch, current_radii))
    return epoch_radii


def train_stop_at_inversion(model: SimpleNN, loader: DataLoader,
                            optimizer: SGD) -> Tuple[Dict[int, SimpleNN], Dict[int, int]]:
    """ Train a model and monitor the radii of class manifolds. When an inversion point is identified for a class, save
    the current state of the model to the 'model' list that is returned by this function.

    :param model: this model will be used to find the inversion point
    :param loader: the program will look for stragglers within the data in this loader
    :param optimizer: used for training
    :return: dictionary mapping an index of a class manifold to a model, which can be used to extract stragglers for
    the given class
    """
    prev_radii, models = {class_idx: torch.tensor(float('inf')) for class_idx in range(10)}, {}
    found_classes = set()  # Keep track of classes for which the inversion point has already been found.
    inversion_points = {}
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
            current_radii = model.radii(loader, found_classes)
            for key in current_radii.keys():
                # For each class see if the radii didn't increase -> reached inversion point. We only check after epoch
                # 20 for the same reasons as in train_model()
                if key not in models.keys() and current_radii[key] > prev_radii[key] and epoch > 20:
                    models[key] = model.to(DEVICE)
                    found_classes.add(key)
                    inversion_points[key] = epoch
            prev_radii = current_radii
        if set(models.keys()) == set(range(10)):
            break
    print(inversion_points)
    print(models.keys())
    return models, inversion_points


def select_stragglers_dataset_level(results: List[Tuple[int, float]], num_stragglers: int,
                                    strategy: str) -> List[int]:
    """ Extracts top 'num_stragglers' uncertain data samples. This works on a dataset-level, hence more hard samples
    might be identified for certain classes.

    :param results: list of tuples of the form (data_sample_index, energy/confidence)
    :param num_stragglers: threshold used to identify hard samples - we look for precisely this many hard samples
    :param strategy: specifies the strategy used for identifying hard samples; only 'stragglers', 'confidence' and
    'energy' allowed
    :return: list of the indices of identified hard samples
    """
    results.sort(key=lambda x: x[1], reverse=(strategy == 'energy'))
    return [x[0] for x in results[:num_stragglers]]


def select_stragglers_class_level(results: List[Tuple[int, float]], stragglers_per_class: List[int], strategy: str,
                                  dataset: TensorDataset) -> List[int]:
    """Extracts top 'num_stragglers' uncertain data samples. This works on a class-level, hence the results should be
     better (more precise) than for the dataset-level implementation.

    :param results: list of tuples of the form (data_sample_index, energy/confidence)
    :param stragglers_per_class: class-level threshold used to identify hard samples - we look for precisely this many
    hard samples within every class
    :param strategy: specifies the strategy used for identifying hard samples; only 'stragglers', 'confidence' and
    'energy' allowed
    :param dataset: TensorDataset that contains the data to be divided into easy and hard samples
    :return: list of the indices of identified hard samples
    """
    targets = [label for _, label in dataset]
    class_results = {i: [] for i in range(10)}
    # Sort the results based on the prediction class
    for idx, score in results:
        class_label = targets[idx]  # Use the passed targets list/tensor
        class_results[class_label.item() if hasattr(class_label, 'item') else class_label].append((idx, score))
    # Extract most uncertain samples on a class-level
    stragglers_indices = []
    for class_label, class_result in class_results.items():
        class_result.sort(key=lambda x: x[1], reverse=(strategy == 'energy'))
        num_stragglers = stragglers_per_class[class_label]
        stragglers_indices.extend([x[0] for x in class_result[:num_stragglers]])
    return stragglers_indices


def calculate_energy(logits: Tensor, targets: Tensor, level: str, temperature: float = 1.0) -> Tensor:
    if level == 'class':
        # Compute energy for each class individually
        unique_classes = targets.unique()
        class_energies = torch.empty(size=(len(logits),), dtype=logits.dtype, device=logits.device)
        for class_label in unique_classes:
            class_mask = (targets == class_label)
            class_logits = logits[class_mask]
            # Calculate class-specific energy
            class_energies[class_mask] = -temperature * torch.logsumexp(class_logits / temperature, dim=1)
        return class_energies
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def identify_hard_samples_with_model_accuracy(gt_indices: List[int], dataset: TensorDataset, stragglers: List[int],
                                              strategy: str, level: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """ This function divides the 'dataset' into hard and easy samples using either confidence- or energy-based method.

    :param gt_indices: list of noisy-label indices (gt stands for ground truth)
    :param dataset: TensorDataset that contains the data to be divided into easy and hard samples
    :param stragglers: list specifying the number of stragglers identified for each class (reference to set threshold
    for confidence- and energy-based methods)
    :param strategy: specifies the strategy used for identifying hard samples; only 'stragglers', 'confidence' and
    'energy' allowed
    :param level: specifies the level at which the energy is computed; it also affects how the hard samples are chose
    (is it class- or dataset-level); only 'dataset' and 'class' allowed
    :return: tuple containing the identified hard and easy samples
    """
    # Initialize necessary variables and models before training
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
            if strategy == 'energy':
                energy = calculate_energy(output, target, level).cpu().numpy()
                results.extend(list(zip(range(len(dataset)), energy)))
            else:
                confidence = output.max(1)[0].cpu().numpy()
                results.extend(list(zip(range(len(dataset)), confidence)))
    # Select the samples that the model is the least confident with either on a dataset- or class-level
    if level == 'dataset':
        stragglers_indices = select_stragglers_dataset_level(results, sum(stragglers), strategy)
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
    # Check how many noisy-labels did the method successfully identify
    if len(gt_indices) > 0:
        accuracy = len(set(total_hard_indices).intersection(gt_indices)) / len(gt_indices) * 100
        print(f'Correctly guessed {accuracy}% of label noise '
              f'({len(set(total_hard_indices).intersection(gt_indices))} out of {len(gt_indices)}).')
    return torch.stack(hard_data), torch.tensor(hard_target), torch.stack(easy_data), torch.tensor(easy_target)


def identify_hard_samples(strategy: str, dataset: TensorDataset, level: str, noise_ratio: float) -> List[Tensor]:
    """ This function divides 'loader' (or 'dataset', depending on the used 'strategy') into hard and easy samples.

    :param strategy: specifies the strategy used for identifying hard samples; only 'stragglers', 'confidence' and
    'energy' allowed
    :param dataset: TensorDataset that contains the data to be divided into easy and hard samples
    :param level: specifies the level at which the energy is computed and how the hard samples are chosen; only
    'dataset' and 'class' allowed
    :param noise_ratio: used when adding label noise to the dataset. Make sure that noise_ratio is in range of [0, 1)
    :return: list containing the identified hard and easy samples
    """
    noisy_indices = []
    if strategy == 'stragglers':
        noisy_indices = introduce_label_noise(dataset, noise_ratio)
    loader = transform_datasets_to_dataloaders(dataset)
    model, optimizer = initialize_model()
    # The following are used to store all stragglers and non-stragglers
    hard_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    hard_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    easy_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    easy_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    # Look for inversion point for each class manifold
    models, _ = train_stop_at_inversion(model, loader, optimizer)
    # Check if stragglers for all classes were found. If not repeat the search
    if set(models.keys()) != set(range(10)):
        print('Have to restart because not all stragglers were found.')
        return identify_hard_samples(strategy, dataset, level, 0.0)
    # This is used to know the distribution of stragglers between classes
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
    if len(noisy_indices) > 0:
        combined_stragglers = torch.any(torch.stack(stragglers), dim=0)
        straggler_indices = set(torch.where(combined_stragglers)[0].cpu().tolist())
        accuracy = len(straggler_indices.intersection(noisy_indices)) / len(noisy_indices) * 100
        print(print(f'Correctly guessed {accuracy:.2f}% of label noise '
              f'({len(set(straggler_indices).intersection(noisy_indices))} out of {len(noisy_indices)}).'))
    # Compute the class-level number of stragglers
    stragglers = [int(tensor.sum().item()) for tensor in stragglers]
    print(f'Found {sum(stragglers)} stragglers.')
    if strategy in ["confidence", "energy"]:
        # Introduce noise and increase the threshold for confidence- and energy-based methods
        noisy_indices = introduce_label_noise(dataset, noise_ratio)
        print(f'Poisoned {len(noisy_indices)} labels.')
        for class_idx in range(len(stragglers)):
            stragglers[class_idx] = stragglers[class_idx] + int(noise_ratio * len(dataset) / len(stragglers))
        print(f'Hence, now the method should find {sum(stragglers)} stragglers.')
        # Identify hard an easy samples using confidence- or energy-based method
        hard_data, hard_target, easy_data, easy_target = (
            identify_hard_samples_with_model_accuracy(noisy_indices, dataset, stragglers, strategy, level))
    return [hard_data, hard_target, easy_data, easy_target]


def create_dataloaders_with_straggler_ratio(hard_data: Tensor, easy_data: Tensor, hard_target: Tensor,
                                            easy_target: Tensor, train_ratio: float, reduce_hard: bool,
                                            remaining_train_ratio: float) -> Tuple[DataLoader, List[DataLoader]]:
    """ This function divides easy and hard data samples into train and test sets.

    :param hard_data: identifies hard samples (data)
    :param hard_target: identified hard samples (target)
    :param easy_data: identified easy samples (data)
    :param easy_target: identified easy samples (target)
    :param train_ratio: percentage of train set to whole dataset
    :param reduce_hard: flag indicating whether we want to see the effect of changing the number of easy (False) or
    hard (True) samples
    :param remaining_train_ratio: ratio of easy/hard samples remaining in the train set (0.1 means that 90% of hard
    samples will be removed from the pool of hard samples when generating train set, when reduce_hard == True)
    :return: returns train loader and 3 test loaders - 1) with all data samples; 2) with only hard data samples; and 3)
    with only easy data samples.
    """
    # Randomly shuffle hard and easy samples
    hard_perm, easy_perm = torch.randperm(hard_data.size(0)), torch.randperm(easy_data.size(0))
    hard_data, hard_target = hard_data[hard_perm], hard_target[hard_perm]
    easy_data, easy_target = easy_data[easy_perm], easy_target[easy_perm]
    # Split data into initial train/test sets based on the train_ratio and make sure that train_ratio is correct
    if not 0 < train_ratio < 1:
        raise ValueError(f'The parameter split_ratio must be in (0, 1); {train_ratio} not allowed.')
    train_size_hard = int(len(hard_data) * train_ratio)
    train_size_easy = int(len(easy_data) * train_ratio)
    hard_train_data, hard_test_data = hard_data[:train_size_hard], hard_data[train_size_hard:]
    hard_train_target, hard_test_target = hard_target[:train_size_hard], hard_target[train_size_hard:]
    easy_train_data, easy_test_data = easy_data[:train_size_easy], easy_data[train_size_easy:]
    easy_train_target, easy_test_target = easy_target[:train_size_easy], easy_target[train_size_easy:]
    # Reduce the number of train samples by remaining_train_ratio
    if not 0 <= remaining_train_ratio <= 1:
        raise ValueError(f'The parameter remaining_train_ratio must be in [0, 1]; {remaining_train_ratio} not allowed.')
    if reduce_hard:
        reduced_hard_train_size = int(train_size_hard * remaining_train_ratio)
        reduced_easy_train_size = train_size_easy
    else:
        reduced_hard_train_size = train_size_hard
        reduced_easy_train_size = int(train_size_easy * remaining_train_ratio)
    # Combine easy and hard samples into train and test data
    train_data = torch.cat((hard_train_data[:reduced_hard_train_size],
                            easy_train_data[:reduced_easy_train_size]), dim=0)
    train_targets = torch.cat((hard_train_target[:reduced_hard_train_size],
                               easy_train_target[:reduced_easy_train_size]), dim=0)
    # Shuffle the final train dataset
    train_permutation = torch.randperm(train_data.size(0))
    train_data, train_targets = train_data[train_permutation], train_targets[train_permutation]
    train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=len(train_data))
    # Create two test sets - one containing only hard samples, and the other only easy samples
    hard_and_easy_test_sets = [(hard_test_data, hard_test_target), (easy_test_data, easy_test_target)]
    full_test_data = torch.cat((hard_and_easy_test_sets[0][0], hard_and_easy_test_sets[1][0]), dim=0)
    full_test_targets = torch.cat((hard_and_easy_test_sets[0][1], hard_and_easy_test_sets[1][1]), dim=0)
    # Create 3 test loaders: 1) with all data samples; 2) with only hard data samples; 3) with only easy data samples
    test_loaders = []
    for data, target in [(full_test_data, full_test_targets)] + hard_and_easy_test_sets:
        test_loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=False)
        test_loaders.append(test_loader)
    return train_loader, test_loaders


def test(model: SimpleNN, loader: DataLoader) -> float:
    """ Measures the accuracy of the 'model' on the test set.

    :param model: model, which performance we want to evaluate
    :param loader: DataLoader containing test data
    :return: accuracy on the test set rounded to 2 decimal places
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return round(100 * correct / total, 2)


def straggler_ratio_vs_generalisation(hard_data: Tensor, hard_target: Tensor, easy_data: Tensor, easy_target: Tensor,
                                      train_ratio: float, reduce_hard: bool, remaining_train_ratios: List[float],
                                      current_accuracies: Dict[str, Dict[float, List]], evaluation_network: str):
    """ In this function we want to measure the effect of changing the number of easy/hard samples on the accuracy on
    the test set for distinct train:test ratio (where train:test ratio is passed as a parameter). The experiments are
    repeated multiple times to ensure that they are initialization-invariant.

    :param hard_data: identifies hard samples (data)
    :param hard_target: identified hard samples (target)
    :param easy_data: identified easy samples (data)
    :param easy_target: identified easy samples (target)
    :param train_ratio: percentage of train set to whole dataset
    :param reduce_hard: flag indicating whether we want to see the effect of changing the number of easy (False) or
    hard (True) samples
    :param remaining_train_ratios: list of ratios of easy/hard samples remaining in the train set (0.1 means that 90% of
    hard samples were removed from the train set before training, when reduce_hard == True)
    :param current_accuracies: used to save accuracies to the outer scope
    :param evaluation_network: this network will be used to measure the performance on hard/easy data
    """
    generalisation_settings = ['full', 'hard', 'easy']
    for remaining_train_ratio in remaining_train_ratios:
        accuracies_for_ratio = [[], [], []]  # Store accuracies for the current ratio across different initializations
        train_loader, test_loaders = create_dataloaders_with_straggler_ratio(hard_data, easy_data, hard_target,
                                                                             easy_target, train_ratio,
                                                                             reduce_hard, remaining_train_ratio)
        # We train multiple times to make sure that the performance is initialization-invariant
        for _ in range(3):
            model, optimizer = initialize_model(evaluation_network=evaluation_network)
            train_model(model, train_loader, optimizer, False)
            # Evaluate the model on test set
            for i in range(3):
                accuracy = test(model, test_loaders[i])
                accuracies_for_ratio[i].append(accuracy)
        # Save the accuracies to the outer scope (outside of this function)
        for i in range(3):
            current_accuracies[generalisation_settings[i]][remaining_train_ratio].extend(accuracies_for_ratio[i])


def plot_radii(all_radii: List[List[Tuple[int, Dict[int, torch.Tensor]]]], dataset_name: str, save: bool = False):
    """ This function plots the radii of class manifolds (generates Figure 2 from our paper).

    :param all_radii: list containing 10 lists - each showing the development of the radii of class manifolds during
    training
    :param dataset_name: used when saving to indicate on which dataset the results come from
    :param save: flag indicating whether to save or not
    """
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(all_radii)))  # Darker to lighter blues
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for run_index in range(len(all_radii)):
        radii = all_radii[run_index]
        for i, ax in enumerate(axes.flatten()):
            y = [radii[j][1][i].cpu() for j in range(len(radii))]
            x = [radii[j][0] for j in range(len(radii))]
            ax.plot(x, y, color=colors[run_index], linewidth=3)
            if run_index == 0:
                ax.set_title(f'Class {i} Radii Over Epoch')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Radius')
                ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f'Figures/radii_on_{dataset_name}.png')
        plt.savefig(f'Figures/radii_on_{dataset_name}.pdf')


def plot_generalisation(train_ratios: list[float], remaining_train_ratios: list[float], reduce_hard: bool,
                        avg_accuracies: Dict[str, Dict[float, List[float]]], strategy: str, level: str,
                        std_accuracies: Dict[str, Dict[float, List[float]]], noise_ratio: float, dataset_name: str):
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(train_ratios)))
    for setting in ['full', 'hard', 'easy']:
        for idx in range(len(train_ratios)):
            ratio = train_ratios[idx]
            plt.errorbar(remaining_train_ratios, avg_accuracies[setting][ratio], marker='o', markersize=5,
                         yerr=std_accuracies[setting][ratio], capsize=5, linewidth=2, color=colors[idx],
                         label=f'Training:Test={int(100 * ratio)}:{100 - int(100 * ratio)}')
        plt.xlabel(f'Proportion of {["Easy", "Hard"][reduce_hard]} Samples Remaining in Training Set', fontsize=14)
        plt.ylabel(f'Accuracy on {setting.capitalize()} Test Set (%)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(title='Training:Test Ratio')
        plt.tight_layout()
        # Generate a title for the figure
        s = ''
        if strategy != 'stragglers':
            s = f'{level}_'
        plt.savefig(
            f'Figures/generalisation_from_{["easy", "hard"][reduce_hard]}_to_{setting}_on_{dataset_name}_using_{s}'
            f'{strategy}_{noise_ratio}noise.png')
        plt.savefig(
            f'Figures/generalisation_from_{["easy", "hard"][reduce_hard]}_to_{setting}_on_{dataset_name}_using_{s}'
            f'{strategy}_{noise_ratio}noise.pdf')
        plt.clf()
