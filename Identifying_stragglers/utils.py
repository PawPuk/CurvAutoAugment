import time

import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

from neural_networks import SimpleNN

PDATA = 8192  # number of elements in the data set
DATA_BLOCK = 1  # Data block to use within the full data set
EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500


def load_data():
    # Define normalization transform
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    return train_dataset, test_dataset, full_dataset


def identify_hard_samples(loader):
    model = SimpleNN(28 * 28, 2, 20, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    stragglers_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    stragglers_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    non_stragglers_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    non_stragglers_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    for data, target in loader:
        print('---------------------------------------')
        print(data.shape, target.shape)
        print('---------------------------------------')
    models = train_stop_at_inversion(model, loader, optimizer, criterion)
    if None in models:  # Check if stragglers for all classes were found. If not repeat the search
        return identify_hard_samples(loader)
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
    return stragglers_data, stragglers_target, non_stragglers_data, non_stragglers_target


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
    train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=len(train_data), shuffle=True)
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


def straggler_ratio_vs_generalisation(reduce_train_ratio, straggler_data, straggler_target, non_straggler_data,
                                      non_straggler_target, split_ratio, reduce_stragglers, loader):
    train_loader, test_loaders = create_dataloaders_with_straggler_ratio(straggler_data, non_straggler_data,
                                                                         straggler_target, non_straggler_target,
                                                                         split_ratio, reduce_train_ratio,
                                                                         reduce_stragglers)
    print(f'Divided data into train ({len(train_loader.dataset)} samples) and test ({len(test_loaders[0].dataset)} '
          f'samples) split.')

    model = SimpleNN(28 * 28, 2, 20, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    train_model(model, loader, optimizer, criterion)
    print(f'Whereas, now it takes {time.time() - t0} seconds.')

    same_loader = None
    for data, targets in loader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        same_loader = DataLoader(TensorDataset(data, targets), batch_size=len(data), shuffle=True)
    model = SimpleNN(28 * 28, 2, 20, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    train_model(model, same_loader, optimizer, criterion)
    print(f'On the other hand, now it takes {time.time() - t0} seconds.')

    # Train the model
    model = SimpleNN(28 * 28, 2, 20, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.to(DEVICE)
    t0 = time.time()
    train_model(model, train_loader, optimizer, criterion)
    print(f'And now it takes {time.time() - t0} seconds.')
    for data, target in train_loader:
        print('---------------------------------------')
        print(data.shape, target.shape)
        print('---------------------------------------')


def train_model(model, train_loader, optimizer, criterion):
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
    return epoch_radii, error_radii


def train_stop_at_inversion(model, train_loader, optimizer, criterion):
    prev_radii, radii, models = ([[torch.tensor(float('inf'))] for _ in range(10)], [None for _ in range(10)],
                                 [None for _ in range(10)])
    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # break
        if epoch % 10 == 0:
            current_radii = model.radii(train_loader)
            for key in current_radii.keys():
                if models[key] is None and current_radii[key][0].item() > prev_radii[key][0].item() and epoch > 20:
                    models[key] = model.to(DEVICE)
                    if None not in models:
                        return models
            prev_radii = current_radii
    return models
