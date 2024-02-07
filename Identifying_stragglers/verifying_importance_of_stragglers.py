import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neural_networks import SimpleNN
from utils import (calculate_percentiles, calculate_statistics, load_data, load_statistics, plot_radii, plot_statistics,
                   test, train_model, train_stop_at_inversion, transform_datasets_to_dataloaders)

# Load the dataset. We copy the batch size from the "Inversion dynamics of class manifolds in deep learning ..." paper
train_dataset, test_dataset, full_dataset = load_data()
train_loader, test_loader, full_loader = transform_datasets_to_dataloaders(
    [train_dataset, test_dataset, full_dataset], 70000)
stragglers = [None for _ in range(10)]
# Instantiate the model, loss function, optimizer and learning rate scheduler
model = SimpleNN(28*28, 2, 40, 1)
model1 = copy.deepcopy(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# Run the training process
num_epochs = 500
models = train_stop_at_inversion(model, full_loader, optimizer, criterion, num_epochs)
straggler_data = torch.tensor([], dtype=torch.float32)
straggler_target = torch.tensor([], dtype=torch.long)
non_straggler_data = torch.tensor([], dtype=torch.float32)
non_straggler_target = torch.tensor([], dtype=torch.long)
for data, target in full_loader:
    for i in range(10):
        if models[i] is not None:
            stragglers[i] = ((torch.argmax(model(data), dim=1) != target) & (target == i))
            current_non_stragglers = (torch.argmax(model(data), dim=1) == target) & (target == i)
            # Concatenate the straggler data and targets
            straggler_data = torch.cat((straggler_data, data[stragglers[i]]), dim=0)
            straggler_target = torch.cat((straggler_target, target[stragglers[i]]), dim=0)
            # Concatenate the non-straggler data and targets
            non_straggler_data = torch.cat((non_straggler_data, data[current_non_stragglers]), dim=0)
            non_straggler_target = torch.cat((non_straggler_target, target[current_non_stragglers]), dim=0)
print(f'The model achieved the test accuracy of {round(test(model, train_loader), 4)}%, and '
      f'{round(test(model, test_loader), 4)}% on the test set')
print(len(straggler_data), len(straggler_target))
print(len(non_straggler_data), len(non_straggler_target))
# Create TensorDataset for stragglers
straggler_dataset = TensorDataset(straggler_data, straggler_target)
# Create DataLoader for stragglers
straggler_loader = DataLoader(straggler_dataset, batch_size=64, shuffle=True)

# Create TensorDataset for non-stragglers
non_straggler_dataset = TensorDataset(non_straggler_data, non_straggler_target)
# Create DataLoader for non-stragglers
non_straggler_loader = DataLoader(non_straggler_dataset, batch_size=64, shuffle=True)
for _ in range(5):
    model = SimpleNN(28*28, 2, 40, 1)
    model1 = copy.deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer1 = optim.SGD(model1.parameters(), lr=0.1)
    _, _ = train_model(model, straggler_loader, optimizer, criterion, num_epochs, False)
    print(f'The model achieved the train accuracy of {round(test(model, straggler_loader), 4)}% when training on '
          f'stragglers, and {round(test(model, full_loader), 4)}% on the whole dataset')
    _, _ = train_model(model1, non_straggler_loader, optimizer1, criterion, num_epochs, False)
    print(f'The model achieved the train accuracy of {round(test(model1, non_straggler_loader), 4)}% when training on '
          f'non-stragglers, and {round(test(model1, full_loader), 4)}% on the whole dataset')
