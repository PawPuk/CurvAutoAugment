import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neural_networks import SimpleNN
from utils import (create_data_splits, extract_top_samples, calculate_statistics, load_data, load_statistics,
                   test, train_model, train_stop_at_inversion, transform_datasets_to_dataloaders)

# Load the dataset. We copy the batch size from the "Inversion dynamics of class manifolds in deep learning ..." paper
train_dataset, test_dataset, full_dataset = load_data()
train_loader, test_loader, full_loader = transform_datasets_to_dataloaders(
    [train_dataset, test_dataset, full_dataset], 70000)
if os.path.exists('statistics.pkl') and False:
    stats = load_statistics('statistics.pkl')
else:
    stats = calculate_statistics(full_loader)
estimated_stragglers = extract_top_samples(stats, full_dataset)
data_splits = create_data_splits(full_dataset, estimated_stragglers)
epochs = 500
for key, splits in data_splits.items():
    train_loader = DataLoader(splits['train'], batch_size=64, shuffle=True)
    test_loader = DataLoader(splits['test'], batch_size=1000, shuffle=False)
    # Reinitialize the model for each statistic
    model = SimpleNN(28*28, 2, 20, 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # Train and evaluate the model
    train_model(model, train_loader, optimizer, criterion, epochs, False)
    print(f'The model trained on stragglers estimated with {key} achieved the train accuracy of '
          f'{round(test(model, train_loader, False), 4)}%, and test accuracy of '
          f'{round(test(model, test_loader, False), 4)}%')
    # Reinitialize the model for each statistic
    model = SimpleNN(28 * 28, 2, 20, 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # Train and evaluate the model
    train_model(model, train_loader, optimizer, criterion, epochs, False)
    print(f'The model trained on non-stragglers estimated with {key} achieved the train accuracy of '
          f'{round(test(model, train_loader, False), 4)}%, and test accuracy of '
          f'{round(test(model, test_loader, False), 4)}%\n{"-"*25}')
