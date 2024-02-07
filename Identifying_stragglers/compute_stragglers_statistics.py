import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from neural_networks import SimpleNN
from utils import (calculate_percentiles, calculate_statistics, load_data, load_statistics, plot_radii, plot_statistics,
                   test, train_model, train_stop_at_inversion, transform_datasets_to_dataloaders)

# Load the dataset. We copy the batch size from the "Inversion dynamics of class manifolds in deep learning ..." paper
train_dataset, test_dataset, full_dataset = load_data()
full_loader = transform_datasets_to_dataloaders([train_dataset, test_dataset], 70000)[0]
# Calculate the statistics necessary for locating stragglers within the given subset of the dataset
if os.path.exists('statistics.pkl') and False:
    stats = load_statistics('statistics.pkl')
else:
    stats = calculate_statistics(full_loader)
plot_statistics(stats)
stragglers = [None for _ in range(10)]
for _ in range(1):
    # Instantiate the model, loss function, optimizer and learning rate scheduler
    model = SimpleNN(28*28, 2, 20, 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # Run the training process
    num_epochs = 500
    models = train_stop_at_inversion(model, full_loader, optimizer, criterion, num_epochs)
    for data, target in full_loader:
        for i in range(10):
            if models[i] is not None:
                stragglers[i] = ((torch.argmax(model(data), dim=1) != target) & (target == i))
                stragglers_stats = [stats[idx] for idx, s in enumerate(stragglers[i]) if s]
                print(f'Found {len(stragglers_stats)} stragglers from class {i} out of '
                      f'{len([s for s in stats if s["class"] == i])} data samples that were used for training.')
                print(calculate_percentiles([s for s in stats if s["class"] == i], stragglers_stats))
                fig, axs = plot_statistics([s for s in stats if s["class"] == i], stragglers_stats, i)
                plot_statistics(stragglers_stats, i=i, fig=fig, axs=axs)
    plt.show()
