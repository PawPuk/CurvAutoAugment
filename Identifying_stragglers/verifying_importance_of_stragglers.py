import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import tqdm

from neural_networks import SimpleNN
from utils import (load_data, identify_hard_samples_with_model_accuracy, straggler_ratio_vs_generalisation,
                   train_stop_at_inversion, transform_datasets_to_dataloaders)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = 'CIFAR10'
# Load the dataset. We copy the batch size from the "Inversion dynamics of class manifolds in deep learning ..." paper
train_dataset, test_dataset, full_dataset = load_data(dataset_name)
train_loader, test_loader, full_loader = transform_datasets_to_dataloaders(
    [train_dataset, test_dataset, full_dataset], 70000)
stragglers = [None for _ in range(10)]
# Instantiate the model, loss function, optimizer and learning rate scheduler
if dataset_name == 'CIFAR10':
    model = SimpleNN(32*32*3, 8, 20, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.02)

else:
    model = SimpleNN(28*28, 2, 20, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
model.to(device)
criterion = nn.CrossEntropyLoss()
# Run the training process
strategy = "stragglers"  # choose from "stragglers", "model", "cluster"
num_epochs = 500
if strategy == 'stragglers':
    models = train_stop_at_inversion(model, test_loader, optimizer, criterion, num_epochs)
# Calculate the number of steps in your gradient
train_ratios = [0.9, 0.8, 0.7, 0.6]
n_ratios = len(train_ratios)
# Define your edge colors and select a colormap for the gradients
straggler_ratios = np.array([0, 0.25, 0.5, 0.75, 1])  # Define the ratios
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(train_ratios)))  # Darker to lighter blues
reduce_stragglers = False  # True/False - see the impact of reducing stragglers/non_stragglers on generalisation
generalisation_settings = ['full', 'stragglers', 'non_stragglers']
total_avg_accuracies = {setting: {ratio: [] for ratio in train_ratios} for setting in generalisation_settings}
total_std_accuracies = {setting: {ratio: [] for ratio in train_ratios} for setting in generalisation_settings}
for idx, train_ratio in tqdm.tqdm(enumerate(train_ratios), desc='Going through different train:test ratios'):
    stragglers_data = torch.tensor([], dtype=torch.float32).to(device)
    stragglers_target = torch.tensor([], dtype=torch.long).to(device)
    non_stragglers_data = torch.tensor([], dtype=torch.float32).to(device)
    non_stragglers_target = torch.tensor([], dtype=torch.long).to(device)
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if strategy == "stragglers":
            for i in range(10):
                if models[i] is not None:
                    stragglers[i] = ((torch.argmax(model(data), dim=1) != target) & (target == i))
                    current_non_stragglers = (torch.argmax(model(data), dim=1) == target) & (target == i)
                    # Concatenate the straggler data and targets
                    stragglers_data = torch.cat((stragglers_data, data[stragglers[i]]), dim=0)
                    stragglers_target = torch.cat((stragglers_target, target[stragglers[i]]), dim=0)
                    # Concatenate the non-straggler data and targets
                    non_stragglers_data = torch.cat((non_stragglers_data, data[current_non_stragglers]), dim=0)
                    non_stragglers_target = torch.cat((non_stragglers_target, target[current_non_stragglers]), dim=0)
        elif strategy == "model":
            stragglers_data, stragglers_target, non_stragglers_data, non_stragglers_target = (
                identify_hard_samples_with_model_accuracy(model, full_dataset, optimizer, criterion, num_epochs))
    print(f'A total of {len(stragglers_data)} stragglers and {len(non_stragglers_data)} non-stragglers were found.')
    avg_accuracies, std_accuracies = straggler_ratio_vs_generalisation(straggler_ratios, stragglers_data,
                                                                       stragglers_target, non_stragglers_data,
                                                                       non_stragglers_target, train_ratio,
                                                                       reduce_stragglers)
    print(f'For train_ratio = {train_ratio} we get average accuracies of {avg_accuracies["full"]}% on full test set,'
          f'{avg_accuracies["stragglers"]}% on test stragglers and {avg_accuracies["non_stragglers"]}% on '
          f'non-stragglers.')
    for setting in generalisation_settings:
        total_avg_accuracies[setting][train_ratio] = avg_accuracies[setting]
        total_std_accuracies[setting][train_ratio] = std_accuracies[setting]
for setting in generalisation_settings:
    for idx in range(len(train_ratios)):
        ratio = train_ratios[idx]
        plt.errorbar(straggler_ratios, total_avg_accuracies[setting][ratio], yerr=total_std_accuracies[setting][ratio],
                     label=f'Train:Test={int(10*ratio)}:{int(10*(1-ratio))}', marker='o', markersize=5, capsize=5,
                     linewidth=2, color=colors[idx])
    plt.xlabel('Proportion of Stragglers Removed from Train Set', fontsize=14)
    plt.ylabel(f'Accuracy on {setting.capitalize()} Test Set (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(title='Train:Test Ratio')
    plt.tight_layout()  # Adjust the padding between and around subplots.
    plt.savefig(f'Figures/generalisation_from_{["non_stragglers", "stragglers"][reduce_stragglers]}_to_{setting}_on_'
                f'{dataset_name}.png')
    plt.savefig(f'Figures/generalisation_from_{["non_stragglers", "stragglers"][reduce_stragglers]}_to_{setting}_on_'
                f'{dataset_name}.pdf')
    plt.clf()
