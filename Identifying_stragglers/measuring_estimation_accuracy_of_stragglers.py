import os

import torch
import torch.nn as nn
import torch.optim as optim

from neural_networks import SimpleNN
from utils import (calculate_statistics, extract_top_samples, load_data, load_statistics, train_stop_at_inversion,
                   transform_datasets_to_dataloaders)

# Load the dataset. We copy the batch size from the "Inversion dynamics of class manifolds in deep learning ..." paper
train_dataset, test_dataset, full_dataset = load_data()
full_loader = transform_datasets_to_dataloaders([train_dataset, test_dataset], 70000)[0]
# Calculate the statistics necessary for locating stragglers within the given subset of the dataset
if os.path.exists('statistics.pkl') and False:
    stats = load_statistics('statistics.pkl')
else:
    stats = calculate_statistics(full_loader)
straggler_data = torch.tensor([], dtype=torch.float32)
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
                current_stragglers = ((torch.argmax(model(data), dim=1) != target) & (target == i))
                straggler_data = torch.cat((straggler_data, data[current_stragglers]), dim=0)
    estimated_stragglers = extract_top_samples(stats, full_dataset)
    common_elements_count = {
        "min_distance_same_class": 0,
        "min_distance_diff_class": 0,
        "k_smallest_same_class": 0,
        "k_smallest_diff_class": 0,
        "avg_distance_same_class": 0,
        "avg_distance_diff_class": 0
    }
    straggler_data_indices = set(straggler_data.numpy())
    for key in estimated_stragglers.keys():
        # Convert the list of indices for the current statistic to a set
        top_samples_indices_set = set(estimated_stragglers[key])
        # Compute the intersection with the straggler data indices
        common_elements = top_samples_indices_set.intersection(straggler_data_indices)
        # Store the count of common elements
        common_elements_count[key] = len(common_elements)
    # Print the number of common elements for each statistic
    for key, count in common_elements_count.items():
        print(f"{key}: {count} common elements")