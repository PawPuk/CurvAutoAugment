import torch
import torch.nn as nn
import torch.optim as optim

from neural_networks import SimpleNN
from utils import load_data, plot_radii, test, train_model, transform_datasets_to_dataloaders

# Load the MNIST dataset. We copy the batch size from "Inversion dynamics of class manifolds in deep learning ..." paper
train_dataset, test_dataset, full_dataset = load_data()
train_loader, test_loader = transform_datasets_to_dataloaders([train_dataset, test_dataset], 8192)
# Instantiate the model, loss function, optimizer and learning rate scheduler
model = SimpleNN(28*28, 3, 32, 1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# Run the training process
num_epochs = 150
epoch_radii = train_model(model, train_loader, optimizer, criterion, num_epochs)
print(f'The model achieved the accuracy of {round(test(model, train_loader), 4)}% on the train set, and '
      f'{round(test(model, test_loader), 4)}% on the test set')
# Plot the results
plot_radii(epoch_radii)
