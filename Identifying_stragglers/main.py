from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

from neural_networks import SimpleNN
from utils import (create_spiral_data_loader, generate_spiral_data, load_data, plot_radii, plot_spiral_data, test,
                   train_model, transform_datasets_to_dataloaders)

"""# Load the dataset. We copy the batch size from the "Inversion dynamics of class manifolds in deep learning ..." paper
train_dataset, test_dataset, full_dataset = load_data()
train_loader, test_loader = transform_datasets_to_dataloaders([train_dataset, test_dataset], 8192)"""
data, labels = generate_spiral_data(1000, 1.2)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=int(len(data)//2),
                                                                    random_state=42)
train_loader, test_loader = (create_spiral_data_loader(train_data, train_labels, 8192),
                             create_spiral_data_loader(test_data, test_labels, 8192))
# Instantiate the model, loss function, optimizer and learning rate scheduler
model = SimpleNN(2, 2, 60, 1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# Run the training process
num_epochs = 500
print(len(train_data), len(test_data))
epoch_radii = train_model(model, train_loader, optimizer, criterion, num_epochs)
print(f'The model achieved the accuracy of {round(test(model, train_loader), 4)}% on the train set, and '
      f'{round(test(model, test_loader), 4)}% on the test set')
plot_spiral_data(train_data, train_labels)
# Plot the results
plot_radii(epoch_radii)
