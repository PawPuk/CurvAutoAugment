import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from neural_networks import SimpleNN
from utils import load_data, plot_radii, test, train_model, transform_datasets_to_dataloaders

dataset_name = 'MNIST'
# Load the dataset. We copy the batch size from the "Inversion dynamics of class manifolds in deep learning ..." paper
train_dataset, test_dataset, full_dataset = load_data(dataset_name)
train_loader, test_loader, full_loader = transform_datasets_to_dataloaders([train_dataset, test_dataset, full_dataset])
all_epoch_radii, all_error_radii = [], []
for _ in tqdm(range(5)):
    # Instantiate the model, loss function, optimizer and learning rate scheduler
    if dataset_name == 'CIFAR10':
        model = SimpleNN(32 * 32 * 3, 8, 20, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.02)
    else:
        model = SimpleNN(28 * 28, 2, 20, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    # Run the training process
    epoch_radii, error_radii = train_model(model, full_loader, optimizer, criterion, test_loader=full_loader)
    print(f'The model achieved the accuracy of {round(test(model, train_loader), 4)}% on the train set, and '
          f'{round(test(model, test_loader), 4)}% on the test set')
    all_epoch_radii.append(epoch_radii)
    all_error_radii.append(error_radii)
# Plot the results
plot_radii('Epoch', all_epoch_radii, dataset_name, True)
plot_radii('Error', all_error_radii, dataset_name, True)
plt.show()
