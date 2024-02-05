from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    full_dataset = ConcatDataset((train_dataset, test_dataset))
    return train_dataset, test_dataset, full_dataset


def transform_datasets_to_dataloaders(list_of_datasets: List[Dataset], batch_size: int) -> Tuple[DataLoader, ...]:
    return tuple([DataLoader(dataset, batch_size=batch_size, shuffle=False) for dataset in list_of_datasets])


def compute_radii(data, model):
    model.eval()  # Set the model to evaluation mode
    radii = {i: [] for i in range(10)}  # Dictionary to store radii of each class manifold
    with torch.no_grad():  # No need to track gradients
        for X, y in data:
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            # Compute the pairwise distances for instances of the same class
            for i in range(10):
                class_mask = y == i
                class_data = X[class_mask]
                if len(class_data) > 1:  # At least 2 instances to compute distance
                    distances = torch.cdist(class_data.view(class_data.size(0), -1),
                                            class_data.view(class_data.size(0), -1), p=2)
                    # We take the upper triangle of the distance matrix, excluding the diagonal
                    upper_tri_distances = distances[np.triu_indices(distances.size(0), k=1)]
                    # Calculate the radii and store them
                    radii[i].append(torch.sqrt(torch.mean(upper_tri_distances ** 2)).item())
                else:
                    print(len(class_data))
                    raise Exception('The manifold has no data samples.')
    # Average the radii for this epoch
    average_radii = {i: np.mean(radii[i]) if radii[i] else 0 for i in radii}
    return average_radii


def train_model(model, train_loader, optimizer, criterion, epochs):
    epoch_radii = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in tqdm(range(epochs)):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            break
        current_radii = model.radii(train_loader)
        epoch_radii.append((epoch, current_radii))
    return epoch_radii


def test(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            break
    return 100 * correct / total


def plot_radii(epoch_radii):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flatten()):
        class_radii = [epoch_radii[j][1][i] for j in range(len(epoch_radii))]
        epochs = [epoch_radii[j][0] for j in range(len(epoch_radii))]
        ax.plot(epochs, class_radii, marker='o')
        ax.set_title(f'Class {i} Radii Over Time')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Radius')
        ax.grid(True)
    plt.tight_layout()
    plt.show()
