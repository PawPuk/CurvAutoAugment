from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import ConcatDataset, Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm


PDATA = 8192  # number of elements in the data set
DATA_BLOCK = 1  # Data block to use within the full data set
EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
tdDATASET = torchvision.datasets.MNIST  # the dataset (MNIST, KMNIST, FashionMNIST, CIFAR10)


def load_data():
    dataset = tdDATASET("data", train=True, download=True,
                        transform=torchvision.transforms.ToTensor())
    in_block = lambda n: (DATA_BLOCK - 1) * PDATA <= n < DATA_BLOCK * PDATA
    data_means = torch.mean(torch.cat([a[0] for n, a in enumerate(dataset) if in_block(n)]), dim=0)
    data_vars = torch.sqrt(torch.var(torch.cat([a[0] for n, a in enumerate(dataset) if in_block(n)]), dim=0))
    transf = lambda x: (x - data_means) / (data_vars + EPSILON)
    transform = transforms.Compose([transforms.ToTensor(), transf])
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    full_dataset = ConcatDataset((train_dataset, test_dataset))
    return train_dataset, test_dataset, full_dataset


def transform_datasets_to_dataloaders(list_of_datasets: List[Dataset], batch_size: int) -> Tuple[DataLoader, ...]:
    return tuple([DataLoader(dataset, batch_size=batch_size, shuffle=False) for dataset in list_of_datasets])


def generate_spiral_data(n_points, noise=0.5):
    """
    Generates a 2D spiral dataset with two classes.
    :param n_points: Number of points per class.
    :param noise: Standard deviation of Gaussian noise added to the data.
    :return: data (features), labels
    """
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (1*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))


def create_spiral_data_loader(data, labels, batch_size=128):
    # Convert the numpy arrays into torch tensors
    data = torch.Tensor(data)
    labels = torch.Tensor(labels).long()  # Labels should be of type Long
    # Create TensorDataset objects
    train_dataset = TensorDataset(data, labels)
    # Create DataLoader objects
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


def train_model(model, train_loader, optimizer, criterion, epochs):
    epoch_radii = []
    error_radii = []
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
        current_train_error = 1 - 0.01*(test(model, train_loader))
        if current_train_error <= 0.5:
            epoch_radii.append((epoch, current_radii))
            error_radii.append((current_train_error, current_radii))
    return epoch_radii, error_radii


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


def plot_radii(X_type, radii):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flatten()):
        y = [radii[j][1][i] for j in range(len(radii))]
        X = [radii[j][0] for j in range(len(radii))]
        ax.plot(X, y, marker='o')
        ax.set_title(f'Class {i} Radii Over {X_type}')
        ax.set_xlabel(X_type)
        ax.set_ylabel('Radius')
        ax.grid(True)
    plt.tight_layout()


def plot_spiral_data(data, labels, title='Spiral Data'):

    """
    Plots the 2D spiral data.
    :param data: Features (data points).
    :param labels: Corresponding labels for the data points.
    :param title: Title for the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Spectral, s=25, edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

