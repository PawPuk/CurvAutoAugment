import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

######################
# DATASET PARAMETERS #
######################
PDATA = 8192  # number of elements in the data set
DATA_BLOCK = 1  # Data block to use within the full data set
EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
tdDATASET = torchvision.datasets.MNIST  # the dataset (MNIST, KMNIST, FashionMNIST, CIFAR10)
######################


# Check if GPU is present and set device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("using GPU")
else:
    device = torch.device("cpu")
    print("using CPU")

# Download the dataset
dataset = tdDATASET("./data", train=True, download=True,
                    transform=torchvision.transforms.ToTensor())

# Standardize the data
in_block = lambda n: (DATA_BLOCK - 1) * PDATA <= n < DATA_BLOCK * PDATA
data_means = torch.mean(torch.cat([a[0] for n, a in enumerate(dataset) if in_block(n)]), dim=0)
data_vars = torch.sqrt(torch.var(torch.cat([a[0] for n, a in enumerate(dataset) if in_block(n)]), dim=0))
transf = lambda x: (x - data_means) / (data_vars + EPSILON)

# the training set
dataset = tdDATASET("./data", train=True, download=True,
                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transf]))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=PDATA, shuffle=False)
# the test set
testset = tdDATASET("./data", train=False, download=True,
                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transf]))
test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)


############ CLASSES ############

# Base NN class
# (computes the metric observables given a latent_representation, a.k.a. the activations of a hidden layer)
class myNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def latent_representation(self, X):
        pass

    def radii(self, data, labels):
        with torch.no_grad():
            X = data[labels == 0], data[labels == 1]
            nump = X[0].shape[0], X[1].shape[0]
            X = self.latent_representation(X[0]), self.latent_representation(X[1])

            # normalization
            X = torch.nn.functional.normalize(X[0], dim=1), torch.nn.functional.normalize(X[1], dim=1)

            # computation of the metric quantities
            Xmean = torch.mean(X[0], dim=0), torch.mean(X[1], dim=0)
            radius = (torch.sqrt(torch.sum(torch.square(X[0] - Xmean[0])) / nump[0]),
                      torch.sqrt(torch.sum(torch.square(X[1] - Xmean[1])) / nump[1]))
            distance = torch.norm(Xmean[0] - Xmean[1]).item()
        return radius, distance


# Derived class implementing a fully-connected NN
# - in_size: size of input
# - K: number of layers
# - N_his: number of units in hidden layers (all equal)
# - latent: ordinal number of hidden layer where the observables are computed
class NN_KHL(myNN):
    def __init__(self, in_size, K, N_hid, latent):
        super().__init__()
        self.latent = latent
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_size, N_hid, bias=True)])
        for _ in range(K - 2):
            self.layers.append(torch.nn.Linear(N_hid, N_hid, bias=True))
        self.layers.append(torch.nn.Linear(N_hid, 2, bias=True))

    def latent_representation(self, X):
        X = X.view(-1, self.layers[0].in_features)
        for l in range(self.latent):
            X = self.layers[l](X)
            X = torch.tanh(X)
        return X

    def forward(self, X):
        X = self.latent_representation(X)
        for l in range(self.latent, len(self.layers)):
            X = self.layers[l](X)
            if l < len(self.layers) - 1:
                X = torch.tanh(X)
        return X


############ UTILITIES ############
# Applies Gaussian noise to a tensor
apply_noise = lambda noise, x: torch.normal(x, noise * torch.ones_like(x))


# Returns data points and lables of the training and test set
def load_data(data_block):
    loader_it = iter(train_loader)

    for _ in range(data_block):
        data, labels = next(loader_it)
    test_data, test_labels = next(iter(test_loader))

    data, labels = data.to(device), labels.to(device)
    test_data, test_labels = test_data.to(device), test_labels.to(device)

    # When using CIFAR10, convert to greyscale (1 channel)
    if tdDATASET == torchvision.datasets.CIFAR10:
        data = data[:, 0, :, :] + data[:, 1, :, :] + data[:, 2, :, :]
        test_data = test_data[:, 0, :, :] + test_data[:, 1, :, :] + test_data[:, 2, :, :]

    # Binarize class labels (NOTE: labels are 0,1 here but +1,-1 in the manuscript)
    labels %= 2
    test_labels %= 2

    return data, labels, test_data, test_labels


# Trains a model and returns errors, the metric quantities, misclassified examples at each epoch
def train_and_measure(model, data, labels, test_data, test_labels, optimizer, criterion, epochs):
    results_run = []
    misclassified_examples_list = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(data), labels)
        loss.backward()
        optimizer.step()

        # Compute errors and metric observables
        train_error = torch.sum(torch.abs(torch.argmax(model(data), dim=1) - labels)).item() / data.shape[0]
        test_error = torch.sum(torch.abs(torch.argmax(model(test_data), dim=1) - test_labels)).item() / len(testset)
        radii, distance = model.radii(data, labels)
        results_run.append([epoch, train_error, test_error, radii[0].item(), radii[1].item(), distance])
        misclassified_examples = torch.argmax(model(data), dim=1) - labels != 0
        misclassified_examples_list.append(misclassified_examples)

    return results_run, misclassified_examples_list


# Trains and stops when the inversion point is reached
def train_stop_at_inversion(model, data, labels, optimizer, criterion):
    radius, radius_prev = 0, 0
    count = 0
    radii = []

    # This cycle trains until the inversion point is reached.
    # The inversion point is reached when the first radius starts increasing.
    # (does not halt during the initial 20 epochs to avoid being fooled by initial fluctuations)
    while radius < radius_prev or count < 20:
        count += 1
        optimizer.zero_grad()
        loss = criterion(model(data), labels)
        loss.backward()
        optimizer.step()

        radius_prev = radius
        (radius, _), _ = model.radii(data, labels)
