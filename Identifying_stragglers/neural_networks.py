# This code is a modified version from https://github.com/marco-gherardi/stragglers
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def latent_representation(self, X):
        pass

    def radii(self, data_loader):
        radii = {i: [] for i in range(10)}
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            with torch.no_grad():
                for i in range(10):
                    class_data = data[target == i]
                    nump = class_data.shape[0]
                    class_data = self.latent_representation(class_data)
                    # normalization (mapping onto the unit sphere)
                    class_data = torch.nn.functional.normalize(class_data, dim=1)
                    # computation of the metric quantities
                    class_data_mean = torch.mean(class_data, dim=0)
                    radii[i].append(torch.sqrt(torch.sum(torch.square(class_data - class_data_mean)) / nump))
            break
        return radii


class SimpleNN(MyNN):
    def __init__(self, in_size, L, N_hid, latent):
        """

        :param in_size: size of input
        :param L: number of hidden layers
        :param N_hid: width of hidden layers
        :param latent: ordinal number of hidden layer where the observables are computed
        """
        super().__init__()
        self.latent = latent
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_size, N_hid, bias=True)])
        for _ in range(L-1):
            self.layers.append(torch.nn.Linear(N_hid, N_hid, bias=True))
        self.layers.append(torch.nn.Linear(N_hid, 10, bias=True))

    def latent_representation(self, X):
        X = X.view(-1, self.layers[0].in_features)
        for layer_index in range(self.latent):
            X = self.layers[layer_index](X)
            X = torch.tanh(X)
        return X

    def forward(self, X):
        X = self.latent_representation(X)
        for layer_index in range(self.latent, len(self.layers)):
            X = self.layers[layer_index](X)
            if layer_index < len(self.layers) - 1:
                X = torch.tanh(X)
        return X
