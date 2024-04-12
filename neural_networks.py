from typing import Dict, Set, Type

# This code is a modified version from https://github.com/marco-gherardi/stragglers
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def latent_representation(self, x: Tensor) -> Tensor:
        pass

    def radii(self, data_loader: DataLoader, sustainability_mask: Set[int]) -> Dict[int, Tensor]:
        """ Computes the radii of the representation of class manifolds.

        :param data_loader: DataLoader containing the data used for experiments
        :param sustainability_mask: used to increase sustainability of the code. When an inversion point was found for
        class i then there is no point in computing the radii of the manifold of class i anymore.
        :return: dictionary containing information on the radii of class manifolds for the given model; keys are the
        indices, and they map to Tensors containing a single float
        """
        radii = {i: torch.tensor(0) for i in range(10) if i not in sustainability_mask}
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            with torch.no_grad():
                for i in set(range(10)).difference(sustainability_mask):
                    class_data = data[target == i]
                    class_data = self.latent_representation(class_data)
                    # normalization (mapping onto the unit sphere)
                    class_data = torch.nn.functional.normalize(class_data, dim=1)
                    # computation of the metric quantities
                    class_data_mean = torch.mean(class_data, dim=0)
                    radii[i] = torch.sqrt(torch.sum(torch.square(class_data - class_data_mean)) / class_data.shape[0])
        return radii


class SimpleNN(MyNN):
    def __init__(self, in_size: int, depth: int, width_hid: int, latent: int, out_size: int = 10):
        """

        :param in_size: size of input
        :param depth: number of hidden layers
        :param width_hid: width of hidden layers
        :param latent: ordinal number of hidden layer where the observables are computed
        :param out_size: size of output - number of classes
        """
        super().__init__()
        if not 0 < latent <= depth + 1:
            raise ValueError(f"The 'latent' parameter must be in (0, depth+1]; "
                             f"0 < {latent} <= {depth + 1} does not hold.")
        self.latent = latent
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_size, width_hid, bias=True)])
        for _ in range(depth - 1):
            self.layers.append(torch.nn.Linear(width_hid, width_hid, bias=True))
        self.layers.append(torch.nn.Linear(width_hid, out_size, bias=True))

    def latent_representation(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.layers[0].in_features)
        for layer_index in range(self.latent):
            x = self.layers[layer_index](x)
            if layer_index < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.latent_representation(x)
        for layer_index in range(self.latent, len(self.layers)):
            x = self.layers[layer_index](x)
            if layer_index < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 1,
                 downsample: nn.Module = None) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, img_channels: int, num_layers: int, block: Type[BasicBlock], num_classes: int = 10) -> None:
        super(ResNet, self).__init__()
        # Assume num_layers == 18 for simplicity
        layers = [2, 2, 2, 2]
        self.expansion = 1

        self.in_channels = 64
        # Adjustments for MNIST: kernel_size=3, stride=1, and possibly remove maxpool.
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=self.in_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Consider removing or adjusting this layer for MNIST
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, self.expansion, downsample)]
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the latent representation after the first hidden layer (layer1)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Assuming the latent representation is the output after layer1.
        # x = self.layer1(x)
        return x

    def radii(self, data_loader: torch.utils.data.DataLoader, sustainability_mask: Set[int]) -> Dict[int, torch.Tensor]:
        """Computes the radii of the representation of class manifolds for the first hidden layer."""
        radii = {i: torch.tensor(0., device=DEVICE) for i in range(10) if i not in sustainability_mask}
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            with torch.no_grad():
                latent_data = self.latent_representation(data)
                # Normalization to map onto the unit sphere is performed here as in MyNN class.
                for i in set(range(10)).difference(sustainability_mask):
                    class_data = latent_data[target == i]
                    class_data = torch.nn.functional.normalize(class_data, p=2, dim=1)
                    class_data_mean = torch.mean(class_data, dim=0)
                    radii[i] = torch.sqrt(torch.sum(torch.square(class_data - class_data_mean)) / class_data.shape[0])
        return radii
