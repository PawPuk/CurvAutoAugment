import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import medmnist
import matplotlib.pyplot as plt


# Function to calculate accuracy
def calculate_accuracy(model, data_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to calculate gradients
        for images, labels in data_loader:
            outputs = model(images)
            labels = labels.squeeze()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Set model back to train mode
    return 100 * correct / total


def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std


# Load the dataset without normalization to compute mean and std
transform = Compose([ToTensor()])  # Temporarily remove normalization to calculate mean and std
train_dataset = medmnist.PneumoniaMNIST(root='./data', split='train', transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# Calculate mean and std
mean, std = calculate_mean_std(train_loader)
# Now, re-define the dataset and loader with normalization using the calculated mean and std
transform_normalized = Compose([ToTensor(), Normalize(mean=mean.numpy(), std=std.numpy())])

train_dataset_normalized = medmnist.PneumoniaMNIST(root='./data', split='train', transform=transform_normalized,
                                                   download=True)
train_loader_normalized = DataLoader(dataset=train_dataset_normalized, batch_size=64, shuffle=True)

# Load validation and test datasets
validation_dataset_normalized = medmnist.PneumoniaMNIST(root='./data', split='val', transform=transform_normalized,
                                                        download=True)
validation_loader_normalized = DataLoader(dataset=validation_dataset_normalized, batch_size=64, shuffle=True)
test_dataset_normalized = medmnist.PneumoniaMNIST(root='./data', split='test', transform=transform_normalized,
                                                  download=True)
test_loader_normalized = DataLoader(dataset=test_dataset_normalized, batch_size=64, shuffle=True)


# 2. Define the Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 20)  # Flatten 28x28 images and feed into the network
        self.fc2 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 2)  # Output layer: 2 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)  # Flatten the image
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc4(x)
        return x


model = MLP()

# 3. Train the Model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
losses = []
# Initialize lists to keep track of accuracies
train_accuracies = []
validation_accuracies = []
test_accuracies = []  # Will only store the final test accuracy

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader_normalized:
        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.squeeze()  # Adjust labels shape
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate accuracies for training, validation, and test sets
    train_accuracy = calculate_accuracy(model, train_loader_normalized)
    validation_accuracy = calculate_accuracy(model, validation_loader_normalized)
    test_accuracy = calculate_accuracy(model, test_loader_normalized)  # Compute test accuracy during training

    # Append accuracies for plotting
    train_accuracies.append(train_accuracy)
    validation_accuracies.append(validation_accuracy)
    test_accuracies.append(test_accuracy)  # Append test accuracy for each epoch

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader_normalized):.4f}, '
          f'Training Accuracy: {train_accuracy:.2f}%, '
          f'Validation Accuracy: {validation_accuracy:.2f}%, '
          f'Test Accuracy: {test_accuracy:.2f}%')


# Plotting
plt.figure(figsize=(15, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(validation_accuracies, label='Validation Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')  # Plot test accuracy as a regular line
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()


