import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Function to add noise to dataset labels
def add_label_noise(dataset, noise_rate=0.1):
    noisy_labels = np.array(dataset.targets)
    num_samples = len(dataset)
    num_noisy_samples = int(noise_rate * num_samples)
    changed_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
    for idx in changed_indices:
        possible_labels = list(range(10))
        possible_labels.remove(noisy_labels[idx])
        noisy_labels[idx] = np.random.choice(possible_labels)
    return noisy_labels, changed_indices


# Neural network model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # MNIST images are 1x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7*7*64, 256)  # The image size is reduced to 7x7 after convolutions and pooling
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))
        x = torch.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 7*7*64)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# Training function
def train(model, device, train_loader, optimizer, epoch, criterion, scheduler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    scheduler.step()


# Evaluate accuracy
def evaluate_accuracy(model, device, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy


def get_model_confidence_and_predictions(model, device, data_loader):
    model.eval()
    confidences = []
    predictions = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_label = torch.max(probs, dim=1)
            confidences.extend(confidence.tolist())
            predictions.extend(predicted_label.tolist())
    return confidences, predictions


# Main script
def main():
    # Load and transform MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Add noise to labels
    noisy_labels, changed_indices = add_label_noise(mnist_train, noise_rate=0.05)
    mnist_train.targets = torch.tensor(noisy_labels, dtype=torch.long)

    # DataLoader
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    # Model, loss function, and optimizer
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training
    for epoch in range(1, 51):  # 50 epochs
        train(model, device, train_loader, optimizer, epoch, criterion, scheduler)

    # Evaluation
    accuracy = evaluate_accuracy(model, device, test_loader)
    print(f'Training Accuracy with Noisy Labels: {accuracy:.2f}%')

    train_loader_single = DataLoader(mnist_train, batch_size=1, shuffle=False)
    confidences, predictions = get_model_confidence_and_predictions(model, device, train_loader_single)

    uncertain_samples_indices = sorted(range(len(confidences)), key=lambda i: confidences[i])

    precision_results = []
    recall_results = []
    samples_counts = range(100, 15001, 100)  # Adjust the range as needed

    for top_n in samples_counts:
        top_uncertain_indices = uncertain_samples_indices[:top_n]
        noise_found = sum(1 for idx in top_uncertain_indices if idx in changed_indices)
        false_positives = top_n - noise_found
        true_positives = noise_found
        actual_noisy_samples = len(changed_indices)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / actual_noisy_samples if actual_noisy_samples > 0 else 0

        precision_results.append(precision * 100)  # Convert to percentage
        recall_results.append(recall * 100)  # Convert to percentage

    # Plot the results
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Precision plot
    axs[0].plot(samples_counts, precision_results, marker='o', color='blue', label='Precision')
    axs[0].set_xlabel('Number of Top Uncertain Samples Examined')
    axs[0].set_ylabel('Precision (%)')
    axs[0].set_title('Precision of Noise Identification')
    axs[0].grid(True)

    # Recall plot
    axs[1].plot(samples_counts, recall_results, marker='o', color='green', label='Recall')
    axs[1].set_xlabel('Number of Top Uncertain Samples Examined')
    axs[1].set_ylabel('Recall (%)')
    axs[1].set_title('Recall of Noise Identification')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
