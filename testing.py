import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import time


# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 20)  # First hidden layer
        self.fc2 = nn.Linear(20, 20)  # Second hidden layer
        self.fc3 = nn.Linear(20, 10)  # Output layer

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load and normalize MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

# Initialize the model and define loss function and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
new_loader = None
# Training loop for 500 epochs
t0 = time.time()
for epoch in range(500):
    for data, target in train_loader:
        if epoch == 0:
            new_loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=True)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Optional: Print loss every 50 epochs
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
print(f'The whole training took {time.time() - t0} seconds.')
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
t0 = time.time()
for epoch in range(500):
    for data, target in new_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Optional: Print loss every 50 epochs
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
print(f'Now, the whole training took {time.time() - t0} seconds.')
# Optional: Test the model (You would need to create a test_loader similar to train_loader for actual testing)
# This is just a placeholder for where you would implement testing.
print('Training complete')
