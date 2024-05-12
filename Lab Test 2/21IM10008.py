import torch
import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt
##EXPERIMENT 1
# Seed for reproducibility
seed_value = 2021  # Joining year
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Setting device to CPU
device = torch.device('cpu')
print("Experiment 1:")
print("Seed value:", seed_value)
print("Device:", device)


##EXPERIMENT 2
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load data
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='lab_test_2_dataset', transform=data_transform)

# Shuffle and split data
train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=seed_value)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=seed_value)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

print("\nExperiment 2:")
print("Overall dataset size:", len(dataset))
print("Training dataset size:", len(train_data))
print("Validation dataset size:", len(val_data))
print("Testing dataset size:", len(test_data))
#     def __init__(self):
#         super(CNNRegression, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(32*8*8, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)
class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1=nn.Conv2d(3,16,3,1,1)
        self.conv2=nn.Conv2d(16,32,3,1,1)
        self.fc1=nn.Linear(32*8*8,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,1)

#     def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.max_pool2d(x, 2, 2)
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.max_pool2d(x, 2, 2)
#         x = x.view(-1, 32*8*8)
#         x = nn.functional.relu(self.fc1(x))
#         x = nn.functional.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    def forward(self,x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1,32*8*8)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#Chosen conventional hyper parameter 
model = CNNRegression().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("\nExperiment 3:")
print("Model architecture:")
print(model)
print("Loss function: Mean Squared Error")
print("Optimizer: Adam")
print("Learning rate: 0.001")

#Model Training
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        running_loss = 0.0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader)

# Plot training and validation losses
num_epochs=25
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()
# Evaluate model on test set
# def evaluate_model(model, criterion, test_loader):
#     model.eval()
#     test_loss = 0.0
#     predictions = []
#     ground_truth = []
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.float().to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs.squeeze(), labels)
#             test_loss += loss.item() * inputs.size(0)
#             predictions.extend(outputs.squeeze().cpu().numpy())
#             ground_truth.extend(labels.cpu().numpy())
    
#     test_loss /= len(test_loader.dataset)
#     return test_loss, predictions, ground_truth

def evaluate_model(model,criterion, text_loader):
    model.eval()
    test_loss = 0.0
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for inputs, labels in text_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs)
            Loss = criterion(outputs.squeeze(),labels)
            test_loss += Loss.item()*inputs.size(0)
            predictions.extend(outputs.squeeze().cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    return test_loss, predictions, ground_truth
    
#Evaluating test set
test_loss, predictions, ground_truth = evaluate_model(model, criterion, test_loader)
print("\nExperiment 5:")
print("Test MSE Loss:", test_loss)

plt.figure(figsize=(8,6))
plt.scatter(ground_truth, predictions, alpha=0.5)
plt.xlabel("Ground Truth Age")
plt.ylabel("Predicted Age")
# plt.title()
plt.title("Scatter Plot of Predicted vs. Ground Truth Ages")
plt.show()

# # Evaluate model on test set
# test_loss, predictions, ground_truth = evaluate_model(model, criterion, test_loader)
# print("\nExperiment 5:")
# print("Test MSE Loss:", test_loss)

# # Plot scatter plot of predicted vs. ground truth ages
# plt.figure(figsize=(8, 6))
# plt.scatter(ground_truth, predictions, alpha=0.5)
# plt.xlabel('Ground Truth Age')
# plt.ylabel('Predicted Age')
# plt.title('Scatter Plot of Predicted vs. Ground Truth Ages')
# plt.show()


