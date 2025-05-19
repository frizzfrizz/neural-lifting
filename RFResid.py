import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import torchvision.models as models

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def initialize_resnet(input_channels, output_size):
    resnet = models.resnet18(pretrained=False)
    resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, output_size)
    return resnet

np.random.seed(42)
torch.manual_seed(42)

n_samples = 1000
vector_size = 3
image_size = 32

X = np.random.rand(n_samples, vector_size, image_size, image_size).astype(np.float32) # random images

resnet_data = models.resnet50(pretrained=False)
resnet_data.fc = nn.Linear(resnet_data.fc.in_features, 1)
resnet_data.eval()

with torch.no_grad():
    X_tensor = torch.tensor(X)
    y = resnet_data(X_tensor).numpy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

torch.manual_seed(123)
input_channels = vector_size
output_size = y.shape[1]
model = initialize_resnet(input_channels, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

phase_1_epochs = 30
phase_3_epochs = 20
patience = 10
min_delta = 1e-4

def phase_1(model, optimizer, criterion, train_loader, val_loader):
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(phase_1_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_outputs = model(X_batch)
                val_loss += criterion(val_outputs, y_batch).item()
        val_loss /= len(val_loader)

        sys.stdout.write(f"\rPhase 1 Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")
        sys.stdout.flush()

        if best_loss - val_loss > min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        best_loss = min(best_loss, val_loss)

        if epochs_without_improvement >= patience:
            print("\nMoving to Phase 2.")
            break

def phase_2(model, train_loader, rf):
    model.eval()
    nn_outputs = []
    y_train = []
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            nn_outputs.append(model(X_batch).numpy())
            y_train.append(y_batch.numpy())
    nn_outputs = np.concatenate(nn_outputs, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    residuals = y_train.ravel() - nn_outputs.ravel()
    rf.fit(nn_outputs, residuals)
    print("\nPhase 2: Random Forest trained.")

def phase_3(model, optimizer, criterion, train_loader, val_loader, rf):
    best_loss = float('inf')
    epochs_without_improvement = 0

    model.eval()
    with torch.no_grad():
        nn_outputs = []
        y_train = []
        for X_batch, y_batch in train_loader:
            nn_outputs.append(model(X_batch).numpy())
            y_train.append(y_batch.numpy())
        nn_outputs = np.concatenate(nn_outputs, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        resid_hat = rf.predict(nn_outputs)
        y_modified = torch.tensor(nn_outputs + resid_hat.reshape(-1, 1), dtype=torch.float32)

    train_dataset_modified = TensorDataset(X_train_tensor, y_modified)
    train_loader_modified = DataLoader(train_dataset_modified, batch_size=batch_size, shuffle=True)

    for epoch in range(phase_3_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_modified_batch in train_loader_modified:
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_modified_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_outputs = model(X_batch)
                val_loss += criterion(val_outputs, y_batch).item()
        val_loss /= len(val_loader)

        sys.stdout.write(f"\rPhase 3 Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")
        sys.stdout.flush()

        if best_loss - val_loss > min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        best_loss = min(best_loss, val_loss)
        
        if epochs_without_improvement >= patience:
            print("\nEnding Phase 3.")
            break
    print()

rf = RandomForestRegressor(n_estimators=100, random_state=42)

train_losses = []
val_losses = []

num_cycles = 6
for cycle in range(num_cycles):
    print(f"\nStarting Cycle {cycle+1}")
    
    phase_1(model, optimizer, criterion, train_loader, val_loader)
    phase_2(model, train_loader, rf)
    phase_3(model, optimizer, criterion, train_loader, val_loader, rf)
    
    model.eval()
    train_loss = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            train_outputs = model(X_batch)
            train_loss += criterion(train_outputs, y_batch).item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        for X_batch, y_batch in val_loader:
            val_outputs = model(X_batch)
            val_loss += criterion(val_outputs, y_batch).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
    
    print(f"Cycle {cycle+1} completed. Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_cycles + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_cycles + 1), val_losses, label='Validation Loss', marker='x')
plt.xlabel('Cycle')
plt.ylabel('Loss')
plt.title('Train and Validation Loss vs Cycles')
plt.legend()
plt.grid(True)
plt.show()