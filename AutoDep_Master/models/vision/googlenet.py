"""
GoogLeNet Training Script for Depression Detection

This script trains a GoogLeNet model for binary classification (depressed vs. non-depressed)
using images from the Twitter dataset. The dataset is loaded using the `TwitterImageDataset` class.

### Key Components:
- **Dataset:** Loads images and applies transformations (resizing, normalization, etc.).
- **Model:** GoogLeNet with a modified fully connected (fc) layer for binary classification.
- **Training:** Uses Adam optimizer and CrossEntropyLoss.
- **Evaluation:** Computes accuracy using a test dataset split.

### How to Run:
1. Ensure dependencies are installed: `pip install torch torchvision scikit-learn`
2. Place this script in the appropriate directory (ensure dataset paths are correct).
3. Run the script: `python googlenet.py`

### Expected Output:
- Training loss per epoch
- Final test accuracy

"""

import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


sys.path.append(os.path.join(os.getcwd(),'dataset'))

from TwitterImageDataset import TwitterImageDataset

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

print("Loading dataset...")
dataset = TwitterImageDataset("data", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Initializing GoogLeNet model...")
model = models.googlenet(pretrained=True)

# (2 classes: depressed/non-depressed)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

print('Training has started...')

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

print("Evaluating model on test data...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute and display metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
