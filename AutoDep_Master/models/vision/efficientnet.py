"""
EfficientNet Training Script for Depression Detection

This script trains an EfficientNet-B7 model for binary classification (depressed vs. non-depressed) 
using images from the Twitter dataset. The dataset is loaded using the `TwitterImageDataset` class.

### Key Components:
- **Dataset:** Loads images and applies transformations (resizing, normalization, etc.).
- **Model:** EfficientNet-B7 with a modified classifier layer for binary classification.
- **Training:** Uses Adam optimizer and CrossEntropyLoss.
- **Evaluation:** Computes accuracy using a test dataset split.

### How to Run:
1. Ensure dependencies are installed: `pip install torch torchvision scikit-learn`
2. Place this script in the appropriate directory (ensure dataset paths are correct).
3. Run the script: `python efficientnet.py`

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
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score

# Determine the absolute path to the AutoDep_Master directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Append the base directory to sys.path
sys.path.append(base_dir)

# Now you can import the module
from datasets.TwitterImageDataset import TwitterImageDataset

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((600, 600)),  # Resize images to 600x600 pixels
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load the full dataset
print("Loading dataset...")
dataset = TwitterImageDataset("../../AutoDep_Master/data", transform=transform)

# Create a subset of the dataset with only 10 samples for quick testing
subset_size = 10
indices = torch.randperm(len(dataset))[:subset_size]
subset = Subset(dataset, indices)

# Split the subset into training (80%) and testing (20%)
train_size = int(0.8 * len(subset))
test_size = len(subset) - train_size
train_subset, test_subset = torch.utils.data.random_split(subset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)  # Adjust batch size as needed
test_loader = DataLoader(test_subset, batch_size=2, shuffle=False)

# Load EfficientNet-B7 model
print("Initializing EfficientNet-B7 model...")
model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)

# Modify the final classifier layer to match binary classification (2 classes: depressed/non-depressed)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

# Move model to available device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Very small learning rate for stability

print('Training has started...')

# Training loop
num_epochs = 5  # Reduced number of epochs for quick testing
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation phase
print("Evaluating model on test data...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():  # Disable gradient computation for efficiency
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # Get class with highest probability
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute and display accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")
