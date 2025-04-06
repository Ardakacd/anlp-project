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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the root directory (AutoDep_Master) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataset.TwitterImageDataset import TwitterImageDataset

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'✅' if torch.cuda.is_available() else '⚠️'} Using device: {device}")
if torch.cuda.is_available():
    print(f"🟢 GPU Name: {torch.cuda.get_device_name(0)}")

print("🔄 Loading dataset...")
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
dataset = TwitterImageDataset(dataset_path, transform=transform)

# Reduce dataset to 1000 samples max
subset_size = min(1000, len(dataset))
indices = torch.randperm(len(dataset))[:subset_size]
subset = Subset(dataset, indices)

# Split into training (80%) and testing (20%)
train_size = int(0.8 * len(subset))
test_size = len(subset) - train_size
train_subset, test_subset = torch.utils.data.random_split(subset, [train_size, test_size])

train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=8, shuffle=False)

print("🔧 Initializing EfficientNet-B7 model...")
model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

print('🚀 Training has started...')
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

    print(f"📉 Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

print("🧪 Evaluating model on test data...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print(f"🎯 Test Accuracy: {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🎯 Recall: {recall:.4f}")
print(f"🎯 F1 Score: {f1:.4f}")

# Save results
results_dir = os.path.join(os.path.dirname(__file__), "../../results/image")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "efficientnet_results.txt")
with open(results_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"💾 Results saved to {results_path}")
