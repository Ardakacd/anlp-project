import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataset.TwitterMultimodalDataset import TwitterMultimodalDataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'✅' if torch.cuda.is_available() else '⚠️'} Using device: {device}")
if torch.cuda.is_available():
    print(f"🟢 GPU Name: {torch.cuda.get_device_name(0)}")

# Load tokenizer and transforms
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load multimodal dataset
print("🔄 Loading TwitterMultimodalDataset...")
dataset = TwitterMultimodalDataset(transform=transform, max_tokens=512)
print(f"✅ Loaded {len(dataset)} multimodal entries.")

# Reduce dataset for speed/testing
subset_size = min(500, len(dataset))
indices = torch.randperm(len(dataset))[:subset_size]
subset = torch.utils.data.Subset(dataset, indices)

train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size
train_dataset, val_dataset = random_split(subset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Define model
class MultimodalBertEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.text_hidden_size = self.bert.config.hidden_size

        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.image_encoder = nn.Sequential(*list(efficientnet.children())[:-1])
        self.image_hidden_size = efficientnet.classifier[1].in_features * 2  # profile + banner

        self.fusion = nn.Sequential(
            nn.Linear(self.text_hidden_size + self.image_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask, profile_img, banner_img):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]  # [CLS]

        profile_feat = self.image_encoder(profile_img).squeeze(-1).squeeze(-1)
        banner_feat = self.image_encoder(banner_img).squeeze(-1).squeeze(-1)
        image_feat = torch.cat([profile_feat, banner_feat], dim=1)

        combined = torch.cat([text_feat, image_feat], dim=1)
        return self.fusion(combined)

# Initialize model
model = MultimodalBertEfficientNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training
print("🚀 Training has started...")
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        profile_img = batch["profile_img"].to(device)
        banner_img = batch["banner_img"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, profile_img, banner_img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"📉 Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluation
print("🧪 Evaluating model...")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        profile_img = batch["profile_img"].to(device)
        banner_img = batch["banner_img"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, profile_img, banner_img)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print(f"🎯 Accuracy: {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🎯 Recall: {recall:.4f}")
print(f"🎯 F1 Score: {f1:.4f}")

# Save results
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
results_dir = os.path.join(project_root, "results", "multimodal")
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, "output.txt")
with open(results_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"📁 Results saved to {results_path}")
