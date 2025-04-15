import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertModel

# Add root path to import dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from dataset.TwitterMultimodalDataset import TwitterMultimodalDataset

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'✅' if torch.cuda.is_available() else '⚠️'} Using device: {device}")
if device.type == "cuda":
    print(f"🟢 GPU Name: {torch.cuda.get_device_name(0)}")

# ✅ Tokenizer and image transforms
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Load dataset
print("🔄 Loading TwitterMultimodalDataset...")
dataset = TwitterMultimodalDataset(transform=transform, max_tokens=512)
print(f"✅ Loaded {len(dataset)} multimodal entries.")

# ⚙️ Use full dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ✅ Define Multimodal Model
class MultimodalDistilBertEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_hidden_size = self.bert.config.hidden_size

        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.image_encoder = nn.Sequential(*list(efficientnet.children())[:-1])
        self.image_hidden_size = efficientnet.classifier[1].in_features * 2

        self.gate = nn.Sequential(
            nn.Linear(self.image_hidden_size, self.image_hidden_size),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.text_hidden_size + self.image_hidden_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask, profile_img, banner_img):
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        profile_feat = self.image_encoder(profile_img).squeeze(-1).squeeze(-1)
        banner_feat = self.image_encoder(banner_img).squeeze(-1).squeeze(-1)
        image_feat = torch.cat([profile_feat, banner_feat], dim=1)

        gated_image_feat = image_feat * self.gate(image_feat)
        fused = torch.cat([text_feat * 0.9, gated_image_feat * 0.1], dim=1)
        return self.fusion(fused)

# ✅ Model setup
model = MultimodalDistilBertEfficientNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# 🔁 Training
print("🚀 Training has started...")
num_epochs = 10
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    print(f"📉 Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 🧰 Evaluation
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
precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

print(f"\n🎯 Accuracy: {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🎯 Recall: {recall:.4f}")
print(f"🎯 F1 Score: {f1:.4f}")

# 📁 Save results
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
results_dir = os.path.join(project_root, "results", "multimodal")
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, "distilbert_efficientnet_results.txt")
with open(results_path, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"📁 Results saved to {results_path}")
