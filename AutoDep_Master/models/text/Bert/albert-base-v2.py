import torch
import sys
import os
import evaluate
from torch.utils.data import random_split
from transformers import AlbertForSequenceClassification, Trainer, TrainingArguments

# Move up 3 levels to reach "AutoDep_Master"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from dataset.TwitterTextDataset import TwitterTextDataset  # Ensure this is correctly imported

os.environ["WANDB_DISABLED"] = "true"  # Disable W&B logging


"""
Run this in order to get your PC to use your GPU (Potentially 10-30x faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""

# ✅ Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if device.type == "cuda":
    print(f"🟢 GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not available. Using CPU.")

# ✅ Corrected dataset path
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
dataset = TwitterTextDataset(dataset_path)  # Use corrected path

# ✅ Debugging: Check if dataset loaded correctly
print(f"✅ Loaded {len(dataset)} text entries from dataset.")

# Ensure dataset is not empty before proceeding
if len(dataset) == 0:
    raise ValueError("🚨 ERROR: The dataset is empty! Check the dataset path and make sure Tweets.csv exists.")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Load ALBERT Model for Binary Classification
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)
model.to(device)  # ✅ Move model to the correct device

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels, average="binary")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="binary")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="binary")["f1"]
    }

training_args = TrainingArguments(
    output_dir="./results/text/albert-base-v2",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./results/text/logs/albert-base-v2",
    logging_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()
print("📊 Evaluation Results:", eval_results)

model.save_pretrained("albert_base_v2_model")
print("✅ Training complete. Model saved.")
