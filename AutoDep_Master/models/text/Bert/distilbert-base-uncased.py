import torch
import os
import sys
import evaluate
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import random_split

# Setup
os.environ["WANDB_DISABLED"] = "true"  # Disable W&B logging

# Add AutoDep_Master root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import dataset
from dataset.TwitterTextDataset import TwitterTextDataset

# Device info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if device.type == "cuda":
    print(f"🟢 GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not available. Using CPU.")

# Resolve dataset path
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
print(f"Resolved dataset path: {dataset_path}")

# Load dataset
dataset = TwitterTextDataset(dataset_path)  # tokenizer is handled internally now
print(f"✅ Loaded {len(dataset)} text entries from dataset.")

# Safety check
if len(dataset) == 0:
    raise ValueError("🚨 ERROR: The dataset is empty! Please check the dataset path.")

# Train-validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Metric function
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

# Training config
training_args = TrainingArguments(
    output_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../results/text/distilbert-base-uncased")),
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../results/text/logs/distilbert-base-uncased")),
    logging_steps=10,
    load_best_model_at_end=True,
)



# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print("📊 Evaluation Results:", eval_results)

# Save
model.save_pretrained("distilbert_base_uncased_model")
tokenizer.save_pretrained("distilbert_base_uncased_model")
print("✅ Training complete. Model saved.")
