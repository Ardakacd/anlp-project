import os
import sys
import torch
import evaluate
from torch.utils.data import random_split
from transformers import AutoTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments

# Move up to AutoDep_Master
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from dataset.TwitterTextDataset import TwitterTextDataset

os.environ["WANDB_DISABLED"] = "true"

# ✅ GPU Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if device.type == "cuda":
    print(f"🟢 GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not available. Using CPU.")

# ✅ Set model name and tokenizer
model_name = "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Load Dataset
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
dataset = TwitterTextDataset(dataset_path, tokenizer=tokenizer)

print(f"✅ Loaded {len(dataset)} text entries from dataset.")
if len(dataset) == 0:
    raise ValueError("🚨 ERROR: Dataset is empty! Check if Tweets.csv exists.")

# ✅ Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ✅ Load Model
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# ✅ Metrics
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

# ✅ Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
results_dir = os.path.join(project_root, "results", "text", "albert-base-v2")
os.makedirs(results_dir, exist_ok=True)

# ✅ Training Args
training_args = TrainingArguments(
    output_dir=results_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=os.path.join(project_root, "results", "text", "logs", "albert-base-v2"),
    logging_steps=10,
    load_best_model_at_end=True,
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ✅ Train & Evaluate
trainer.train()
eval_results = trainer.evaluate()

# ✅ Pretty Print
accuracy = eval_results.get("eval_accuracy", 0.0)
precision = eval_results.get("eval_precision", 0.0)
recall = eval_results.get("eval_recall", 0.0)
f1 = eval_results.get("eval_f1", 0.0)

print("\n📊 Evaluation Results:")
print(f"🎯 Test Accuracy: {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🎯 Recall: {recall:.4f}")
print(f"🎯 F1 Score: {f1:.4f}")

# ✅ Save Results
results_path = os.path.join(results_dir, "output.txt")
with open(results_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print(f"📁 Results saved to {results_path}")
model.save_pretrained(os.path.join(results_dir, "model"))
