import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler
)

# Load dataset from JSON
with open("full_dataset.json") as f:
    data = json.load(f)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load PubMedBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

# Custom Dataset
class MentalHealthDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = " ".join(item["texts"])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }

# Split dataset
train_data, val_data = train_test_split(data, test_size=0.2, stratify=[x["label"] for x in data], random_state=42)
train_loader = DataLoader(MentalHealthDataset(train_data, tokenizer), batch_size=8, shuffle=True)
val_loader = DataLoader(MentalHealthDataset(val_data, tokenizer), batch_size=8)

# Load PubMedBERT model
model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", num_labels=2)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)


# Weighted loss
class_counts = np.bincount([item["label"] for item in train_data])
class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Scheduler
num_training_steps = len(train_loader) * 5  # 5 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
model.train()
for epoch in range(5):
    print(f"\nEpoch {epoch+1}")
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = loss_fn(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

# Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

# Results
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
