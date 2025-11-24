import json
from pathlib import Path
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate


#configuration
DATA_PATH = Path("C:/Users/rogelio/Desktop/CS446PROJECT/bias_fairness_detection_app/safety_export_2025-11-19T06-50-08-530Z.jsonl")
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "safety-detector-model"

#load JSONL into a DataFrame
rows = []
with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

#filter to rows with labels
df = df[df["isUnsafe"].notna()].copy()
df["label"] = df["isUnsafe"].astype(int)  # True means 1, False means 0
df["text"] = df["text"].fillna("") #fill NaN texts with empty strings


print("Label counts:")
print(df["label"].value_counts())

#as soon as you label more than 20–50 items…
#switch back to stratify=df["label"] because because balanced splits really matter once we have some real training data.
train_df, eval_df = train_test_split(
    df[["text", "label"]], test_size=0.2, random_state=42
)

#dataset
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
eval_ds = Dataset.from_pandas(eval_df, preserve_index=False)

#tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=256
    )

#tokenize datasets
train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)

#remove text columns
train_ds = train_ds.remove_columns(["text"])
eval_ds = eval_ds.remove_columns(["text"])

#set format for PyTorch
train_ds.set_format("torch")
eval_ds.set_format("torch")

#model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

#metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

#compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

#training args
args = TrainingArguments(
    OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
)

#trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

#train
trainer.train()

#save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved model to {OUTPUT_DIR}")