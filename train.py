import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(5000))  # optional: subsample for speed

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Format for PyTorch
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(4000)),
    eval_dataset=tokenized_dataset["train"].select(range(4000, 5000)),
)

# Train
trainer.train()

# Save
model.save_pretrained("bert-imdb")
tokenizer.save_pretrained("bert-imdb")