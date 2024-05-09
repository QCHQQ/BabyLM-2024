from datasets import load_dataset
from transformers import BertTokenizer
from transformers import BertForMaskedLM, Trainer, TrainingArguments
from datasets import load_metric
import torch

if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")

train_files = [
    "data/train_100M/bnc_spoken.train",
    "data/train_100M/childes.train",
    "data/train_100M/gutenberg.train",
    "data/train_100M/open_subtitles.train",
    "data/train_100M/simple_wiki.train",
    "data/train_100M/switchboard.train"
]

test_files = [
    "data/test/bnc_spoken.test",
    "data/test/childes.test",
    "data/test/gutenberg.test",
    "data/test/open_subtitles.test",
    "data/test/simple_wiki.test",
    "data/test/switchboard.test"
]

train_dataset = load_dataset("text", data_files=train_files)
test_dataset = load_dataset("text", data_files=test_files)

train_dataset = train_dataset["train"]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    result = tokenizer(examples["text"], padding="max_length", truncation=True)
    
    # Randomly mask tokens
    input_ids = result["input_ids"]
    labels = input_ids.copy()
    
    labels_tensor = torch.tensor(labels)  # Convert labels to a tensor
    probability_matrix = torch.full(labels_tensor.shape, 0.15)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels_tensor[~masked_indices] = -100
    
    result["input_ids"] = input_ids
    result["labels"] = labels_tensor.tolist()  # Convert labels back to a list
    
    return result

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

model = BertForMaskedLM.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2.0e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=True,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    
)

trainer.train()

# Save the trained model
model_save_path = "./trained_model"
trainer.save_model(model_save_path)

