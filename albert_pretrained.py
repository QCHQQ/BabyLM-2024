from datasets import load_dataset
from transformers import AlbertTokenizer, AlbertForMaskedLM, Trainer, TrainingArguments
import torch

if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")

train_files = [
    # "data/train_10M/bnc_spoken.train",
    # "data/train_10M/childes.train",
    # "data/train_10M/gutenberg.train",
    # "data/train_10M/open_subtitles.train",
    # "data/train_10M/simple_wiki.train",
    "data/train_10M/switchboard.train"
]

test_files = [
    # "data/test/bnc_spoken.test",
    # "data/test/childes.test",
    # "data/test/gutenberg.test",
    # "data/test/open_subtitles.test",
    # "data/test/simple_wiki.test",
    "data/test/switchboard.test"
]


train_dataset = load_dataset("text", data_files=train_files)
test_dataset = load_dataset("text", data_files=test_files)

train_dataset = train_dataset["train"]

tokenizer = AlbertTokenizer.from_config("albert-base-v2")
model = AlbertForMaskedLM.from_pretrained("albert-base-v2")


def tokenize_function(examples):
    result = tokenizer(examples["text"], padding="max_length", truncation=True)
    input_ids = result["input_ids"]
    labels = input_ids.copy()

    labels_tensor = torch.tensor(labels)
    probability_matrix = torch.full(labels_tensor.shape, 0.15)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels_tensor[~masked_indices] = -100

    result["input_ids"] = input_ids
    result["labels"] = labels_tensor.tolist()

    return result


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2.0e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=False,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

trainer.train()

model_save_path = "./trained_albert_model"
trainer.save_model(model_save_path)