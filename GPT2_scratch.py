from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from tokenizers import Tokenizer

if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")

train_files = [
    "text_data/train_10M/bnc_spoken.train",
    "text_data/train_10M/childes.train",
    "text_data/train_10M/gutenberg.train",
    "text_data/train_10M/open_subtitles.train",
    "text_data/train_10M/simple_wiki.train",
    "text_data/train_10M/switchboard.train"
]
test_files = [
    "text_data/test/bnc_spoken.test",
    "text_data/test/childes.test",
    "text_data/test/gutenberg.test",
    "text_data/test/open_subtitles.test",
    "text_data/test/simple_wiki.test",
    "text_data/test/switchboard.test"
]

tokenizer = Tokenizer.from_file("trained_tokenizer.json")


train_dataset = load_dataset("text", data_files=train_files)
test_dataset = load_dataset("text", data_files=test_files)
train_dataset = train_dataset["train"]

config = GPT2Config(
    vocab_size=30000,
    n_positions=512,
    n_embd=256,
    n_layer=4,
    n_head=4,
    resid_pdrop=0.2,
    embd_pdrop=0.2,
    attn_pdrop=0.2
)
model = GPT2LMHeadModel(config=config)


def tokenize_function(examples):
    token_ids = []
    attention_masks = []
    labels = []
    max_length = 512

    for text in examples["text"]:
        output = tokenizer.encode(text)
        ids = output.ids
        if len(ids) > max_length:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        padding_length = max_length - len(ids)
        ids += [0] * padding_length
        mask += [0] * padding_length
        token_ids.append(ids)
        attention_masks.append(mask)
        labels_tensor = torch.tensor(ids)
        probability_matrix = torch.full(labels_tensor.shape, 0.15)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels_tensor[~masked_indices] = -100
        labels.append(labels_tensor.tolist())

    return {"input_ids": token_ids, "attention_mask": attention_masks, "labels": labels}

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

trainer.train()

model_save_path = "./trained_distilGPT2_model"
trainer.save_model(model_save_path)
