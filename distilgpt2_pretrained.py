from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from tokenizers import Tokenizer
import torch

# login()
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

train_dataset = load_dataset("text", data_files=train_files)["train"]
test_dataset = load_dataset("text", data_files=test_files)["train"]

tokenizer = GPT2LMHeadModel.from_pretrained('distilbert/distilgpt2')
model = GPT2Config.from_pretrained('distilbert/distilgpt2')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="longest", truncation=True, return_tensors="pt")

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
)

trainer.train()

model_path = "./trained_gpt2_base"
trainer.save_model(model_path)

