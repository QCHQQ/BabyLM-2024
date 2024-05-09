from datasets import load_dataset
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments
import torch
from tokenizers import Tokenizer

if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")

train_files = [
    "data/train_10M/bnc_spoken.train",
    "data/train_10M/childes.train",
    "data/train_10M/gutenberg.train",
    "data/train_10M/open_subtitles.train",
    "data/train_10M/simple_wiki.train",
    "data/train_10M/switchboard.train"
]

test_files = [
    "data/test/bnc_spoken.test",
    "data/test/childes.test",
    "data/test/gutenberg.test",
    "data/test/open_subtitles.test",
    "data/test/simple_wiki.test",
    "data/test/switchboard.test"
]

# 加载数据集
train_dataset = load_dataset("text", data_files=train_files)
test_dataset = load_dataset("text", data_files=test_files)
train_dataset = train_dataset["train"]

# 加载自训练的 tokenizer
tokenizer = Tokenizer.from_file("trained_tokenizer.json")

# 配置模型参数
config = BertConfig(
    vocab_size=30000,  # 根据 tokenizer 调整词汇表大小
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024
)

# 初始化模型
model = BertForMaskedLM(config=config)

# 自定义 tokenize 函数
def tokenize_function(examples):
    token_ids = []
    attention_masks = []
    labels = []
    for text in examples["text"]:
        output = tokenizer.encode(text)
        ids = output.ids
        mask = [1] * len(ids)
        max_length = 512
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

# 应用 tokenize 函数
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # 根据 GPU 支持决定是否使用 fp16
    save_strategy="no"
)

# 初始化训练器并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)
trainer.train()

# 保存模型
model_save_path = "./trained_bert_model"
trainer.save_model(model_save_path)