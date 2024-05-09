# from datasets import load_dataset
# from transformers import AlbertTokenizer, AlbertForMaskedLM, Trainer, TrainingArguments
# import torch
#
# if torch.cuda.is_available():
#     print("GPU is available.")
# else:
#     print("GPU is not available.")
#
# train_files = [
#     # "data/train_10M/bnc_spoken.train",
#     # "data/train_10M/childes.train",
#     # "data/train_10M/gutenberg.train",
#     # "data/train_10M/open_subtitles.train",
#     # "data/train_10M/simple_wiki.train",
#     "data/train_10M/switchboard.train"
# ]
#
# test_files = [
#     # "data/test/bnc_spoken.test",
#     # "data/test/childes.test",
#     # "data/test/gutenberg.test",
#     # "data/test/open_subtitles.test",
#     # "data/test/simple_wiki.test",
#     "data/test/switchboard.test"
# ]
#
#
# train_dataset = load_dataset("text", data_files=train_files)
# test_dataset = load_dataset("text", data_files=test_files)
#
# train_dataset = train_dataset["train"]
#
# tokenizer = AlbertTokenizer.from_config("albert-base-v2")
# model = AlbertForMaskedLM.from_pretrained("albert-base-v2")
#
#
# def tokenize_function(examples):
#     result = tokenizer(examples["text"], padding="max_length", truncation=True)
#     input_ids = result["input_ids"]
#     labels = input_ids.copy()
#
#     labels_tensor = torch.tensor(labels)
#     probability_matrix = torch.full(labels_tensor.shape, 0.15)
#     masked_indices = torch.bernoulli(probability_matrix).bool()
#     labels_tensor[~masked_indices] = -100
#
#     result["input_ids"] = input_ids
#     result["labels"] = labels_tensor.tolist()
#
#     return result
#
#
# tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
# tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
#
#
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2.0e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     fp16=False,
#     save_strategy="no",
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_test_dataset,
# )
#
# trainer.train()
#
# model_save_path = "./trained_albert_model"
# trainer.save_model(model_save_path)
#
# from datasets import load_dataset
# from transformers import AlbertTokenizer, AlbertConfig, AlbertForMaskedLM, Trainer, TrainingArguments
# import torch
#
# if torch.cuda.is_available():
#     print("GPU is available.")
# else:
#     print("GPU is not available.")
#
# train_files = [
#     # "data/train_10M/bnc_spoken.train",
#     # "data/train_10M/childes.train",
#     # "data/train_10M/gutenberg.train",
#     # "data/train_10M/open_subtitles.train",
#     # "data/train_10M/simple_wiki.train",
#     "data/train_10M/switchboard.train"
# ]
#
# test_files = [
#     # "data/test/bnc_spoken.test",
#     # "data/test/childes.test",
#     # "data/test/gutenberg.test",
#     # "data/test/open_subtitles.test",
#     # "data/test/simple_wiki.test",
#     "data/test/switchboard.test"
# ]
#
# train_dataset = load_dataset("text", data_files=train_files)
# test_dataset = load_dataset("text", data_files=test_files)
#
# train_dataset = train_dataset["train"]
#
# # Configuring the tokenizer
# tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
#
# # Configuring the model with custom parameters
# config = AlbertConfig(
#     vocab_size=30000,  # Example of changing the vocab size
#     embedding_size=128,
#     hidden_size=256,
#     num_hidden_layers=12,
#     num_attention_heads=8,
#     intermediate_size=1024
# )
# model = AlbertForMaskedLM(config=config)
# # Initializing the model with the custom configuration
#
# def tokenize_function(examples):
#     result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
#     input_ids = result["input_ids"]
#     labels = input_ids.copy()
#
#     labels_tensor = torch.tensor(labels)
#     probability_matrix = torch.full(labels_tensor.shape, 0.15)
#     masked_indices = torch.bernoulli(probability_matrix).bool()
#     labels_tensor[~masked_indices] = -100
#
#     result["input_ids"] = input_ids
#     result["labels"] = labels_tensor.tolist()
#
#     return result
#
# tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
# tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
#
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     fp16=torch.cuda.is_available(),  # Use fp16 if GPU is available
#     save_strategy="no"
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_test_dataset,
# )
#
# trainer.train()
#
# model_save_path = "./trained_albert_model"
# trainer.save_model(model_save_path)



# # 配置模型参数
# config = AlbertConfig(
#     vocab_size=35000,  # 根据 tokenizer 调整词汇表大小
#     embedding_size=32,
#     hidden_size=64,
#     num_hidden_layers=3,
#     num_attention_heads=2,
#     intermediate_size=256
# )


from datasets import load_dataset
from transformers import AlbertConfig, AlbertForMaskedLM, Trainer, TrainingArguments
import torch
from tokenizers import Tokenizer

# 检查 GPU 可用性
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")

# 训练和测试文件路径

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
config = AlbertConfig(
    vocab_size=35000,  # 根据 tokenizer 调整词汇表大小
    embedding_size=4,
    hidden_size=8,
    num_hidden_layers=3,
    num_attention_heads=2,
    intermediate_size=16
)

# 初始化模型
model = AlbertForMaskedLM(config=config)


# 自定义 tokenize 函数
def tokenize_function(examples):
    token_ids = []
    attention_masks = []
    labels = []

    max_length = 512  # 定义最大长度

    for text in examples["text"]:
        output = tokenizer.encode(text)

        # 截断或填充 ids 到 max_length
        ids = output.ids[:max_length]  # 截断
        attention_mask = [1] * len(ids)  # 有效位为 1

        # 如果长度小于 max_length，则填充
        padding_length = max_length - len(ids)
        ids += [0] * padding_length  # 填充 id
        attention_mask += [0] * padding_length  # 填充 mask

        token_ids.append(ids)
        attention_masks.append(attention_mask)

        # 创建标签，对于 MLM 任务，通常使用相同的 ids
        labels_tensor = torch.tensor(ids)
        probability_matrix = torch.full(labels_tensor.shape, 0.15)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels_tensor[~masked_indices] = -100  # 不预测的位置标记为 -100
        labels.append(labels_tensor.tolist())

    return {
        "input_ids": token_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }


# 应用 tokenize 函数
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)



# 设置训练参数
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

# 初始化训练器并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

trainer.train()

# 保存模型
model_save_path = "./trained_albert_model"
trainer.save_model(model_save_path)
