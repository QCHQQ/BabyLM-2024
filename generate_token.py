from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

train_files = [
    "data/train_10M/bnc_spoken.train",
    "data/train_10M/childes.train",
    "data/train_10M/gutenberg.train",
    "data/train_10M/open_subtitles.train",
    "data/train_10M/simple_wiki.train",
    "data/train_10M/switchboard.train"
]

def read_and_write_text_files(file_paths, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        out_file.write(line)
            except IOError as e:
                print(f"Error reading {file_path}: {e}")

output_file = 'all_train_data.txt'
read_and_write_text_files(train_files, output_file)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.train([output_file], trainer)

tokenizer.save("trained_tokenizer.json")

