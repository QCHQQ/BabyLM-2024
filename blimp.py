import os
import json
import torch
from transformers import AlbertForMaskedLM, GPT2LMHeadModel, BertForMaskedLM
from tokenizers import Tokenizer


model = BertForMaskedLM.from_pretrained("./trained_bert_scratche5ori_model")
tokenizer = Tokenizer.from_file("trained_tokenizer.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

blimp_dir = "./blimp/data 2"
accuracy_scores = {}
all_predictions = {}

for filename in os.listdir(blimp_dir):
    if filename.endswith(".jsonl"):
        phenomenon = os.path.splitext(filename)[0]
        file_path = os.path.join(blimp_dir, filename)

        with open(file_path, "r") as file:
            blimp_data = [json.loads(line) for line in file]

        correct_predictions = 0
        total_samples = 0
        phenomenon_predictions = []

        for sentence_pair in blimp_data:
            sentence_good = sentence_pair["sentence_good"]
            sentence_bad = sentence_pair["sentence_bad"]

            inputs_good = tokenizer.encode(sentence_good).ids
            inputs_bad = tokenizer.encode(sentence_bad).ids

            inputs_good = torch.tensor([inputs_good]).to(device)
            inputs_bad = torch.tensor([inputs_bad]).to(device)

            with torch.no_grad():
                outputs_good = model(inputs_good)
                outputs_bad = model(inputs_bad)
                logits_good = outputs_good.logits
                logits_bad = outputs_bad.logits

            prob_good = torch.softmax(logits_good, dim=-1).mean()
            prob_bad = torch.softmax(logits_bad, dim=-1).mean()

            if prob_good > prob_bad:
                correct_predictions += 1
            total_samples += 1

            phenomenon_predictions.append({
                "sentence_good": sentence_good,
                "sentence_bad": sentence_bad,
                "prob_good": prob_good.item(),
                "prob_bad": prob_bad.item()
            })

        accuracy = correct_predictions / total_samples
        accuracy_scores[phenomenon] = accuracy
        all_predictions[phenomenon] = phenomenon_predictions

blimp_score = sum(accuracy_scores.values()) / len(accuracy_scores)

print("Evaluation Results:")
print(f"BLiMP Score: {blimp_score:.4f}")
print("\nAccuracy Scores:")
for phenomenon, accuracy in accuracy_scores.items():
    print(f"{phenomenon}: {accuracy:.4f}")

with open("blimp_predictions.json", "w") as file:
    json.dump(all_predictions, file, indent=4)
print("\nPredictions saved to blimp_predictions.json")
