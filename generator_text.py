from transformers import GPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
# tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("trained_distilgpt2")

text = "John asked Mary to leave because he was tired,"
encoded_input = tokenizer(text, return_tensors='pt')
output = model.generate(encoded_input['input_ids'], max_length=100, num_return_sequences=1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
