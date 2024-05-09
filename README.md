# BabyLM - COSC 5402 Final Project

Team member: Tianyi Bao, Shuheng Gong, Chenhao Qi

## Setup

1. You will need to download the dataset provided by BabyLM Challenge first and put it in directory ```/data```.
2. We generated our own tokenizer using ```tokenizer.py``` and saved the generated file as ```trained_tokenizer.json```.
3. ```*_pretrained.py``` are training files of pretrained models fine-tuned using the dataset. ```*_scratch.py``` are training files of models trained from scratch using the dataset.
4. ```blimp.py``` is used for evaulation on BLiMP score.
5. ```generator_text.py``` is used for generating texts using ```distilgpt2```. 
6. Under releases are our trained models. Load them to ```blimp.py``` to get BLiMP scores. 
