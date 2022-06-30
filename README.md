## SupMPN: Supervised Multiple Positives and Negatives Contrastive Learning for Semantic Textual Similarity
This repository contains the code, pre-trained models and training data for our paper.

## SupMPN chekpoints
Our released models are listed in below table with average results on STS and Transfer Learning Tasks. 

| Model name                             | Avg. On STS tasks | Avg. On Transfer Learning tasks |
| -------------------------------------- | ----------------- | ------------------------------- |
| SoDehghan/supmpn-bert-base-uncased     |       82.07       |                     86.96       |
| SoDehghan/supmpn-bert-large-uncased    |       83.15       |                     87.75       |


## Use SupMPN with HuggingFace 
You can import our models by using HuggingFace's Transformers.
```
!pip install -U transformers
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SoDehghan/supmpn-bert-base-uncased")
SupMPN_model =  AutoModel.from_pretrained("SoDehghan/supmpn-bert-base-uncased")
