## SupMPN: Supervised Multiple Positives and Negatives Contrastive Learning for Semantic Textual Similarity
This repository contains the code, pre-trained models and training data for our paper

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
supmpn =  AutoModel.from_pretrained("SoDehghan/supmpn-bert-large-uncased")
supmpn.eval()
```

## Semantic Textual Similarity with SupMPN
```
import numpy as np
!pip3 install seaborn
import seaborn as sns

sentences = ["It will snow tomorrow", "Heavy snowfall is expected tomorrow" , "Tomorrow will be sunny",
             "The athlete excelled the others", "The athlete won" , "The athlete fell behind the others",
             'This phone looks great', "This phone's speakers deliver clear sound" , 'This phone looks worn out',
             "The women are eating seafood" , "The women are having lunch", "The women are not eating seafood"
             ]  
def cosine(u, v):
    sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return round(sim,2)

def create_similarity_matrix (encodes):
  values = []
  for i in range (len(encodes)) :
    v=[]
    for sent in encodes:
       sim = cosine(encodes[i], sent)
       v.append(sim)
    values.append(v)  
  return values

def plot_similarity (labels, corr, title , rotation):
  sns.set(font_scale=1)
  g = sns.heatmap(corr, xticklabels = labels , yticklabels = labels, vmin=0, vmax =1, cmap = "YlOrRd", annot =False, annot_kws={"size": 8}, cbar = True ) 
  g.set_xticklabels (labels, rotation = rotation)
  g.set_title ("SupMPN-bert-base)

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
# Get the embeddings
with torch.no_grad():
    embeddings = supmpn(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    
corr = create_similarity_matrix(embeddings)
plot_similarity(sentences, corr, title , 90)
```

## Download datasets
Download training data for SupMPN from HuggingFace
```
!wget https://huggingface.co/datasets/SoDehghan/datasets-for-supmpn/all_snli_mnli_for_supmpn.csv

!wget https://huggingface.co/datasets/SoDehghan/datasets-for-supmpn/sub_snli_for_supmpn_8k.csv

```

## Train your model using SupMPN
### Clone SupMPN
```
!git clone https://github.com/SoDehghan/SupMPN.git
```


## Train your model using SupMPNLR (our objective function) via Sentence transformers

```
from sentence_transformers import SentenceTransformer, models, SentencesDataset
from sentence_transformers import datasets
import SupMPN
from SupMPN.loss import SupervisedMultiplePositivesNegativesRankingLoss


word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=150)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode = None, pooling_mode_mean_tokens = True,
                               pooling_mode_cls_token = False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# traini model
num_epochs = 3
train_loss = SupervisedMultiplePositivesNegativesRankingLoss(model = model, pos_count=5)
train_dataloader = datasets.NoDuplicatesDataLoader (training_data, batch_size=32)
warmup_steps = 0 # math.ceil(len(train_dataloader) * num_epochs * 0.05)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps)
model.save(model_save_path)
```

# Acknowledgements
loss function Codes are adapted from the repos of the EMNLP19 paper [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://github.com/UKPLab/sentence-transformers)
