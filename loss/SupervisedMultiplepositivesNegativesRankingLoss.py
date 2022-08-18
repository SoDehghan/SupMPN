#SupMPNRL
"""
Created on Fri Jul  1 12:43:13 2022
@author: Somaiyeh Dehghan,   mail: so.dehghan87@gmail.com,   Yildiz Technical University
loss function for our paper: "SupMPN: Supervised Multiple Positives and Negatives Contrastive Learning for Semantic Textual Similarity"
"""

import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import uti

class SupervisedMultiplePositivesNegativesRankingLoss(nn.Module):  
    """
        This loss is an extentaion of MultipleNegativesRankingLoss from https://www.sbert.net/
        
        This loss expects as input a batch consisting of sentence triplets as  (anchor, multiple positives, multiple negatives) 
        (a_1; p_11, p_12, ...p_1P ; n_11, n_12,...,n_1Q) , (a_2; p_21, p_22, ...p_2P ; n_21, n_22,...,n_2Q) , ...
        where we assume that (p_i1,...p_iP) are multiple positives for anchor a_i  and  (n_i1,...n_iQ) are multiple negatives for anchor a_i
        
        For each a_i, it uses its negatives (n_i1,...n_iQ) and all other positives (p_j1,...p_jP) and all other negatives (n_j1,...n_jQ) as negative samples (i!=j)
        
        This loss function works great to train embeddings for Textual Semantic Similarity where you have multiple triplets as (anchor sentences, positive sentences, negative sentences)

        Number of positives and negative can be different (P!=Q)
        Number of positievs sampels is specified using pos_count parametre, otherwise its defult value is 1  
        
    """
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim, pos_count: int = 1):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(SupervisedMultiplePositivesNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.pos_count= pos_count

    #-------------------------------------------------------------
    def compute_loss(self, reps):
        loss = []
        positives_index = [*range(1,self.pos_count+1,1)]
        embeddings_a = reps[0]
        batch_size = (embeddings_a.size(dim=0))
        for i in positives_index :
              pn_reps = reps[i:i+1] + reps[1:i] + reps[i+1:]
              embeddings_b =  torch.cat(pn_reps[0:])
              scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
              labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)   
              
              for i in range (batch_size):
                  mask_index = [*range(labels[i]+batch_size, batch_size*self.pos_count, batch_size)]
                  for j in mask_index :
                        mask[i][j] = 0
              scores[mask==0]=0
              loss.append(self.cross_entropy_loss(scores, labels))
        losses = sum(loss)/self.pos_count
        
        return losses
     #--------------------------------------------------------------

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        losses = self.compute_loss(reps)
        return  losses

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}

