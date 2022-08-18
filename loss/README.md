This loss is an extentaion of MultipleNegativesRankingLoss from https://www.sbert.net/
        
        This loss expects as input a batch consisting of sentence triplets as  (anchor, multiple positives, multiple negatives) 
        (a_1; p_11, p_12, ...p_1P ; n_11, n_12,...,n_1Q) , (a_2; p_21, p_22, ...p_2P ; n_21, n_22,...,n_2Q) , ...
        where we assume that (p_i1,...p_iP) are multiple positives for anchor a_i  and  (n_i1,...n_iQ) are multiple negatives for anchor a_i
        
        For each a_i, it uses its negatives (n_i1,...n_iQ) and all other positives (p_j1,...p_jP) and all other negatives (n_j1,...n_jQ) as negative samples (i!=j)
        
        This loss function works great to train embeddings for Textual Semantic Similarity where you have multiple triplets as (anchor sentences, positive sentences, negative sentences)
        Number of positives and negative can be different (P!=Q)
        Number of positievs sampels is specified using pos_count parametre, otherwise its defult value is 1 
