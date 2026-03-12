import os
# import re
from typing import Union, Optional
from torch import Tensor
import numpy as np
import pandas as pd
from .mask_utils import random_mask, missing_mask, specify_mask
import random


def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]


def get_min_metabolites(adata):
    matrix = adata.X
    nan_mask = np.isnan(matrix)
    n_var_with_values = np.sum((matrix != 0) & (~nan_mask), axis=1)
    return n_var_with_values.min()


class MetFoundation_Tokenizer:
    def __init__(
            self,
            VOCAB_Identifiers:list=[],
            VOCAB_FILES_PATH:str=None,
            **kwargs
        ):
  
        ## special tokens: utilizing trainable embeddings
        
        ## vacabulary files- iden: identifier
        if len(VOCAB_Identifiers)>0:
            self.iden_tokens = VOCAB_Identifiers
        else:
            self.iden_tokens = load_vocab_file(VOCAB_FILES_PATH)
        


        self._id_to_token_iden = dict(enumerate(self.iden_tokens))
        self._token_to_id_iden = {tok: ind for ind, tok in enumerate(self.iden_tokens)}
        
 
    ####
    # id <-> token
    # self.unk_token is removed
    ####
    def id_to_token_iden(self, index: int) -> str:
        # return self._id_to_token_iden.get(index, self.unk_token)
        return self._id_to_token_iden.get(index)
    
    def token_to_id_iden(self, token: str) -> int:
        # return self._token_to_id_iden.get(token, self._token_to_id_iden.get(self.unk_token))
        return self._token_to_id_iden.get(token)


    ####
    # get the information of vocabularies
    ####
    @property
    def vocab_size_identifiers(self):
        return len(self.iden_tokens)


   
        
    def concentration_mask(self, concentration=list[float], add_mask:Optional[str]=None, mask_ratio:Optional[float]=.15, mask_specify=None) -> list[str]:
        idx_mask = None
        arr_conc = np.array(concentration)
        arr_mask = np.zeros(arr_conc.shape)

        if add_mask == 'random':
            idx_mask, _ = random_mask(arr_conc, mask_ratio=mask_ratio)
        elif add_mask == 'missing':
            idx_mask, _ = missing_mask(arr_conc)
        elif add_mask == 'specify':
            idx_mask = np.where(mask_specify)[0] 
            
        arr_mask[idx_mask] = 1  

        return arr_mask
    

    ####
    # pipeline
    ####
    def tokenize_from_anndata(self, adata, padding='longest', masking='random', data_layer='log_normalized',
                              masking_ratios=None, masking_specify=None, 
                              max_length=512, mode='train', 
                              return_tensor=False, device='cpu'): # non_coremetabo=None,
        '''
        Perform the whole logit for tokenization from anndata object
        In: 
            adata: anndata object
            padding:'longest' - padding to max sequence in batch 
                    'max_length' - padding to max model input length 
            masking:'random' (default): randomly masking (<MAS> token) concentration for model pre-training
                    'missing': masking missing (<MAS> token) concentration values for model inference
                    'specify': masking specific concentration values by providing a masked matrix 'masking_specify'
            masking_ratios: Tensor[float]: only enabled if masking='random'
            max_length: integer: only enabled if padding='max_length'
            masking_specify: tensor: only enabled if masking='specify'
        Out:  
            {input_ids: {identifier:tensor, concentration:tensor}, padding_mask: tensor}
            metabo_conc: concentration values as labels
        '''
        ## Loading 
        n_samples = adata.n_obs                                 # the number of input samples 
        metabo_conc = adata.layers[data_layer]    

        ## Batch operation: tokens                      
        mask_collect = np.empty((0, adata.n_vars))           
        metabo_conc_copy = metabo_conc
        for i in range(n_samples):
            if masking == 'random':
                if isinstance(masking_ratios, Tensor):
                    conc_mask = self.concentration_mask(metabo_conc_copy[i], add_mask=masking, mask_ratio=masking_ratios[i]) # dynamic mask ratios for model training
            elif masking == 'specify':
                conc_mask = self.concentration_mask(metabo_conc_copy[i], add_mask=masking, mask_specify=masking_specify[i]) 
            else:
                conc_mask = self.concentration_mask(metabo_conc_copy[i], add_mask=masking)

            mask_collect = np.vstack([mask_collect, np.array(conc_mask)])                                      
        
        mask_collect = mask_collect.astype('int')                
        conc_collect = metabo_conc.copy()
        iden_collect = np.array([self._token_to_id_iden.get(id_) for id_ in adata.var_names])
        iden_collect = np.tile(iden_collect, (n_samples, 1))

     
        if mode != 'inference':  ## Non-zero & zero element sorting                        
            conc_collect, iden_collect, mask_collect = self.push_zeros_to_end(conc_collect,  # this would be ignored during model inference (zeros/NaN are all masked)
                                                                            iden_collect,                      
                                                                            mask_collect) 
            
            ## Reduce the redundancy of metabolites (with zero/nan across samples from this batch) to align the shape (columns) with the maximum length (input) in per batch
            nonzero_columns = np.any(~np.isnan(conc_collect), axis=0) # 
            conc_collect = conc_collect[:, nonzero_columns]
            iden_collect = iden_collect[:, nonzero_columns]
            mask_collect = mask_collect[:, nonzero_columns]
            
            ## Padding
            if masking!='missing':
                padding_mask = np.where(np.isnan(conc_collect), 1, 0) # np.where(A,B,C): if A, then B, otherwise C
            else:
                padding_mask = np.zeros(conc_collect.shape)        
        else:
            padding_mask = np.zeros(conc_collect.shape) # disable padding during inference
            
            
        if padding == 'max_length': # padding to max model input length
            if padding_mask.shape[1] < max_length:
                end_padding = np.ones((padding_mask.shape[0], int(max_length-padding_mask.shape[1]) ))
                padding_mask = np.concatenate((padding_mask, end_padding), axis=1)
        elif padding == 'longest':  # padding to max sequence in batch
            pass
        else:
            raise TypeError('Padding is required for the model! Check the setting of the tokenizer.')
        
        ## Format: Tensor or Numpy array
        if return_tensor:
            iden_collect = Tensor(iden_collect).long().to(device)
            conc_collect = Tensor(conc_collect).float().to(device)
            mask_collect = Tensor(mask_collect).long().to(device)
            padding_mask = Tensor(padding_mask).long().to(device)
  
        ## Compression of the processed data
        out_dict = {}
        out_dict['input_ids'] = {'identifier': iden_collect,
                                'concentration': conc_collect}
        out_dict['masking_mask'] = mask_collect
        out_dict['padding_mask'] = padding_mask
        return out_dict, conc_collect  



    def save_vocab_file(self, token_list, save_dir, filename_prefix=None):
        vocab_file = os.path.join(save_dir, (filename_prefix + "_" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(token_list))
        return (vocab_file,)



    def push_zeros_to_end(self, conc, iden, conc_mask): 

        if conc.shape != iden.shape:
            raise ValueError("Input metabolites' concentration and identifier tokens must have the same shape.")

        # Initialize arrays to hold the results
        conc_reordered = np.zeros_like(conc)
        embID_reordered = np.zeros_like(iden)
        conc_mask_reordered = np.zeros_like(conc_mask)

        
        # Iterate over each row in the arrays: each row represent each sample
        for i, (row_conc, row_iden, row_conc_mask) in enumerate(zip(conc, iden, conc_mask)):
     
            # Mask of non-Nan elements in the current row of a
            mask = ~np.isnan(row_conc)

            # Reorder the current row of A by filtering non-zeros and then appending zeros
            conc_reordered[i] = np.concatenate((row_conc[mask], row_conc[~mask]))    

            # Apply the same order to the current row of b
            embID_reordered[i] = np.concatenate((row_iden[mask], row_iden[~mask]))
            
            # Apply the same order to the current row of c
            conc_mask_reordered[i] = np.concatenate((row_conc_mask[mask], row_conc_mask[~mask]))

        return conc_reordered, embID_reordered, conc_mask_reordered



   



if __name__ == '__main__':
    ###### module debug

    pass