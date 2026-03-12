import numpy as np
import torch
from torch import Tensor
from typing import Optional, Union
import math
import torch.nn.functional as F


def random_mask(arr_conc, mask_ratio:float=.15):
    '''
    Randomly masking concentration (excluding missing values) for model pre-training.
    arr_conc: a concentration vector of a sample
    mask_ratio: the percentage of concentration tokens to be masked
    '''
    # get the idx of elements excluding zeros and NaN
    idx_nonzero = np.nonzero(~np.isnan(arr_conc) & (arr_conc != 0))[0]
    # get the number of elements to be masked
    n_mask_tokens = int( len(idx_nonzero) * mask_ratio )
    if n_mask_tokens == 0:
        n_mask_tokens = 1   # at least masking one tokens (if the mask ratio is very small, n_mask_tokens would be zero)
    # get the idx of masked elements
    idx_mask = np.random.choice(idx_nonzero, size=n_mask_tokens, replace=False)
    # replace the values of masked elements with NaN (making masked elements are not involved in the concentration value binning)
    arr_conc[idx_mask] = np.nan
    return idx_mask, arr_conc



def missing_mask(arr_conc):
    '''
    Masking concentration missing values for model generation.
    arr_conc: a concentration vector of a sample
    '''    
    # get the idx of zeros (and NaN) elements
    idx_zero_nan = np.nonzero(np.isnan(arr_conc) | (arr_conc == 0))[0]
    # mask all missing values
    idx_mask = idx_zero_nan
    # replace the values of masked elements with NaN (making masked elements are not involved in the concentration value binning) 
    arr_conc[idx_mask] = np.nan
    return idx_mask, arr_conc



def specify_mask(arr_conc, idx_mask):
    arr_conc[idx_mask] = np.nan
    return arr_conc



def top_k_nonzero_indices(tensor, k:Union[int, Tensor]=10, descending:bool=True):
    batch_size, num_cols = tensor.shape
    row_indices = []
    col_indices = []

    for i in range(batch_size):
        row = tensor[i]
        nonzero_indices = torch.nonzero(row, as_tuple=True)[0]
        
        if len(nonzero_indices) == 0:
            row_indices.append(torch.tensor([]))
            col_indices.append(torch.tensor([]))
            continue
        
        ## dynamic or fixed k
        # fix k: for model training; dynamic k: for model generation
        if isinstance(k, Tensor):
            topk = k[i]
        elif isinstance(k, int):
            topk = k

        nonzero_values = row[nonzero_indices]
        top_k_indices = nonzero_indices[torch.argsort(nonzero_values, descending=descending)[:topk]]
        
        row_indices.append(torch.full((len(top_k_indices),), i, dtype=torch.long))
        col_indices.append(top_k_indices)
    
    return torch.concat(row_indices,dim=0), torch.concat(col_indices,dim=0)



def top_k_masked_tokens_by_attn(attn, idx_masked, k):
    '''
    top-k masked tokens with the highest attention scores
    attn: the attention scores of tokens
    idx_masked: the indices of all masked tokens
    k: integer (training) or list[int] (for generation); the number of predicted masked tokens to be retained
    ''' 
    attn_maskedtokens = torch.zeros_like(attn)
    for i in range(len(idx_masked[0])):
        attn_maskedtokens[ idx_masked[0][i],idx_masked[1][i] ] = attn[ idx_masked[0][i],idx_masked[1][i] ]
    idx_masked = top_k_nonzero_indices(attn_maskedtokens, k=k)
    return idx_masked



def top_k_masked_tokens_by_sim(inputs, embs, tokenizer, k, device='cpu'):
    '''
    adding...
    ''' 
   
    ids_specialtokens = [tokenizer.token_to_id_conc(token) for token in tokenizer.all_special_tokens]               # id of special tokens in the concentration vocabulary
    idx_known = ~ torch.isin( inputs['input_ids']['concentration'], torch.Tensor(ids_specialtokens).to(device) )    # obtain the ids of known concentration
    idx_mask = inputs['input_ids']['concentration'] == tokenizer.token_to_id_conc(tokenizer.mask_token)             # obtain the ids of masked concentration
    iden_metabo = inputs['input_ids']['identifier']                                                                 # identifiers for retrieving prior information (e.g., text embedding) 
    buffer = torch.zeros_like(inputs['input_ids']['concentration']).float().to(device)                              # used to record the averaging similarities for each masked token to known tokens   
    

    # for each sample
    for i in range(len(idx_known)): 
        known = idx_known[i]
        metabo = iden_metabo[i]
        mask = idx_mask[i]

        known_metabo = metabo[known]    # set 1: known
        masked_metabo = metabo[mask]    # set 2: to be ranked by distance to set 1

        # known and masked 
        iden_known_metabo = [tokenizer.id_to_token_iden(int(metabo)) for metabo in known_metabo ]
        iden_masked_metabo = [tokenizer.id_to_token_iden(int(metabo)) for metabo in masked_metabo ]
    

        emb_known = embs[iden_known_metabo].T
        emb_maksed = embs[iden_masked_metabo].T
        
        # tensor -> gpu accelerate
        t_emb_known = torch.Tensor(emb_known.values).to(device).unsqueeze(1)
        t_emb_maksed = torch.Tensor(emb_maksed.values).to(device).unsqueeze(0)

        similarity_matrix = F.cosine_similarity(t_emb_known, t_emb_maksed, dim=2)
        similarities = torch.mean(similarity_matrix, dim=0)

        # record similarities for masked tokens
        buffer[i, mask] = similarities
        
    idx_masked = top_k_nonzero_indices(buffer, k=k)
    
    return idx_masked



def top_k_masked_tokens_by_missingratio(inputs, tokenizer, k, missing_ratio, device='cpu'):
    '''
    adding...
    ''' 
    idx_mask = inputs['input_ids']['concentration'] == tokenizer.token_to_id_conc(tokenizer.mask_token)             # obtain the ids of masked concentration
    iden_metabo = inputs['input_ids']['identifier']                                                                 # identifiers for retrieving prior information (e.g., text embedding) 
    buffer_missingratio = torch.zeros_like(inputs['input_ids']['concentration']).float().to(device)                 # used to record the missing ratio of each masked token 

    # for each sample
    for i in range(len(idx_mask)): 
        metabo = iden_metabo[i]
        mask = idx_mask[i]

        masked_metabo = metabo[mask]  
        iden_masked_metabo = [tokenizer.id_to_token_iden(int(metabo)) for metabo in masked_metabo ]
    
        # record missing ratio for masked tokens
        missing_ratio_masked = missing_ratio.loc[iden_masked_metabo]
        buffer_missingratio[i, mask] = torch.Tensor( missing_ratio_masked.values.ravel() ).to(device)

    
    idx_masked = top_k_nonzero_indices(buffer_missingratio, k=k)

    return idx_masked



def _generate_mask_matrix(
        conc_tokens: Tensor,
        mask_id: int,
        device: Optional[torch.device] = None
    ) -> Tensor:
    """Generate a mask matrix for <MAS> tokens.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = conc_tokens.device

    collect = []
    for i in range(conc_tokens.shape[0]):   # for each sample
        # 2D masked matrix
        seq_len = conc_tokens.shape[1]
        casual_mask = torch.zeros(seq_len,seq_len) 
        # columns corresponding to masked tokens
        casual_mask[:, torch.where(conc_tokens[i] == mask_id)[0]] = 1
        # masked token can 'see' itself (receiving attention from itself)
        casual_mask[torch.where(conc_tokens[i] == mask_id)[0], torch.where(conc_tokens[i] == mask_id)[0]] = 0 

        collect.append(casual_mask)
 
    # 3D masked matrix (for batch operation)
    mask_matrix = torch.stack(collect,dim=0).to(device)
    return mask_matrix


def _generate_mask_matrix_VocabFree(
        conc_tokens: Tensor,
        mask_matrix: Tensor,
        num_stable:int=0,
        device: Optional[torch.device] = None
    ) -> Tensor:
    if device is None:
        device = conc_tokens.device

    collect = []
    for i in range(conc_tokens.shape[0]):   # for each sample
        # 2D masked matrix
        seq_len = mask_matrix.shape[1]
        
        if num_stable!=0:
            casual_mask = torch.zeros(seq_len+num_stable+1,seq_len+num_stable+1)  #num_stable+1: plus 1 for CLS
            # columns corresponding to masked tokens
            casual_mask[:, torch.where(mask_matrix[i] == 1)[0]+ (num_stable+1)] = 1
            # masked token can 'see' itself (receiving attention from itself)
            casual_mask[torch.where(mask_matrix[i] == 1)[0]+ (num_stable+1), torch.where(mask_matrix[i] == 1)[0]+ (num_stable+1)] = 0 


        else:   
            casual_mask = torch.zeros(seq_len,seq_len) 
            # columns corresponding to masked tokens
            casual_mask[:, torch.where(mask_matrix[i] == 1)[0]] = 1
            # masked token can 'see' itself (receiving attention from itself)
            casual_mask[torch.where(mask_matrix[i] == 1)[0], torch.where(mask_matrix[i] == 1)[0]] = 0 

        collect.append(casual_mask)


    # 3D masked matrix (for batch operation)
    mask_matrix = torch.stack(collect,dim=0).to(device)
    return mask_matrix


class Mask_Schedule:        
    def __init__(self, max_mask_ratio:float=.5, mode:str='train',gamma_func:str='cosine', n_iterations:int=10):
        if mode == 'train':
            self.max_mask_ratio=max_mask_ratio
        elif mode == 'generation':
            self.max_mask_ratio=1.0
        self.mode=mode
        self.f_type=gamma_func
        self.T=n_iterations

    def get_ratio(self, n_samples:Optional[int]=None):
        if self.mode=='train':
            r = torch.rand(n_samples) 
            mask_ratio = self._mask_schedule(r)
        elif self.mode=='generation': # TODO: if the number of masked tokens is quit small, we need to adjust 'n_iterations' or dont use autegressive generation, just prediction like BERT
            r = torch.linspace(0, 1, self.T+1)
            mask_ratio = self._mask_schedule(r)[:-1] # -1: the last element is 0, which means we finished the generation with the mask ratio of 0; remove it
        else:
            raise TypeError(f"check the mode of Mask_Schedule.get_ratio; {self.mode} is not supported.")
        return mask_ratio


    def _mask_schedule(self, r):
        '''
        The mask scheduling function (gamma), described in MaskGIT: https://arxiv.org/pdf/2202.04200
        Ref to: https://github.com/valeoai/Maskgit-pytorch/blob/main/Trainer/vit.py#L136 

        r: randomly sampled from [0,1) in training; t/T in testing for t-th iteration
        f_type: 'cosine' (default)
        max_mask_ratio: [0,1], default:0.5 (1.0 for model generation); to scale the ratio of mask schedule (for model training)
        '''
        if self.f_type == "root":              # root scheduler
            mask_ratio = 1 - (r ** .5)
        elif self.f_type == "linear":          # linear scheduler
            mask_ratio = 1 - r
        elif self.f_type == "square":          # square scheduler
            mask_ratio = 1 - (r ** 2)
        elif self.f_type == "cosine":          # cosine scheduler
            mask_ratio = torch.cos(r * math.pi * 0.5)
        elif self.f_type == "arccos":          # arc cosine scheduler
            mask_ratio = torch.arccos(r) / (math.pi * 0.5)
        mask_ratio[mask_ratio<0] = 0 # for cosine(1)
        return mask_ratio * self.max_mask_ratio


    def get_num_generate_tokens(self, scheduling_mask_ratios, inputs, id_mask_token):
        # get the mask tokens of each sample in this inputs
        mask_tokens_each_sample = inputs['input_ids']['concentration'] == id_mask_token
        # get the number of the mask tokens for each sample in this inputs
        num_mask_tokens_each_sample = mask_tokens_each_sample.sum(dim=1)
        # get the number of the mask tokens in each interation for each sample in this inputs
        num_mask_tokens_each_sample_each_iter = torch.floor( torch.matmul(num_mask_tokens_each_sample.reshape(-1,1).float() , scheduling_mask_ratios.reshape(1,-1)) ).long() 
        # get the number of the generate tokens in each interation for each sample in this inputs
        difference = num_mask_tokens_each_sample_each_iter[:, :-1] -  num_mask_tokens_each_sample_each_iter[:, 1:]
        last_column = num_mask_tokens_each_sample_each_iter[:, -1:]
        num_generate_tokens_each_sample_each_iter = torch.cat((difference, last_column), dim=1)
        return num_generate_tokens_each_sample_each_iter


if __name__ == '__main__':
    print("Masking")
  
