import torch
from torch.utils.data import Dataset, DataLoader
import anndata as ad 
import numpy as np
import os
import pandas as pd

class AnnDataset(Dataset):
    def __init__(self, adata):
        self.adata = adata

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        ## modify here to fit the input data 
        return self.adata[idx]  # data sampling


def data_collection(data): 
    concated_ad = ad.concat(data)
    # concated_ad.var = data[0].var # avoid .var loss
    # concated_ad.varm = data[0].varm # avoid .var loss

    return concated_ad 


def _load_dataset_from_adata(adata,  batch_size=128, shuffle=False):
    dataset = AnnDataset(adata)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collection)
    return dataloader


def dataset_split_adata(adata, save_split=None, val_ratio=None):
    def _dataset_split_adata(adata, ratio:float=0.1):
        indices = np.arange(adata.n_obs)
        np.random.shuffle(indices)
        n_set1 = int(adata.n_obs * ratio)
        set1_indices = indices[:n_set1]
        set2_indices = indices[n_set1:]
        return adata[set1_indices], adata[set2_indices]

    val_data=None
    test_data, train_data = _dataset_split_adata(adata, ratio=0.1)
    if val_ratio:
        val_data, train_data = _dataset_split_adata(train_data, ratio=val_ratio)

    ## save split datasets for analysis and competing with baseline methods
    if save_split:
        if not os.path.exists(save_split):
            os.makedirs(save_split)

        train_data.write(os.path.join(save_split, 'train.h5ad'), compression="gzip")
        test_data.write(os.path.join(save_split, 'test.h5ad'), compression="gzip")
        if val_ratio:
            val_data.write(os.path.join(save_split, 'val.h5ad'), compression="gzip")
    
    return train_data, val_data, test_data


def load_dataset_from_dir(data_path:str, metabo_prior:bool=False, with_batchID:bool=False, batch_size:int=128, device:str='cpu'):
    train_data = ad.read_h5ad(os.path.join(data_path,'train.h5ad')) 
    val_data = ad.read_h5ad(os.path.join(data_path,'val.h5ad')) 

    # metabolite metadata
    metabo_embs = None
    if metabo_prior == True:
        # metabo_meta = train_data.var
        # mol_embs = pd.DataFrame(metabo_meta.varm['molecular embedding'].T, columns=adata.var_names)       # prior knowledge: molecular propertities (embedded by Chemical Checker)
        # text_embs = pd.DataFrame(metabo_meta.varm['text embedding'].T, columns=adata.var_names)           # prior knowledge: text description (embedded by ChatGPT)
        mol_embs = torch.Tensor(train_data.varm['molecular embedding']).to(device)       # prior knowledge: molecular propertities (embedded by Chemical Checker)
        text_embs = torch.Tensor(train_data.varm['text embedding']).to(device)           # prior knowledge: text description (embedded by ChatGPT)
        metabo_embs = {'mol_embs': mol_embs, 'text_embs': text_embs}                                      # wrap as a dict object
    
    # sample metadata
    batch_vocab = None
    if with_batchID == True:
        sample_meta = train_data.obs
        if 'Project' in sample_meta.keys():
            batch_vocab = list(set(sample_meta['Project'].values))


    train_dataloader = _load_dataset_from_adata(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = _load_dataset_from_adata(val_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, train_data, val_data, metabo_embs, batch_vocab



def load_dataset_from_adata(data_path:str, metabo_prior:bool=False, with_batchID:bool=False, shuffle:bool=False, batch_size:int=128, device:str='cpu'):
    adata = ad.read_h5ad(data_path) 

    # metabolite metadata
    metabo_embs = None
    if metabo_prior == True:
        mol_embs = torch.Tensor(adata.varm['molecular embedding']).to(device)       # prior knowledge: molecular propertities (embedded by Chemical Checker)
        text_embs = torch.Tensor(adata.varm['text embedding']).to(device)           # prior knowledge: text description (embedded by ChatGPT)
        metabo_embs = {'mol_embs': mol_embs, 'text_embs': text_embs}                                      # wrap as a dict object
    
    # sample metadata
    batch_vocab = None
    if with_batchID == True:
        sample_meta = adata.obs
        if 'Project' in sample_meta.keys():
            batch_vocab = list(set(sample_meta['Project'].values))

    dataloder = _load_dataset_from_adata(adata, batch_size=batch_size, shuffle=shuffle)
    return dataloder, adata, metabo_embs, batch_vocab



def load_dataset_from_dir_NMR(data_path:str, batch_size:int=128, device:str='cpu'):
    train_data = ad.read_h5ad(os.path.join(data_path,'train.h5ad')) 
    val_data = ad.read_h5ad(os.path.join(data_path,'val.h5ad')) 

    train_dataloader = _load_dataset_from_adata(train_data, batch_size=batch_size, shuffle=False)
    val_dataloader = _load_dataset_from_adata(val_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, train_data, val_data



def load_dataset_from_adata_NMR(data_path:str, shuffle:bool=False, batch_size:int=128, specify_eid=None, specific_col=None, device:str='cpu'):
    adata = ad.read_h5ad(data_path) 
    if isinstance(specify_eid,list):
        adata = adata[specify_eid]
    if specific_col:
        adata = adata[:, specific_col]
    dataloder = _load_dataset_from_adata(adata, batch_size=batch_size, shuffle=shuffle)
    return dataloder, adata



if __name__ == '__main__':
    ###### module debug
    print("MetaboLM dataset")
