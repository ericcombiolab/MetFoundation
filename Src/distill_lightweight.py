import os
import numpy as np
import torch
import torch.nn.functional as F
from metfoundation_torch.models import  MetFoundation_Lightweight_Survival
from metfoundation_torch.dataset import  load_dataset_from_adata_NMR
from typing import Optional
from tqdm import tqdm
import json
import argparse
from utils import *
import pandas as pd
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from torchsurv.loss import cox
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from collections import OrderedDict

import warnings    
warnings.filterwarnings("ignore", category=UserWarning)


def compute_classification_metrics(y_true, y_pred, labels=None, include_confusion_matrix=False):
    """
    Compute common evaluation metrics for multi-class classification tasks.

    Parameters:
        y_true (array-like): True labels, shape (n_samples,)
        y_pred (array-like): Predicted labels, shape (n_samples,)
        labels (list, optional): List of class labels (to specify order or include classes not present in samples)
        include_confusion_matrix (bool): Whether to return the confusion matrix

    Returns:
        dict: Dictionary containing the metrics
    """
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Macro and weighted average F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=labels)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=labels)
    
    # Precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels
    )
    
    # Get unique labels (in order of appearance or use provided labels)
    unique_labels = labels if labels is not None else sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    # Build per-class metrics
    per_class_metrics = OrderedDict()
    for i, label in enumerate(unique_labels):
        per_class_metrics[str(label)] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1-score': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        }
    
    results = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': per_class_metrics,
    }
    
    if include_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        results['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization
    
    return results


def pdist(e, squared=False):
    """
    Compute the Euclidean distance matrix between all row vectors in tensor e.
    If squared=True, return the squared distances.
    """
    # e: (B, D) -> B batch size, D embedding dimension
    e_square = e.pow(2).sum(dim=1) # (B,)
    prod = torch.mm(e, e.t()) # (B, B)
    
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    dist_square = (e_square.unsqueeze(0) + e_square.unsqueeze(1) - 2 * prod).clamp(min=1e-8)
    
    if squared:
        return dist_square
    else:
        # Return distance
        return torch.sqrt(dist_square)

def compute_rkd_distance_loss(pred_embs, true_embs):
    """
    Compute RKD distance loss (Relational Knowledge Distillation - Distance Loss)
    
    Args:
        pred_embs (torch.Tensor): Student embeddings (B, D)
        true_embs (torch.Tensor): Teacher embeddings (B, D)
        
    Returns:
        torch.Tensor: RKD distance loss
    """
    
    # 1. Compute distance matrix of Teacher embeddings
    with torch.no_grad():
        # RKD paper suggests normalizing Teacher distances
        # to ensure loss is insensitive to batch size and embedding dimension changes
        t_dist = pdist(true_embs, squared=False)
        mean_t_dist = t_dist[t_dist > 0].mean() # Exclude diagonal zeros
        t_dist = t_dist / mean_t_dist

    # 2. Compute distance matrix of Student embeddings
    s_dist = pdist(pred_embs, squared=False)
    # Student distances use the same normalization factor
    s_dist = s_dist / mean_t_dist
    
    # 3. Compute L1 absolute difference of distances
    # Exclude diagonal zeros, only compute loss for off-diagonal elements
    loss = F.l1_loss(s_dist, t_dist, reduction='mean') # Use L1 loss (|a-b|)
    
    return loss

def compute_rkd_angle_loss(pred_embs, true_embs):
    """
    Compute RKD angle loss (Relational Knowledge Distillation - Angle Loss)
    
    Args:
        pred_embs (torch.Tensor): Student embeddings (B, D)
        true_embs (torch.Tensor): Teacher embeddings (B, D)
        
    Returns:
        torch.Tensor: RKD angle loss
    """
    B, D = pred_embs.shape
    
    # 1. Compute vector differences
    # diff_s[i, k, :] = pred_embs[i] - pred_embs[k] (note index k is the apex)
    # Use unsqueeze(1) broadcasting for subtraction: (B, 1, D) - (1, B, D) -> (B, B, D)
    diff_s = pred_embs.unsqueeze(1) - pred_embs.unsqueeze(0)  # (B, B, D)
    diff_t = true_embs.unsqueeze(1) - true_embs.unsqueeze(0)  # (B, B, D)

    # RKD Angle Loss focuses on triplet angles <(z_i - z_k), (z_j - z_k)>, k is the apex.
    # 
    # Correct dimensions:
    # diff_s[i, k, :] is vector z_i - z_k
    # diff_s[j, k, :] is vector z_j - z_k
    # 
    # We need a (B, B, B) tensor where [i, k, j] stores <z_i-z_k, z_j-z_k>

    # 2. Compute numerator - dot product <v1, v2>
    # Use einsum to compute dot products of i-k and j-k vectors for all k.
    # i, j, k are batch indices, d is dimension index.
    # 'ikd, jkd -> ikj' computes <diff_s[i, k, :], diff_s[j, k, :]>, result shape (B, B, B)
    # Note: To match original paper indexing order, we use i, j, k as B indices.
    # dot_s[i, k, j] = <z_i - z_k, z_j - z_k>
    dot_s = torch.einsum('ikd, jkd -> ikj', diff_s, diff_s)
    dot_t = torch.einsum('ikd, jkd -> ikj', diff_t, diff_t)

    # 3. Compute denominator - ||v1|| * ||v2||
    # Squared norm: ||z_i - z_k||^2
    norm_s_sq = diff_s.pow(2).sum(dim=2) # (B, B) -> [i, k]
    norm_t_sq = diff_t.pow(2).sum(dim=2) # (B, B) -> [i, k]

    # Denominator: ||z_i - z_k|| * ||z_j - z_k||
    # Use einsum: 'ik, jk -> ikj' to compute product of norm_s_sq[i, k] * norm_s_sq[j, k]
    # Then take sqrt to get norm product
    # denominator_s[i, k, j] = ||z_i - z_k|| * ||z_j - z_k||
    denominator_s = torch.sqrt(
        torch.einsum('ik, jk -> ikj', norm_s_sq, norm_s_sq) + 1e-8
    ) # (B, B, B)
    denominator_t = torch.sqrt(
        torch.einsum('ik, jk -> ikj', norm_t_sq, norm_t_sq) + 1e-8
    ) # (B, B, B)
    
    # 4. Compute angle cosine values
    cos_s = dot_s / denominator_s
    cos_t = dot_t / denominator_t
    
    # 5. Compute L1 loss
    # RKD paper usually suggests excluding invalid terms (i=k or j=k or i=j)
    # But for simplicity, we first use mean(), consider masking if results are poor.
    loss = F.l1_loss(cos_s, cos_t, reduction='mean')
    
    return loss


def compute_contrastive_loss(pred_embs, true_embs, temperature=0.07):
    """
    Compute contrastive loss (InfoNCE) within a mini-batch
    contrastive learning only based on student models itself
    """
    batch_size = pred_embs.shape[0]
    
    # Normalize embeddings
    pred_embs = torch.nn.functional.normalize(pred_embs, p=2, dim=1)
    true_embs = torch.nn.functional.normalize(true_embs, p=2, dim=1)
    
    # Compute similarity matrix: pred_embs @ true_embs.T
    # Shape: [batch_size, batch_size]
    similarity_matrix = torch.matmul(pred_embs, true_embs.T) / temperature
    # similarity_matrix = torch.matmul(pred_embs, pred_embs.T) / temperature # Self-contrastive learning on student model
    
    # Create labels: diagonal elements are positives
    labels = torch.arange(batch_size).to(pred_embs.device)
    
    # Compute cross entropy loss
    # For each row, the correct class is the diagonal element
    loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
    
    return loss


def compute_loss(pred_embs, true_embs, target, criterion, mse_criterion, loss_mode, alpha, temperature=0.07):
    """
    Compute loss based on the specified loss mode
    """
    # commonly used loss functions
    if loss_mode == 'cosine':
        loss = criterion(pred_embs, true_embs, target)
    elif loss_mode == 'mse':
        loss = mse_criterion(pred_embs, true_embs)
    elif loss_mode == 'cosine_mse':
        cosine_loss = criterion(pred_embs, true_embs, target)
        mse_loss = mse_criterion(pred_embs, true_embs)
        loss = cosine_loss + alpha * mse_loss
        
    # with RKD  
    elif loss_mode == 'rkd':
        distance_loss = compute_rkd_distance_loss(pred_embs, true_embs)  
        angle_loss = compute_rkd_angle_loss(pred_embs, true_embs)
        loss = distance_loss + angle_loss
    elif loss_mode == 'rkd_mse': 
        mse_loss = mse_criterion(pred_embs, true_embs)
        distance_loss = compute_rkd_distance_loss(pred_embs, true_embs)  
        angle_loss = compute_rkd_angle_loss(pred_embs, true_embs)
        rkd_loss = distance_loss + angle_loss
        loss = rkd_loss + alpha * mse_loss
        
    # with contrastive learning (InfoNCE loss)
    elif loss_mode == 'contrastive':
        loss = compute_contrastive_loss(pred_embs, true_embs, temperature)
    elif loss_mode == 'contrastive_mse':
        contrastive_loss = compute_contrastive_loss(pred_embs, true_embs, temperature)
        mse_loss = mse_criterion(pred_embs, true_embs)
        loss = contrastive_loss + alpha * mse_loss
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")
    
    return loss


def train(
        model,
        dataloader,
        val_dataloader,       
        lr: float=0.0001,
        n_epoch: int=20,    
        n_toler: Optional[int] = None,
        save_dir: str = './',
        device: Optional[str] = None,
        loss_mode: str = 'cosine',
        alpha: float = 0.5, # balance between loss
        temperature: float = 0.07, # temperature for contrastive learning
        cox_weight: float = 0.01,
        emb_weight: float = 1.0,
    ):
    
    
    optim = torch.optim.AdamW(model.parameters(), lr=lr , betas=(0.9, 0.98), eps=1e-07)   
    criterion = torch.nn.CosineEmbeddingLoss()
    mse_criterion = torch.nn.MSELoss()
    # smooth_l1_criterion = torch.nn.SmoothL1Loss()
    
    train_loss_epoch =[]
    val_loss_epoch = []
    val_subtype_loss_epoch = []
    best_val_loss = 999999
    watchdog = 0


    for epoch in range(n_epoch):
        model.train()
        step_loss_collect = []

        for data in tqdm(dataloader):

            optim.zero_grad()
            
            age = data.obs['Age at assessment (estimated)'].copy()
            age_normalized = torch.Tensor(age.values / 100).to(device)
            
         
            survival_info = get_survival_info(data, return_tensor=True, device=device)

            
            X = torch.Tensor(np.nan_to_num(data.layers['Z-score normalized'].copy(), nan=0.0)).to(device)
            y = torch.Tensor( data.obsm['metabolomic embedding'].copy() ).to(device)
            target = torch.ones(X.size(0)).to(device) 
            
            out = model(X, age_normalized)
            embs = out['embs']
            risk = out['logit_risk']
            
            loss = compute_loss(embs, y, target, criterion, mse_criterion, loss_mode, alpha, temperature)
            loss =  emb_weight * loss + cox_weight * cox.neg_partial_log_likelihood(risk, survival_info['event'], survival_info['time'])

            if 'logit_subtype' in out.keys():
                subtype = out['logit_subtype']
                subtype_labels = data.obs['metabolic subtype'].values
                subtype_labels_tensor = torch.LongTensor(subtype_labels).to(device)
                subtype_loss = F.cross_entropy(subtype, subtype_labels_tensor)
                loss += subtype_loss  # add subtype classification loss


            loss.backward()
            optim.step()
            step_loss_collect.append(loss.item())



            
        train_loss_epoch.append(np.mean(step_loss_collect))


        model.eval()
        val_loss_collect = []
        val_subtype_loss_collect = []
        
    
        for data_val in val_dataloader:

            X_val = torch.Tensor(np.nan_to_num(data_val.layers['Z-score normalized'].copy(), nan=0.0)).to(device)               
            y_val = torch.Tensor( data_val.obsm['metabolomic embedding'].copy() ).to(device)
            target_val = torch.ones(X_val.size(0)).to(device) 
            
            age = data_val.obs['Age at assessment (estimated)'].copy()
            age_normalized = torch.Tensor(age.values / 100).to(device)
            # age_normalized = age_normalized.unsqueeze(1)
            
            with torch.no_grad():
                out_val = model(X_val, age_normalized)
                risk = out_val['logit_risk']
                embs_val = out_val['embs']
                
            survival_info = get_survival_info(data_val, return_tensor=True, device=device)
            loss_val = compute_loss(embs_val, y_val, target_val, criterion, mse_criterion, loss_mode, alpha, temperature)
            loss_val = emb_weight * loss_val + cox_weight * cox.neg_partial_log_likelihood(risk, survival_info['event'], survival_info['time'])
            
            if 'logit_subtype' in out_val.keys():
                subtype = out_val['logit_subtype']
                subtype_labels = data_val.obs['metabolic subtype'].values
                subtype_labels_tensor = torch.LongTensor(subtype_labels).to(device)
                subtype_loss = F.cross_entropy(subtype, subtype_labels_tensor)
                loss_val += subtype_loss  # add subtype classification loss

                val_subtype_loss_collect.append(subtype_loss.item())
            
            val_loss_collect.append(loss_val.item())
            
        if len(val_subtype_loss_collect) > 0:
            val_subtype_loss_epoch.append( np.mean(val_subtype_loss_collect) )


        val_loss_epoch.append(np.mean(val_loss_collect))

        if len(val_subtype_loss_epoch) > 0:
            print(f"Epoch {epoch+1}/{n_epoch} | Train Loss: {train_loss_epoch[-1]} | Val Loss: {val_loss_epoch[-1]} | Val Subtype Loss: {val_subtype_loss_epoch[-1]}")
        else:
            print(f"Epoch {epoch+1}/{n_epoch} | Train Loss: {train_loss_epoch[-1]} | Val Loss: {val_loss_epoch[-1]}")

        # Early stopping
        if val_loss_epoch[-1] < best_val_loss:
            best_val_loss = val_loss_epoch[-1]
            watchdog = 0
            # torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))
            model.save_distilled(os.path.join(save_dir, 'model_weights.pth'))
        else:
            watchdog += 1
            if n_toler is not None and watchdog >= n_toler:
                print("Early stopping triggered.")
                break


        


def get_survival_info(adata, return_tensor=False, device='cpu'):
    label = {}
    event = adata.obs['Death event'].values
    time = adata.obs['Death event time'].values

    if return_tensor:
        event = torch.Tensor(event).float().to(device)
        time = torch.Tensor(time).float().to(device)

    label['event'] = event
    label['time'] = time

    return label

def testing(
        model,
        dataloader,
        test_adata,
        save_dir: str = './',
        device: Optional[str] = None
    ):
    cindex = ConcordanceIndex()
    auc = Auc()

    model.eval()

    # embs_collect = []
    risk_collect = []
    survival_event_collect = []
    survival_time_collect = []
    true_subtype_collect = []
    pred_subtype_collect = []
    for data in tqdm(dataloader):
        age = data.obs['Age at assessment (estimated)'].copy()
        age_normalized = torch.Tensor(age.values / 100).to(device)
        # age_normalized = age_normalized.unsqueeze(1)
        
        X = torch.Tensor(np.nan_to_num(data.layers['Z-score normalized'].copy(), nan=0.0)).to(device)
        
        survival_info = get_survival_info(data, return_tensor=True, device=device)
        
        with torch.no_grad():
            out = model(X, age_normalized)
        # embs = out['embs'] 
        risk = out['logit_risk']
        # embs_collect.append(embs.cpu().detach().numpy())
        risk_collect.append(risk.data.flatten().cpu().detach())
        survival_event_collect.append(survival_info['event'].cpu().detach())
        survival_time_collect.append(survival_info['time'].cpu().detach())

        if 'logit_subtype' in out.keys():
            subtype = out['logit_subtype']
            pred_subtypes = torch.argmax(subtype, dim=1).cpu().detach().numpy()
            true_subtypes = data.obs['metabolic subtype'].values
            true_subtype_collect.append(true_subtypes)
            pred_subtype_collect.append(pred_subtypes)
        
        
        
    if len(pred_subtype_collect) > 0:    
        true_subtype_collect = np.concatenate(true_subtype_collect, axis=0)
        pred_subtype_collect = np.concatenate(pred_subtype_collect, axis=0)
        metrics = compute_classification_metrics(
                    y_true=true_subtype_collect,
                    y_pred=pred_subtype_collect,
                    include_confusion_matrix=True
                )
        print("Accuracy:", metrics['accuracy'])
        print("Macro F1:", metrics['macro_f1'])
        print("Per-class metrics:")
        for cls, scores in metrics['per_class'].items():
            print(f"Class {cls}: Precision={scores['precision']:.3f}, Recall={scores['recall']:.3f}, F1={scores['f1-score']:.3f}")
        
        with open(os.path.join(save_dir, 'classification_metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
            
    # # save embeddings generated from lightweight models for evaluation
    # embs_all = np.concatenate(embs_collect, axis=0)
    # pickle.dump(embs_all, open(os.path.join(save_dir, 'embeddings_test.pkl'), 'wb')) 

    # save risk scores
    risk_all = torch.cat(risk_collect, dim=0)
    pd.DataFrame({'eid':test_adata.obs_names, 'risk': risk_all.numpy()}).to_csv(os.path.join(save_dir,'prediction.csv'), index=False)

    # save survival information
    survival_event_collect = torch.cat(survival_event_collect, dim=0)
    survival_time_collect = torch.cat(survival_time_collect, dim=0)


    test_cidx = cindex(risk_all, survival_event_collect.bool(), survival_time_collect)
    test_cidx_ci = cindex.confidence_interval()

    test_auc = auc(risk_all, survival_event_collect.bool(), survival_time_collect, new_time=10)[0]
    f = open(os.path.join(save_dir, 'cindex.txt'), 'w')
    f.write(f"{test_cidx}\n")
    f.write(f"{test_cidx_ci[0]}\n")
    f.write(f"{test_cidx_ci[1]}\n")
    f.close()

    f = open(os.path.join(save_dir, 'auc.txt'), 'w')
    f.write(f"{test_auc.data}\n")
    f.close()
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--train_config', type=str, help='.')
    args = parser.parse_args()

    with open(args.train_config, 'r') as file:
        train_settings = json.load(file)
        
    # params for model training
    lr = train_settings['lr']
    batch_size = train_settings['batch_size']
    n_epoch = train_settings['n_epoch']
    n_toler = train_settings['n_toler']
    data_path = train_settings['data_path']

    # params of model architecture
    drop_out = train_settings['drop_out']
    d_model = train_settings['d_model']

    # loss parameters
    loss_mode = train_settings.get('loss_mode', 'cosine')
    alpha = train_settings.get('alpha', 0.5)
    temperature = train_settings.get('temperature', 0.07)
    cox_weight = train_settings.get('cox_weight', 0.01)
    emb_weight = train_settings.get('emb_weight', 1.0)
    

    # save setting
    save_note = train_settings['save_note']
  
    save_dir = os.path.join(train_settings['save_dir'], f'{save_note}')
    
    risk_head_path = train_settings['risk_head_path']
    
    ####################################################################################
    # creata dir for saving models and results
    create_directory(save_dir)

    # randomseed and cuda
    set_seeds(3047)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data loading 
    # train_loader, train_data = load_dataset_from_adata_NMR(os.path.join(data_path, 'train.h5ad'),
    #                                             shuffle=True,
    #                                             batch_size=batch_size,
    #                                             device=device)
    # val_loader, _ = load_dataset_from_adata_NMR(os.path.join(data_path, 'val.h5ad'),
    #                                             shuffle=True,
    #                                             batch_size=batch_size,                                        
    #                                             device=device)
    
    # n_features = train_data.n_vars

    # test_loader, test_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'test.h5ad'),
    #                                                 shuffle=False,
    #                                                 batch_size=batch_size,                                         
    #                                                 device=device)
    
    
    # fake data for testing code functionality
    train_loader, train_data = load_dataset_from_adata_NMR(os.path.join(data_path, 'fake_val.h5ad'),
                                                shuffle=True,
                                                batch_size=batch_size,
                                                device=device)
    val_loader, _ = load_dataset_from_adata_NMR(os.path.join(data_path, 'fake_val.h5ad'),
                                                shuffle=True,
                                                batch_size=batch_size,                                        
                                                device=device)
    
    n_features = train_data.n_vars

    test_loader, test_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'fake_val.h5ad'),
                                                    shuffle=False,
                                                    batch_size=batch_size,                                         
                                                    device=device)


    # model
    model_conf = {
        'n_features': n_features,
        'd_model': d_model,
        'dropout': drop_out
    }
    
    if 'metabolic subtype' in train_data.obs.columns:
        n_subtypes = train_data.obs['metabolic subtype'].nunique()
        model_conf['n_subtypes'] = n_subtypes
        print(f"Number of metabolic subtypes: {n_subtypes}")
    
    model = MetFoundation_Lightweight_Survival(model_conf)

    model._load_risk_head_weights(os.path.join(risk_head_path,'model_weights.pth')) 
    
    # Freeze risk_head parameters
    for param in model.risk_head.parameters():
        param.requires_grad = False
    print("Frozen risk_head parameters - they will not be updated during training")
    
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())  # size of all model
    print(f'Total parameters:\t{total_params}')

    distilled_params = sum(p.numel() for p in model.lightweight_model.parameters())  # size of the lightweight model
    print(f'Distilled model parameters:\t{distilled_params}')
    
    risk_head_params = sum(p.numel() for p in model.risk_head.parameters())  # size of risk_head (frozen)
    print(f'Risk head parameters (frozen):\t{risk_head_params}')



   
    train(
        model,
        train_loader,
        val_loader,  
        lr=lr,
        n_epoch=n_epoch, 
        n_toler=n_toler,
        save_dir=save_dir,
        device=device,
        loss_mode=loss_mode,
        alpha=alpha,
        temperature=temperature,
        cox_weight=cox_weight,
        emb_weight=emb_weight
    )
    
    test_model = MetFoundation_Lightweight_Survival(model_conf)
    test_model.from_distilled(os.path.join(save_dir, 'model_weights.pth'))
    test_model.to(device)
    testing(test_model, test_loader, test_adata,save_dir, device)
    




# python distill_lightweight.py --train_config ./distill_config/blood_Distill.json
