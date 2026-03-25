import os
import torch
from metfoundation_torch.models import RiskPredictionHead
from metfoundation_torch.dataset import load_dataset_from_adata_NMR

import pandas as pd
from tqdm import tqdm
from typing import Optional
import json
import argparse

from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc


from utils import *
    
import warnings    
warnings.filterwarnings("ignore", category=UserWarning)


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




def train(
        model,
        train_loader,
        val_loader,      
        lr: float=0.0001,
        n_epoch: int=20, 
        save_dir: str = './',
        n_toler: Optional[int] = None,
    ):
    
    cindex = ConcordanceIndex()

    # optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr , betas=(0.9, 0.98), eps=1e-07,weight_decay=0.1)   


    train_loss_epoch =[]
    train_cidx_epoch =[]


    val_loss_epoch = []
    val_cidx_epoch =[]

    
    # best_val_loss = 999999
    best_val_cidx = 0 
    watchdog = 0


    for epoch in range(n_epoch):

        # switch model into training mode
        model.train() 

        risk_collect = []
        loss_collect = []
        survival_event_collect = []
        survival_time_collect = []
 
        for data in tqdm(train_loader,desc=f"Epoch {epoch} Training",total=len(train_loader)):


            optim.zero_grad()                                               # optimizer step: clean cache gradients

            # inputs = torch.Tensor(data.X).to(device)
            inputs = torch.Tensor(data.obsm['metabolomic embedding']).to(device)
            
      
            # get chronological age for survival model
            age = data.obs['Age at assessment (estimated)'].copy()
            age_normalized = torch.Tensor(age.values / 100).to(device)
    

            survival_info = get_survival_info(data, return_tensor=True, device=device)

            outputs = model(inputs, age_normalized)                         # model forward: fuse age and predict

            risk = outputs[0]

     
            loss = cox.neg_partial_log_likelihood(risk, survival_info['event'], survival_info['time'])

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optim.step()

            risk_collect.append(risk.data.flatten().cpu().detach())
            loss_collect.append(loss.data.cpu().detach().numpy())
            survival_event_collect.append(survival_info['event'].cpu().detach())
            survival_time_collect.append(survival_info['time'].cpu().detach())
  
            


        risk_collect = torch.cat(risk_collect, dim=0)    
        survival_event_collect = torch.cat(survival_event_collect, dim=0)
        survival_time_collect = torch.cat(survival_time_collect, dim=0)

        train_cidx = cindex(risk_collect, survival_event_collect.bool(), survival_time_collect)
        
        train_cidx_epoch.append(train_cidx)
        train_loss_epoch.append(np.mean(loss_collect))
         


        ## validation
        val_risk_collect = []
        val_loss_collect = []
        val_survival_event_collect = []
        val_survival_time_collect = []
  
        model.eval() 
        for data in tqdm(val_loader,desc=f"Epoch {epoch} Validation",total=len(val_loader)):



            # inputs = torch.Tensor(data.X).to(device)
            inputs = torch.Tensor(data.obsm['metabolomic embedding']).to(device)
                    
  
            
            # get chronological age for survival model
            age = data.obs['Age at assessment (estimated)'].copy()
            age_normalized = torch.Tensor(age.values / 100).to(device)
    
            survival_info = get_survival_info(data, return_tensor=True, device=device)
            with torch.no_grad():  
                outputs = model(inputs, age_normalized)                     # model forward: fuse age and predict
          
                risk = outputs[0]
            loss = cox.neg_partial_log_likelihood(risk, survival_info['event'], survival_info['time'])
            val_risk_collect.append(risk.data.flatten().cpu().detach())
            val_loss_collect.append(loss.data.cpu().detach().numpy())
            val_survival_event_collect.append(survival_info['event'].cpu().detach())
            val_survival_time_collect.append(survival_info['time'].cpu().detach())


        val_risk_collect = torch.cat(val_risk_collect, dim=0)    
        val_survival_event_collect = torch.cat(val_survival_event_collect, dim=0)
        val_survival_time_collect = torch.cat(val_survival_time_collect, dim=0)

        val_cidx = cindex(val_risk_collect, val_survival_event_collect.bool(), val_survival_time_collect)
        val_cidx_epoch.append(val_cidx)
        val_loss_epoch.append(np.mean(val_loss_collect))

 

        

        print(f"epoch: {epoch}; train loss:{train_loss_epoch[-1]}, C-index:{train_cidx}; val loss:{val_loss_epoch[-1]}, C-index:{val_cidx}")


        ## model saving and early stopping
        if isinstance(n_toler, int):
            watchdog +=1 
     
                
            if val_cidx_epoch[-1] > best_val_cidx:
                watchdog = 0
                best_val_cidx = val_cidx_epoch[-1]
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))  
                
                  
            if watchdog >= n_toler:
                save_txt_single_column(train_loss_epoch, save_dir=save_dir, filename='train_loss.txt')
                save_txt_single_column(val_loss_epoch, save_dir=save_dir, filename='val_loss.txt')
                save_txt_single_column(train_cidx_epoch, save_dir=save_dir, filename='train_cidx.txt')
                save_txt_single_column(val_cidx_epoch, save_dir=save_dir, filename='val_cidx.txt')
                break
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))

            
        ## clean cache
        torch.cuda.empty_cache()
        
        
    save_txt_single_column(train_loss_epoch, save_dir=save_dir, filename='train_loss.txt')
    save_txt_single_column(val_loss_epoch, save_dir=save_dir, filename='val_loss.txt')
    save_txt_single_column(train_cidx_epoch, save_dir=save_dir, filename='train_cidx.txt')
    save_txt_single_column(val_cidx_epoch, save_dir=save_dir, filename='val_cidx.txt')

def test(
        model,
        test_adata,
        test_loader,
        save_dir,
        device='cpu'
        ):
    cindex = ConcordanceIndex()
    auc = Auc()     
    
    model.eval() 
 
    test_risk_collect = []
    test_survival_event_collect = []
    test_survival_time_collect = []

    for data in tqdm(test_loader,desc=f"Testing",total=len(test_loader)):

        
        # inputs = torch.Tensor(data.X).to(device)
        inputs = torch.Tensor(data.obsm['metabolomic embedding']).to(device)
   

        # get chronological age for survival model
        age = data.obs['Age at assessment (estimated)'].copy()
        age_normalized = torch.Tensor(age.values / 100).to(device)

        survival_info = get_survival_info(data, return_tensor=True, device=device)
        with torch.no_grad():  
            outputs = model(inputs, age_normalized)                         # model forward: fuse age and predict
            risk = outputs[0]

        test_risk_collect.append(risk.data.flatten().cpu().detach())
        test_survival_event_collect.append(survival_info['event'].cpu().detach())
        test_survival_time_collect.append(survival_info['time'].cpu().detach())


    test_risk_collect = torch.cat(test_risk_collect, dim=0)
    test_survival_event_collect = torch.cat(test_survival_event_collect, dim=0)
    test_survival_time_collect = torch.cat(test_survival_time_collect, dim=0)


    # C-index and 10-year AUC in hold-out test set
    test_cidx = cindex(test_risk_collect, test_survival_event_collect.bool(), test_survival_time_collect)
    test_cidx_ci = cindex.confidence_interval()
    test_auc = auc(test_risk_collect, test_survival_event_collect.bool(), test_survival_time_collect, new_time=10)[0]

    # save risk scores
    pd.DataFrame({'eid':test_adata.obs_names, 'risk': test_risk_collect.numpy()}).to_csv(os.path.join(save_dir,'prediction.csv'), index=False)

    # save C-index
    f = open(os.path.join(save_dir, 'cindex.txt'), 'w')
    f.write(f"{test_cidx}\n")
    f.write(f"{test_cidx_ci[0]}\n")
    f.write(f"{test_cidx_ci[1]}\n")
    f.close()
    
    f = open(os.path.join(save_dir, 'auc.txt'), 'w')
    f.write(f"{test_auc.data}\n")
    f.close()


    return risk, test_cidx

 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--model_dir', type=str, help='.')
    parser.add_argument('--save_dir', type=str, help='.')
    parser.add_argument('--data_path', type=str, help='.')
    parser.add_argument('--batch_size', type=int, default=4096,required=False, help='.')

    args = parser.parse_args()

    # loading setting file
    model_dir = args.model_dir
    model_config = os.path.join(model_dir, 'config.json')
    with open(model_config, 'r') as file:
        model_config = json.load(file)
        
    # params for model inference
    batch_size = args.batch_size              
    data_path = args.data_path
    save_dir = args.save_dir


    create_directory(save_dir)                              # the dir of evaluation result

    set_seeds(3047)                                         # random seed

    device = "cuda" if torch.cuda.is_available() else "cpu" 


    # train_loader, train_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'train.h5ad'),
    #                                                     shuffle=True,
    #                                                     batch_size=batch_size,
    #                                                     device=device)    
    # val_loader, val_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'val.h5ad'),
    #                                                     shuffle=True,
    #                                                     batch_size=batch_size,
    #                                                     device=device)
    # test_loader, test_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'test.h5ad'),
    #                                                     shuffle=False,
    #                                                     batch_size=batch_size,
                                                        # device=device)

    # fake data for demonstration
    train_loader, train_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'fake_val.h5ad'),
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        device=device)    
    val_loader, val_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'fake_val.h5ad'),
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        device=device)
    test_loader, test_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'fake_val.h5ad'),
                                                        shuffle=False,
                                                        batch_size=batch_size,
                                                        device=device)


  
    # Create complete age-fused prediction model
    model = RiskPredictionHead(
        d_model=model_config['d_model'],
        num_classes=1,
        fusion_mode='add',
        activation=None,
        transform_layer=False
    )
    


    model.to(device)   
  

    train(
        model,
        train_loader,
        val_loader,
        lr=0.0001,
        n_epoch=1000, 
        n_toler= 5,
        save_dir=save_dir,
    )
    

    # Load trained model
    test_model = RiskPredictionHead(
        d_model=model_config['d_model'],
        num_classes=1,
        fusion_mode='add',
        activation=None,
        transform_layer=False
    )
    test_model.load_state_dict(torch.load(os.path.join(save_dir, 'model_weights.pth'), map_location=device))
    test_model.to(device)  

    test(
        test_model,
        test_adata,
        test_loader,
        save_dir=save_dir,
        device=device
        )






# python finetune_UKBNMR_Mortality.py \
# --save_dir ../Finetuned_Weights/Demo_Fine_Tune \
# --model_dir ../Pretrained_Weights/Demo_Pre_Train \
# --data_path ../Data/UKB_Blood

