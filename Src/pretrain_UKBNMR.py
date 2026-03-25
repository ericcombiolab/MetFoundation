import os
# import anndata as ad 
import numpy as np
import torch
# import torch.nn.functional as F
from metfoundation_torch.tokenizer import MetFoundation_Tokenizer
from metfoundation_torch.models import MetFoundation_ForPreTrain
from metfoundation_torch.dataset import  load_dataset_from_dir_NMR, load_dataset_from_adata_NMR
from metfoundation_torch.mask_utils import Mask_Schedule

from typing import Optional
import math

import json
from collections import defaultdict

import argparse

import wandb

from utils import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



class WarmupLR:
    def __init__(self, optimizer, max_lr, num_warm, num_allsteps,decay_type='linear') -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        self.lr = max_lr
        self.num_step = 0
        self.num_allsteps = num_allsteps
        self.decay_type = decay_type

    def __compute(self, lr) -> float: 
        if self.num_step <= self.num_warm:
            initial_lr = lr*0.1
            return initial_lr + (lr - initial_lr) * (self.num_step / self.num_warm)
        else:   # linear decay
            if self.decay_type == 'linear':
                return lr * (1- ( (self.num_step-self.num_warm) / (self.num_allsteps-self.num_warm) ) )
            elif self.decay_type == 'cosine':
                return lr * 0.5 * (1 + math.cos(math.pi * (self.num_step-self.num_warm) / (self.num_allsteps-self.num_warm) ))

    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr] 
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]

    def get_lr(self):
        return self.lr


    
def get_multitask_labels(adata, return_tensor=False, device='cpu'):
    label = {}

    age = adata.obs['Age at assessment (estimated)'].values / 100     # normalizing value scale
    sex = adata.obs['Sex'].values 
    bmi = adata.obs['Body mass index (BMI)'].values / 100       # normalizing value scale
    

    if return_tensor:
        age = torch.Tensor(age).float().to(device)
        sex = torch.Tensor(sex).float().to(device)
        bmi = torch.Tensor(bmi).float().to(device)
        
    label['age'] = age
    label['sex'] = sex
    label['bmi'] = bmi
    

    return label


def train(
        model,
        dataloader,
        n_train,
        val_dataloader,        
        lr: float=0.0001,
        n_epoch: int=20, 
        save_dir: str = './',
        max_mr: float = .5,
        n_toler: Optional[int] = None,
        f_loss: str = 'MAE',
        f_gamma: str = 'cosine',
        alpha: float=0.5,
        multitask_config:dict=None
    ):
    
    # mask strategy
    mask_scheduler = Mask_Schedule(mode='train', max_mask_ratio=max_mr, gamma_func=f_gamma)

    # objective: main
    if f_loss=='MAE':
        criterion = torch.nn.L1Loss(reduction='mean')
    elif f_loss=='MSE':
        criterion = torch.nn.MSELoss(reduction='mean')
    elif f_loss=='SmoothL1Loss':
        criterion = torch.nn.SmoothL1Loss(reduction='mean')
    else:
        raise TypeError("Loss function type is not supported.")
    
    # objective: multi-task
    criterion_multitask = {}
    for name, config in multitask_config.items():
        if config[2] == 'mse':
            criterion_multitask[name] = torch.nn.MSELoss(reduction='none')
        elif config[2] == 'bce':
            criterion_multitask[name] = torch.nn.BCELoss(reduction='none')
        elif config[2] == 'mae':
            criterion_multitask[name] = torch.nn.L1Loss(reduction='none')
    # optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr , betas=(0.9, 0.98), eps=1e-07)   


    # learning rate schedule
    n_step_epoch = math.ceil( n_train / batch_size )
    n_total_steps = n_epoch * n_step_epoch
    n_warm_steps = int(n_total_steps/10)
    scheduler = WarmupLR(optimizer=optim, 
                        max_lr=[lr], 
                        num_warm=n_warm_steps, 
                        num_allsteps=n_total_steps,
                        decay_type='cosine')


    train_loss_epoch =[]
    train_conc_loss_epoch =[]
    train_multitask_loss_epoch =defaultdict(list)


    val_loss_epoch = []
    val_conc_loss_epoch = []
    val_multitask_loss_epoch =defaultdict(list)
    
    best_val_loss = 999999
    watchdog = 0

    step_count = 0
    for epoch in range(n_epoch):

        # switch model into training mode
        model.train() 

        # loss buffer
        step_loss_collect = []
        step_conc_loss_collect = []
        step_multitask_loss_collect = defaultdict(list)

        for data in dataloader:
                
            optim.zero_grad()                                               # optimizer step: clean cache gradients
            
            mask_ratios = mask_scheduler.get_ratio(len(data))               # mask ratio scheduling
      
            inputs, label = Tokenizer.tokenize_from_anndata(data, padding='longest', masking='random', data_layer='Z-score normalized',
                                                            masking_ratios=mask_ratios, return_tensor=True, device=device)
            
        
            multitask_label = get_multitask_labels(data, return_tensor=True, device=device)

        
            outputs = model(inputs)                                         # model forward
   
           
            idx_masked = torch.where(inputs['masking_mask']==1)
            predicted = outputs['logit_conc'][:,1:]                         # remove the prediction for <CLS> token
            loss = criterion(predicted[idx_masked], label[idx_masked])             
            step_conc_loss_collect.append(loss.data.cpu().detach().numpy())


            if len(multitask_config) > 0:                                   # multi-task loss
                loss_multitask = 0    
                for name, config in multitask_config.items():
              
                    idx_notnan = torch.where( ~torch.isnan(multitask_label[name]) )
                    loss_multitask_buffer = criterion_multitask[name](outputs[f"logit_{name}"].flatten()[idx_notnan], multitask_label[name][idx_notnan])
                    loss_multitask_buffer = torch.mean(loss_multitask_buffer)      
        
        
                    step_multitask_loss_collect[name].append(loss_multitask_buffer.data.cpu().detach().numpy()) 
                    loss_multitask += config[3] * loss_multitask_buffer     # weight each additional task  
                          
                loss = alpha*loss + (1-alpha)*loss_multitask                # weight main task 


            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optim.step()     
            
            
            step_loss_collect.append(loss.data.cpu().detach().numpy())
                 
            scheduler.step()    # learning rate warmup & decay

          
            step_count+=1       # step counting
    
       
            # print the loss of each step
            loss_info = [f'step {str(step_count)} loss:\t{str(step_loss_collect[-1])}', f'conc:\t{str(step_conc_loss_collect[-1])}']
            train_info = {'learning_rate': optim.param_groups[0]['lr'],'loss':step_loss_collect[-1], 'conc':step_conc_loss_collect[-1]}
            
            if len(multitask_config) > 0: 
                for name, _ in multitask_config.items():
                    loss_info.append(f'{name}:\t{str(step_multitask_loss_collect[name][-1])}')
                    train_info[name] = step_multitask_loss_collect[name][-1]
        
            print('\t'.join(loss_info))
            
            if wandb_monitor==True:
                wandb.log(train_info)

            
     

        # metric collection during model training epoch
        train_loss_epoch.append(np.mean(np.array(step_loss_collect)))
        train_conc_loss_epoch.append(np.mean(np.array(step_conc_loss_collect)))


        ## save training loss
        save_txt_single_column(train_loss_epoch, save_dir=save_dir, filename='train_loss.txt')
        save_txt_single_column(train_conc_loss_epoch, save_dir=save_dir, filename='train_conc_loss.txt')
        

        ## validation
        val_loss, val_loss_conc, val_loss_multitask = validation(model, 
                                                                val_dataloader, 
                                                                mask_scheduler,
                                                                criterion,
                                                                criterion_multitask, 
                                                                multitask_config,
                                                                alpha)
                                            
        val_loss_epoch.append(val_loss)
        val_conc_loss_epoch.append(val_loss_conc)
        print(f'epoch:\t{str(epoch)}, val loss:\t{str(val_loss_epoch[-1])}' )
        save_txt_single_column(val_loss_epoch, save_dir=save_dir, filename='val_loss.txt')
        save_txt_single_column(val_conc_loss_epoch, save_dir=save_dir, filename='val_conc_loss.txt')
        

        ## save multi-task loss (train and val)
        if len(multitask_config) > 0: 
            for name, config in multitask_config.items():
                train_multitask_loss_epoch[name].append(np.mean(np.array(step_multitask_loss_collect[name])))
                save_txt_single_column(train_multitask_loss_epoch[name], save_dir=save_dir, filename=f"train_{name}_loss.txt")
                
                val_multitask_loss_epoch[name].append(val_loss_multitask[name])       
                save_txt_single_column(val_multitask_loss_epoch[name], save_dir=save_dir, filename=f"val_{name}_loss.txt")
                


        ## model saving and early stopping
        if isinstance(n_toler, int):
            watchdog +=1 
            if val_loss < best_val_loss:
                watchdog = 0
                best_val_loss = val_loss      
                model.save_pretrained( save_path=os.path.join(save_dir,'model_weights.pth'))
                
                # break #debug
                
            if watchdog >= n_toler:
                break
        else:
            model.save_pretrained( save_path=os.path.join(save_dir,'model_weights.pth'))

        ## clean cache
        torch.cuda.empty_cache()
            
    


def validation(
    model,
    val_dataloader,
    mask_scheduler,
    criterion,
    criterion_multitask,  
    multitask_config,    
    alpha: float=0.5
    ):


    # switch model into evalution mode
    model.eval() 

    step_loss_collect = []
    conc_loss_collect = []
    multitask_loss_collect = defaultdict(list)




    for data in val_dataloader:
        # mask ratio scheduling
        mask_ratios = mask_scheduler.get_ratio(len(data))
    
        inputs, label = Tokenizer.tokenize_from_anndata(data, padding='longest', masking='random', data_layer='Z-score normalized',
                                                        masking_ratios=mask_ratios, return_tensor=True, device=device)
        multitask_label = get_multitask_labels(data, return_tensor=True, device=device)

        with torch.no_grad():
            outputs = model(inputs)                                     # model forward
    
        idx_masked = torch.where(inputs['masking_mask']==1)
        predicted = outputs['logit_conc'][:,1:]                         # remove the prediction for <CLS> token
        loss = criterion(predicted[idx_masked], label[idx_masked])             
        conc_loss_collect.append(loss.data.cpu().detach().numpy())
    
        
        if len(multitask_config) > 0:                                   # multi-task loss
            loss_multitask = 0    
            for name, config in multitask_config.items(): 
                idx_notnan = torch.where( ~torch.isnan(multitask_label[name]) )
                loss_multitask_buffer = criterion_multitask[name](outputs[f"logit_{name}"].flatten()[idx_notnan], multitask_label[name][idx_notnan])
                loss_multitask_buffer = torch.mean(loss_multitask_buffer)      


                multitask_loss_collect[name].append(loss_multitask_buffer.data.cpu().detach().numpy()) 
                loss_multitask += config[3] * loss_multitask_buffer     # weight each additional task  
                        
            loss = alpha*loss + (1-alpha)*loss_multitask                # weight main task 

        step_loss_collect.append(loss.data.cpu().detach().numpy())
        
        
    


    val_loss = np.mean(np.array(step_loss_collect))
    val_loss_conc =  np.mean(np.array(conc_loss_collect))
    val_info = {'val_loss':val_loss, 'val_loss_conc':val_loss_conc}
           
    val_loss_multitask = {}
    if len(multitask_config) > 0: 
        for name, _ in multitask_config.items():
            val_loss_multitask[name] = np.mean(np.array(multitask_loss_collect[name])) 
            val_info[f"val_loss_{name}"] = val_loss_multitask[name]

    if wandb_monitor==True:
        wandb.log(val_info)
        
    return val_loss, val_loss_conc, val_loss_multitask


    

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
    max_mr = train_settings['max_mr']                         # the maximum of mask ratios
    f_loss = train_settings['f_loss']                         # 'MAE', 'MSE', 'SmoothL1Loss'
    data_path = train_settings['data_path']
    f_gamma = train_settings['f_gamma']                       # scheduling mask ratios for each mini-batch training
    wandb_monitor = train_settings['wandb_monitor']
    multi_task = train_settings['multi_task']                 # multi-task learning
    alpha = train_settings['alpha']                           # weight main/auxilary tasks


    # params of model architecture
    drop_out = train_settings['drop_out']
    attn_mode = train_settings['attn_mode'] 
    
    n_heads = train_settings['n_heads']
    n_blocks = train_settings['n_blocks']
    d_ff = train_settings['d_ff']
    d_model = train_settings['d_model']
    f_act = train_settings['f_act']


    # save setting
    save_note = train_settings['save_note']
    save_dir = os.path.join(train_settings['save_dir'], f'{save_note}')
    

    ####################################################################################
    # creata dir for saving models and results
    create_directory(save_dir)

    # randomseed and cuda
    set_seeds(3047)
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # if wandb_monitor==True:
        # wandb.login(key=) 
        # wandb.init(project="")

    # data loading 
    # train_loader, val_loader, train_data, _,  = load_dataset_from_dir_NMR(data_path,                                                                                   
    #                                                                     batch_size=batch_size,
    #                                                                     device=device)
    
    
    # fake data for demonstration
    train_loader, train_data = load_dataset_from_adata_NMR(os.path.join(data_path,'fake_val.h5ad'), 
                                batch_size=batch_size, 
                                device=device)   
    val_loader, val_data = load_dataset_from_adata_NMR(os.path.join(data_path,'fake_val.h5ad'), 
                                batch_size=batch_size, 
                                device=device) 
    
    
    num_train = len(train_loader.dataset) # pyright: ignore[reportArgumentType]
    metabo_id = train_data.var_names.values.tolist()

        
    # tokenization setting
    Tokenizer = MetFoundation_Tokenizer(VOCAB_Identifiers=metabo_id) 


    # embedding module setting
    EmbeddingModule_conf = {
                "n_vocabs": {'identifier': Tokenizer.vocab_size_identifiers}
                }
    
    # save tokenizer
    save_tokenizer(Tokenizer,save_dir=save_dir)      


    # model setting
    Model_conf = {            
                "n_heads": n_heads,
                "n_blocks": n_blocks,
                "d_ff": d_ff,
                "d_model": d_model,
                "dropout": drop_out,
                "activation":f_act,
                "need_weights":True,
                "average_attn_weights":True,
                "attn_mode": attn_mode
                }


    
    model = MetFoundation_ForPreTrain(EmbeddingModule_conf, Model_conf)
    
    # [num_classes, activation, loss, loss_weight, transform layer]
    if multi_task==True:
        multitask_config = {'age':[1, None,'mae', 0.4, False],
                            'sex':[1, 'sigmoid', 'bce', 0.2, False],
                            'bmi':[1,None, 'mse', 0.4, False]
                            }
        model.set_multitask_heads(multitask_config)
    else:
        multitask_config = {}   


  
    # check the model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters:\t{total_params}')

    Model_conf_all = Model_conf
    Model_conf_all['n_params'] = total_params
    
    if multi_task==True:
        Model_conf_all['multitask_config'] = multitask_config

    save_dict_2_json(Model_conf_all, filename='config.json', save_dir=save_dir)


    # GPU acceleration
    model.to(device)   


    # train
    train(model,   
        train_loader,
        num_train,
        val_loader, 
        lr,
        n_epoch,
        save_dir,
        max_mr,
        n_toler,
        f_loss,
        f_gamma,
        alpha=alpha,
        multitask_config=multitask_config
        )
    
    
    
    # if wandb_monitor==True:
    #     wandb.finish()
        
        
        

# python pretrain_UKBNMR.py --train_config ./pretrain_config_NMR/mlmtask.json


