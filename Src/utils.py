import os
import torch
import numpy as np
import json
import pickle

def set_seeds(seed_val=3047): 
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)




def save_txt_single_column(data, save_dir='./', filename='save_test.txt'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    f = open(os.path.join(save_dir, filename),'w')
    for i in data:
        f.write(str(i)+'\n')
    f.close()
    
    
def read_txt_single_column(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file {file_path} dont exists")
        
    collect = []
    f = open(file_path,'r')
    for line in f:
        collect.append( line.strip())
    f.close()
    return collect 


def saving_result(label_collect, pred_collect, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f1 = open(os.path.join(save_dir, str(epoch)+'_y.txt'),'w')
    f2 = open(os.path.join(save_dir, str(epoch)+'_y_hat.txt'),'w')
    for i in range(len(label_collect)):
        f1.write(str(label_collect[i])+'\n')
        f2.write(str(pred_collect[i])+'\n')
    f1.close()
    f2.close()



def keep_nonNaN_values(arr, n, random_seed=42):
    np.random.seed(random_seed) 
    
    # Create a mask with the same shape as arr, initialized to False
    mask = np.full(arr.shape, False)

    # Iterate over each row in the array
    for i, row in enumerate(arr):
        # Get the indices of non-NaN values
        non_nan_indices = np.where(~np.isnan(row))[0]

        # If there are more non-NaN values than n, keep n random ones
        if len(non_nan_indices) > n:
            # Randomly select n indices to keep
            indices_to_keep = np.random.choice(non_nan_indices, n, replace=False)

            # Determine indices to replace with NaN
            indices_to_replace = np.setdiff1d(non_nan_indices, indices_to_keep)

            # Replace the values at the selected indices with NaN
            arr[i, indices_to_replace] = np.nan

            # Update the mask for the replaced positions
            mask[i, indices_to_replace] = True

    return arr, mask



def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_dict_2_json(data, filename, save_dir):
    with open(os.path.join(save_dir, filename), 'w') as json_file:
        json.dump(data, json_file, indent=4)


def save_tokenizer(tokenizer, filename='tokenizer.pkl', save_dir='./'):
    with open(os.path.join(save_dir, filename), 'wb') as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(path):
    with open(path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer