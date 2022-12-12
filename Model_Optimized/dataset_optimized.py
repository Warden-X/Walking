# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:08:04 2021

@author: comeo
"""

import torch
import pickle
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler



class Num_Dataset(Dataset):
    def __init__(self, root, transform=None): 
        p = Path(root) 
        data_path = []
        for file in p.rglob('*.pkl'):
            data_path.append(str(file))  
        self.data_path = data_path   
        self.transform = transform
    	
    def __len__(self):
        return len(self.data_path)
    	
    def __getitem__(self, idx):
        data_path = self.data_path[idx] 
        # PD 1   Normal 0
        label = 1 if '\\1-' in data_path else 0 
        with open(data_path, 'rb') as file_obj:
            data = pickle.load(file_obj)
        data = data[[0,2,5],:,:]
        if self.transform:
            data = self.transform(data)

        return data, label 


def dataset_bulid(root, batch_size, num_v):
    
    dataset = Num_Dataset(root)

    pd_indices = list(range(975))
    nor_indices = list(range(975, 1832))
    random.seed(1)
    random.shuffle(pd_indices)
    random.seed(2)
    random.shuffle(nor_indices)
    
    test_indices = pd_indices[0:195] + nor_indices[0:171]
    
    valid_indices = pd_indices[(195+78*(num_v-1)):(195+78*num_v)] \
                    + nor_indices[(171+68*(num_v-1)):(171+68*num_v)]   
                    
    train_indices = pd_indices[195:(195+78*(num_v-1))] + pd_indices[(195+78*num_v)::] \
                    + nor_indices[171:(171+68*(num_v-1))] + nor_indices[(171+68*num_v)::]

    train_sampler = SubsetRandomSampler(train_indices, generator=torch.Generator().manual_seed(21474836))
    valid_sampler = SubsetRandomSampler(valid_indices, generator=torch.Generator().manual_seed(21474836))
    test_sampler = SubsetRandomSampler(test_indices, generator=torch.Generator().manual_seed(21474836))
    
    train_set = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_set = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_set = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    
    #                   PD       Normal       Rate
    #   Train_set       702        618        53.2%
    #   Valid_set       78          68        53.2%
    #   Test_set        195        171        53.2%

    return train_set, valid_set, test_set

















