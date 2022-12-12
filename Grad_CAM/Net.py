# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:31:12 2022

@author: comeo
"""

import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
import pickle



config = {
    "font.family":'Times New Roman',  
    "axes.linewidth":'1.2',
    "savefig.dpi": 300  
}
plt.rcParams.update(config) 



cfg = {'Model2': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M']}

class Model(nn.Module):
    def __init__(self, model_name='Model2'):
        super().__init__()
        self.Features = self.GetLayer(model_name)
        self.Classifier = nn.Sequential(
            nn.Linear(in_features=32*8*12, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2),
        )


    def forward(self, x):
        x = self.Features(x)
        x = x.view(x.shape[0], -1)
        x = self.Classifier(x)
        return x
    
    
    def GetLayer(self, model_name):
        Layer_List = cfg[model_name]
        Layer = []
        in_channel = 6
        for x in Layer_List:
            if x == 'M':
                Layer.append(nn.MaxPool2d(2, 2))
            else:
                # Batch_Normalization
                Layer += [nn.Conv2d(in_channel, x, 3, padding=1), 
                          nn.BatchNorm2d(x), nn.ReLU()]
                in_channel = x
                
        return nn.Sequential(*Layer)


def dataset_load(root):
    p = Path(root) 
    data_path_all = []
    for file in p.rglob('*.pkl'):
        data_path_all.append(str(file))  

    labels = []
    data_all = []
	
    for idx in range(len(data_path_all)):
        data_path = data_path_all[idx]
        # PD 1   Normal 0
        label = 1 if '\\1-' in data_path else 0 
        with open(data_path, 'rb') as file_obj:
            data = pickle.load(file_obj)
            
        labels.append(label)
        data_all.append(data)

    return data_all, labels
















