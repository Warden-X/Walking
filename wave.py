# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:52:16 2021

@author: comeo
"""


#%% dataset 64*100  overlap: 50%

# 1832 = PD + Normal = 975 + 857       975/1832 = 53.2%

import matplotlib.pyplot as plt
import numpy as np
import pywt
import pickle
import seaborn as sns
from pathlib import Path


with open(r'E:\Desktop\GitHub\NN\Walking\saved\walking_pd.pkl', 'rb') as file_obj:
    walking_pd = pickle.load(file_obj)

with open(r'E:\Desktop\GitHub\NN\Walking\saved\walking_nor.pkl', 'rb') as file_obj:
    walking_nor = pickle.load(file_obj)

path = 'E:/Desktop/数据集/行走数据/dataset/dataset_64_100/'  

dataset_all = [walking_pd, walking_nor]
marks = ['/1-', '/0-']   # mark = '/0-'   PD 1  Normal 0  
overlap = [50, 50] # step 

# c = []

for i, dataset in enumerate(dataset_all):
    mark = marks[i]
    for sensor_num in range(len(dataset)):  
    
        p = Path(path+str(sensor_num+1)+'/')
        p.mkdir(parents=True, exist_ok=True)  
        
        for sample_num in range(len(dataset[sensor_num])):  
            data = dataset[sensor_num][sample_num]
            sampling_rate = 50
            time = 0.02*len(data)-0.001
            t = np.arange(0,time,0.02)
            
            wavename = 'morl'
            totalscal = 64
            fc = pywt.central_frequency(wavename)
            cparam = 2 * fc * totalscal
            scales = cparam / np.arange(totalscal, 0, -1)
            [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1/sampling_rate, axis=0)
            cwtmatr = abs(cwtmatr)  

            cwtmatr = cwtmatr.transpose(2, 0, 1).astype('float32')  
            idex_len= np.size(cwtmatr,2)
            
            # specified length  64*100
            num = overlap[i]
            for j in range((idex_len-100)//num+1): 
                cwtmatr_ = cwtmatr[:,:,num*j:(num*j+100)]
                save_path = str(p)+mark+str(sample_num+1)+'-'+str(j+1)+'.pkl'
                with open(save_path, 'wb') as file_obj:
                    pickle.dump(cwtmatr_, file_obj)   








