# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 08:30:38 2022

@author: comeo
"""

#%% ACC

import torch
import numpy as np
from matplotlib import pyplot as plt



config = {
    "font.family":'Times New Roman',  
    "axes.linewidth":'1.2',
    "savefig.dpi": 300  
}
plt.rcParams.update(config)  

plt.figure(figsize=(4,4))

results = torch.load('E:/Desktop/GitHub/NN/results/results_64_100_BN_2/5results.pt')
results = np.array(results)/100
mean_value = np.mean(results,axis=1)
std_value = np.std(results, axis=1)

width = 0.5  
capsize = 10 

x = np.arange(len(mean_value))
plt.bar(x, mean_value, yerr = std_value, capsize=capsize, width=width, 
         edgecolor='#ff6600', color='white', lw=3, hatch='ooo')


tick_label = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5']
# y_ticks = np.linspace(0, 1, 11)

plt.xticks(x, tick_label, fontsize=11) 
plt.yticks(fontsize=11)  
plt.ylim(0.85, 1.008)
plt.title('Accuracy Comparison of Models', fontsize=12)
    
plt.xlabel('Sensor Number', fontsize=12)  
plt.ylabel('Accruacy', fontsize=12)  
plt.subplots_adjust(left=0.15, right=0.95)  

save_dir = 'E:/Desktop/小论文/行走CNN/图/材料/ACC'+'.png'
plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)



#%% AUC

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import torch
from torch import nn
import matplotlib.pyplot as plt

from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from dataset import dataset_bulid



config = {
    "font.family":'Times New Roman', 
    "axes.linewidth":'1.2',
    "savefig.dpi": 300  
}
plt.rcParams.update(config)  


## Network Architecture
cfg = {'Model2': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M']}

class Model(nn.Module):
    def __init__(self, model_name):
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


def prob_func(net, test_set, device):
    """Output predicted values and labels of testset sample"""
    
    labels = []
    predict_prob = []
    
    net.eval()
    with torch.no_grad():
        for data in test_set:
            
            im, label = data
            im = im.to(device)
            label = label.to(device)
            outputs = net(im)
            prob = outputs[:,1].numpy()
            
            predict_prob.extend(prob)
            labels.extend(label.numpy())
            
    return np.array(predict_prob), np.array(labels)


def prob_cv(sensor_num):
    """ 10-fold cross validation"""
    
    batch_size = 1
    root = 'E:/Desktop/数据集/行走数据/dataset/dataset_64_100/'+str(sensor_num) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predict_prob_all = []
    labels_all = []
    
    for j in range(10): 
        
        num_v = j + 1 
        save_path = 'E:/Desktop/GitHub/NN/results/results_64_100_BN_2/'+ \
                            str(sensor_num)+'/'+ str(num_v)+'/'
     
        net = Model('Model2')
        net.load_state_dict(torch.load(save_path+'checkpoint.pt'), strict=False) 

        train_set, valid_set, test_set = dataset_bulid(root, batch_size, num_v)
        
        predict_prob, labels = prob_func(net, test_set, device) 
        
        predict_prob_all.append(predict_prob)
        labels_all.append(labels)

    return np.array(predict_prob_all), np.array(labels_all)



def plot_roc(predict_prob_all, labels_all):
    """ROC"""
    
    tpr_all = []
    roc_auc_all = []
    mean_fpr = np.linspace(0, 1, 1001)  
    
    for i in range(len(predict_prob_all)):
        
        fpr, tpr, thresholds  = roc_curve(labels_all[i], predict_prob_all[i])
        roc_auc=auc(fpr, tpr)  # AUC
        
        tpr_all.append(np.interp(mean_fpr, fpr, tpr))
        tpr_all[-1][0] = 0.0  
        roc_auc_all.append(roc_auc)
        
        plt.plot(fpr, tpr, lw=1, alpha=0.6,
             label='ROC fold %d (AUC = %0.4f)' % (i+1, roc_auc))


    mean_tpr = np.mean(tpr_all, axis=0)
    mean_tpr[-1] = 1.0 
    mean_roc_auc = np.mean(roc_auc_all)
    std_auc = np.std(roc_auc_all)

    
    plt.plot(mean_fpr, mean_tpr,'b',
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_roc_auc, std_auc))
    
    std_tpr = np.std(tpr_all, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.6,
                     label=r'$\pm$ Standard Deviation')
    
    plt.plot([0,1],[0,1],'r--', label='Random')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xticks(fontsize=11) 
    plt.yticks(fontsize=11)  
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    plt.title('ROC Curve of Sensor %d' % sensor_num, fontsize=12)
    plt.legend(loc='lower right',fontsize=9)
    plt.show()

    return roc_auc_all


def anova(data):
    df_data = pd.DataFrame(np.array(data).T, columns=['s1', 's2', 's3', 's4', 's5'])
    df_data_melt = df_data.melt()
    df_data_melt.columns = ['Sensor','AUC']
    
    model = ols('AUC~C(Sensor)',data=df_data_melt).fit()
    anova_table = anova_lm(model, typ = 2)
    print(anova_table)





def tukey_hsd(data):
    df_data = pd.DataFrame(np.array(data).T, columns=['s1', 's2', 's3', 's4', 's5'])
    df_data_melt = df_data.melt()
    df_data_melt.columns = ['Sensor','AUC']

    mc = MultiComparison(df_data_melt['AUC'], df_data_melt['Sensor'])
    tukey_result = mc.tukeyhsd(alpha = 0.05)
    print(tukey_result)
    


for i in range(5):
    sensor_num = i+1
    predict_prob_all, labels_all = prob_cv(sensor_num)
    plt.figure(figsize=(4,4))
    roc_auc_sensor = plot_roc(predict_prob_all, labels_all)
    save_dir = 'E:/Desktop/小论文/行走CNN/图/材料/Sensor'+str(sensor_num)+'.png'
    plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()


roc_auc_sensors = []
for i in range(5):
    sensor_num = i+1
    predict_prob_all, labels_all = prob_cv(sensor_num)
    plt.figure(figsize=(4,4))
    roc_auc_sensor = plot_roc(predict_prob_all, labels_all)
    plt.close()
    
    roc_auc_sensors.append(roc_auc_sensor)

# one-way analysis of variance
tukey_hsd(roc_auc_sensors)




#%% Raw Data & CWT


import matplotlib.pyplot as plt
import numpy as np
import pywt
import pickle
import seaborn as sns
from pathlib import Path




def windows_pisition(dataset, sensor_num, overlap):
    sensor_num = sensor_num - 1
    data_sensor = dataset[sensor_num]
    a = np.array([(data_sensor[i].shape[0]-100)//overlap + 1 for i in range(len(data_sensor))])
    w_position = np.cumsum(a)

    return w_position


def plot_raw_data(data):
    """Draw the curve of acceleration and angular velocity"""

    config = {
        "font.family":'Times New Roman', 
        "savefig.dpi": 300  
    }
    plt.rcParams.update(config)  
    
    # acceleration
    x = [i*0.02 for i in range(len(data))]    # The x-coordinate is time(s)
    
    plt.figure(figsize=(4.6,3.5))
    plt.plot(x, data[:,0])
    plt.plot(x, data[:,1])
    plt.plot(x, data[:,2])
    
    # plt.legend(['$\mathregular{a_x}$', '$\mathregular{a_y}$', 
    #             '$\mathregular{a_z}$'], fontsize=18, ncol=3, loc='lower center')  # 添加图例
    
    plt.xlabel('Time(s)', fontsize=22)  
    plt.ylabel('Acceleration(g)', fontsize=22)  
    plt.xlim(-0.05, 5.1)  
    plt.ylim(-0.8, 2.0)
    plt.xticks(fontsize=18)  
    plt.yticks(fontsize=18)  
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.22) 
    plt.show()
    
    #  angular velocity
    plt.figure(figsize=(4.6,3.5))
    plt.plot(x, data[:,3])
    plt.plot(x, data[:,4])
    plt.plot(x, data[:,5])
    
    # plt.legend(['$\mathregular{ω_x}$', '$\mathregular{ω_y}$', 
    #             '$\mathregular{ω_z}$'], fontsize=18, ncol=3, loc='lower center')  
    plt.xlabel('Time(s)', fontsize=22)  
    plt.ylabel('Angular Velocity(rad/s)', fontsize=22)  
    plt.xlim(-0.05, 5.1)  
    plt.ylim(-1.5, 1.5)
    plt.xticks(fontsize=18)  
    plt.yticks(fontsize=18)  
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.22) 
    plt.show()


def cwt_solving(data):
    """CWT"""
        
    sampling_rate = 50

    wavename = 'morl'
    totalscal = 64
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 0, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1/sampling_rate, axis=0)
    cwtmatr = abs(cwtmatr)  
         
    cwtmatr = cwtmatr.transpose(2, 0, 1).astype('float32')  
    
    data_1 = cwtmatr[0:3,:,:]
    data_1[np.where(data_1>1)] = 1  
    cwtmatr[0:3,:,:] = (data_1 - 0.5)/0.5 
    
    data_2 = cwtmatr[3:6,:,:]
    data_2[np.where(data_2>1.5)] = 1.5  
    cwtmatr[3:6,:,:] = (data_2/1.5 - 0.5)/0.5 
    
    return cwtmatr
    


with open(r'E:\Desktop\GitHub\NN\Walking\saved\walking_pd.pkl', 'rb') as file_obj:
    walking_pd = pickle.load(file_obj)

with open(r'E:\Desktop\GitHub\NN\Walking\saved\walking_nor.pkl', 'rb') as file_obj:
    walking_hc = pickle.load(file_obj)


pd_position = windows_pisition(walking_pd, 3, 50)
hc_position = windows_pisition(walking_hc, 3, 30)

# [305 622 893 1596]
data_pd = walking_pd[2][125][70::,:]
data_hc = walking_hc[2][120][20::,:]

"""Raw Data"""
plot_raw_data(data_pd)
plot_raw_data(data_hc)

"""CWT"""
cwtmatr_pd = cwt_solving(data_pd)
cwtmatr_hc = cwt_solving(data_hc)



for i in range(6):
    plt.figure(figsize=(4, 2.5))
    sns.heatmap(cwtmatr_pd[i], vmin=-1, vmax=1, cmap='Spectral_r', cbar=False)
    
    # colorbar 
    # ax = sns.heatmap(cwtmatr_pd[i], vmin=-1, vmax=1, cmap='Spectral_r', cbar=True)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=20)

    plt.xticks([0,50,100,150,200,250],[0,1,2,3,4,5], fontsize=20, rotation=0)  
    plt.yticks([0,13,25,38,51,64], [25,20,15,10,5,0],fontsize=20, rotation=0)
    
    plt.xlabel('Time(s)', fontsize=22)  
    plt.ylabel('Frequency(Hz)', fontsize=22) 
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2)  
    
    save_dir = 'E:/Desktop/小论文/行走CNN/图/材料/Raw Data/PD_C'+str(i)+'.png'
    plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()


#%% Optimized Model AUC


import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import torch
from torch import nn

from dataset_optimized import dataset_bulid




## Network Architecture
cfg = {'Model2': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M']}

class Model(nn.Module):
    def __init__(self, model_name):
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
        in_channel = 3

        for x in Layer_List:
            if x == 'M':
                Layer.append(nn.MaxPool2d(2, 2))
            else:
                
                # Batch_Normalization
                Layer += [nn.Conv2d(in_channel, x, 3, padding=1), 
                          nn.BatchNorm2d(x), nn.ReLU()]
                
                in_channel = x
        return nn.Sequential(*Layer)


def prob_func(net, test_set, device):
    """Output predicted values and labels of testset sample"""
    
    labels = []
    predict_prob = []
    
    net.eval()
    with torch.no_grad():
        for data in test_set:
            
            im, label = data
            im = im.to(device)
            label = label.to(device)
            outputs = net(im)
            prob = outputs[:,1].numpy()
            
            predict_prob.extend(prob)
            labels.extend(label.numpy())
            
    return np.array(predict_prob), np.array(labels)


def prob_cv(sensor_num):

    batch_size = 1
    root = 'E:/Desktop/数据集/行走数据/dataset/dataset_64_100/'+str(sensor_num) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predict_prob_all = []
    labels_all = []
    
    for j in range(10): 
        
        num_v = j + 1 
        save_path = 'E:/Desktop/GitHub/NN/results/results_optimized/'+ \
                            str(sensor_num)+'/'+ str(num_v)+'/'
     
        net = Model('Model2')
        net.load_state_dict(torch.load(save_path+'checkpoint.pt'), strict=False) 

        train_set, valid_set, test_set = dataset_bulid(root, batch_size, num_v)
        
        predict_prob, labels = prob_func(net, test_set, device) 
        
        predict_prob_all.append(predict_prob)
        labels_all.append(labels)

    return np.array(predict_prob_all), np.array(labels_all)


def roc_auc(predict_prob_all, labels_all):
    roc_auc_all = []
    for i in range(len(predict_prob_all)):
        fpr, tpr, thresholds  = roc_curve(labels_all[i], predict_prob_all[i])
        roc_auc=auc(fpr, tpr)  # AUC
        roc_auc_all.append(roc_auc)

    return roc_auc_all





sensor_num = 3
predict_prob_all, labels_all = prob_cv(sensor_num)

roc_auc_all = roc_auc(predict_prob_all, labels_all)

mean_roc_auc = np.mean(roc_auc_all)
std_auc = np.std(roc_auc_all)

















