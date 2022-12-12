# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 08:20:13 2022

@author: comeo
"""

import numpy as np
import pickle as pkl
import torch
import cv2
import matplotlib.pyplot as plt

from Net import Model, dataset_load



class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al. 
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''
    def __init__(self, model, target_layers, use_cuda=False):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers
        
        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)
        
        self.activations = []
        self.grads = []
        
    def forward_hook(self, module, input, output):
        self.activations.append(output[0])
        
    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())
        
    def calculate_cam(self, model_input, label):
        if self.use_cuda:
            device = torch.device('cuda')
            self.model.to(device)                 # Module.to() is in-place method 
            model_input = model_input.to(device)  # Tensor.to() is not a in-place method
        self.model.eval()
        
        # forward
        y_hat = self.model(model_input)
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)

        # backward
        model.zero_grad()
        y_c = y_hat[0, max_class]
        y_c.backward()
        
        # get activations and gradients
        activations = self.activations[0].cpu().data.numpy().squeeze()
        grads = self.grads[0].cpu().data.numpy().squeeze()
        
        # calculate weights
        weights = np.mean(grads.reshape(grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1, 1)
        cam = (weights * activations).sum(axis=0)
        cam = np.maximum(cam, 0) # ReLU
        cam = cam / cam.max()
        

        if max_class != label:
            c_en = 1
        else:
            c_en = 0
            
        return cam, c_en
    
    @staticmethod
    def show_cam_on_image(data, cam):
        # image: [H,W,C]
        h, w = data.shape[1:3]
        
        cam = cv2.resize(cam, (w,h))
        cam = cam / cam.max()
        heatmap = cv2.applyColorMap((255*cam).astype(np.uint8), cv2.COLORMAP_JET) # [H,W,C]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        heatmap = heatmap / heatmap.max()
        
        plt.figure(figsize=(4,2.5))
        plt.imshow((heatmap*255).astype(np.uint8))
        # plt.colorbar(shrink=0.8)
        
        plt.xticks([0,20,40,60,80,100],[0,0.4,0.8,1.2,1.6,2.0],fontsize=13)  
        plt.yticks([0,13,25,38,51,64], [25,20,15,10,5,0],fontsize=13)  
            
        plt.xlabel('Time(s)', fontsize=14)  
        plt.ylabel('Frequency(Hz)', fontsize=14)  

        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2)  
        plt.show()
        
        


#%% main

if __name__ == '__main__':
    
    num_v_optimal = [7, 3, 7, 2, 10]  
    error_idxes_all = []  
    
    for i in range(5):
        # 0、parameter setting
        sensor_num = i+1  
        root = 'E:/Desktop/数据集/行走数据/dataset/dataset_64_100/'+str(sensor_num) 
        data_all, labels = dataset_load(root)
        num_v = num_v_optimal[i] 
        save_path = 'E:/Desktop/GitHub/NN/results/results_64_100_BN_2/'+ \
                            str(sensor_num)+'/'+ str(num_v)+'/'
    
        # 1、loading model
        model = Model('Model2')
        model.load_state_dict(torch.load(save_path+'checkpoint.pt'), strict=False) 
        model.eval()
    
        # 2、Tensor
        # [305 622 893 1596]
        
        # idxes =  [94, 305, 428, 622, 836, 893, 982, 1578, 1596, 1679]
        
        # idxes = list(range(857))  # HC
        idxes = list(range(857, 1832))  # PD
        
        error_idxes = []  
        
        for idx in idxes:
            data = data_all[idx]
            input_tensor = torch.tensor(np.array([data]))
            label = labels[idx]
            
            # 3、CAM
            grad_cam = GradCAM(model, model.Features[-1], False)
            cam, c_en = grad_cam.calculate_cam(input_tensor, label)
            GradCAM.show_cam_on_image(data, cam)
            
            # 4、save
            save_dir = 'E:/Desktop/小论文/行走CNN/图/材料/Grad-CAM/PD/Sp'+ \
                str(idx)+'_S'+str(sensor_num)+'.png'
            plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close()
            
            if c_en == 1:
                error_idxes.append(idx)

        error_idxes_all.append(error_idxes)

    with open('E:/Desktop/小论文/行走CNN/图/材料/Grad-CAM/Error/error_idxes_PD.pkl', 'wb') as file_obj:
        pkl.dump(error_idxes_all, file_obj)  




#%% Misclassified visual map

import pickle as pkl

with open('E:/Desktop/小论文/行走CNN/图/材料/Grad-CAM/Error/error_idxes_PD.pkl', 'rb') as file_obj:
    error_idxes_PD = pkl.load(file_obj) 

with open('E:/Desktop/小论文/行走CNN/图/材料/Grad-CAM/Error/error_idxes_HC.pkl', 'rb') as file_obj:
    error_idxes_HC = pkl.load(file_obj)   

    
num_v_optimal = [7, 3, 7, 2, 10]  
error_idxes_all = [] 

for i in range(5):
    # 0、parameter setting
    sensor_num = i+1  
    root = 'E:/Desktop/数据集/行走数据/dataset/dataset_64_100/'+str(sensor_num) 
    
    data_all, labels = dataset_load(root)
    
    num_v = num_v_optimal[i]  
    # num_v = 5
    
    save_path = 'E:/Desktop/GitHub/NN/results/results_64_100_BN_2/'+ \
                        str(sensor_num)+'/'+ str(num_v)+'/'

    # 1. loading model
    model = Model('Model2')
    model.load_state_dict(torch.load(save_path+'checkpoint.pt'), strict=False) 
    model.eval()

    # 2. Tensor

    idxes = error_idxes_HC[i] # HC
    # idxes = error_idxes_PD[i]  # PD
    
    error_idxes = []  
    
    for idx in idxes:
        data = data_all[idx]
        input_tensor = torch.tensor(np.array([data]))
        label = labels[idx]
        
        # 3. CAM
        grad_cam = GradCAM(model, model.Features[-1], False)
        cam, c_en = grad_cam.calculate_cam(input_tensor, label)
        GradCAM.show_cam_on_image(data, cam)
        
        # 4. save
        save_dir = 'E:/Desktop/小论文/行走CNN/图/材料/Grad-CAM/Error/HC_PD/Sp'+ \
            str(idx)+'_S'+str(sensor_num)+'.png'
        plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        
        if c_en == 1:
            error_idxes.append(idx)

    error_idxes_all.append(error_idxes)




