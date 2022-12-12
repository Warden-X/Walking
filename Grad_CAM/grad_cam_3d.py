# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:46:31 2022

@author: comeo
"""


import numpy as np
import torch
import cv2
import random
import matplotlib.pyplot as plt

from Net_3d import Model, dataset_load



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
        y_c = y_hat[0, label]
        y_c.backward()
        
        # get activations and gradients
        activations = self.activations[0].cpu().data.numpy().squeeze()
        grads = self.grads[0].cpu().data.numpy().squeeze()
        
        # calculate weights
        weights = np.mean(grads.reshape(grads.shape[0],grads.shape[1], -1), axis=2)
        weights = weights.reshape(weights.shape[0], weights.shape[1],1, 1)
        cam = (weights * activations).sum(axis=0)
        cam = np.maximum(cam, 0) # ReLU
        cam = cam / cam.max()
        

        if max_class != label:
            cam = 0*cam
            
        return cam
    
    @staticmethod
    def show_cam_on_image(data, cam):
        # image: [H,W,C]
        d, h, w = data.shape[1:4]
        
        for i in range(d):
            cam_one = cv2.resize(cam[i], (w,h))
            cam_one = cam_one / cam_one.max()
            heatmap = cv2.applyColorMap((255*cam_one).astype(np.uint8), cv2.COLORMAP_JET) # [H,W,C]
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
            
            save_dir = 'E:/Desktop/小论文/行走CNN/图/材料/Grad-CAM-3D/Sp'+ \
                str(idx)+'_S'+str(sensor_num)+'_C'+str(i+1)+'.png'
            plt.savefig(save_dir, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close()
        


#%% main

if __name__ == '__main__':
    
    
    # 0、parameter setting
    sensor_num = 3 
    root = 'E:/Desktop/数据集/行走数据/dataset/dataset_64_100/'+str(sensor_num) 
    
    data_all, labels = dataset_load(root)
    
    num_v = 5  
    
    save_path = 'E:/Desktop/GitHub/NN/results/results_3d/'+ \
                        str(sensor_num)+'/'+ str(num_v)+'/'

    # 1、loading model
    model = Model('Model2')
    model.load_state_dict(torch.load(save_path+'checkpoint.pt'), strict=False) 
    model.eval()

    # 2、Tensor
    # [215, 456, 1573, 1736]
    
    HC_idx = [random.randint(0,857) for i in range(30)]
    PD_idx = [random.randint(857,1832) for i in range(30)]   
    
    idxes = []
    idxes.extend(HC_idx)
    idxes.extend(PD_idx)
    
    for idx in idxes:
        data = data_all[idx]
        input_tensor = torch.tensor(np.array([data]))
        label = labels[idx]
        
        # 3、CAM
        grad_cam = GradCAM(model, model.Features[-1], False)
        cam = grad_cam.calculate_cam(input_tensor, label)
        
        # 4、save
        GradCAM.show_cam_on_image(data, cam)
        
        
        




















