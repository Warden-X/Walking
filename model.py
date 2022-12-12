# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 08:29:20 2021

@author: comeo
"""


import torch
from torch import nn
import time
from pathlib import Path
from dataset import dataset_bulid
from pytorchtools import EarlyStopping



## Network Architecture
cfg = {'Model2': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M']}

class Model(nn.Module):
    def __init__(self, model_name, init_weights=True):
        super().__init__()
        self.Features = self.GetLayer(model_name)
        self.Classifier = nn.Sequential(
            nn.Linear(in_features=32*8*12, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2),
        )
        if init_weights:
            self._initialize_weights()


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
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)



def train(net, num_epochs, valid_set, device):   
    Train_Loss = []
    Train_Accuracy = []
    Valid_accuracy  =[]
    Valid_loss = []
    
    for epoch in range(num_epochs):
        start = time.time()
        train_loss = 0
        correct = 0
        total = 0
        net.train()
        
        for i, (im, label) in enumerate(train_set):

            im = im.to(device)
            label = label.to(device)

            output = net(im)
            loss = criterion(output, label)
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()

            train_loss += loss.item()  
            total += label.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()

        end = time.time()
        T_loss_one = train_loss * batch_size / total
        T_acc_one = correct / total * 100
        current_Ir = scheduler.get_last_lr()[0]
        
        if valid_set is not None:
            V_acc_one, V_loss_one = validation(valid_set, net, device)
            epoch_str = ('[S%d N%d E%d]  T_loss: %.4f  T_acc: %.2f %%  '
                'V_loss: %.4f  V_acc: %.2f %%  Ir: %.5f  time %.1f' \
                %(sensor_num, num_v, epoch + 1, T_loss_one, T_acc_one, 
                  V_loss_one, V_acc_one, current_Ir, end-start))
            
        else:
            epoch_str = ('[S%d N%d E%d]  T_loss: %.4f  T_acc: %.2f %%  '
                         'Ir: %.5f  time %.1f' %(sensor_num, num_v, epoch + 1, 
                        T_loss_one, T_acc_one, current_Ir, end-start))
            
        print(epoch_str)
        
        Train_Loss.append(T_loss_one)
        Train_Accuracy.append(T_acc_one)
        Valid_accuracy.append(V_acc_one)
        Valid_loss.append(V_loss_one)  
                
        scheduler.step()  
        
        torch.save(Train_Loss, save_path+'Train_Loss.pt')
        torch.save(Train_Accuracy, save_path+'Train_Accuracy.pt')
        torch.save(Valid_accuracy, save_path+'Valid_accuracy.pt')
        torch.save(Valid_loss, save_path+'Valid_loss.pt')
            
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(V_loss_one, net)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    net.load_state_dict(torch.load(save_path+'checkpoint.pt'))
        
    return net, Train_Loss, Train_Accuracy, Valid_accuracy, Valid_loss 


def validation(valid_data, net, device):
    valid_loss = 0
    valid_acc = 0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for im, label in valid_data:
            
            im = im.to(device)
            label = label.to(device)
                    
            total += label.size(0)
            output = net(im)
            loss = criterion(output, label)
            valid_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            valid_acc += (predicted == label).sum().item()
    
    valid_acc = 100 * valid_acc / total
    valid_loss = valid_loss * batch_size / total
    
    return valid_acc, valid_loss 


def test(test_set, net, device):
    correct = 0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for data in test_set:
            
            im, label = data
            im = im.to(device)
            label = label.to(device)
            outputs = net(im)
            
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    accuracy = 100 * correct / total
    print('Accuracy: ', accuracy)

    return accuracy



#%% main
if __name__ == '__main__':
    
    results = [] # Save the classification results of all datasets
    for i in range(5): # Individual sensor dataset
        sensor_num = i + 1 
        accuracy_all = [] 
        
        for j in range(10): # 10-fold cross validation
            
            num_v = j + 1 
            root = 'E:/Desktop/数据集/行走数据/dataset/dataset_64_100/'+str(sensor_num) 
            
            save_path = 'E:/Desktop/GitHub/NN/results/results_64_100_BN_2/'+ \
                                str(sensor_num)+'/'+ str(num_v)+'/'
            p = Path(save_path)
            p.mkdir(parents=True, exist_ok=True)  
            
            net = Model('Model2', init_weights=True)
            # print(net)
            batch_size = 8
            
            # SGD
            # optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0005) 
            # Adam
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), 
                                        eps=1e-08, weight_decay=0.001)
             
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: 1 / (epoch+1))
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=5e-5)
            
            criterion = nn.CrossEntropyLoss() 
            
            # Early_Stopping
            patience = 30	
            early_stopping = EarlyStopping(patience=patience, verbose=True,
                                           path=save_path+'checkpoint.pt')  
        
            train_set, valid_set, test_set = dataset_bulid(root, batch_size, num_v)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net.to(device)
            net, T_Loss, T_Acc, V_acc, V_loss = train(net, 100, valid_set, device)
            accuracy = test(test_set, net, device)
            
            accuracy_all.append(accuracy) 
        
        torch.save(accuracy_all, save_path[0:-2]+'accuracy_all.pt')
        results.append(accuracy_all)
    torch.save(results, save_path[0:-4]+'results.pt')
    
        
      







