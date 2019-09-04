#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import helper
from threading import Thread
import time
import os
import subprocess
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import helper
from torch import nn


# In[2]:



transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
                                      transforms.ToTensor()])
#train_data = datasets.ImageFolder(data_dir,transform=train_transforms)
data = datasets.ImageFolder("samples",transform=transforms)
len_data = len(data)
valid_size=0.2
indices = list(range(len_data))
np.random.shuffle(indices)
split = int(np.floor(valid_size * len_data))
valid_idx =indices[:split]

# define samplers for obtaining training and validation batches
valid_sampler = SubsetRandomSampler(valid_idx)

#trainLoader =torch.utils.data.DataLoader(train_data,batch_size=64, sampler=train_sampler)
validLoader =torch.utils.data.DataLoader(data,batch_size=64, sampler=valid_sampler)
#train_gpu = torch.cuda.is_available()


# In[3]:



class MyNetwork(nn.Module):
        
        def __init__(self):
            super(MyNetwork, self).__init__()
            #image is in 100*100*1
            self.conv1 = nn.Conv2d(1 ,10,3,padding=1)
            #image is in 50*50*10
            self.conv2 = nn.Conv2d(10,50,3,padding=1)
            #image is in 25*25*50
            self.conv3 = nn.Conv2d(50,100,3,padding=1)
            #image is in 12*12*100
            self.conv4 = nn.Conv2d(100,150,3,padding=1)
            #image is in 6*6*150
            self.conv5 = nn.Conv2d(150,200,3,padding=1)
            #image is in 3*3*200
            self.pool=nn.MaxPool2d(2,2)
            self.dropout=nn.Dropout(p=0.25)
            #fully connected layer 
            self.fc1=nn.Linear(1800,900)
            self.fc2=nn.Linear(900,500)
            self.fc3=nn.Linear(500,24)
            self.dropout=nn.Dropout(p=0.05) 
        def forward(self,x):
            x=self.pool(F.relu(self.conv1(x)))
            x=self.pool(F.relu(self.conv2(x)))
            x=self.pool(F.relu(self.conv3(x)))
            x=self.pool(F.relu(self.conv4(x)))
            x=self.pool(F.relu(self.conv5(x)))
            x=x.view(-1,3*3*200)
            x=self.dropout(F.relu(self.fc1(x)))
            x=self.dropout(F.relu(self.fc2(x)))
            x=F.log_softmax(self.fc3(x),dim=1)
            return x
        def load_checkpoint(self,name_file="checkpoint.pt"):
            checkpoint=torch.load(name_file) 
            self.load_state_dict(checkpoint)


# In[4]:


import helper
model = MyNetwork()
model.load_checkpoint("checkpoint_last.pt")
#checkpoint = torch.load("checkpoint_last.pt")
#model.load_state_dict(checkpoint)

images, labels=next(iter(validLoader))
img = images[1,:]
print(img.shape)
ps = torch.exp(model(images[1,:].unsqueeze(0)))
# Plot the image and probabilities
#helper.imshow(img)
t = ['A','B','C','D','E','F','G','H','I','k','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
#t =list(reversed(t))
_,top=ps.topk(1,dim=1)
print(" expected = ",t[labels[1].item()]," result=",t[top.item()] )
helper.view_classify(img,ps, version='ALPHABET',title=t[labels[1].item()])


# In[ ]:




