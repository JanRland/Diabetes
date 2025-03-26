"""
This script contains the deep neural network for diabetes prediction. 

@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""

import torch 
import torch.nn as nn
from AdaptiveNormalization import AdaptiveNorm

class DeepNetBlock(nn.Module):
    def __init__(self, inputF, outputF):
        super(DeepNetBlock, self).__init__()
        self.l1=nn.Linear(outputF, outputF)
        self.norm1=AdaptiveNorm((1,outputF))
        
        self.l2=nn.Linear(outputF, outputF)
        self.norm2=AdaptiveNorm((1,outputF))
        self.scale=None
        if inputF!=outputF:
            self.scale=nn.Linear(inputF, outputF)
        
        self.l3=nn.Linear(outputF, outputF)
        self.relu=nn.ReLU()
        
    def forward(self, x):
        if self.scale!=None:
            x=self.scale(x)
        xRes=x
        x=self.l1(x)
        x=self.relu(x)
        x=self.norm1(x)
        

        x=self.l2(x)
        x=self.relu(x)
        x=self.norm2(x)

        x=self.l3(x)
        
        x=x+xRes
        return x
        
        
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        nFeatures=7
        
        self.norm=AdaptiveNorm((1,nFeatures))
        self.l1 = nn.Linear(nFeatures, nFeatures)
        self.l2 =  nn.Linear(nFeatures, nFeatures*2)
        self.l3 =  nn.Linear(nFeatures*2, nFeatures*2)
        self.l4 =  nn.Linear(nFeatures*2, nFeatures*2)
        self.l5 =  nn.Linear(nFeatures*2, nFeatures)
        self.l6 =  nn.Linear(nFeatures, nFeatures)
        self.activation=nn.Tanh()
        
        self.b1 = DeepNetBlock(nFeatures, nFeatures)
        self.b2 = DeepNetBlock(nFeatures, nFeatures*2)
        self.b3 = DeepNetBlock(nFeatures*2, nFeatures*2)
        self.b5 = DeepNetBlock(nFeatures*2, nFeatures)

        
        self.fc = nn.Linear(nFeatures, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        

    def forward(self, x):
        
        x=self.norm(x)
        x=self.l1(x)
        x=self.relu(x)
        
        x=self.b1(x)
        x=self.b2(x)
        x=self.b3(x)
        x=self.b5(x)

        x=self.relu(x)
        x=self.fc(x)
        x=self.sigmoid(x)
        
        return x 
