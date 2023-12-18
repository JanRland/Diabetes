#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:45:18 2023

@author: jan
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import StratifiedKFold


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
device ="cpu"


class Diabetes(nn.Module):
    def __init__(self, activation=nn.Tanh(), output=nn.Sigmoid(),
                 input_size=4, hidden_size=1, classes=1):
        
        super(Diabetes, self).__init__()
        self.input_size = input_size

        # Here we initialize our activation and set up our two linear layers
        self.activation = activation
        self.h1 = nn.Tanh()
        self.h2 = nn.Tanh()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, classes)
        self.fc3 = nn.Linear(hidden_size, classes)
        self.fc4 = nn.Linear(hidden_size, classes)
        self.output = output

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.h1(x)
        #x = self.fc3(x)
        #x = self.h2(x)
        #x = self.fc4(x)
        x = self.output(x)

        return x