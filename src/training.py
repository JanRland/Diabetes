#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:53:47 2023

@author: jan
"""

import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np
import pickle as pkl
from DataAugment import Augmentation
import sys
from Diabetes import Diabetes


#
# Note: upsampling only during training crossvalidation
#


torch.manual_seed(10)
np.random.seed(42)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
device ="cpu"

#
# Set the flags
#
crossVal=True
crossValN=10
nEpochs=20000
epochFeedback=1000
saveResults=True
result_ID=6
GaussAug=False
GaussAugMeanStrength=0.01
GaussAugStdStrength=0.01
SmoteAug=True
augFactor=2
saveFold=6

#
# Set the NN
#
Net=Diabetes()
Net.double()

#
# Set training Data
#
data=np.load('data.npy')

#
# Balance if necassary
#
a=Augmentation(data)
#a.downSampleData()
#data=a.data
#a.shuffleData()

if not crossVal:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    
    for i, (train_index, test_index) in enumerate(sss.split(a.data[:,1:], a.data[:,0])):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        print(a.data[test_index][:,0].shape)
        print(np.where(a.data[test_index][:,0]>0)[0].shape[0])



    trainXnp=a.data[train_index][:,1:]
    trainYnp=a.data[train_index][:,0]
    trainX=torch.tensor(a.data[test_index][:,1:]).double()
    trainY=torch.tensor(a.data[test_index][:,0]).double()
else:
    trainXnp=data[:,1:]
    trainYnp=data[:,0]
    trainX=torch.tensor(data[:,1:]).double()
    trainY=torch.tensor(data[:,0]).double()
#
# Set loss function 
#
loss_function = nn.MSELoss()
#loss_function = nn.CrossEntropyLoss()

#
# Set optimizer
#
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.SGD(Net.parameters(), lr=0.02)

#
# Initialise the weights
#
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
        
        
Net.apply(init_weights)

if __name__ == "__main__":
    # cross validation flag
    if crossVal:
        skf = StratifiedKFold(n_splits=crossValN, random_state=None, shuffle=False)
        
        interpo = np.linspace(0, 1, 100)
        iter_aucs = []
        fprs = []
        tprs = []
        train_metrices=[]
        val_metrices=[]
        
        if saveResults:
            torch.save(Net.state_dict(), "NNinit_"+str(result_ID)+'.pth')
        for fold, (train_index, test_index) in enumerate(skf.split(trainXnp, trainYnp)):
            xNpy, yNpy = trainXnp[train_index], trainYnp[train_index] 
            #
            # Data Augmentation
            #
            
            #
            # Gauß noise
            #
            if GaussAug:
                
                crossData=np.column_stack((yNpy,xNpy))
                a.setData(crossData)
                af, bf=a.getDistribution()
                at=round(af*a.nSamples)
                bt=round(bf*a.nSamples)
                print(at, bt)
                extraSample=round(augFactor*bt-bt) #1.5 times minority class
                print(extraSample)
                print(a.getDistribution())
                augmentedData=a.GausNoise(extraSample, strengthMean=GaussAugMeanStrength, strengthStd=GaussAugStdStrength)
                
                validRows=np.where(a.data[:,0]>0)[0]
                nRows=validRows.shape[0]
                print(validRows, a.data[validRows[0],:])
                unNoisedData=np.zeros((extraSample, 5))
                for i in range(extraSample):
                    randomN=np.random.randint(0, extraSample)
                    unNoisedData[i, :]=a.data[randomN, :]
                
                augmentedSamples=randomN+augmentedData[extraSample:,:]
                augmentedSamples[:,0]=1
                
                a.mergeData(augmentedSamples)
                a.downSampleData()
                xNpy=a.data[:, 1:]
                yNpy=a.data[:,0]
                print(a.getDistribution())
                
                
            #
            # SMOTE
            #
            if SmoteAug:
                crossData=np.column_stack((yNpy,xNpy))
                a.setData(crossData)
                af, bf=a.getDistribution()
                at=round(af*a.nSamples)
                bt=round(bf*a.nSamples)
                print(at, bt)
                extraSample=round(augFactor*bt-bt) #1.5 times minority class
                print(extraSample)
                print(a.getDistribution())
                
                imbClass0=a.data[np.where(a.data[:,0]==0)][0:(extraSample+bt),:]
                imbClass1=a.data[np.where(a.data[:,0]==1)]
                
                imbData=np.vstack((imbClass0,imbClass1))
                a.setData(imbData)
                
                x,y=a.smoteUp()
                res=np.zeros((x.shape[0], x.shape[1]+1))
                res[:,0]=y
                res[:,1:]=x
                
                xNpy=res[:, 1:]
                yNpy=res[:,0]
                
                a.setData(np.column_stack((yNpy, xNpy)))
            
            #
            # Continue training
            #
            x=torch.tensor(xNpy).double()
            y=torch.tensor(yNpy).double()
            
            xVal, yVal = trainX[test_index], trainY[test_index] 
            x = x.to(device)
            y = y.to(device)
            y = y.unsqueeze(1)
            xVal = xVal.to(device)
            yVal = yVal.to(device)
            yVal = yVal.unsqueeze(1)
            
            
            fold_metric=[]
            f1_metric=[]
            auc_metric=[]
            loss_metric=[]
            
            fold_metric_val=[]
            f1_metric_val=[]
            auc_metric_val=[]
            loss_metric_val=[]
            
            for epoch in range(nEpochs):
                optimizer.zero_grad()
                
                y_pred = Net(x)             # Perform a forward pass on the network with inputs
                loss = loss_function(y_pred, y) # calculate the loss with the network predictions and ground Truth
                loss.backward()             # Perform a backward pass to calculate the gradients
                optimizer.step()            # Optimise the network parameters with calculated gradients
            
                # calculate validation out_put_values, prediction_values, target values
                Ynp = (y.detach().numpy()[:,0]).astype(float)
                y_prednp = (y_pred.detach().numpy()[:,0]).astype(float)
                fpr, tpr, th = roc_curve(Ynp, y_prednp)
        
                roc_auc = auc(fpr, tpr)
                #f1 = f1_score(Ynp, y_prednp)
                
                #f1_metric.append(f1)
                auc_metric.append(roc_auc)
                loss_metric.append(loss.item())
                # Print statistics to console
                if epoch%epochFeedback==0:
                    print("Epoch %d, Iteration %5d] AUC: %.3f LOSS: %.3f" % (epoch+1, fold+1, roc_auc, loss.item()))
                
                # validation
                yVal_pred = Net(xVal)
                loss = loss_function(yVal_pred, yVal)
                
                YValnp = yVal.detach().numpy()[:,0]
                yVal_prednp = yVal_pred.detach().numpy()[:,0]
                
                fpr, tpr, th = roc_curve(YValnp, yVal_prednp)
                roc_auc = auc(fpr, tpr)
                #f1 = f1_score(YValnp, yVal_prednp)
                
                #f1_metric_val.append(f1)
                auc_metric_val.append(roc_auc)
                loss_metric_val.append(loss.item())
                
                if epoch%epochFeedback==0:        
                    print("Val AUC: %.3f LOSS: %.3f" % (roc_auc, loss.item()))
                

            
            #fold_metric.append(f1_metric)
            fold_metric.append(auc_metric)
            fold_metric.append(loss_metric)
            
            train_metrices.append(fold_metric)
            
            interpo_tpr = np.interp(interpo, fpr, tpr)
            interpo_fpr = np.interp(interpo, tpr, fpr)
        
            tprs.append(interpo_tpr)
            fprs.append(interpo_fpr)
            
            #fold_metric_val.append(f1_metric)
            fold_metric_val.append(auc_metric_val)
            fold_metric_val.append(loss_metric_val)
            
            val_metrices.append(fold_metric_val)
            
            if saveResults:
                if fold==saveFold:
                    torch.save(Net.state_dict(), "NNres_"+str(result_ID)+'.pth')
            

        if saveResults:
            metrices=[train_metrices, tprs, fprs, val_metrices]
            #torch.save(Net.state_dict(), "NNres_"+str(result_ID)+'.pth')
            with open(('metrices_' + str(result_ID) + '.pkl'), 'wb') as f:
                pkl.dump(metrices, f)
            
    else:
            
        
        interpo = np.linspace(0, 1, 100)
        iter_aucs = []
        fprs = []
        tprs = []
        train_metrices=[]
        val_metrices=[]
        
        if saveResults:
            torch.save(Net.state_dict(), "NNinit_"+str(result_ID)+'.pth')

        xNpy, yNpy = trainXnp, trainYnp
        #
        # Data Augmentation
        #
        
        #
        # Gauß noise
        #
        if GaussAug:
            
            crossData=np.column_stack((yNpy,xNpy))
            a.setData(crossData)
            af, bf=a.getDistribution()
            at=round(af*a.nSamples)
            bt=round(bf*a.nSamples)
            print(at, bt)
            extraSample=round(augFactor*bt-bt) #1.5 times minority class
            print(extraSample)
            print(a.getDistribution())
            augmentedData=a.GausNoise(extraSample, strengthMean=GaussAugMeanStrength, strengthStd=GaussAugStdStrength)
            
            validRows=np.where(a.data[:,0]>0)[0]
            nRows=validRows.shape[0]
            print(validRows, a.data[validRows[0],:])
            unNoisedData=np.zeros((extraSample, 5))
            for i in range(extraSample):
                randomN=np.random.randint(0, extraSample)
                unNoisedData[i, :]=a.data[randomN, :]
            
            augmentedSamples=randomN+augmentedData[extraSample:,:]
            augmentedSamples[:,0]=1
            
            a.mergeData(augmentedSamples)
            a.downSampleData()
            xNpy=a.data[:, 1:]
            yNpy=a.data[:,0]
            print(a.getDistribution())
            
            
        #
        # SMOTE
        #
        if SmoteAug:
            crossData=np.column_stack((yNpy,xNpy))
            a.setData(crossData)
            af, bf=a.getDistribution()
            at=round(af*a.nSamples)
            bt=round(bf*a.nSamples)
            print(at, bt)
            extraSample=round(augFactor*bt-bt) #1.5 times minority class
            print(extraSample)
            print(a.getDistribution())
            
            imbClass0=a.data[np.where(a.data[:,0]==0)][0:(extraSample+bt),:]
            imbClass1=a.data[np.where(a.data[:,0]==1)]
            
            imbData=np.vstack((imbClass0,imbClass1))
            a.setData(imbData)
            
            x,y=a.smoteUp()
            res=np.zeros((x.shape[0], x.shape[1]+1))
            res[:,0]=y
            res[:,1:]=x
            
            xNpy=res[:, 1:]
            yNpy=res[:,0]
            
            a.setData(np.column_stack((yNpy, xNpy)))
        
        #
        # Continue training
        #
        x=torch.tensor(xNpy).double()
        y=torch.tensor(yNpy).double()
        
        xVal, yVal = trainX, trainY
        x = x.to(device)
        y = y.to(device)
        y = y.unsqueeze(1)
        xVal = xVal.to(device)
        yVal = yVal.to(device)
        yVal = yVal.unsqueeze(1)
        
        
        fold_metric=[]
        f1_metric=[]
        auc_metric=[]
        loss_metric=[]
        
        fold_metric_val=[]
        f1_metric_val=[]
        auc_metric_val=[]
        loss_metric_val=[]
        
        for epoch in range(nEpochs):
            optimizer.zero_grad()
            
            y_pred = Net(x)             # Perform a forward pass on the network with inputs
            loss = loss_function(y_pred, y) # calculate the loss with the network predictions and ground Truth
            loss.backward()             # Perform a backward pass to calculate the gradients
            optimizer.step()            # Optimise the network parameters with calculated gradients
        
            # calculate validation out_put_values, prediction_values, target values
            Ynp = (y.detach().numpy()[:,0]).astype(float)
            y_prednp = (y_pred.detach().numpy()[:,0]).astype(float)
            fpr, tpr, th = roc_curve(Ynp, y_prednp)
    
            roc_auc = auc(fpr, tpr)
            #f1 = f1_score(Ynp, y_prednp)
            
            #f1_metric.append(f1)
            auc_metric.append(roc_auc)
            loss_metric.append(loss.item())
            # Print statistics to console
            if epoch%epochFeedback==0:
                print("Epoch %d] AUC: %.3f LOSS: %.3f" % (epoch+1, roc_auc, loss.item()))
            
            # validation
            yVal_pred = Net(xVal)
            loss = loss_function(yVal_pred, yVal)
            
            YValnp = yVal.detach().numpy()[:,0]
            yVal_prednp = yVal_pred.detach().numpy()[:,0]
            
            fpr, tpr, th = roc_curve(YValnp, yVal_prednp)
            roc_auc = auc(fpr, tpr)
            #f1 = f1_score(YValnp, yVal_prednp)
            
            #f1_metric_val.append(f1)
            auc_metric_val.append(roc_auc)
            loss_metric_val.append(loss.item())
            
            if epoch%epochFeedback==0:        
                print("Val AUC: %.3f LOSS: %.3f" % (roc_auc, loss.item()))
            

        
        #fold_metric.append(f1_metric)
        fold_metric.append(auc_metric)
        fold_metric.append(loss_metric)
        
        train_metrices.append(fold_metric)
        
        interpo_tpr = np.interp(interpo, fpr, tpr)
        interpo_fpr = np.interp(interpo, tpr, fpr)
    
        tprs.append(interpo_tpr)
        fprs.append(interpo_fpr)
        
        #fold_metric_val.append(f1_metric)
        fold_metric_val.append(auc_metric_val)
        fold_metric_val.append(loss_metric_val)
        
        val_metrices.append(fold_metric_val)
        

        if saveResults:
            metrices=[train_metrices, tprs, fprs, val_metrices]
            torch.save(Net.state_dict(), "NNres_"+str(result_ID)+'.pth')
            with open(('metrices_' + str(result_ID) + '.pkl'), 'wb') as f:
                pkl.dump(metrices, f)
                  
            
            
            
