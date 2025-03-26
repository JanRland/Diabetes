"""
This script contains the training strategies. This script was also utilized for the grid search.  

@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, matthews_corrcoef


from PreProc import removeNan, writeTrainTest
from model import CustomModel
from util import init_weights, train_loop, test_eval
from upsampling import GNUS, SMOTEUp, DownSample, UpSample, StratUpSample, IdsToData, StatisticNormalization


if __name__ == "__main__":
    


    # Set seeds to ensure reproducibility
    #
    torch.manual_seed(16) 
    np.random.seed(16)  

    device="cpu"
    normalize=False
    
    #
    # Tags are used to identify the saved results
    # tags[0]: boolean, crossVal
    # tags[1]: boolean, GaussAug
    # tags[2]: boolean, SmoteAug
    # tags[3]: boolean, StratifyData
    # tags[4]: boolean, DownSampleData
    # tags[5]: boolean, UpSampling
    # tags[6]: int, number of epochs
    # tags[7]: int, id
    #
    tags=[0,0,0,0,0,0,0,1111]
    
    #
    # Apply cross validation
    #
    crossVal=False
    if crossVal:
        tags[0]=1
    
    #
    # Save best result
    #
    saveResults=True
    
    #
    # Apply Gaussian noise upsampling
    #
    GaussAug=True
    if GaussAug:
        tags[1]=1
    
    #
    # Apply SMOTE upsampling
    #
    SmoteAug=False
    if SmoteAug:
        tags[2]=1
        
    #
    # Apply stratification for age and sex
    #
    StratifyData=False
    if StratifyData:
        tags[3]=1
        
    #
    # Apply downsampling to stratify
    #
    DownSampleData=False
    if DownSampleData:
        tags[4]=1
        
    #
    # Apply random upsampling
    #
    UpSampleData=True
    if UpSampleData:
        tags[5]=1
    
    #
    # Number of folds, epochs and reports
    #
    crossValN=10
    nEpochs=2000
    epochFeedback=25
    tags[6]=nEpochs

    #
    # Strength of Gaussian upsampling
    #
    GaussAugMeanStrength=0.001
    GaussAugStdStrength=0.0001
    
    #
    # Remove NaNs from dataset
    #
    data=removeNan("data.csv")
    
    #
    # Split final evaluation test split, 10% is testing
    #
    N_original=data.shape[0]
    #test, train=writeTrainTest(data, N_original)
    
    test=np.load("test.npy")
    train=np.load("train.npy")[:,:9]
    
    
    xTest, yTest=test[:,2:], test[:,1]
    
    #np.save("test.npy", test)
    #np.save("train.npy", train)
    
    
    testData=torch.tensor(test[:,2:]).double()
    testLabel=torch.tensor(test[:,1]).double().unsqueeze(1)
    trainXnp=train[:,2:] # features
    trainYnp=train[:,1] # label
    print("Data Shape: ")
    print(train.shape)
    
    #
    # Set Model, apply Xavier initialization
    #     
    model=CustomModel()
    model.double()
    model.apply(init_weights)
    
    #
    # Feature Importance
    #
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    featureName=["age", "sex", "height", "weight", "taille", "beer", "smoke", "hba1n", "chol", "hbn"]
    
    #
    # Set training Parameters
    #
    loss_function = nn.BCELoss()#nn.BCELoss()#nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min", factor=0.5, patience=100,verbose=True)
    #
    # Training for cross validation
    #
    if crossVal:
        #
        # Define split
        #
        skf = StratifiedKFold(n_splits=crossValN, random_state=None, shuffle=False)

        #
        # Save results in lists
        #
        interpo = np.linspace(0, 1, 100)
        iter_aucs = []
        fprs = []
        tprs = []
        train_metrices=[]
        val_metrices=[]
        
        for fold, (train_index, test_index) in enumerate(skf.split(trainXnp, trainYnp)):
            
            #
            # Fold data
            #
            print("FOLD: " + str(fold))
            xNpy, yNpy = trainXnp[train_index], trainYnp[train_index] 
            data=train[train_index]
            fold_metric=[]
            auc_metric=[]
            loss_metric=[]
            
            fold_metric_val=[]
            auc_metric_val=[]
            loss_metric_val=[]
            
            #
            # Normalization of the data
            #
            if normalize:
                xNpy=StatisticNormalization(data[:,2:], 1)

            
            #
            # Apply Gaussian upsampling, SMOTE upsampling, stratification, or 
            # stratified downsampling
            #
            if StratifyData:
                class_1, class_2 = StratUpSample(data, 2, 3, 8)
                
                if GaussAug: 
                    newData_1=GNUS(data, class_1, 1)
                    newData_2=GNUS(data, class_2, 1)
                    
                    newData=np.vstack((newData_1, newData_2))
                    np.random.shuffle(newData)
                    xNpy=newData[:,2:]
                    yNpy=newData[:,1]
                    
                else:
                    classes=class_1.tolist()+class_2.tolist()
                    newData=IdsToData(data, classes)
                    np.random.shuffle(newData)
                    xNpy=newData[:,2:]
                    yNpy=newData[:,1]
            
            
            elif SmoteAug:
                xNpy, yNpy=SMOTEUp(xNpy, yNpy)
            
            elif DownSampleData:
                class_1, class_2 = DownSample(data)
                
                if GaussAug: 
                    newData_1=GNUS(data, class_1, 1)
                    newData_2=GNUS(data, class_2, 1)
                    
                    newData=np.vstack((newData_1, newData_2))
                    np.random.shuffle(newData)
                    xNpy=newData[:,2:]
                    yNpy=newData[:,1]
                else:
                    classes=class_1.tolist()+class_2.tolist()
                    newData=IdsToData(data, classes)
                    np.random.shuffle(newData)
                    xNpy=newData[:,2:]
                    yNpy=newData[:,1]
                
            elif UpSampleData:
                N=int(max(np.sum(yNpy), yNpy.shape[0]-np.sum(yNpy)))
                class_1, class_2 = UpSample(data, N)
                
                if GaussAug: 
                    newData_1=GNUS(data, class_1, 1)
                    newData_2=GNUS(data, class_2, 1)
                    
                    newData=np.vstack((newData_1, newData_2))
                    np.random.shuffle(newData)
                    xNpy=newData[:,2:]
                    yNpy=newData[:,1]
                else:
                    classes=class_1.tolist()+class_2.tolist()
                    newData=IdsToData(data, classes)
                    np.random.shuffle(newData)
                    xNpy=newData[:,2:]
                    yNpy=newData[:,1]
            print("New Data Shape: ")
            #print(newData.shape)
                
            
            #
            # Define the training and validation set of the split
            #
            x=torch.tensor(xNpy).double()
            y=torch.tensor(yNpy).double()
            xVal, yVal = torch.tensor(trainXnp[test_index]).double(), torch.tensor(trainYnp[test_index]).double()
            x = x.to(device)
            y = y.to(device)
            y = y.unsqueeze(1)
            xVal = xVal.to(device)
            yVal = yVal.to(device)
            yVal = yVal.unsqueeze(1)
            
            for f in range(xNpy.shape[1]):
                if f==1 or f==6:
                    pass
                else:
                    model.norm.m.data[0,f]=torch.tensor(np.mean(xNpy[:,f]))
                    model.norm.v.data[0,f]=torch.tensor(np.var(xNpy[:,f]))
            
            saveDir="fold_" + str(fold) + "_" + "_".join((str(tags[0]), str(tags[1]),str(tags[2]),str(tags[3]),str(tags[4]),str(tags[5]),str(int(tags[6])), str(tags[7]))) + ".pth"
            loss_train, auc_train, loss_val, auc_val, fpr, tpr, modelDict = train_loop(model, nEpochs, epochFeedback, loss_function, optimizer, scheduler, x, y, xVal, yVal, saveDir, device)
            
            
            fold_metric.append(auc_train)
            fold_metric.append(loss_train)
            train_metrices.append(fold_metric)
            
            interpo_tpr = np.interp(interpo, fpr, tpr)
            interpo_fpr = np.interp(interpo, tpr, fpr)
            tprs.append(interpo_tpr)
            fprs.append(interpo_fpr)
            
            fold_metric_val.append(auc_val)
            fold_metric_val.append(loss_val)
            val_metrices.append(fold_metric_val)
    
        if saveResults:
            metrices=[train_metrices, tprs, fprs, val_metrices]
            saveDir="res_" + "_".join((str(tags[0]), str(tags[1]),str(tags[2]),str(tags[3]),str(tags[4]),str(tags[5]),str(int(tags[6])), str(tags[7]))) + ".pkl"
            print(saveDir)
            with open((saveDir), 'wb') as f:
                pkl.dump(metrices, f)
            torch.save(modelDict, "network_" + "_".join((str(tags[0]), str(tags[1]),str(tags[2]),str(tags[3]),str(tags[4]),str(tags[5]),str(int(tags[6])), str(tags[7]))) + ".pth")
    
    
    else:
        #
        # Save results in lists
        #
        interpo = np.linspace(0, 1, 100)
        iter_aucs = []
        fprs = []
        tprs = []
        train_metrices=[]
        val_metrices=[]
        
        #np.save("train.npy",train)
        val, train=writeTrainTest(train, N_original)
        xNpy, yNpy=train[:,2:], train[:,1]
        xVal, yVal = torch.tensor(val[:,2:]).double(), torch.tensor(val[:,1]).double()
        data=train

        #
        # Normalization of the data
        #
        if normalize:
            data[:,2:]=StatisticNormalization(data[:,2:], 1)
            xNpy=data[:,2:]
        
        #
        # Apply Gaussian upsampling, SMOTE upsampling, stratification, or 
        # stratified downsampling
        #
        if StratifyData:
            class_1, class_2 = StratUpSample(data, 2, 3, 8)
            
            if GaussAug: 
                newData_1=GNUS(data, class_1, 1)
                newData_2=GNUS(data, class_2, 1)
                
                newData=np.vstack((newData_1, newData_2))
                np.random.shuffle(newData)
                xNpy=newData[:,2:]
                yNpy=newData[:,1]
                
            else:
                classes=class_1.tolist()+class_2.tolist()
                newData=IdsToData(data, classes)
        
        
        elif SmoteAug:
            xNpy, yNpy=SMOTEUp(xNpy, yNpy)
        
        elif DownSampleData:
            class_1, class_2 = DownSample(data)
            
            if GaussAug: 
                newData_1=GNUS(data, class_1, 1)
                newData_2=GNUS(data, class_2, 1)
                
                newData=np.vstack((newData_1, newData_2))
                np.random.shuffle(newData)
                xNpy=newData[:,2:]
                yNpy=newData[:,1]
            else:
                classes=class_1.tolist()+class_2.tolist()
                newData=IdsToData(data, classes)
            
        elif UpSampleData:
            N=int(max(np.sum(yNpy), yNpy.shape[0]-np.sum(yNpy)))
            print(np.sum(yNpy))
            print(yNpy.shape[0]-np.sum(yNpy))
            class_1, class_2 = UpSample(data, int(2.*N))
            print(class_1.shape)
            print(class_2.shape)
            
            if GaussAug: 
                newData_1=GNUS(data, class_1, 1)
                newData_2=GNUS(data, class_2, 1)
                print(newData_1.shape)
                print(newData_2.shape)
                
                
                newData=np.vstack((newData_1, newData_2))
                
                print(newData.shape)
                print(np.sum(newData[:,1]==1))
                print(np.sum(newData[:,1]==0))
                np.random.shuffle(newData)
                xNpy=newData[:,2:]
                yNpy=newData[:,1]
            else:
                classes=class_1.tolist()+class_2.tolist()
                newData=IdsToData(data, classes)
                
        if normalize:
            xNpy=StatisticNormalization(xNpy, 1)
        #
        # Define the training and validation set of the split
        #
        train2=np.column_stack((yNpy, xNpy))
        #np.save("train_4params.npy",train2)
        x=torch.tensor(xNpy).double()
        y=torch.tensor(yNpy).double()
        x = x.to(device)
        y = y.to(device)
        y = y.unsqueeze(1)
        xVal = xVal.to(device)
        yVal = yVal.to(device)
        yVal = yVal.unsqueeze(1)
        
        rf.fit(xNpy, yNpy)
        importances = rf.feature_importances_
        p=[(x, round(y,2)) for y, x in reversed(sorted(zip(importances, featureName)))]
        for elm in p:
            print(elm[0] + " & " + str(elm[1]) + ' \\\\')
        
        
        for f in range(xNpy.shape[1]):
            if f==1 or f==6:
                pass
            else:
                model.norm.m.data[0,f]=torch.tensor(np.mean(xNpy[:,f]))
                model.norm.v.data[0,f]=torch.tensor(np.var(xNpy[:,f]))
        saveDir="result_training" + "_" + "_".join((str(tags[0]), str(tags[1]),str(tags[2]),str(tags[3]),str(tags[4]),str(tags[5]),str(int(tags[6])), str(tags[7]))) + ".pth"
        loss_train, auc_train, loss_val, auc_val, fpr, tpr, modelDict = train_loop(model, nEpochs, epochFeedback, loss_function, optimizer, scheduler, x, y, xVal, yVal, saveDir, device)
        
        
        train_metrices.append(auc_train)
        train_metrices.append(loss_train)
        
        interpo_tpr = np.interp(interpo, fpr, tpr)
        interpo_fpr = np.interp(interpo, tpr, fpr)
        tprs.append(interpo_tpr)
        fprs.append(interpo_fpr)
        
        val_metrices.append(auc_val)
        val_metrices.append(loss_val)
        
        print(auc_val.index(np.amax(auc_val)))
    
        if saveResults:
            metrices=[train_metrices, tprs, fprs, val_metrices]
            saveDir="results_train_" + "_".join((str(tags[0]), str(tags[1]),str(tags[2]),str(tags[3]),str(tags[4]),str(tags[5]),str(int(tags[6])), str(tags[7]))) + ".pkl"
            with open((saveDir), 'wb') as f:
                pkl.dump(metrices, f)
            #torch.save(modelDict, "network_" + "_".join((str(tags[0]), str(tags[1]),str(tags[2]),str(tags[3]),str(tags[4]),str(tags[5]),str(int(tags[6])), str(tags[7]))) + ".pth")
            torch.save(modelDict, "network_7params.pth")
                
        #
        # Final Evaluation
        #
        model.load_state_dict(modelDict)
        model.eval()
        print(testData.shape)
        test_loss, test_fpr, test_tpr, test_auc = test_eval(model, loss_function, testData, testLabel)
        print("Test AUC:")
        print(test_auc)
        if saveResults:
            metrices=[test_loss, test_tpr, test_fpr, test_auc]
            saveDir="results_test_" + "_".join((str(tags[0]), str(tags[1]),str(tags[2]),str(tags[3]),str(tags[4]),str(tags[5]),str(int(tags[6])), str(tags[7]))) + ".pkl"
            with open((saveDir), 'wb') as f:
                pkl.dump(metrices, f)
