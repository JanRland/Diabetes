"""
This script contains the data preprocessing of the  Heinz Nixdorf RECALL Study dataset.

@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""

import pandas as pd
import numpy as np



def removeNan(dataFile):
    data=pd.read_csv(dataFile)
    
    
    age=np.array(data["alter"])
    sex=np.array(data["sex"])
    height=np.array(data["groesse"])
    weight=np.array(data["gewicht"])
    taille=np.array(data["taille"])
    diab=np.array(data["diab3"])
    beer=np.array(data["FE_BIER"])
    dias=np.array(data["DIA_OM"])
    sys=np.array(data["SYS_OM"])
    ery=np.array(data["ERYn"])
    chol=np.array(data["CHOLn"])
    bilin=np.array(data["BILIn"])
    hba1n=np.array(data["HBA1n"])
    trig=np.array(data["TRIGn"])
    hbn=np.array(data["HBn"])
    meal=np.array(data["ba_mahl"])
    smoke=np.array(data["xraustat"])
    ids=np.array(data["probidneu"])
    
    diab=np.where(diab>0, 1, 0)
    sex=sex-1
    beer =np.where(beer>0, beer, 0)
    smoke=np.where(smoke>0, 1, 0)
    
    #newData=np.column_stack((ids, diab, age, sex, height, weight, taille, beer, smoke))
    #newData=np.column_stack((ids, diab, age, sex, height, weight, taille, beer, dias, sys, ery, chol, bilin, hba1n, trig, hbn, meal, smoke))
    newData=np.column_stack((ids, diab, age, sex, height, weight, taille, beer, smoke))
    newData = newData[~np.ma.fix_invalid(newData).mask.any(axis=1)]
    
    return newData

def getRandom(idList,N):
    testSet=np.random.choice(idList, N, replace=False)
    return testSet

def writeTrainTest(data, N_original):
    N=int(0.1*N_original)
    print(N)
    ids=data[:,0]
    testIds=getRandom(ids, N).tolist()

    testData=[]
    trainData=[]
    for v in data:
        if v[0] in testIds:
            testData.append(v.tolist())
        else:
            trainData.append(v.tolist())
    
    testData=np.array(testData)
    trainData=np.array(trainData)
    return testData, trainData
