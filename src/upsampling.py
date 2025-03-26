"""
This script contains the upsampling/augmentation strategies for the training. 

@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""

import numpy as np
import imblearn

def getDist(x, label, ids):
    """
    

    Parameters
    ----------
    x : List
        List containing all ids in training set. The dimension is N.
    label : Numpy Array
        Array containing all binary labels. The dimension is N and each 
        element is in set {0,1}.
    ids : Numpy Array
        Array containing all ids in entire data set. The dimension is N.

    Returns
    -------
    N_m : Int
        Number of elements with label 1.
    N_f : Int
        Number of elements with label 0.

    """
    labelList=[]
    for i in x:
        labelList.append(int(label[np.where(ids==i)][0]))
    N=len(x)
    N_m=np.sum(labelList)
    N_f= N-N_m
    return N_m, N_f

def stratify(x, sex, ids):
    xM, xF =getDist(x, sex, ids)
    newX=x.copy()
    
    if xM>xF:
        fList=[]
        for i in x:
            if sex[np.where(ids==i)]==0:
                fList.append(i)
        newX+=np.random.choice(fList, xM-xF).tolist()
    elif xF>xM:
        mList=[]
        for i in x:
            if sex[np.where(ids==i)]==1:
                mList.append(i)
        newX+=np.random.choice(mList, xF-xM).tolist()
    else:
        pass
    return newX

def upsample(x):
    newX=x.copy()
    newX+=np.random.choice(x, 1).tolist()
    return newX


def downsample(x, sex, ids):
    xM, xF =getDist(x, sex, ids)
    res=0
    
    if xF>xM:
        fList=[]
        for i in x:
            #print(sex[np.where(ids==i)], np.where(ids==i), i)
            if sex[np.where(ids==i)]==0:
                fList.append(i)
        res=np.random.choice(fList, 1).tolist()
    elif xM>xF:
        mList=[]
        for i in x:
            #print(sex[np.where(ids==i)], np.where(ids==i), i)
            if sex[np.where(ids==i)]==1:
                mList.append(i)
        res=np.random.choice(mList,1).tolist()
    else:
        res=np.random.choice(x,1).tolist()
    return res


def getStratifiedSubset(N, x, sex, ids):
    k=len(x)
    res=x.copy()
    if k<N:
        startUps=False
        while k<N:
            if startUps:
                res=upsample(res)
                res=stratify(res, sex, ids)
            else:
                res=stratify(res, sex, ids)
                startUps=True
            k=len(res)
    elif k>N:
        while k>N:
            res.remove(downsample(res, sex, ids)[0])
            k=len(res)
            #print(k)
    res=stratify(res, sex, ids)
    return res

def GaussianDraw(features, ignore=None):
    
    #
    # Calculate statistical moments
    #
    moment_1=[]
    moment_2=[]
    n_features=features.shape[1]
    for i in range(n_features):
        moment_1.append(np.mean(features[:,i]))
        moment_2.append(np.std(features[:,i]))
    if i==3:
        moment_1=np.array(moment_1)*np.array([0.00001, 0, 0.00001, 0.00001])#, 0.00001, 0.00001, 0.00001])
        moment_2=np.array(moment_2)*np.array([0.001, 0.001, 0.001, 0.001])#, 0.001, 0.001, 0.001])
        drawGauss=np.random.normal(moment_1, moment_2)
        if ignore is None:
            pass
        else:
            drawGauss[ignore]=0
            drawGauss[-1]=0
    elif i==6:
        moment_1=np.array(moment_1)*np.array([0.0001, 0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
        moment_2=np.array(moment_2)*np.array([0.001, 0.05, 0.001, 0.001, 0.001, 0.001, 0.001])
        drawGauss=np.random.normal(moment_1, moment_2)
        if ignore is None:
            pass
        else:
            drawGauss[ignore]=0
            drawGauss[-1]=0
        
    return drawGauss
    
def SMOTEUp(data, label):
    oversample = imblearn.over_sampling.SMOTE()
    x,y = oversample.fit_resample(data, label)
    return x,y
    
def DownSample(data):
    
    #
    # Get ids, labels, age and sex
    #
    ids=data[:,0]
    labels=data[:,1]
    
    #
    # Find majority class 
    #
    target_a=ids[np.where(labels==1)]
    target_b=ids[np.where(labels==0)]
    target_a2=target_a.copy().tolist()
    target_b2=target_b.copy().tolist()
    a,b=getDist(ids, labels, ids)           
    while a>b:
        np.random.shuffle(target_a2)
        target_a2.pop()
        a=len(target_a2)
        
    while b>a:
        np.random.shuffle(target_b2)
        target_b2.pop()
        b=len(target_b2)
    
    target_a2=np.array(target_a2)
    target_b2=np.array(target_b2)
    
    return target_a2, target_b2

def UpSample(data, N):
    
    #
    # Get ids, labels, age and sex
    #
    ids=data[:,0]
    labels=data[:,1]
    
    #
    # Find majority class 
    #
    target_a=ids[np.where(labels==1)]
    target_b=ids[np.where(labels==0)]
    target_a2=target_a.copy().tolist()
    target_b2=target_b.copy().tolist()
    
    a,b=getDist(ids, labels, ids)
    
    #
    # Upsample
    #
    target_a2+=np.random.choice(target_a2, N-a).tolist()
    target_b2+=np.random.choice(target_b2, N-b).tolist()
    
    target_a2=np.array(target_a2)
    target_b2=np.array(target_b2)
    
    return target_a2, target_b2

def GNUS(data, ids_hat, ignore):
    """
    Make sure the "data" parameter contains only target samples and NOT all!

    Parameters
    ----------
    data : Numpy Array
        Array containing ...
    ids_hat : TYPE
        DESCRIPTION.
    ignore : TYPE
        DESCRIPTION.

    Returns
    -------
    newData : TYPE
        DESCRIPTION.

    """
    #
    # Get ids, labels and features 
    #    
    ids=data[:,0]
    labels=data[:,1]
    features=data[:,2:]
    
    newData=[]
    duplicateCheck=[]
    for i in ids_hat:
        cFeature=features[np.where(ids==i)[0][0], :]
        if i in duplicateCheck:
            newFeature=[i]+labels[np.where(ids==i)].tolist()+(cFeature+GaussianDraw(features, ignore)).tolist()
            newData.append(newFeature)
        else:
            duplicateCheck.append(i)
            newFeature=[i]+labels[np.where(ids==i)].tolist()+cFeature.tolist()
            newData.append(newFeature)
    newData=np.array(newData)
    return newData
    
def StratUpSample(data, age_index, sex_index, age_split):
    """
    

    Parameters
    ----------
    data : Numpy Array
        Multidimensional array containing ids, classes and features. 
        In dim [:,0] must be the classes and in [:,1] the labels. 
    age_index : Int
        Couloumn index of the age feature.
    sex_index : Int
        Couloumn index of the sex feature.
    age_split : Int
        Number of age groups that shall be generated. 

    Returns
    -------
    SaveUpSampled : Numpy Array
        Array containing the ids of the upsampled data points for class 1
    SaveUpSampled2 : Numpy Array
        Array containing the ids of the upsampled data points for class 0

    """
    
    #
    # Get ids, labels, age and sex
    #
    ids=data[:,0]
    labels=data[:,1]
    age=data[:,age_index]
    sex=data[:,sex_index]
    
    #
    # Find majority class 
    #
    a,b=getDist(ids, labels, ids)
    
    #
    # Start balancing a (label 1) and then b (label 0) to match the balanced set a
    #
    if a>b: 
        #
        # Dissect data into age groups
        #
        age_a=age[np.where(labels==1)]
        ids_a=ids[np.where(labels==1)]
        sex_a=sex[np.where(labels==1)]
        
        minAge=np.amin(age_a)
        maxAge=np.amax(age_a)
        interval=(maxAge-minAge)/age_split
        age_group=[]
        for i in range(age_split):
            if i==(age_split-1):
                condition=np.where((age_a>=minAge+i*interval)&(age_a<=minAge+(i+1)*interval))
            else:
                condition=np.where((age_a>=minAge+i*interval)&(age_a<=minAge+(i+1)*interval))
            age_group.append(ids_a[condition].tolist())
            
        #
        # Find maximal upsampled group by stratifying for age
        #
        N_max=0
        for i,s in enumerate(age_group):
            N_s = len(stratify(s, sex_a, ids_a))
            if N_s>N_max:
                N_max=N_s
        
        #
        # Stratify all age groups according to the maximal number N_max
        #
        SaveUpSampled=np.zeros((age_split*N_max))
        for i,s in enumerate(age_group):
            target=stratify(s, sex_a, ids_a)
            k=len(target)
        
            while k<N_max:
                target=upsample(target)
                target=stratify(target, sex_a, ids_a)
                k=len(target)
        
            SaveUpSampled[i*N_max:(i+1)*N_max]=target
        
        #
        # Start balancing b (label 0)
        #
        #
        # Dissect data into age groups
        #
        age_b=age[np.where(labels==0)]
        ids_b=ids[np.where(labels==0)]
        sex_b=sex[np.where(labels==0)]
        
        minAge=np.amin(age_b)
        maxAge=np.amax(age_b)
        interval=(maxAge-minAge)/age_split
        age_group=[]
        for i in range(age_split):
            if i==(age_split-1):
                condition=np.where((age_b>=minAge+i*interval)&(age_b<=minAge+(i+1)*interval))
            else:
                condition=np.where((age_b>=minAge+i*interval)&(age_b<=minAge+(i+1)*interval))
            age_group.append(ids_b[condition].tolist())
            
        #
        # Find maximal upsampled group by stratifying for age
        #
        N_max=0
        for i,s in enumerate(age_group):
            N_s = len(stratify(s, sex_b, ids_b))
            if N_s>N_max:
                N_max=N_s
                
        #
        # Stratify all age groups according to the maximal number N_max
        #
        SaveUpSampled2=np.zeros((age_split*N_max))
        for i,s in enumerate(age_group):
            target=stratify(s, sex_b, ids_b)
            k=len(target)
        
            while k<N_max:
                target=upsample(target)
                target=stratify(target, sex_b, ids_b)
                k=len(target)
        
            SaveUpSampled2[i*N_max:(i+1)*N_max]=target
    #
    # Start balancing b (label 0) and then a (label 1) to match the balanced set b
    #        
    elif b>=a:
        #
        # Dissect data into age groups 
        #
        age_a=age[np.where(labels==0)]
        ids_a=ids[np.where(labels==0)]
        sex_a=sex[np.where(labels==0)]
        
        minAge=np.amin(age_a)
        maxAge=np.amax(age_a)
        interval=(maxAge-minAge)/age_split
        age_group=[]
        for i in range(age_split):
            if i==(age_split-1):
                condition=np.where((age_a>=minAge+i*interval)&(age_a<=minAge+(i+1)*interval))
            else:
                condition=np.where((age_a>=minAge+i*interval)&(age_a<=minAge+(i+1)*interval))
            age_group.append(ids_a[condition].tolist())
            
        #
        # Find maximal upsampled group by stratifying for age
        #
        N_max=0
        for i,s in enumerate(age_group):
            N_s = len(stratify(s, sex_a, ids_a))
            if N_s>N_max:
                N_max=N_s
        
        #
        # Stratify all age groups according to the maximal number N_max
        #
        SaveUpSampled2=np.zeros((age_split*N_max))
        for i,s in enumerate(age_group):
            target=stratify(s, sex_a, ids_a)
            k=len(target)
        
            while k<N_max:
                target=upsample(target)
                target=stratify(target, sex_a, ids_a)
                k=len(target)
        
            SaveUpSampled2[i*N_max:(i+1)*N_max]=target
        
        #
        # Start balancing b (label 0)
        #
        #
        # Dissect data into age groups
        #
        age_b=age[np.where(labels==1)]
        ids_b=ids[np.where(labels==1)]
        sex_b=sex[np.where(labels==1)]
        
        minAge=np.amin(age_b)
        maxAge=np.amax(age_b)
        interval=(maxAge-minAge)/age_split
        age_group=[]
        for i in range(age_split):
            if i==(age_split-1):
                condition=np.where((age_b>=minAge+i*interval)&(age_b<=minAge+(i+1)*interval))
            else:
                condition=np.where((age_b>=minAge+i*interval)&(age_b<=minAge+(i+1)*interval))
            age_group.append(ids_b[condition].tolist())
            
        #
        # Find maximal upsampled group by stratifying for age
        #
        N_max=0
        for i,s in enumerate(age_group):
            N_s = len(stratify(s, sex_b, ids_b))
            if N_s>N_max:
                N_max=N_s
                
        #
        # Stratify all age groups according to the maximal number N_max
        #
        SaveUpSampled=np.zeros((age_split*N_max))
        for i,s in enumerate(age_group):
            target=stratify(s, sex_b, ids_b)
            k=len(target)
        
            while k<N_max:
                target=upsample(target)
                target=stratify(target, sex_b, ids_b)
                k=len(target)
        
            SaveUpSampled[i*N_max:(i+1)*N_max]=target        
    
        return SaveUpSampled, SaveUpSampled2
    
    
def StatisticNormalization(features, ignore=None):
    moment_1=[]
    moment_2=[]
    n_features=features.shape[1]
    for i in range(n_features):
        moment_1.append(np.mean(features[:,i]))
        moment_2.append(np.std(features[:,i]))
    moment_1=np.array(moment_1)
    moment_2=np.array(moment_2)
    if ignore is None:
        pass
    else:
        moment_1[ignore]=0
        moment_2[ignore]=1
    print(moment_1)
    print(moment_2)
    newX=(features-moment_1)/moment_2
    return newX

def IdsToData(data, ids_hat):

    ids=data[:,0]
    newData=[]
    for i in ids_hat:
        newData.append(data[np.where(ids==i)[0][0], :])
        
    newData=np.array(newData)
    return newData
