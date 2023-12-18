# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import imblearn


class Augmentation:
    def __init__(self, data, target=0, dataType="normal", targetType="binary"):
        """
    

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        target : TYPE
            DESCRIPTION.
        dataType : TYPE, optional
            DESCRIPTION. The default is "normal".

        Returns
        -------
        None.

        """
        self.dataType=dataType
        self.targetType=targetType
        self.data=data
        self.dim=0
        self.nSamples=0
        self.target=target
        
        #
        # TODO: Add more data types
        #
        if dataType=="normal":
            self.nSamples=self.data.shape[0]
            self.dim=self.data.shape[1]
                       
        self.featuresOnly=None
        if self.target==0:
            self.featuresOnly=data[:,1:]
        else:
            self.featuresOnly=data[:,0]
            for i in range(1,self.dim):
                if i==self.target:
                    pass
                else:
                    self.featuresOnly=np.vstack(self.featuresOnly, data[:,i])
           
        
        print("Samples: " + str(self.nSamples))
        print("Features: " + str(self.dim))
        
    def GausNoise(self, N, strengthMean=0.001, strengthStd=0.001):
        
        augmentedData=np.zeros((2*N, self.dim))
        if self.dataType=="normal" and self.targetType=="binary":
            targetValues=self.data[:, self.target]
            for i in [0,1]:
                meanValues=[]
                stdValues=[]
                reducedData=self.data[np.where(targetValues==i)]
                for feature in range(self.dim):
                    if feature==self.target:
                        pass
                    else:
                        meanValues.append(np.mean(reducedData[:,feature])*strengthMean)
                        stdValues.append(np.std(reducedData[:, feature])*strengthStd)
                augmentedData[int(N)*i:int(N)*(i+1), 0] = np.zeros(int(N))+i
                augmentedData[int(N)*i:int(N)*(i+1), 1:] = np.random.normal(meanValues, stdValues, (N,self.dim-1))

        return augmentedData
    
    def smoteUp(self):
        """
        ONLY WORKS IF TARGET IS IN COLUMN 1

        Returns
        -------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        """
        oversample = imblearn.over_sampling.SMOTE()
        x,y = oversample.fit_resample(self.data[:,1:], self.data[:, self.target])
        return x,y
    
    def splitData(self, split=0.9, stratified=False):
        train, test = 0,0
        if stratified:
            pass
                    
        else:
            train, test = self.data[:(int(split*self.nSamples)),:], self.data[(int(split*self.nSamples)):,:]
        return train, test
    
    def normalizeData(self, targetNorm=None):
        if (self.targetType=="binary" and targetNorm==None) or (self.targetType=="regression" and targetNorm==None):
            for feature in range(self.dim):
                if feature==self.target:
                    pass
                else:
                    self.data[:,feature]=(self.data[:,feature]-np.mean(self.data[:,feature]))/np.var(self.data[:,feature].astype(float))
                    
        elif self.targetType=="regression" and targetNorm=="normal":
            for feature in range(self.dim):
                self.data[:,feature]=(self.data[:,feature]-np.mean(self.data[:,feature]))/np.var(self.data[:,feature].astype(float))
                
        elif self.targetType=="regression" and targetNorm=="scaled":
            for feature in range(self.dim):
                if feature==self.target:
                    self.data[:,feature]=(self.data[:,feature]-np.min(self.data[:,feature]))/(np.min(self.data[:,feature])-np.max(self.data[:,feature]))
                else:
                    self.data[:,feature]=(self.data[:,feature]-np.mean(self.data[:,feature]))/np.var(self.data[:,feature].astype(float))
                
    
    def shuffleData(self):
        np.random.shuffle(self.data)
    
    def mergeData(self, data):
        self.data=np.vstack((self.data, data))
        if self.dataType=="normal":
            self.nSamples=self.data.shape[0]
            self.dim=self.data.shape[1]
                       
        self.featuresOnly=None
        if self.target==0:
            self.featuresOnly=data[:,1:]
        else:
            self.featuresOnly=data[:,0]
            for i in range(1,self.dim):
                if i==self.target:
                    pass
                else:
                    self.featuresOnly=np.vstack(self.featuresOnly, data[:,i])
        
        print("Samples: " + str(self.nSamples))
        print("Features: " + str(self.dim))
        
    def downSampleData(self, fraction=1):
        if self.targetType=='binary':
            balancedData=[]
            target=self.data[:,self.target]
            class1=np.where(target>0)[0].shape[0]
            class2=np.where(target==0)[0].shape[0]
            
            if class1<=class2:
                minorClass=class1
                minorityClass=1
                majorityClass=0
            else:
                minorClass=class2
                minorityClass=0
                majorityClass=1
            count=0
            for sample in self.data:
            
                if sample[self.target]==minorityClass:
                    balancedData.append(sample)
                    
                if (sample[self.target]==majorityClass and count<(minorClass*fraction)):
                    balancedData.append(sample)
                    count+=1
            balancedData=np.array(balancedData).astype(float)
            self.data=balancedData

    
    def setData(self, data):
        self.data=data
        #
        # TODO: Add more data types
        #
        if self.dataType=="normal":
            self.nSamples=self.data.shape[0]
            self.dim=self.data.shape[1]
                       
        self.featuresOnly=None
        if self.target==0:
            self.featuresOnly=data[:,1:]
        else:
            self.featuresOnly=data[:,0]
            for i in range(1,self.dim):
                if i==self.target:
                    pass
                else:
                    self.featuresOnly=np.vstack(self.featuresOnly, data[:,i])
        
        print("Samples: " + str(self.nSamples))
        print("Features: " + str(self.dim))
     
        
    def getDistribution(self):
        target=self.data[:,self.target]
        Nsamples=float(target.shape[0])
        
        if self.targetType=="binary":
            class1=np.where(target>0)[0].shape[0]
            class2=np.where(target==0)[0].shape[0]
            res=class2/Nsamples, class1/Nsamples
            print(class2, class1)
        if self.targetType=="regression":
            res=self.data[:,self.target]
        
        return res
        
    
        
