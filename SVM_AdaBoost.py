#!/usr/bin/env python
# coding: utf-8




from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn import preprocessing
import random
from sklearn.preprocessing import  MinMaxScaler,StandardScaler
min_max_scaler=MinMaxScaler()
import os
from sklearn.model_selection import train_test_split,KFold

class  SVC_AdaBoost():
#     def __init__(self,X_train,y_train,C=100,gamma=1):   
    def __init__(self,X_train,y_train,C=100,gamma=1):   
        self.X_train,self.y_train=X_train,y_train
        self.N=self.X_train.shape[0]
        self.C=C
#         self.gamma=gamma
        self.T=15
        self.beta=np.ones([1,self.N])/self.N     
        self.Q=[]
        self.CLASS=[]
        
    def SVC_AdaBoost_(self,):
        self.iter_=0
        for i in range(self.T):
            self.iter_=self.iter_+1
            if np.sum(self.beta[0])==0:    
                break
            clf = SVC(C=self.C,kernel='linear',random_state=42).fit(self.X_train,self.y_train,sample_weight=self.beta.reshape(-1))
            self.CLASS.append(clf) 
            clf_pre=clf.predict(self.X_train)
            
            #calculte error err
            I0=np.zeros([1,self.N])      
            I0[0][clf_pre!=self.y_train]=1            
            #err=np.dot(I0,self.beta.T)[0][0]
            err=np.dot(I0,self.beta[0].T)
            
            #calculate q
            if err>0.5:                     
                print('iter_',self.iter_)
                break
            elif 0<err<=0.5:
                q=0.5*np.log((1-err)/err)
            elif err==0:
                break                                             
            self.Q.append(q)

            #update beta
            I1=np.ones(self.N)
            I1[clf_pre==self.y_train]=-1
            beta0=self.beta[0]*np.exp(I1*q)
            self.beta=np.array([list(beta0/np.sum(beta0))])        
    
    def strong_pythesis_predict(self,X_test):
        if self.iter_==1:
            clf_pre_fin= self.CLASS[0].predict(X_test)
        else:
            i=0
            cla_w_pre=0
            for q,clf in zip(self.Q,self.CLASS):
                clf_pre=clf.predict(X_test)
                if i==0:
                    clf_w_pre=q*clf_pre
                    i+=1
                else :
                    clf_w_pre=clf_w_pre+q*clf_pre
               
                    i+=1
            clf_pre_fin=np.sign(clf_w_pre)
        return clf_pre_fin






