#!/usr/bin/env python
# coding: utf-8


import numpy as np
from sklearn.metrics.pairwise import rbf_kernel,sigmoid_kernel,laplacian_kernel
import copy
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
import pandas as pd
from sklearn import preprocessing
import random
from sklearn.preprocessing import  MinMaxScaler,StandardScaler
min_max_scaler=MinMaxScaler()
import os
import math
import cvxopt as cp
from sklearn import metrics
from cvxopt import matrix, solvers


class Ada_L1MKLSVC_FISTA():
    def __init__(self,X_train,y_train,ker_list,initial_d,gamma,L,C):
        self.X_train,self.y_train=X_train,y_train
        #self.X_test,self.y_test=X_test,y_test
        self.ker_list=ker_list
        
        self.N=self.X_train.shape[0]
        self.M=len(self.ker_list)
        
        self.gamma=gamma
        self.C=C
        
        self.L=L
        self.yita=2
        self.a=1
        
        self.tol=0.009
        self.max_iter=7
        
        self.d=initial_d
        self.beta=np.ones([1,self.N])/self.N

    def linear_kernel(self,X,Y):
        return np.dot(X,Y.T)

    ##多项式核
    def poly_kernel(self,X,Y,n):
        return (np.dot(X,Y.T)+1)**n

    ##高斯核
    def rbf_kernel1(self,X,Y,l):
        return rbf_kernel(X=X,Y=Y,gamma=l)
    
    def sigmoid_kernel1(self,X,Y,l):
        return sigmoid_kernel(X=X,Y=Y,gamma=l)
    
    def laplacian_kernel1(self,X,Y,l):
        return laplacian_kernel(X=X,Y=Y,gamma=l)
    

    
    '''进行算法的编写'''
    '''根据ker_list计算核矩阵'''
    def cal_ker_mat(self,X,Y):
        ker_mat_list=[]
        for i in range(self.M):
            if self.ker_list[i][0]=='linear':
                ker_mat_list.append(self.linear_kernel(X,Y))
            elif self.ker_list[i][0]=='poly':
                ker_mat_list.append(self.poly_kernel(X,Y,self.ker_list[i][1]))
            elif self.ker_list[i][0]=='rbf':
                ker_mat_list.append(self.rbf_kernel1(X,Y,self.ker_list[i][1]))
            elif self.ker_list[i][0]=='sigmoid':
                ker_mat_list.append(self.sigmoid_kernel1(X,Y,self.ker_list[i][1]))
            elif self.ker_list[i][0]=='laplacian':
                ker_mat_list.append(self.laplacian_kernel1(X,Y,self.ker_list[i][1]))
        return ker_mat_list

    #构造加权的核函数
    def MKL_ker(self,ker_mat_list,d):
        for i in range(self.M):
            if i==0:
                MKL_mat=ker_mat_list[i]*d[i]
            elif i>0:
                MKL_mat=MKL_mat+ker_mat_list[i]*d[i]
        return MKL_mat
    

    def cvx_svc(self,d,beta):
        C_beta=self.C*beta 
        MKL_K=self.MKL_ker(self.ker_mat_list,d)
        P = MKL_K * (self.y_train.reshape(-1,1) @ self.y_train.reshape(1,-1))
        q = -1*np.ones(self.X_train.shape[0]).reshape(-1,1)
        
        G = np.zeros((2*self.X_train.shape[0],self.X_train.shape[0]))
        G[0:self.X_train.shape[0]] = - np.identity(self.X_train.shape[0])
        G[self.X_train.shape[0]:] = np.identity(self.X_train.shape[0])
        h = np.zeros(2*self.X_train.shape[0])
        h[self.X_train.shape[0]:] = C_beta
        h = h.reshape(-1,1)
        
        A = self.y_train.reshape(1,-1)
        b = np.zeros(1).reshape(-1,1)
        
        [P,q,G,h,A,b] = [matrix(i,i.shape,"d") for i in [P,q,G,h,A,b]]
        cp.solvers.options['show_progress'] = False
        sol=cp.solvers.qp(P,q,G,h,A,b)
    
        alpha_list=np.array(sol['x'])
        alpha=alpha_list[:self.N].T
        Jd=sol['primal objective']
        return alpha,-Jd
    
    ##编写目标函数的梯度向量
    def cal_dJd(self,alpha,d):
        dJd=[]
        for i in range(self.M):
            MKL_K=self.MKL_ker(self.ker_mat_list,d)
            y1=self.y_train.reshape(1,-1)
            y2=np.array([self.y_train]).T
            P=np.dot(y2,y1)*self.ker_mat_list
            dJdi=-0.5*np.dot(np.dot(alpha,P[i]),alpha.T)[0][0]
            dJd.append(dJdi)
        return np.array(dJd)

#编写F函数
    def cal_F(self,d):
        sol=self.cvx_svc(d,self.beta)
        Jd=sol[1]
        return Jd+self.gamma*np.sum(np.abs(d))

#编写Q函数
    def cal_Q(self,d1,d2,L,Jd,dJd):
        return Jd+np.sum((d1-d2)*dJd)+0.5*L*np.sum((d1-d2)**2)+self.gamma*np.sum(np.abs(d1))

#计算更新算子
    def cal_p_L_h(self,h,L,dJd):
        Phi=np.abs(h-dJd/L)
        mPhi=Phi-self.gamma/L
        mPhi[mPhi<0]=0
        d=mPhi*np.sign(Phi)
        return d

    def find_it(self,h,L,d):
        it=0
        alpha,Jd=self.cvx_svc(h,self.beta)
        dJd=self.cal_dJd(alpha,d)
        while 1:
            L_bat=self.yita**it*L
            h1=self.cal_p_L_h(h,L_bat,dJd)
            F_h1=self.cal_F(h1)
            Q_h1_h=self.cal_Q(h1,h,L_bat,Jd,dJd)
            it=it+1
            if F_h1<=Q_h1_h:
                break
            print('it',it)
        return L_bat,h1
    
    
    def strong_pythesis_predict(self,X_test,model=1):
        ker_mat_list=self.cal_ker_mat(X_test,self.X_train)
        if model==1:
            i=0
            y_pre_w1=0
            for q, reg, d in zip(self.Q,self.REG,self.D):
                if q==0:
                    continue
                MKL_d=self.MKL_ker(ker_mat_list,d)
                y_pre_d=reg.predict(MKL_d)
                
                if i==0:
                    y_pre_w1=q*y_pre_d
                elif i>0:
                    y_pre_w1=y_pre_w1+q*y_pre_d
                i=i+1
                #print(i)
            y_pre_fin=np.sign(y_pre_w1)
        elif model==2:
            reg_fin=self.REG[-1]
            MKL_d=self.MKL_ker(ker_mat_list,self.D[-1])
            y_pre_fin=reg_fin.predict(MKL_d)
        return y_pre_fin
    
    def Ada_L1MKLSVC_FISTA_(self,):
        
        #计算每个核函数对应的核矩阵 
        self.ker_mat_list=self.cal_ker_mat(self.X_train,self.X_train)
        
        h=self.d
        self.iter_=0
        self.D=[]
        self.Q=[]
        self.REG=[]
        self.BETA=[]
        while 1:
            print('迭代次数:',self.iter_)
            
            '''update beta'''
 
            MKL_mat_fin=self.MKL_ker(self.ker_mat_list,self.d)
            reg=SVC(C=self.C,kernel='precomputed',random_state=42).fit(MKL_mat_fin,self.y_train,sample_weight=self.beta[0])
            self.REG.append(reg) 
            self.BETA.append(self.beta) 
            self.D.append(self.d) 
            reg_pre=reg.predict(MKL_mat_fin)
            auc=roc_auc_score(self.y_train,reg_pre)
            #print(auc)
            #calculate q
            if auc<0.5:                       
                break
            elif 0.5<=auc<1:
                q=0.3*auc
            elif auc==1:
                break                                             
            self.Q.append(q)
#             print(self.Q)
            #update beta
            I1=np.ones(self.N)
            I1[reg_pre==self.y_train]=-1
            beta0=self.beta[0]*np.exp(I1*q)
            self.beta=np.array([list(beta0/np.sum(beta0))])
#             print(self.beta)
            ##搜索最小整数it对应的L以及更新d
            d_old=copy.copy(self.d)
            self.L,self.d=self.find_it(h,self.L,self.d)
            self.d[self.d<0]=0
            
            ##根据alpha1_alpha计算Jd和dJd
            if np.max(np.abs(self.d-d_old))<=self.tol or self.iter_>=self.max_iter:
                print('停止条件1(权重的绝对值变化)：',np.max(np.abs(self.d-d_old)))
                print('停止条件2(迭代的次数)：',self.iter_)
                break
            #print(np.max(np.abs(self.d-d_old)))
            
            #更新a
            a_old=copy.copy(self.a)
            self.a=0.5*(1+np.sqrt(1+4*a_old**2))
            
            #更新h
            h=self.d+((a_old-1)/self.a)*(self.d-d_old)
            
            self.iter_+=1






