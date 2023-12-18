# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 21:34:31 2022

@author: tuxueyong
"""

import numpy as np
import random
import pandas as pd
from scipy.linalg import sqrtm  # for finding the squared root of Sigma
import os
import glob
import warnings
warnings.filterwarnings("ignore")


os.chdir(r'E:\02实验\98ML-AC-code')

ret=pd.read_csv('data/ret_clean.csv',index_col=0).astype(float)/100 #读取收益   
ret.index.name='date'

def get_weight(path,method):
    '''
    path:权重路径
    method:'MV'、'SPO'
    '''
    file = glob.glob(os.path.join(path, "*.csv"))
    f = []  ##存放因子+超额收益
    weight=pd.DataFrame() #columns=ret.columns.astype(float)
    for i in range(len(file)):
        f_data=pd.read_csv(file[i], header=0, index_col=0)
        f_data.columns=f_data.columns.astype(float)
        weight=pd.concat([weight,f_data],axis=0,join='outer')
        
    wr=weight.reset_index()      
    weight=wr.sort_values(wr.columns[0],ascending=[True]).set_index(wr.columns[0])
    weight.to_csv('result/middle/weight_all/weight_'+method+'.csv')
    
    return weight
    
#%%整理LSTM-AC的投资权重
method_list=['weightLSTM_5_False_MV_False','weightLSTM_5_True_MV_False','weightLSTM_10_True_MV_False']
for i in range(len(method_list)):
    path = 'result/'+method_list[i]+'/'
    method=method_list[i]
    get_weight(path,method) 
    
#%%整理LSTM-AC的投资权重
method_list=['weightRNN_5_False_MV_False','weightRNN_5_True_MV_False','weightRNN_10_True_MV_False']
for i in range(len(method_list)):
    path = 'result/'+method_list[i]+'/'
    method=method_list[i]
    get_weight(path,method) 
    
 

#%%整理DFN-AC的投资权重
def get_weight_sum(path,method):
    '''
    path:权重路径
    method:'MV'、'SPO'
    '''
    file = glob.glob(os.path.join(path, "*.csv"))
    weight=pd.DataFrame() #columns=ret.columns.astype(float)
    for i in range(len(file)):
        f_data=pd.read_csv(file[i], header=0, index_col=0)
        f_data.columns=f_data.columns.astype(float)
        weight=pd.concat([weight,f_data],axis=0,join='outer')
        
    wr=weight.reset_index()      
    weight=wr.sort_values(wr.columns[0],ascending=[True]).set_index(wr.columns[0])
    return weight


weig0=0
for i in [100,101,102,103,104]:
    method='weight_PPNN_5_True_MV_False_'+str(i)  
    path = 'result/'+method+'/'
    print(path)
    weig=get_weight_sum(path,method) 
    weig0=weig0+weig
weig0=weig0/5  
weig0.to_csv('result/middle/weight_all/weight_'+method+'.csv')    
    

weig0=0
for i in [100,101,102,103,104]:
    method='weight_PPNN_5_False_MV_False_'+str(i)  
    path = 'result/'+method+'/'
    print(path)
    weig=get_weight_sum(path,method) 
    weig0=weig0+weig
weig0=weig0/5    
weig0.to_csv('result/middle/weight_all/weight_'+method+'.csv')

 
weig0=0
for i in [100,101,102,103,104]:
    method='weight_PPNN_10_True_MV_False_'+str(i)  
    path = 'result/'+method+'/'
    print(path)
    weig=get_weight_sum(path,method)   
    weig0=weig0+weig
weig0=weig0/5 
weig0.to_csv('result/middle/weight_all/weight_'+method+'.csv')  


 