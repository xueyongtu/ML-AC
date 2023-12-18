# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:33:11 2022

@author: whufi
"""
 
import os
import numpy as np
import random
import pandas as pd
import warnings
random.seed(135)
warnings.filterwarnings("ignore")

# os.chdir(r'E:\02实验\98ML-AC-code')  ##设置文件路径

#%% 主要代码

 
##获取等权投资组合
def get_equal_weight(ret):
    ret_m=pd.melt(ret.reset_index(),id_vars=ret.reset_index().columns[0])
    ret_m.columns=['date','permno','ret']
    ret_m['month']=ret_m['date'].apply(lambda x:str(x)[:6])
    ret_m=ret_m.dropna(axis=0).astype(float)
    count=ret_m.groupby(['date'])['ret'].count().reset_index()
    retd=pd.merge(ret_m,count,on=['date'],how='left')
    
    retd['weight']=1/retd['ret_y']
    weight1=pd.pivot(retd,index='date',columns='permno',values='weight')
    weight2=pd.concat([pd.DataFrame(columns=ret.columns.astype(float)),weight1],axis=0,join='outer')
    weight2=pd.concat([pd.DataFrame(index=ret.index),weight2],axis=1,join='outer')
    weight2.columns=weight2.columns.astype(float)    

    return  weight2

 

def get_tc(j,theta0,rets,chs,weights,cr=0.005):
    '''
    j:从1开始，假设第一期到第二期无交易费用
    cr:表示费率，一般取0.005
    '''
    
    r=rets.iloc[j:j+1,:].values
    c=chs[chs.date==rets.index[j]].sort_values('permno').iloc[:,2:].fillna(0).values
    wbar=weights.iloc[j:j+1,:].fillna(0).values
    w0bar=weights.iloc[j-1:j,:].fillna(0).values
    c0=chs[chs.date==rets.index[j-1]].sort_values('permno').iloc[:,2:].fillna(0).values
    r0=rets.iloc[j-1:j,:].values
    r00=rets.iloc[j-1:j,:].fillna(0).values
    
    wp=np.multiply(w0bar + np.dot(theta0.T,c0.T)/np.sum(~np.isnan(r0),axis=1),(1+r00))
    w=wbar + np.dot(theta0.T,c.T)/np.sum(~np.isnan(r),axis=1) 

    ##固定比列的交易成本
    lc= c/np.sum(~np.isnan(r),axis=1) - np.multiply(c0/np.sum(~np.isnan(r0),axis=1), 1+np.repeat(r00,c0.shape[1],axis=0).T)   
    tc1=cr*np.dot(np.sign(w-wp),lc)
    
    return tc1.T



def power_utility(r,gamma=5):   ##越大越好
    return np.power(1+r,1-gamma)/(1-gamma)

 

#根据资产的特征和收益数据，获得特征的权重系数theta
def train(rets,chs,weights,gamma,lambda1,rho,cr, allow_short_selling,utility_function,cost_type):  ##rho=1,L1; rho=0,L2 ; cr:cost rate费率 ; allow_short_selling默认无卖空约束
    
    retsv=rets.fillna(0).values
    weightsv=weights.fillna(0).values 
    rcs=np.hstack([np.dot(retsv[j:j+1,:],chs[chs.date==rets.index[j]].sort_values('permno').iloc[:,2:].fillna(0).values).T/np.sum(~np.isnan(rets.iloc[j:j+1,:]),axis=1).values  for  j in range(len(rets)) ])     
    rbs=np.hstack([np.dot(retsv[j:j+1,:],weightsv[j:j+1,:].T) for  j in range(len(rets))])    
    sigmac=np.cov(rcs)
    
    cmean=np.mean(rcs,axis=1)
    cmeanm=np.vstack([cmean for x in range(rcs.shape[1])]).T
    sigmabc=np.dot(rbs-np.mean(rbs), (rcs-cmeanm).T)/(rcs.shape[1]-1)
    
    if rcs.shape[0]==1:
        theta=np.dot(1/sigmac,(cmean.reshape(len(cmean),1)/gamma-sigmabc.reshape(len(cmean),1) ))
    else: ##inv求逆函数要求矩阵是二维的      
        theta=np.dot(np.linalg.inv(sigmac),(cmean.reshape(len(cmean),1)/gamma-sigmabc.reshape(len(cmean),1) ))
    
    return theta
    
 


 
#根据特征的权重系数theta，进而获得投资组合权重w
def test(theta,rett,weightt,chs,allow_short_selling):   
    
    rets=rett
    weightsv=weightt.fillna(0).values 
 
    #是否允许卖空
    if  allow_short_selling==True: #允许卖空
        w=weightsv + np.vstack([np.dot(theta.T,chs[chs.date==rets.index[j]].sort_values('permno').iloc[:,2:].fillna(0).values.T)/np.sum(~np.isnan(rets.iloc[j:j+1,:]),axis=1).values  for  j in range(len(rets)) ])     
        # r= rbs+ np.dot(theta.T,rcs)
    elif  allow_short_selling==False: #不允许卖空
        w=weightsv + np.vstack([np.dot(theta.T,chs[chs.date==rets.index[j]].sort_values('permno').iloc[:,2:].fillna(0).values.T)/np.sum(~np.isnan(rets.iloc[j:j+1,:]),axis=1).values  for  j in range(len(rets)) ])     
        w[w<0]=0             
        wsum=np.sum(w,axis=1).reshape(w.shape[0],1).repeat(w.shape[1],axis=1)
        w=w/wsum
    
    return w
     



def get_weights(i,ret,ch,weight0,gamma,rho,cr,allow_short_selling,utility_function,cost_type):
    trw=7
    viw=3
   
    #样本集
    weights=weight0.iloc[i:i+12*(trw+viw),:] ##市值加权的投资组合    
    
    rets1=pd.melt(ret.iloc[i:i+12*(trw+viw+1),:].reset_index(),id_vars='date')
    rets1.columns=['date','permno','ret']
    rets1=rets1.sort_values(by=['date']).astype(float)
    chs=pd.merge(rets1[['date','permno']],ch,how='left',on=['date','permno'])  ##训练验证测试所用的特征
    
    retm=ret.iloc[i:i+12*(trw+viw),:]
    lambdaopt=0
    
    theta=train(ret.iloc[i:i+12*(trw+viw),:],chs,weight0.iloc[i:i+12*(trw+viw),:],gamma,lambdaopt,rho,cr, allow_short_selling=allow_short_selling,utility_function=utility_function,cost_type=cost_type)
    print(theta)
    ##测试集
    rett=ret.iloc[i+12*(trw+viw):i+12*(trw+viw+1),:]    
    weightt=weight0.iloc[i+12*(trw+viw):i+12*(trw+viw+1),:]    
    wp=test(theta,rett,weightt,chs,allow_short_selling)
    wp=pd.DataFrame(wp,index=weightt.index,columns=weightt.columns)
    return wp
 

def get_result(ret,ch,methodname,gamma,rho,cr, allow_short_selling,utility_function,cost_type):
    trww=7*12
    viww=3*12
    teww=1*12  
    weight0= get_equal_weight(ret)
    weight=pd.DataFrame()
    for i in range(0,len(ret)-trww-viww-teww+1,12):
        print(i)
        w=get_weights(i,ret,ch,weight0,gamma,rho,cr, allow_short_selling=allow_short_selling,utility_function=utility_function,cost_type=cost_type)  
        weight=pd.concat([weight,w],axis=0,join='outer')
    weight.to_csv('result/middle/weight_all/weight_'+methodname+'_'+str(rho)+'_'+str(gamma)+'_'+str(allow_short_selling)+'_'+str(utility_function)+'_'+str(cost_type)+'.csv')



#%%#所需数据，代码调用
ret=pd.read_csv('data/ret_clean.csv',index_col=0).astype(float)/100 #读取收益   
ret.index.name='date'
ch=pd.read_csv('data/char.csv',index_col=0).astype(float) #读取特征
ch=ch.sort_values(by=['date','permno'],ascending= True)


# %%单个特征检验的权重输出
# for i in range(ch.shape[1]-2):
#     para=[ch.columns[2+i]]+[0,5,0.005,True,'MV',False]
#     chm=pd.concat([ch.iloc[:,:2],ch.iloc[:,i+2]],axis=1)
#     get_result(ret,chm,methodname=para[0],rho=para[1],gamma=para[2],cr=para[3], allow_short_selling=para[4],utility_function=para[5],cost_type=para[6])

 
#1.主结果的投资组合权重 2.卖空约束下的投资组合权重 3. 风险厌恶系数等于10的投资组合权重
#%%#OLS-AC的权重输出
para_list=[['OLS',0,5,0.005,True,'MV',False],['OLS',0,5,0.005,False,'MV',False],['OLS',0,10,0.005,True,'MV',False]] #,
for para in para_list:
    print(para)
    get_result(ret,ch,methodname=para[0],rho=para[1],gamma=para[2],cr=para[3], allow_short_selling=para[4],utility_function=para[5],cost_type=para[6])

 
#%%#OLS-5C的权重输出
ch=ch[['date','permno','01_size', '19_mom12', '29_BM', '59_ROE', '43_AG']]
para_list=[['ff5',0,5,0.005,True,'MV',False],['ff5',0,5,0.005,False,'MV',False],['ff5',0,10,0.005,True,'MV',False]] 
for para in para_list:
    get_result(ret,ch,methodname=para[0],rho=para[1],gamma=para[2],cr=para[3], allow_short_selling=para[4],utility_function=para[5],cost_type=para[6])


#%%#OLS-3C的权重输出
ch1=ch[['date','permno','01_size', '19_mom12', '29_BM']]
para_list=[['ff3',0,5,0.005,True,'MV',False],['ff3',0,5,0.005,False,'MV',False],['ff3',0,10,0.005,True,'MV',False]] 
for para in para_list:
    get_result(ret,ch1,methodname=para[0],rho=para[1],gamma=para[2],cr=para[3], allow_short_selling=para[4],utility_function=para[5],cost_type=para[6])


 



