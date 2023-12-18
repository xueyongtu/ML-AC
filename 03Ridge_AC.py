# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:28:20 2022

@author: whufi
"""
import os
import numpy as np
import random
import pandas as pd
import warnings
from sklearn.model_selection import KFold
random.seed(135)
warnings.filterwarnings("ignore")

# os.chdir(r'E:\02实验\98ML-AC-code')  ##设置文件路径 
#%%#等权投资组合
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



def power_utility(r,gamma=5):    
    return np.power(1+r,1-gamma)/(1-gamma)

def loss(rets,chs,weights,theta,gamma,lambda1,rho,cr, allow_short_selling,utility_function,cost_type):
    
    retsv=rets.fillna(0).values
    weightsv=weights.fillna(0).values 
 
    #是否允许卖空
    w=weightsv + np.vstack([np.dot(theta.T,chs[chs.date==rets.index[j]].sort_values('permno').iloc[:,2:].fillna(0).values.T)/np.sum(~np.isnan(rets.iloc[j:j+1,:]),axis=1).values  for  j in range(len(rets)) ])         
 
    retw=w*retsv
    r= np.sum(retw ,axis=1)   #.reshape(1,len(wsum))
    
 
        
    #是否更换效用函数  
    if utility_function=='crra':
        utility=-np.mean(power_utility(r,gamma))+lambda1*rho*np.linalg.norm(theta,1)+lambda1*(1-rho)/2*np.linalg.norm(theta,2)
    elif utility_function=='MV':
        sigma=np.std(r,ddof=1)
        utility= gamma/2*sigma-np.mean(r)+lambda1*rho*np.linalg.norm(theta,1)+lambda1*(1-rho)/2*np.linalg.norm(theta,2)
 
    if cost_type==False:
        utility=utility
    elif cost_type==True:
        wp=w*(1+retsv)
        tc=np.mean(np.sum(np.abs(w[1:,:]-wp[:-1,:]),axis=1))  ##计算交易成本 
        utility=utility+cr*tc
    
    return utility
       

def train(rets,chs,weights,gamma,lambda1,rho,cr, allow_short_selling,utility_function,cost_type):  ##rho=1,L1; rho=0,L2 ; cr:cost rate费率 ; allow_short_selling默认无卖空约束
    
    retsv=rets.fillna(0).values
    weightsv=weights.fillna(0).values 
    rcs=np.hstack([np.dot(retsv[j:j+1,:],chs[chs.date==rets.index[j]].sort_values('permno').iloc[:,2:].fillna(0).values).T/np.sum(~np.isnan(rets.iloc[j:j+1,:]),axis=1).values  for  j in range(len(rets)) ])     
    rbs=np.hstack([np.dot(retsv[j:j+1,:],weightsv[j:j+1,:].T) for  j in range(len(rets))])    
    sigmac=np.cov(rcs)
    
    cmean=np.mean(rcs,axis=1)
    cmeanm=np.vstack([cmean for x in range(rcs.shape[1])]).T
    sigmabc=np.dot(rbs-np.mean(rbs), (rcs-cmeanm).T)/(rcs.shape[1]-1)
    uc=cmean
    
    k=len(ch.columns)-2  ##特征的个数
    theta0=np.ones((k,1))*1.5
    eps=10**(-8)
    
    t=1
    beta1=0.9
    beta2=0.999
    alpha=0.2 ##学习率 
    
 
    utility0=100
    
    
    for u in range(100):   
        print('batch',u)
 
        
        batch_size=2  ##如何设置
        
        batch_starts=[start for start in range(1,len(rets)-batch_size,batch_size)]
        random.shuffle(batch_starts)
        
        m0=0
        v0=0
        
        for p in batch_starts:
            print('第多少轮次梯度',t)
            
            
            if utility_function=='crra':
                gra=-np.power(1+rbs+ np.dot(theta0.T,rcs),-gamma).dot(rcs.T).T + lambda1*rho*np.sign(theta0)+lambda1*(1-rho)*theta0   ##梯度
            elif utility_function=='MV': 
                gra=gamma * np.dot(sigmac,theta0) + gamma * sigmabc.T -uc.reshape(len(uc),1)+ lambda1*rho*np.sign(theta0)+lambda1*(1-rho)*theta0   ##梯度
                
            
                
            if cost_type==False:   #不考虑交易成本
                gra=gra  ##梯度
            elif cost_type==True:  #考虑交易成本
                tc=0
                for j in range(p,p+batch_size):
                    # print('成本',j)
                    tc1=get_tc(j,theta0,rets,chs,weights,cr)
                    tc=tc+tc1
                gra=gra+tc   ##梯度
                
            m=beta1*m0 +(1-beta1)*gra
            v=beta2*v0+(1-beta2)*np.dot(gra.T,gra)
            
            beta1t=beta1**t
            beta2t=beta2**t
            
            mh=m/(1-beta1t)
            vh=v/(1-beta2t)
            
            theta= theta0 -alpha*mh/(np.sqrt(vh)+eps)
            utility=loss(rets,chs,weights,theta,gamma,lambda1,rho,cr, allow_short_selling=allow_short_selling,utility_function=utility_function,cost_type=cost_type)
            print('utility',utility)
  
            
            if utility>utility0:
                break
                          
            if np.linalg.norm(theta-theta0) <= 10**(-5) or np.linalg.norm(utility-utility0)<= 10**(-5)  :
                print(np.linalg.norm(utility-utility0))
                break  
            
            theta0=theta
            utility0=utility
            t=t+1
            
    return theta


 

def test(theta,rett,weightt,chs,allow_short_selling):   
    
    rets=rett
    weightsv=weightt.fillna(0).values 
 
    #是否允许卖空
    if  allow_short_selling==True: #允许卖空
        w=weightsv + np.vstack([np.dot(theta.T,chs[chs.date==rets.index[j]].sort_values('permno').iloc[:,2:].fillna(0).values.T)/np.sum(~np.isnan(rets.iloc[j:j+1,:]),axis=1).values  for  j in range(len(rets)) ])     
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
    
    if i==0:##首先仅考虑对第一期进行超参数调整
        kf=KFold(n_splits=5)
        umax=10000000  #最小效用
        global lambdaopt
        lambdaopt=0
        lambda_list=[0.001,0.0001]# 正则化项的项的参数是待调参数，范围可以从 [0.001,0.0001]
        for lambda1 in lambda_list:
            print(lambda1)
            
            ##五折交叉验证进行优化lambda
            util_list=[]
            for rets_index,retv_index in kf.split(retm):
                print(rets_index.shape,retv_index.shape)
                theta=train(retm.iloc[rets_index,:],chs,weights.iloc[rets_index,:],gamma,lambda1,rho,cr, allow_short_selling=allow_short_selling,utility_function=utility_function,cost_type=cost_type)  
                
                util=loss(retm.iloc[retv_index,:],chs,weights.iloc[retv_index,:],theta,gamma,lambda1,rho,cr, allow_short_selling=allow_short_selling,utility_function=utility_function,cost_type=cost_type)
                util_list.append(util)
 
            util_mean=np.mean(util_list)
            if util_mean<umax:
                lambdaopt=lambda1
                umax=util_mean
                
        print('lambda',lambdaopt)
        pd.DataFrame([lambdaopt],columns=['lambda1']).to_csv('result/parameter/RPP'+'_'+str(rho)+'_'+str(gamma)+'_'+str(allow_short_selling)+'_'+str(utility_function)+'_'+str(cost_type)+'.csv')
        
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
    weight.to_csv('result/middle/weight_all/weight_'+methodname+'_'+str(rho)+'_'+str(gamma)+'_'+str(allow_short_selling)+'_'+str(utility_function)+'_'+str(cost_type)+str(lambdaopt)+'.csv')


#%%#所需数据，代码调用
ret=pd.read_csv('data/ret_clean.csv',index_col=0).astype(float)/100 #读取收益   
ret.index.name='date'
ch=pd.read_csv('data/char.csv',index_col=0).astype(float) #读取特征
ch=ch.sort_values(by=['date','permno'],ascending= True)

#1.主结果的投资组合权重 2.卖空约束下的投资组合权重 3. 风险厌恶系数等于10的投资组合权重
para_list=[['ridge',0,5,0.005,True,'MV',False],['ridge',0,5,0.005,False,'MV',False],['ridge',0,10,0.005,True,'MV',False]]
for para in para_list:
    get_result(ret,ch,methodname=para[0],rho=para[1],gamma=para[2],cr=para[3], allow_short_selling=para[4],utility_function=para[5],cost_type=para[6])

 






