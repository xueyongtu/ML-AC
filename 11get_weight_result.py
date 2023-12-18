# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:35:37 2022

@author: whufi
"""

##计算基本的权重指标
 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import glob, os
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import scale

os.chdir(r'E:\02实验\98ML-AC-code')

ret=pd.read_csv('data/ret_clean.csv',index_col=0).astype(float)/100 #读取收益   
ret.index.name='date'
ret=ret.astype(float)

##中国市场的三因子和四因子数据
ch3=pd.read_csv(r'data\CH_3_update_20211231.csv').iloc[7:,:]
ch4=pd.read_csv(r'data\CH_4_fac_update_20211231.csv').iloc[8:,:]

def process_ch(ch):
    ch.columns=ch.iloc[0,:]
    ch=ch.iloc[1:,:]
    ch.iloc[:,0]= ch.iloc[:,0].apply(lambda x:str(x)[:6])
    ch=ch.astype(float)
    ch=ch.set_index('mnthdt')/100
    return ch

ch3=process_ch(ch3)
ch4=process_ch(ch4)

#用于计算回撤
def get_cumret(ret):
    '''
    输入每一期的收益:列为因子收益，行为日期
    输出从起始日期的每日累计收益：列为因子累计收益，行为日期
    '''
    cumpd=pd.DataFrame(columns=ret.columns,index=ret.index)
    ret=ret.fillna(0)
    for j in range(ret.shape[1]):
        for i in range(ret.shape[0]):
            if i==0:
                cumpd.iloc[i,j]=ret.iloc[i,j]+1
            else:
                cumpd.iloc[i,j]=(ret.iloc[i,j]+1)*cumpd.iloc[i-1,j]
    return cumpd


def get_DD(ret):
    '''
    输入每一期的收益:列为因子收益，行为日期
    输出:每一列因子的最大回撤
    '''
    ret=get_cumret(ret)
    dd=pd.DataFrame(columns=ret.columns,index=ret.index)
    for j in range(ret.shape[1]):
        for i in range(ret.shape[0]):
            dd.iloc[i,j] = (np.max(ret.iloc[:i+1,j])-ret.iloc[i,j])/np.max(ret.iloc[:i+1,j])
        
    MDD=np.max(dd,axis=0)
    return MDD


#%%计算每个月的权重的最大、最小

def get_single_result(weight,ret,ch3,ch4,methodname):
    weightraw=weight.copy()
    weight=weight.fillna(0)
 
    retn=pd.concat([pd.DataFrame(index=weight.index),ret],axis=1,join='inner')
    count=retn.count(axis=1)
 
    wn=weight.copy()
    wn[wn>=0]=0 
    wn[wn<0]=1 #用于统计负权重的占比
    
    #平均绝对权重
    wabs=np.sum(np.abs(weight),axis=1)
    wabsmean=np.mean(wabs/count)*100  ##先横截面平均、再时间序列上求平均
     
    #平均最大绝对权重
    wmax=np.max(np.abs(weight),axis=1)  
    wmaxmean=np.mean(wmax)*100    ##权重变动范围还比较小
    
    #平均最小权重
    wmin=np.min(weightraw,axis=1)  
    wminmean=np.mean(wmin)*100    ## 
    
    #平均负权重之和
    wneg=np.sum(weight[weight<0],axis=1)
    wnegmean=np.mean(wneg)
    
    #平均负权重的占比
    wnegn=np.sum(wn,axis=1)
    wnegmeannum=np.mean(wnegn/count)
    
    ##平均权重变动之和（类似于换手率的概念）
    retn=pd.concat([pd.DataFrame(index=weight.index),ret],axis=1,join='inner')
    wplus=weight*(1+retn)
    wplus.index=wplus.reset_index().iloc[:,0].shift(-1)
    wplus=wplus.iloc[:-1,:]
    weightn=weight.iloc[1:,:]
    wwabs=np.sum(np.abs(weightn-wplus),axis=1)
    wwabsmean=np.mean(wwabs)
    
    ##定义均值、方差、夏普比、CER
    retport=np.sum(weight*retn,axis=1)
    mean=np.mean(retport)*12
    std=np.std(retport,ddof=1)*np.sqrt(12)
    sr=mean/std
    cer=mean-std*std/2 
    
    mdd=get_DD(pd.DataFrame(retport))[0]
    
    
    skew=stats.skew(retport)#使用stats计算偏度
    kurtosis = stats.kurtosis(retport)#使用stats计算峰度
    
    ch3=pd.concat([pd.DataFrame(retport,columns=['retp']),ch3],axis=1,join='inner')
    ch4=pd.concat([pd.DataFrame(retport,columns=['retp']),ch4],axis=1,join='inner')
    
    #ch3-α检验
    ch3test = smf.ols('retp~mktrf+SMB+VMG',ch3).fit(cov_type = 'HAC',cov_kwds = {'maxlags':5})
    ch3_a = ch3test.params[0]
    ch3_t = ch3test.tvalues[0]         
    res=[cer,wabsmean,wmaxmean,wminmean,wnegmean,wnegmeannum,wwabsmean,mean,mdd,std,skew,kurtosis,sr,ch3_a,
         ch3_t]
    
    result=pd.DataFrame(res,columns=[methodname],index=['CER','w_abs','w_max','w_min','w_neg','w_negnum','ww_abs','Mean','MDD','StdDev',
                               'Skew','Kurt','SR','CH3_alpha','CH3_t'])
    return result


def get_all_result(path,respath,name):  ##为多个数据集所用
    os.chdir(path)
    file = glob.glob(os.path.join("*.csv"))
    result_all=pd.DataFrame(index=['CER','w_abs','w_max','w_min','w_neg','w_negnum','ww_abs','Mean','MDD',
                                   'StdDev','Skew','Kurt','SR','CH3_alpha','CH3_t'])

    for i in range(len(file)):
        # i=0
        weight=pd.read_csv(file[i],index_col=0)
        result=get_single_result(weight,ret,ch3,ch4,file[i][:-4])
        result_all=pd.concat([result_all,result],axis=1)        
    result_all.to_csv(respath+'/result_'+name+'.csv') ##所有结果

 
#%%#输出结果
respath=r'E:\02实验\98ML-AC-code\result\final'
path=r'E:\02实验\98ML-AC-code\result\middle\weight_all'
name='all_results'
get_all_result(path,respath,name)

 



















 