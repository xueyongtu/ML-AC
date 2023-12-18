# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:51:01 2022
@author: whufi
"""


import torch
from torch import nn
from torch.nn import init
import numpy as np
import torch.utils.data as Data
 
import random
import pandas as pd
import os
import warnings
from pytorchtools_change import EarlyStoppings  ##点开pytorchtools，复制里面的代码，即可新建pytorchtools
import joblib
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
# os.chdir(r'E:\02实验\98ML-AC-code')  ##设置文件路径

#%%读取数据
ret=pd.read_csv('data/ret_clean.csv',index_col=0).astype(float)/100 #读取收益 
ch=pd.read_csv('data/char.csv',index_col=0).astype(float) #读取特征


def get_data(ret,ch):
    '''
    对收益、特征数据进行调整
    '''
    ret.index.name='date'
    
    ##为了保证每一期的股票数量的位置相同，生成全部的date+permno
    new=pd.melt(ret.fillna(0).reset_index(),id_vars='date').astype(float)
    new.columns=['date','permno','ret']
    new=new[['date','permno']]
    new=new.sort_values(by=['date','permno'])    
   
    ret0=pd.melt(ret.reset_index(),id_vars='date').astype(float)
    ret0.columns=['date','permno','ret']
    ch=ch.fillna(0)    
        
    retch=pd.merge(new,ch,how='left',on=['date','permno']).fillna(0)
    retch=pd.merge(retch,ret0,how='left',on=['date','permno'])

    data_list=[]
    for i in ret.index:
        data_list.append(retch[retch.date==i])   ##看看此处的数据有没有乱,没乱
        
    return data_list

data_list=get_data(ret,ch)
 

 


# #获得数据迭代器   
def load_batch(data_arrays, batch_size, N, is_train=True):
    '''
    自定义生成训练集中的batch数据集，为了保持横截面的样本顺序
    '''
    batch_starts=[start for start in range(0, int(data_arrays[0].shape[0]+1-batch_size),batch_size)]
    
    if is_train:
        random.shuffle(batch_starts)
    
    cr_list=[]

    for j in batch_starts :
        cr_list.append([data_arrays[0][j:(j+batch_size)],data_arrays[1][j:(j+batch_size)]])
        
    if np.max(batch_starts)+batch_size < data_arrays[0].shape[0]:
        cr_list.append([data_arrays[0][(np.max(batch_starts)+batch_size):],data_arrays[1][(np.max(batch_starts)+batch_size):]])
    
    return cr_list


###简化标准化过程

class Normalization(nn.Module):
    '''
    标准化每一个截面
    '''
    def __init__(self):
        super(Normalization, self).__init__()
        # self.r=r
        # self.N=N
        
    def forward(self,x):
        xs=torch.empty(x.shape) 
    
        for i in range(x.shape[0]):
            mean=torch.mean(x[i,:,:],axis=0) 
            std=torch.std(x[i,:,:],axis=0) 
            xs[i,:,:]=(x[i,:,:]-mean)/std
        
        return xs



# Neural Network Model  
class Net(nn.Module):
    #初始化网络结构
    def __init__(self, input_size=95, hidden_size=32, num_classes=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   #输入层，线性（liner）关系  ,bias=False
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)   #隐藏层，使用ReLU函数  
        self.norm1=Normalization()  
        self.dropout1 = nn.Dropout(p=0.1)  # dropout训练
        
        
        self.fc2 = nn.Linear(hidden_size, 16)   #输入层，线性（liner）关系  ,bias=False
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)   #隐藏层，使用ReLU函数   
        self.norm2=Normalization()  
        self.dropout2 = nn.Dropout(p=0.1)  # dropout训练        
        self.fc4 = nn.Linear(16, num_classes,bias=False)  #输出层，线性（liner）关系 ,bias=False
 
    
    #forword 参数传递函数，网络中数据的流动
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.norm1(out) 
        out=self.dropout1(out)
        
        
        out = self.fc2(out)
        out = self.relu2(out)        
        out = self.norm2(out) 
        out=self.dropout2(out)   
        out = self.fc4(out)
        return out

net=Net()
    
    
def loss(net,w_hat,r,gamma,N,lambda1,allow_short_selling=True,utility_function='crra',cost_type=False):
    '''
    w_hat:上一层的输出，为权重的一部分，等价于y_pred
    r:对应的收益数据
    '''
    weigh0=1/(~torch.isnan(r)).sum(axis=1)   ##计算等权投资组合的权重
    w_hat_mul=torch.where(torch.isnan(r),torch.full_like(r, np.nan), w_hat)
    w_hatm=torch.where(torch.isnan(r),torch.full_like(r, 0), w_hat)
    
    ws=torch.sum(w_hatm,axis=1)
    num=torch.sum(~torch.isnan(r),axis=1)
    wn=ws/num
    
      
    w0=torch.empty(w_hat.shape) 
    wh=torch.empty(w_hat.shape) 
    
    for j in range(w_hat.shape[0]):
        w0[j]=torch.where(torch.isnan(r[j]),torch.full_like(r[j], 0), weigh0[j])
        wh[j]=(w_hat_mul[j]-wn[j])/num[j]
    wh=torch.where(torch.isnan(wh),torch.full_like(wh, 0), wh)

    w=w0+wh
    if allow_short_selling==True:  ##没有权重约束
        w=w
    elif allow_short_selling==False: ##卖空约束
        w=w.clamp(0,1)    
    w=w/torch.sum(w,axis=1).repeat(1,w.shape[1]).reshape(w.shape)   ##重新要求权重之和等于1 
 
    r0=torch.where(torch.isnan(r),torch.full_like(r, 0), r)
    retw=r0.mul(w)
 
    mret=retw.sum(axis=1)
    if utility_function=='crra':   #指数效用函数
        utility=-torch.pow(1+mret,1-gamma)/(1-gamma)
        utilitymean=torch.mean(utility)
        
    elif utility_function=='MV':  #均值方差效用函数
        sigma=torch.var(mret)
        utilitymean=sigma.mul(gamma/2)-torch.mean(mret)
        
    tc=torch.mean(torch.sum(torch.abs(w[1:,:,:]-retw[:-1,:,:]),axis=1),axis=0) ##计算交易成本       
    if cost_type==False:
        loss=utilitymean
    elif cost_type==True:
        loss=utilitymean+0.005*tc
              
 
    ##加入正则化项    
    l1_reg = torch.tensor(0.)
    for param in net.parameters():
        # print(param)
        l1_reg += torch.sum(torch.abs(param))
    loss = loss+ lambda1 * l1_reg
    
    return loss

 

#记录列表（list），存储训练集和测试集上经过每一轮次，loss的变化
def train_model(train_iter,valid_iter,tc,tr,vc,vr,gamma,N,loss,input_size,hidden_size,num_classes,
                                               num_epochs,batch_size,params=None,lr=None,lambda1=None,optimizer=None,allow_short_selling=True,utility_function='crra',cost_type=False,seed=100):
    train_loss=[]
    valid_loss=[]
    
    p_list='FDNPP'+str(allow_short_selling)+utility_function+str(cost_type)+str(seed)
    
    early_stopping = EarlyStoppings(para_list=p_list,patience=7, verbose=True)
    
    for epoch in range(num_epochs):#外循环控制循环轮次
        #step1在训练集上，进行小批量梯度下降更新参数
                
        for c,r in train_iter:#内循环控制训练批次
            w_hat = net(c.to(torch.float32))
            l = loss(net,w_hat,r.to(torch.float32),gamma,N,lambda1,allow_short_selling=allow_short_selling,utility_function=utility_function,cost_type=cost_type)
            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
                    
            #梯度回传       
            l.backward()
            
            
            if optimizer is None:
                torch.optim.SGD(net.parameters(),lr,batch_size)
                # SGD(params,lr,batch_size)
            else:
                optimizer.step()            
 
        train_loss.append((loss(net,net(tc.to(torch.float32)),tr.to(torch.float32),gamma,N,lambda1,allow_short_selling=allow_short_selling,utility_function=utility_function,cost_type=cost_type)).item())#loss本身就默认了取平均值！
        valid_loss.append((loss(net,net(vc.to(torch.float32)),vr.to(torch.float32),gamma,N,lambda1,allow_short_selling=allow_short_selling,utility_function=utility_function,cost_type=cost_type)).item())
        
        print("epoch %d,train_loss %.6f,valid_loss %.6f"%(epoch+1,train_loss[epoch],valid_loss[epoch])) 
 
        valid_lossave = np.average(valid_loss)   
        early_stopping(valid_loss[epoch], net,para_list=p_list)
 
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    net.load_state_dict(torch.load(p_list+'checkpoint.pt'))
    
    return net, train_loss, valid_loss




def test_model(netopt,ec,er,N,allow_short_selling=True):
    w_hat=netopt(ec.to(torch.float32))
    weigh0=1/(~torch.isnan(er)).sum(axis=1)   ##计算等权投资组合的权重
    w_hat_mul=torch.where(torch.isnan(er),torch.full_like(er, np.nan), w_hat)
    w_hatm=torch.where(torch.isnan(er),torch.full_like(er, 0), w_hat)
    
    ws=torch.sum(w_hatm,axis=1)
    num=torch.sum(~torch.isnan(er),axis=1)
    wn=ws/num
    
    
    w0=torch.empty(w_hat.shape) 
    wh=torch.empty(w_hat.shape) 
        
    for j in range(w_hat.shape[0]):
        w0[j]=torch.where(torch.isnan(er[j]),torch.full_like(er[j], 0), weigh0[j])
        wh[j]=(w_hat_mul[j]-wn[j])/num[j]   #times
    

    wh=torch.where(torch.isnan(wh),torch.full_like(wh, 0), wh)    
    w=w0+wh
    
    if allow_short_selling==True:  ##没有权重约束
        w=w
    elif allow_short_selling==False: ##卖空约束
        w=w.clamp(0,1)
    w=w/torch.sum(w,axis=1).repeat(1,w.shape[1]).reshape(w.shape)   ##重新要求权重之和等于1   
    return w
 
    
def get_weights(i,data_list,ret,gamma,allow_short_selling,utility_function,cost_type,seed):
    
    '''
    i：以12为倍数
    '''
    batch_size =10  # 设置小批量大小  
    num_epochs = 100  #100

    trw=7*12
    viw=3*12
    tew=1*12
    N=data_list[0].shape[0]          ##股票数量 
   
    weight_index=ret.index[i+trw+viw:i+trw+viw+tew]
    
    input_size=data_list[0].shape[1]-3 ##特征数量
    
    hidden_size=32
    num_classes=1
    
    allow_short_selling=allow_short_selling
    utility_function=utility_function
    cost_type=cost_type
 
        
    ##训练集
    tc=torch.tensor(np.stack([x.iloc[:,2:-1] for x in data_list[i:i+trw]])) 
    tr=torch.tensor(np.stack([x.iloc[:,-1:] for x in data_list[i:i+trw]]))     
    
    #验证集
    vc=torch.tensor(np.stack([x.iloc[:,2:-1] for x in data_list[i+trw:i+trw+viw]])) 
    vr=torch.tensor(np.stack([x.iloc[:,-1:] for x in data_list[i+trw:i+trw+viw]]))
    
    
    #测试集
    ec=torch.tensor(np.stack([x.iloc[:,2:-1] for x in data_list[i+trw+viw:i+trw+viw+tew]])) 
    er=torch.tensor(np.stack([x.iloc[:,-1:] for x in data_list[i+trw+viw:i+trw+viw+tew]]))
    
    ##形成batch数据
    train_iter = load_batch([tc,tr], batch_size, N, is_train=False)
    valid_iter = load_batch([vc,vr], batch_size, N, is_train=False)
    

    
    if i==0:##首先仅考虑对第一期进行超参数调整
        data=data_list[i:i+trw+viw] ##用于训练和验证的数据
    
        kf=KFold(n_splits=5)
        umax=10000000  #最小效用
        global lambdaopt
        global lropt
        lambdaopt=0
        lropt=0.01
        
        lr_list=[0.01,0.001] # 正则化项的项的参数是待调参数，范围可以从 [0.01,0.1]
        lambda_list=[0.0001]
        # lambda_list=[0.00001,0.0001,0.001]
        for lambda1 in lambda_list:
            for lr in lr_list:
                print(lambda1,lr)
            
                ##五折交叉验证进行优化lambda
                valoss_list=[]
                for rets_index,retv_index in kf.split(data):
                    print(rets_index.shape,retv_index.shape)
                    
                    #训练集
                    ttc=torch.tensor(np.stack([data[x].iloc[:,2:-1] for x in rets_index])) 
                    ttr=torch.tensor(np.stack([data[x].iloc[:,-1:] for x in rets_index]))
                    
                    #验证集
                    vvc=torch.tensor(np.stack([data[x].iloc[:,2:-1] for x in retv_index])) 
                    vvr=torch.tensor(np.stack([data[x].iloc[:,-1:] for x in retv_index])) 
                    
                    tt_iter = load_batch([ttc,ttr], batch_size, N, is_train=False)
                    vv_iter = load_batch([vvc,vvr], batch_size, N, is_train=False)
                    
                    optimizer =torch.optim.Adam(net.parameters(),lr=lr,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)  
                    netopt, train_loss, valid_loss=train_model(tt_iter,vv_iter,ttc,ttr,vvc,vvr,gamma,N,loss,input_size,hidden_size,num_classes,
                                                               num_epochs,batch_size,params=None,lr=lr,lambda1=torch.tensor(lambda1),optimizer=optimizer,allow_short_selling=allow_short_selling,utility_function=utility_function,cost_type=cost_type,seed=seed)
                    valoss_list.append(valid_loss[-8])
                util_mean=np.mean(valoss_list)
                
                # print(util_mean)
                
                if util_mean<umax:
                    lambdaopt=lambda1
                    lropt=lr
                    umax=util_mean
                
        print('lambda',lambdaopt,'lr',lropt)
        pd.DataFrame([lambdaopt,lropt],index=['lambda1','lr']).to_csv('result/parameter/DFNPP'+'_'+str(allow_short_selling)+'_'+str(utility_function)+'_'+str(cost_type)+'.csv')
    
    
    

    
    # optimizer = torch.optim.SGD(net.parameters(),lr)
    optimizer =torch.optim.Adam(net.parameters(),lr=lropt,betas=(0.9, 0.999),eps=1e-08,weight_decay=0.0001,amsgrad=False)    
        
    
    netopt, train_loss, valid_loss=train_model(train_iter,valid_iter,tc,tr,vc,vr,gamma,N,loss,input_size,hidden_size,num_classes,
                                               num_epochs,batch_size,params=None,lr=lropt,lambda1=torch.tensor(lambdaopt),optimizer=optimizer,allow_short_selling=allow_short_selling,utility_function=utility_function,cost_type=cost_type,seed=seed)
    
    w=test_model(netopt,ec,er.to(torch.float32),N,allow_short_selling)
 
    path = 'result/weight_PPNN'+'_'+str(gamma)+'_'+str(allow_short_selling)+'_'+str(utility_function)+'_'+str(cost_type)+'_'+str(seed)
    if not os.path.exists(path):
        os.mkdir(path)
    pd.DataFrame(w.squeeze().detach().numpy(),index=weight_index,columns=ret.columns).to_csv(path+'/'+str(i)+'.csv')


 

def get_res(seed):
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch+CPU
    torch.cuda.manual_seed(seed)  # torch+GPU
 
    #1.原始模型 2.卖空约束 3.更换效用函数 
    para_list=[[5,True,'MV',False],[5,False,'MV',False],[10,True,'MV',False]] 
        
    for para in para_list:
        trw=7*12
        viw=3*12
        tew=1*12  
        for i in range(0,len(data_list)-trw-viw-tew+1,12):
            print(i)
            get_weights(i,data_list,ret,gamma=para[0],allow_short_selling=para[1],utility_function=para[2],cost_type=para[3],seed=seed)    
            
            
#%%#投资组合权重为5次结果的平均，采用多线性方法调用           
if __name__ == '__main__':
    
    value_list=[]
    for tim in range(100,105):
        value_list.append(tim)    
    n_jobs = len(value_list)
    joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(get_res)(d) for d in value_list)
    
    
 
    
    
    
    
    
 



