import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .model import *


def groundTruth(x,y):
    x,y = torch.tensor(x).float(), torch.tensor(y).float()
    data = torch.cat([x,y], 1)
    A = torch.tensor([[1,-0.1],[2.5, 0.7]])
    xA = torch.mm(x,A)
    xAy = torch.tensor([torch.dot(p1,p2) for p1, p2 in zip(xA,y)]).reshape(len(x),1)
    label = xAy
    return data,label

def rand(sp,lo,hi):
    return torch.tensor(np.random.rand(*sp)*(hi-lo)+lo, dtype=torch.float32)



def get_X_y(lo=-1,hi=2,my_type=None,in_shape=[1], balance=0, N=200):
# def get_X_y(lo=-5,hi=5,N=200):
    if my_type is None:
        my_type = [ 'bilinear',
                'x2',
                '-x2',
                'kx',
                'original',
                '1/x',
                'sinx',
                'decayer',
                'const_-1',
                'causDecay_6x6',
                'timeDecay',
                ] [-1]

    if my_type=='bilinear':
        return groundTruth(rand((N,2),lo,hi),rand((N,2),lo,hi))
    if my_type=='x2':
        data = rand([N,1],lo,hi)
        label = (data**2)
        return data, label
    if my_type=='-x2':
        data = rand([N,1],lo,hi)
        label = -(data**2)
        return data, label
    if my_type=='kx':
        data = rand([N,1],lo,hi)
        label = (data+3)
        return data, label
    if my_type=='1/x':
        data1 = rand([N,1],lo,-0.1)
        data2 = rand([N,1],0.1,hi)
        data = torch.cat([data1,data2],dim=0)
        label = (1/data)
        return data, label
    if my_type=='sinx':
        data = rand([N,1],lo,hi)
        label = torch.sin(data*4)
        return data, label
    if my_type=='causDecay_newyork':
        data = rand([N,1],lo,hi)
        data_ = data/1e4
        data_ = data_ + 0.15
        label = -data_*2 - F.softplus(-data_*600)*0.02 + balance
        return data, label
    if my_type=='causDecay_6x6':
        data = rand([N,1],lo,hi)
        data_ = data/1e4
        label = -data_*20 - F.softplus(-data_*100)*0.6 + balance
        # label =  -F.softplus(-data_*1)*1
        return data, label
    if 'const' in my_type:
        in_shape = [N] + in_shape        
        data = rand(in_shape,lo,hi)
        v = my_type.split('_')[-1]
        label = data*0+float(v)
        return data, label
    else:
        raise NotImplementedError




class DecayFun(nn.Module):
    def __init__(self, N_decayer, ini_range, decay_k_ini, decayer_train_beta, decayers_beta_ini,**w):
        super().__init__()
        

        decayers = torch.linspace(ini_range[0],ini_range[1],N_decayer, device=DEVICE)
        ks = torch.ones(N_decayer, device=DEVICE)*decay_k_ini
        betas = torch.ones(N_decayer, device=DEVICE)*decayers_beta_ini

        self.N_decayer = N_decayer
        self.decayers = nn.Parameter(decayers, requires_grad=True)
        self.ks = nn.Parameter(ks, requires_grad=True)
        self.betas = nn.Parameter(betas, requires_grad=bool(decayer_train_beta))


    def forward(self, x):
        # x:        whatever shape
        # return:   same as input
        res = x*0
        for i in range(self.N_decayer):
            d = self.decayers[i]
            k = self.ks[i]
            beta = self.betas[i]
            if d>=0:
                res += F.softplus(x-d, beta.item())*k
            else:
                res += F.softplus(d-x, beta.item())*k
        return res






def pretrain_dept_decayer(dname, lo=-1,hi=2,my_type=None,n_feature=1,prep_str='',N_epochs=500, lr = 0.01, balance=0):
    n_hidden=(20,20)
    n_output=1
    plt.ion()  
    plt.show()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = getMLP([n_feature]+list(n_hidden)+[n_output],prep_str=prep_str)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # original lr = 0.2
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    rLoss=[]
    rLossTest = []
    for t in range(N_epochs):
        data, label = get_X_y(lo,hi,my_type,[n_feature],balance)

        prediction = net(data)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
    
        if t % 20 == 0:
    
            if type(net)==DecayFun:
                print('\n   decayers:\n',net.decayers)
                print('\n   ks:\n',net.ks)
                print('\n  betas\n',net.betas)
    
    
            data, label = get_X_y(lo,hi,my_type,[n_feature],balance)
            prediction = net(data)
            plt.cla()
            plt.plot(data.data.numpy(), label.data.numpy(),'|', label='GT')
            
            plt.plot(data.data.numpy(), prediction.data.numpy(), 'r.', lw=1, label='Pred')
            plt.text(data.data.numpy().reshape(-1)[0], label.data.numpy().reshape(-1)[0], f'Loss={ loss.data.numpy():.3f}, @{t}', fontdict={'size': 20, 'color':  'blue'})
            if type(net)==DecayFun:
                dec = net.decayers.detach().numpy()
                x=np.linspace(-5,5,len(dec))
                plt.plot(x,dec,'x-')
                plt.plot(x,x*0-1)
                plt.plot(x,x*0+2)
            plt.pause(0.1)
    torch.save(net.state_dict(), f'{dname}')


if __name__ == "__main":
    if which_roadmap=='newyork':
        height, width = 7, 28
        lohi = [-1e4,1e4]
    dname = f'pre_m1@{which_roadmap}'
    pretrain_dept_decayer(dname, lo=lohi[0],hi=lohi[1],my_type=f'coneDecay_{which_roadmap}',n_feature=1,prep_str='/1e4', N_epochs=1000, lr = 0.02,balance=10)


