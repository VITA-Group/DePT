from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import pickle
import math
import itertools
import time
from time import time as timer
import os
from pprint import pprint as prt
from collections import OrderedDict




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class getMLP_11(nn.Module):
    def __init__(self, neurons, prep_str=''):
        super().__init__()
        assert neurons[0]==neurons[-1]==1
        self.mlp = getMLP(neurons, prep_str=prep_str)
    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.mlp(x).squeeze(-1)

class Prep(nn.Module):
    def __init__(self, prep_str):
        super().__init__()
        self.prep_str = prep_str
    def forward(self, x):
        x = eval('x'+self.prep_str)
        return x


def getMLP(neurons, activation=nn.GELU, bias=True, dropout=0.1, last_dropout=False, normfun='layernorm',prep_str=''):
    nn_list = [Prep(prep_str)]
    if len(neurons) in [0,1]:
        return nn.Identity()
    if len(neurons) == 2:
        return nn.Linear(*neurons)

    nn_list = []
    n = len(neurons)-1
    for i in range(n-1):
        if normfun=='layernorm':
            norm = nn.LayerNorm(neurons[i+1])
        elif normfun=='batchnorm':
            norm = nn.BatchNorm1d(neurons[i+1])
        else:
            norm = nn.Identity()
        nn_list.extend([nn.Linear(neurons[i], neurons[i+1], bias=bias), norm, activation(), nn.Dropout(dropout)])
    
    nn_list.extend([nn.Linear(neurons[n-1], neurons[n], bias=bias)])
    if last_dropout:
        nn_list.extend([nn.Dropout(dropout)])
    mlp = nn.Sequential(*nn_list)
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    return mlp



def shuffle_Xy(X,y):
    # input are all list, difficult to use index
    # usage:
    # X=[1,2,3,4]
    # y=[10,20,30,40,]
    # print(X,y)
    # X, y = shuffle_Xy(X, y)
    # print(X,y)
    z=list(zip(X,y))
    np.random.shuffle(z)
    X,y = zip(*z)
    return X,y


def get_infinite_iter(dataset, batch_size=1,num_workers=2,shuffle=True,sampler=None, pin_memory=None, **args):
    if sampler is None:
        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))))
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(list(range(len(dataset))))
    
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler = sampler,
            # shuffle=True,
            num_workers=num_workers,
            persistent_workers = True if num_workers>0 else False,            
            pin_memory=pin_memory,
            # collate_fn=dataset.collate_fn,
        )
    
    dataloaderIter = dataloader.__iter__()
   
    class InfIter(dataloaderIter.__class__):
        def __init__(self,loader):
            super().__init__(loader)
            self.loader = loader
        def __next__(self):

            try:
                return super().__next__()

            except StopIteration:
                self._reset(self.loader)
                return super().__next__()

        def sample(self):
            data_batch, label_batch = self.__next__()
            return data_batch.to(DEVICE), label_batch.to(DEVICE)

    infIter = InfIter(dataloader)
    return infIter





def plot_ci(arr, vx=[], is_std=True, ttl='', xlb='',ylb='',semilogy=False, viz_un_log=False):
    arr = np.asarray(arr)
    if len(arr.shape)==1:  arr = arr.reshape(1,-1)
    rdcolor = plt.get_cmap('viridis')(np.random.rand())  # 随机颜色

    mean = np.mean(arr,axis=0)
    if is_std:
        ci = np.std(arr,axis=0)
        lowci = mean-ci*is_std
        hici = mean+ci*is_std
    else:
        lowci = np.min(arr,axis=0)
        hici = np.max(arr,axis=0)
    # plt.plot(mean, color = '#539caf')
    if viz_un_log:
        mean=np.exp(mean)
        lowci=np.exp(lowci)
        hici=np.exp(hici)
    if vx == []:
        vx_=np.arange(len(mean))
    if semilogy:
        plt.semilogy(vx_, mean, color = rdcolor)
    else:
        plt.plot(vx_, mean, color = rdcolor)
    plt.fill_between(vx_, lowci, hici, color = rdcolor, alpha = 0.4)
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    if list(vx): plt.xticks(vx)
    plt.title(ttl)
    return




last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
    TOTAL_BAR_LENGTH = 65.
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    # L.append('  Step: %s' % format_time(step_time))
    # L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f











def load_model(net, cwd, verbose=True, strict=True, multiGPU=False):
    def load_multiGPUModel(network, cwd):
        network_dict = torch.load(cwd)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in network_dict.items():
            namekey = k[7:] # remove `module.`
            new_state_dict[namekey] = v
        # load params
        network.load_state_dict(new_state_dict)

    def load_singleGPUModel(network, cwd):
        network_dict = torch.load(cwd, map_location=lambda storage, loc: storage)
        network.load_state_dict(network_dict, strict=strict)

    if os.path.exists(cwd):
        if not multiGPU:
            load_singleGPUModel(net, cwd)
        else:
            load_multiGPUModel(net, cwd)

        if verbose: print(f"---››››  LOAD success! from {cwd}\n\n\n")
    else:
        if verbose: print(f"---››››  !!! FileNotFound when load_model: {cwd}\n\n\n")


def save_model(net, cwd):  # 2020-05-20
    torch.save(net.state_dict(), cwd)
    print(f"‹‹‹‹‹‹‹---  Saved @ :{cwd}\n\n\n")



def bestGPU(gpu_verbose=True, **w):

    import GPUtil
    Gpus = GPUtil.getGPUs()
    Ngpu = 4
    mems, loads = [], []
    for ig, gpu in enumerate(Gpus):
        memUtil = gpu.memoryUtil*100
        load = gpu.load*100
        mems.append(memUtil)
        loads.append(load)
        if gpu_verbose: 
            print(f'gpu-{ig}:   Memory: {memUtil:.2f}%   |   load: {load:.2f}% ')
    bestMem = np.argmin(mems)
    bestLoad = np.argmin(loads)
    best = bestMem
    if gpu_verbose: print(f'//////   Will Use GPU - {best}  //////')
    # print(type(best))

    return int(best)



def lp(fname):
    with open(fname,'rb') as f:
        dic = pickle.load(f)
    return dic


def sp(fname,dic):
    with open(fname,'wb') as f:
        pickle.dump(dic, f)
    return






