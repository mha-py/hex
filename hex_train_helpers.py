

import torch
import numpy as np



GPU = True



##### Batch generator #####


def encode_oh(x, N=6):
    'Encodes the board in one hot format (blue stone, red stone, no stone)'
    if N==6:
        x_oh = np.zeros((3, N, N))
        for n in range(3):
            x_oh[n, :, :] = (x == n)  # eigentlich x%3!! aber das netz kann damit umgehen!
        return x_oh
    else:
        x_oh = np.zeros((3, N, N))
        for n in range(3):
            x_oh[n, :, :] = (x%3 == n)  # fehler behoben!
        assert np.all(np.sum(x_oh, 0)==1)
        return x_oh
        
    

def np2t(*args):
    'Converts a numpy array to a torch array'
    res = [torch.from_numpy(np.array(x, dtype='float32')) for x in args]
    if GPU:
        res = [x.cuda() for x in res]
        
    if len(res)==1:
        return res[0]
    else:
        return res


def t2np(*args):
    'Converts a torch array to a numpy array'
    res = [x.detach().cpu().numpy() for x in args]
    
    if len(res)==1:
        return res[0]
    else:
        return res


batch2torch = np2t


def onehot(k, n):
    'One hot vector of length n, one at entry k.'
    res = np.zeros(n, dtype='int32')
    res[k] = 1.
    return res


def onehot2D(ind, shape):
    'One hot vector of length n, one at entry k.'
    res = np.zeros(shape)
    res[ind] = 1.
    return res


def update_mt(mt, n, tau):
    # updates the mean teacher by the network
    mtdict = mt.state_dict()
    ndict = n.state_dict()
    for k in mtdict.keys():
        mtdict[k] = tau * mtdict[k] + (1-tau) * ndict[k]
    mt.load_state_dict(mtdict)
    
def average_net(target, net1, net2, t=.5):
    # averages weights of two nets
    d1 = net1.state_dict()
    d2 = net2.state_dict()
    for k in d1.keys():
        d1[k] = (1-t)*d1[k] + t*d2[k]
    target.load_state_dict(d1)
    
def reinit_net(net):
    # reinitializes the weights of a net
    for layer in net.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


            
# Taken from https://github.com/alphadl/lookahead.pytorch/blob/master/lookahead.py
from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
    
    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)
            
            
            
            
            
            
'''
##### Neural Network #####


import torch
from torch import nn
from tqdm.notebook import tqdm


relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()
avgpool = nn.AvgPool2d(2)
softmax = torch.nn.Softmax(dim=1)
tanh = torch.nn.Tanh()

def softmax2d(y):
    'Softmax for 1xhxw applications'
    # softmax = torch.nn.Softmax(dim=1)
    b, c, h, w = y.shape
    y = y.view(b, c*h*w)
    y = softmax(y)
    y = y.view(b, c, h, w)
    return y


#### Block Definitions ####

class ResBlock(nn.Module):
    def __init__(self, n, sz=3, bn=True):
        'Simple Resnet Block with two convolutionals'
        super().__init__()
        self.bn = bn
        if bn:
            self.conv1 = nn.Conv2d(n, n, sz, padding=sz//2, bias=False)
            self.conv2 = nn.Conv2d(n, n, sz, padding=sz//2, bias=False)
            self.bn = nn.BatchNorm2d(n)
        else:
            self.conv1 = nn.Conv2d(n, n, sz, padding=sz//2, bias=True)
            self.conv2 = nn.Conv2d(n, n, sz, padding=sz//2, bias=False)
            
        
    def forward(self, x):
        if self.bn:
            xr = x
            x = self.conv1(x)
            x = self.bn(x)
            x = relu(x)
            x = self.conv2(x)
            x = xr + x
        else:
            xr = x
            x = self.conv1(x)
            x = relu(x)
            x = self.conv2(x)
            x = xr + x
        return x
    
    
class ResBlockDown(nn.Module):
    def __init__(self, nin, nout):
        'Downscaling Resnet Block'
        super().__init__()
        self.conv_res = nn.Conv2d(nin, nout, 1)
        self.conv1 = nn.Conv2d(nin, nout, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(nout, nout, 3, padding=1)
        
    def forward(self, x):
        xr = avgpool(x)
        xr = self.conv_res(xr)
        x  = self.conv1(x)
        x  = relu(x)
        x  = self.conv2(x)
        x  = xr + x
        return x
    
    
class ResBlockUp(nn.Module):
    def __init__(self, nin, nout):
        'Upscaling Resnet Block'
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
    def forward(self, x):
        x = self.conv(x)
        return x

    


    
# Medium CNN Net

class Net(nn.Module):
    def __init__(self, n=64, ksize=3, bn=True, dropout_rate=0.2):
        super().__init__()
        
        # Initiate all layers of this network
        # shared layers
        self.conv1 = nn.Conv2d(3, n, 3, padding=1)
        self.resblock1 = ResBlock(n, sz=ksize, bn=bn)
        self.resblock21 = ResBlock(2*n, sz=ksize, bn=bn)
        self.resblock22 = ResBlock(2*n, sz=ksize, bn=bn)
        self.resblock23 = ResBlock(2*n, sz=ksize, bn=bn)
        self.down1 = ResBlockDown(n, 2*n)
        
        # policy head
        self.p_resblock3 = ResBlock(n, sz=ksize, bn=bn)
        self.p_up1 = ResBlockUp(2*n, n)
        self.p_conv_m1 = nn.Conv2d(n, 1, 3, padding=1)
        
        # value head
        self.v_dense1 = nn.Linear((N//2)**2 * 2*n, 1024)
        self.v_dense2 = nn.Linear(1024, 512)
        self.v_dense3 = nn.Linear(512, 1)
        self.v_dropout = nn.Dropout(p=dropout_rate)
        
        
        # Move network to GPU
        if GPU:
            self.cuda()
        
    def forward(self, x, use_softmax=True):
        x = relu(self.conv1(x))
        x = self.resblock1(x)
        
        xskip_1 = x
        
        x = self.down1(x)
        x = self.resblock21(x)
        x = self.resblock22(x)
        x = self.resblock23(x)
        
        # p-Block
        y1 = self.p_up1(x)
        y1 += xskip_1
        y1 = self.p_resblock3(y1)
        
        y1 = self.p_conv_m1(y1)
        if use_softmax:
            b, c, h, w = y1.shape
            y1 = y1.view(b, c*h*w)
            y1 = softmax(y1)
            y1 = y1.view(b, c, h, w)
        else:
            y1 = sigmoid(y1)
        
        # v-Block
        n, c, h, w = x.shape
        x = x.view(n, c*h*w)
        y2 = relu(self.v_dense1(x))
        y2 = self.v_dropout(y2)
        y2 = relu(self.v_dense2(y2))
        y2 = self.v_dropout(y2)
        y2 = tanh(self.v_dense3(y2))
        
        return y1, y2
    
    
    def predict(self, numpy_x):
        'Takes a numpy array and give out one, i. e. 10x10 -> 10x10'
        if len(numpy_x)==2:
            numpy_x = numpy_x[None,:]
        x = torch.from_numpy(numpy_x[None,:].astype('float32'))
        if GPU:
            x = x.cuda()
        y1, y2 = self(x)
        y1 = y1[0,0,:,:].detach().cpu().numpy()
        y2 = y2[0].detach().cpu().numpy()
        return y1, y2
    

    
#### Shallow CNN ####

class Net_shallow(nn.Module):
    def __init__(self, n=48, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, n, 2, stride=2)
        self.conv2 = nn.Conv2d(n, n, 3, padding=1)
        self.conv3 = nn.ConvTranspose2d(n, 1, 2, stride=2)
        
        # value head
        self.v_dense1 = nn.Linear((N//2)**2 * n, 512)
        self.v_dense2 = nn.Linear(512, 1)
        self.v_dropout = nn.Dropout(p=dropout_rate)
        # Move network to GPU
        if GPU:
            self.cuda()
        
    def forward(self, x, use_softmax=True):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        
        # p-Block
        y1 = self.conv3(x)
        if use_softmax:
            y1 = softmax2d(y1)
        else:
            y1 = sigmoid(y1)
        
        # v-Block
        n, c, h, w = x.shape
        x = x.view(n, c*h*w)
        y2 = relu(self.v_dense1(x))
        y2 = self.v_dropout(y2)
        y2 = tanh(self.v_dense2(y2))
            
        return y1, y2
    
    def predict(self, numpy_x):
        'Takes a numpy array and give out one, i. e. 10x10 -> 10x10'
        if len(numpy_x)==2:
            numpy_x = numpy_x[None,:]
        x = torch.from_numpy(numpy_x[None,:].astype('float32'))
        if GPU:
            x = x.cuda()
        y1, y2 = self(x)
        y1 = y1[0,0,:,:].detach().cpu().numpy()
        y2 = y2[0].detach().cpu().numpy()
        return y1, y2
    

class Net_Dense(nn.Module):
    def __init__(self, n=256):
        super().__init__()
        self.dense1 = nn.Linear(3*6*6, n)
        self.dense2 = nn.Linear(n, n)
        self.dense3 = nn.Linear(n, n)
        self.dense_p = nn.Linear(n, 6*6)
        self.dense_v = nn.Linear(n, 1)
        # Move network to GPU
        if GPU:
            self.cuda()
        
    def forward(self, x):
        x = x.view(-1, 3*6*6)
        x = relu(self.dense1(x))
        x = relu(self.dense2(x))
        x = relu(self.dense3(x))
        p = softmax(self.dense_p(x)).view(-1, 1, 6, 6)
        v = tanh(self.dense_v(x))
        return p, v
    
    def predict(self, numpy_x):
        'Takes a numpy array and give out one, i. e. 10x10 -> 10x10'
        if len(numpy_x)==2:
            numpy_x = numpy_x[None,:]
        x = torch.from_numpy(numpy_x[None,:].astype('float32'))
        if GPU:
            x = x.cuda()
        y1, y2 = self(x)
        y1 = y1[0,0,:,:].detach().cpu().numpy()
        y2 = y2[0].detach().cpu().numpy()
        return y1, y2
'''