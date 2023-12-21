'''
This file provides a simple neural network heuristic for the minimax ai.
(c) 3.9.2020 mha
'''

import numpy as np
import pickle
import torch
from torch import nn
from hex_train_helpers import *

N = 6
GPU = True
if not GPU:
    print('Warning! GPU is off!')
    


relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()
avgpool = nn.AvgPool2d(2)
softmax = torch.nn.Softmax(dim=1)
tanh = torch.nn.Tanh()


def softmax2d(y):
    'Softmax for bxcxhxw applications'
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
    
    
######### RETRAIN: dropout rausnehmen, encode fehler beheben, use softmax wegnehmen als option
######### umbenennen encode zu preprocess
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
    
    @staticmethod
    def encode_oh(x):
        'Encodes the board in one hot format (blue stone, red stone, no stone)'
        x_oh = np.zeros((3, N, N))
        for n in range(3):
            x_oh[n, :, :] = (x == n)
        return x_oh
    
    
    def predict(self, numpy_x):
        'Takes a numpy array and give out v and a p array.'
        numpy_x = self.encode_oh(numpy_x)
        x = torch.from_numpy(numpy_x[None,:].astype('float32'))
        if GPU:
            x = x.cuda()
        y1, y2 = self(x)
        y1 = y1[0,0,:,:].detach().cpu().numpy()
        y2 = y2.item()
        return y1, y2
    
    
    def predictStack(self, numpy_x):
        'Does a prediction on a stack of states'
        b = len(numpy_x)
        xs = []
        for nx in numpy_x:
            xs += [self.encode_oh(nx)]
        xs = np.asarray(xs)
        xs = torch.from_numpy(xs.astype('float32'))
        if GPU:
            xs = xs.cuda()
        y1, y2 = self(xs)
        y1 = y1[:,0,:,:].detach().cpu().numpy()
        y2 = y2[:,0].detach().cpu().numpy()
        return y1, y2
    
    


class HexHeuristicsNNet:
    '''Heuristic for the MiniMax Class.
    Wraps the neural network and caches its results
    '''
    def __init__(self):
        self.hash_v = dict()
        self.hash_p = dict()
        
        self.nnet = Net()
        self.nnet.load_state_dict(torch.load('net.dat'))
        self.nnet.eval()
        
    def getHeuristicValue(self, board):
        '''Returns the heuristic value of the board. Values will be hashed.'''
        s = board.tostring()
        if s not in self.hash_v:
            self._createHash(board)
        return self.hash_v[s]
    
    def getSortedMoves(self, board, moves, red):
        '''Returns a list of moves which are heuristically sorted from best to worst'''
        s = board.tostring()
        if s not in self.hash_p:
            self._createHash(board)
        p = self.hash_p[s]
        return sorted(moves, key=lambda m: p[m[0],m[1]], reverse=True)
        
    def _createHash(self, board):
        '''Calculates the heuristic value and the onpath information for the board and saves into the hash tables.'''
        p, v = self.nnet.predict(board)
        s = board.tostring()
        self.hash_p[s] = p
        self.hash_v[s] = v