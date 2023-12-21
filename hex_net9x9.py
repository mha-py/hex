from hex_helpers import *
from hex_train_helpers import *
from hex_mcts import *
from hex_pathfinding import *


def _getspiral(N):
    'Gibt eine 2D-Spirale zurück'
    i, j = 0, 0
    l = [(0,0)]
    maxi, maxj, mini, minj = 0, 0, 0, 0
    while True:
        while i < maxi+1:
            i += +1
            l += [(i,j)]
        maxi += 1
        
        while j < maxj+1:
            j += +1
            l += [(i,j)]
        maxj += 1
        if len(l) >= N**2:
            break
        
        while i > mini-1:
            i += -1
            l += [(i,j)]
        mini += -1
        
        while j > minj-1:
            j += -1
            l += [(i,j)]
        minj += -1
        if len(l) >= N**2:
            break
            
    l = l[:N**2]
    l = [ (i-mini, j-minj) for (i,j) in l ] # 2D coordinates from inner to outer
    l2 = [ (N*i+j) for (i,j) in l ]      # flattened coordinates form inner to outer
    return l2

_spirals = [ _getspiral(k) for k in range(20) ]



act = torch.nn.ReLU()



#  These are changed compared to hex_nnet version for 6x6 games!
class ResBlock(nn.Module):
    def __init__(self, n, sz=3, bn=True):
        'Like a ResNet Block but without the residual yet added'
        super().__init__()
        self.bn = bn
        if bn:
            self.conv1 = nn.Conv2d(n, n, sz, padding=sz//2)
            self.conv2 = nn.Conv2d(n, n, sz, padding=sz//2)
            self.bn1 = nn.BatchNorm2d(n)
            self.bn2 = nn.BatchNorm2d(n)
        else:
            self.conv1 = nn.Conv2d(n, n, sz, padding=sz//2)
            self.conv2 = nn.Conv2d(n, n, sz, padding=sz//2)
            
        
    def forward(self, x):
        x0 = x
        if self.bn:
            x = self.bn1(x)
            x = act(x)
            x = self.conv1(x)
            x = self.bn2(x)
            x = act(x)
            x = self.conv2(x)
        else:
            x = act(x)
            x = self.conv1(x)
            x = act(x)
            x = self.conv2(x)
        return x0 + x
    
    

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
        x  = act(x)
        x  = self.conv2(x)
        x  = xr + x
        return x
    
    
class ResBlockUp(nn.Module):
    def __init__(self, nin, nout):
        'Upscaling Resnet Block (actually not a Resnet Block)'
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
    def forward(self, x):
        x = self.conv(x)
        return x
    
    
    
    
    
    
def expandtoeven(x):
    'Expands a tensor such that its multiple of two'
    b, c, h, w = x.shape
    if h%2==0:
        return x
    return torch.nn.functional.pad(x, (0, 1, 0, 1))

def addskip(x, xskip):
    'Adds x and skip connection, adjusts the shape to the skip connection'
    b, c, h, w = xskip.shape
    b, c, hp,wp= x.shape
    r = (hp-h)//2
    s = hp-h - r
    # r and s add up to hp-h = wp-w
    x = x[:,:,r:-s,r:-s]
    return x+xskip
    
    
    
    


class NetAdaptive(nn.Module):
    def __init__(self, path=None, bsize=9, n=64):
        super().__init__()
        
        ksize = 3
        # shared body
        self.conv1 = nn.Conv2d(5, n, ksize, padding=ksize//2)
        '''self.resblock1 = ResBlock(n, sz=ksize)
        self.resblock2 = ResBlock(n, sz=ksize)
        self.resblock3 = ResBlock(n, sz=ksize)
        self.resblock4 = ResBlock(n, sz=ksize)
        self.resblock5 = ResBlock(n, sz=ksize)
        self.resblock6 = ResBlock(n, sz=ksize)
        self.resblock7 = ResBlock(n, sz=ksize)'''
        
        self.resdown_0to1 = ResBlockDown(n, 2*n)
        self.resdown_1to2 = ResBlockDown(2*n, 2*n)
        
        self.resblock01 = ResBlock(n, sz=ksize)
        self.resblock11 = ResBlock(2*n, sz=ksize)
        self.resblock12 = ResBlock(2*n, sz=ksize)
        self.resblock21 = ResBlock(2*n, sz=ksize)
        self.resblock22 = ResBlock(2*n, sz=ksize)
        
        
        # policy head
        self.policy_resup_2to1 = ResBlockUp(2*n, 2*n)
        self.policy_resup_1to0 = ResBlockUp(2*n, n)
        self.policy_last = nn.Conv2d(n, 1, 3, padding=1)
        
        # value head
        self.v_dense1 = nn.Linear(225 * n, 512)
        self.v_dense2 = nn.Linear(512, 256)
        self.v_dense3 = nn.Linear(256, 100)
        
        self.dropout = 0.
        
        # mix of values (linear combination of 100 values)
        self.alphasV = nn.Linear(100, 1)
        ## self.alphasV.data.fill_(0.0) # init as zeros
        self.alphas = nn.Parameter(torch.ones(1, 100))
        
        
        # Move network to GPU
        if GPU:
            self.cuda()
            
        if path is not None:
            self.load_state_dict(torch.load(path), strict=False)
            
    '''
    def mix(self, y):
        # linear combination of many addvs into one V, and rescale additional vs
        V = tanh(self.alphasV(y))
        addv = tanh(self.alphas*10 * y)
        return V, addv # has big V in it
        '''
        
    def forward(self, x, validmask=1):
        
        x = act(self.conv1(x))
        x = self.resblock01(x)
        xskip0 = x
        x = expandtoeven(x)
        x = self.resdown_0to1(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        xskip1 = x
        x = expandtoeven(x)
        x = self.resdown_1to2(x)
        x = self.resblock21(x)
        x = self.resblock22(x)
        
        
        # p-Head
        y = x
        y = self.policy_resup_2to1(y)
        y = addskip(y, xskip1)
        y = self.policy_resup_1to0(y)
        y = addskip(y, xskip0)
        y = self.policy_last(y)
        
        y = validmask*y + (1-validmask)*(-10**10) # herausmaskieren
            
        b, c, h, w = y.shape
        p = softmax(y.view(b, c*h*w))
        p = p.view(b, c, h, w)
        
        # v-Head
        n, c, h, w = x.shape
        y = x.view(n, c, h*w) # flattened in space dimensions
        indices = torch.tensor(_spirals[w]).cuda() # take spiral for N**2 (here it is N=w) and use as indices
        y = torch.index_select(y, 2, indices)
        
        y = torch.nn.functional.linear(y.view(n, c*h*w), self.v_dense1.weight[:, :c*h*w], self.v_dense1.bias) # adjust the number of weights to the input number of channels
        y = act(y)
        
        if self.dropout>0: y = torch.nn.functional.dropout(y, self.dropout)
        y = act(self.v_dense2(y))
        if self.dropout>0: y = torch.nn.functional.dropout(y, self.dropout)
        y = self.v_dense3(y)
        v = tanh(y)
        
        # Additional v tuned
        # linear combination of many addvs into one V, and rescale additional vs
        y = y.detach()
        V = tanh(self.alphasV(y))
        vtuned = tanh(self.alphas*10 * y)
        
        return p, v, vtuned, V
    
    
    def preprocess(self, board):
        'Preprocess board, i. e. encode one hot and add additional features to ease net calculations'
        N = len(board)
        x = np.zeros((5, N, N))
        for n in range(3):
            x[n, :, :] = (board%3 == n)
       ## x[3] = voltage(board.T).T
       ## x[4] = voltage(-board)
        vd = (board==0)[None,:]
        return x, vd
    
    
    def preprocessStack(self, boards):
        'Preprocess a stack of boards, i. e. encode one hot and add additional features to ease net calculations'
        xs = []
        vds = []
        for b in boards:
            x, vd = self.preprocess(b)
            xs.append(x)
            vds.append(vd)
        return xs, vds
    
    
    def predict(self, board):
        'Prediction for a single board'
        p, v = self.predictStack([board])
        return p[0], v[0]
    
    
    def predictStack(self, boards):
        'Version Normal, ohne bigV und vtuned'
        'Does a prediction on a stack of states'
        xs, vd = self.preprocessStack(boards)
        xs, vd = np2t(xs, vd)
            
        # Evaluat at nn
        with torch.no_grad():
            p, v, vtuned, V = self(xs, vd)
            
            # Mix many vs together into one main V
            # replace 0th value with main value "big V".
            #v = torch.roll(vtuned, 1, 1) # shift one to the right in dim=1
            ## v[:,0] = V[:,0]
            
        return t2np(p[:,0,:,:], vtuned)
    
    