from hex_helpers import *
from hex_train_helpers import *
from hex_mcts import *
from hex_pathfinding import *

N = 13
bsize = 13


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



class Net13x13(nn.Module):
    def __init__(self, path=None, n=64, ksize=3):
        super().__init__()
        
        # Initiate all layers of this network
        # shared layers
        self.conv1 = nn.Conv2d(5, n, 3, padding=1)
        self.down1 = ResBlockDown(n, 2*n)
        self.down2 = ResBlockDown(2*n, 4*n)
        self.resblock11 = ResBlock(n, sz=ksize)
        self.resblock21 = ResBlock(2*n, sz=ksize)
        self.resblock31 = ResBlock(4*n, sz=ksize)
        self.resblock32 = ResBlock(4*n, sz=ksize)
        
        # policy head
        self.p_resblock = ResBlock(n, sz=ksize)
        self.p_up1 = ResBlockUp(2*n, n)
        self.p_up2 = ResBlockUp(4*n, 2*n)
        self.p_conv_m1 = nn.Conv2d(n, 1, 3, padding=1)
        
        # value head
        self.v_dense1 = nn.Linear(64*n, 40*n)
        self.v_dense2 = nn.Linear(40*n, 20*n)
        self.v_dense3 = nn.Linear(20*n, 100)
        
        self.dropout = 0.
        
        # mix of values (linear combination of 100 values)
        self.alphasV = nn.Linear(100, 1)
       ## self.alphasV.data.fill_(0.0) # init as zeros
        self.alphas = nn.Parameter(torch.Tensor(1, 100))
        
        
        # Move network to GPU
        if GPU:
            self.cuda()
            
        if path is not None:
            self.load_state_dict(torch.load(path))
            
    '''
    def mix(self, y):
        # linear combination of many addvs into one V, and rescale additional vs
        V = tanh(self.alphasV(y))
        addv = tanh(self.alphas*10 * y)
        return V, addv # has big V in it
        '''
        
    def forward(self, x, validmasks):
        
        # Padding auf richtige Größe, zweier Potenz
        _x = x
        x = torch.zeros((len(x), 5, 16, 16))
        if GPU: x = x.cuda()
        # rot
        x[:,1,:,:1] = +1
        x[:,1,:,N+1:] = +1
        # blau
        x[:,2,:1,:] = +1
        x[:,2,N+1:,:] = +1
        x[:, :, 1:N+1, 1:N+1] = _x
            
        vdm = torch.zeros((len(x), 1, 16, 16))
        if GPU: vdm = vdm.cuda()
        vdm[:, :, 1:N+1, 1:N+1] = validmasks
                
        
        x = act(self.conv1(x))
        x = self.resblock11(x)
        
        xskip_1 = x
        x = self.down1(x)
        x = self.resblock21(x)
        
        xskip_2 = x
        x = self.down2(x)
        x = self.resblock31(x)
        x = self.resblock32(x)
        
        # p-Head
        # x.shape = 256, 2, 2
        y = self.p_up2(x)
        y += xskip_2
        # y.shape = 128, 4, 4
        
        y = self.p_up1(y)
        y += xskip_1
        y = self.p_resblock(y)
        # y.shape = 64, 8, 8
        
        y = self.p_conv_m1(y)
        # y.shape = 1, 8, 8
        
        if type(validmasks) is not type(None):
            y = vdm*y + (1-vdm)*(-10**10) # herausmaskieren
            
        b, c, h, w = y.shape
        y = y.view(b, c*h*w)
        y = softmax(y)
        y = y.view(b, c, h, w)
        p = y
        p = p[:,:,1:N+1,1:N+1]
        
        # v-Head
        n, c, h, w = x.shape
        y = x.view(n, c*h*w)
        if self.dropout>0: y = torch.nn.functional.dropout(y, self.dropout)
        y = relu(self.v_dense1(y))
        if self.dropout>0: y = torch.nn.functional.dropout(y, self.dropout)
        y = relu(self.v_dense2(y))
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
        x = np.zeros((5, N, N))
        for n in range(3):
            x[n, :, :] = (board%3 == n)
        x[3] = voltage(board.T).T
        x[4] = voltage(-board)
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
        'Does a prediction on a stack of states'
        xs, vd = self.preprocessStack(boards)
        xs, vd = np2t(xs, vd)
            
        # Evaluat at nn
        with torch.no_grad():
            p, v, vtuned, V = self(xs, vd)
            
            # Mix many vs together into one main V
            # replace 0th value with main value "big V".
            v = torch.roll(vtuned, 1, 1) # shift one to the right in dim=1
            v[:,0] = V[:,0]
            
        return t2np(p[:,0,:,:], v)
    
    