


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

    



    
#### Medium CNN Net ####

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
    





### TODO: Einstellungen richtig wählen, Argumente weniger werden lassen
def train(net=None, nsuperv=20000, aug_ds=True, aug_gs=None, mixed_lbls=False, cons_ds=False, cons_gs=None, cons_mteacher=False, mteacher=False, tau=0.9, teacher_function=None, verbose=0):
    '''Trainiert das Netzwerk mit angegebenen Einstellungen
    aug_ds: Augmentation durch Drehspiegelung (Flip in x und y)
    aug_gs: Augmentation durch Gaußsches Rauschen
    mixed_lbls: Samples und Labels mischen
    cons_ds: Konsistenzbedingung unter Drehspiegelung auf dem unsupervised Set
    cons_gs: Konsistenzbedingung unter Gaußschem Raushen auf unsupervised Set
    cons_mteacher: Die Konsistenzbedingungen werden mit einem Mean Teacher (statt dem Netzwerk selbst) berechnet.
    mteacher: Konsistenz in der Zeit, zusätzlich zu aug und cons.
    teacher_function: Eine Funktion, nach der auf dem unsupervised Set das Netzwerk trainiert wird. Wenn None wird der MTeacher genutzt'''

    if net is None:
        net = Net(n=64)
    optimizer = torch.optim.Adam(lr=1e-3, params=net.parameters(), weight_decay=1e-4)
    bg_s = batchgen_supervised(nsuperv=nsuperv, verbose=False)
    bg_u = batchgen_unsupervised(nsuperv=nsuperv, verbose=False)
    BCE = nn.BCELoss()
    MSE = nn.MSELoss()
    L1Loss = nn.L1Loss()
    LogLoss = lambda yp, yt: torch.mean(-yt*torch.log(yp+1e-4))
    losses = []
    semlosses = []
    mtlosses = []

    assert not (cons_mteacher or mteacher) or not teacher_function, 'Only one possible!'

    # mean teacher
    if cons_mteacher or mteacher:
        net_teacher = Net(n=64) # identisches Netzwerk zu oben
        net_teacher.load_state_dict(net.state_dict())
        teacher_function = net_teacher
    else:
        net_teacher = None

    # Validation batches
    val = [ batch2torch(*getValbatch(i)) for i in range(3) ]
    b_val, p_val, v_val = zip(*val)

    
    # Trainings loop
    vloss_min = np.Inf
    for k in tqdm(range(4000)):
        optimizer.zero_grad()

        # ==== Supervised step ====
        b, tp, tv = next(bg_s)
        if mixed_lbls:
            b2, tp2, tv2 = next(bg_s)
            s = np.random.beta(0.75, 0.75, size=50)
            s = s[:,None,None,None]
            b = s*b+(1-s)*b2
            tp = s*tp+(1-s)*tp2
            s = s[:,:,0,0]
            tv = s*tv+(1-s)*tv2
        b, tp, tv = batch2torch(b, tp, tv)
        
        if aug_ds:
            if np.random.rand()<0.5:
                b = torch.flip(b, [2, 3])
                tp = torch.flip(tp, [2, 3])
                
        if aug_gs:
            assert aug_gs != True, 'Use aug_gs as a coefficient before randn! Values like 1/8 are appropriate!'
            b = b + aug_gs * torch.randn_like(b)
            
        pp, pv = net(b)
        # For softmax LogLoss, for sigmoid BCE
        loss = LogLoss(pp.view(-1), tp.view(-1)) + MSE(pv, tv)
        loss.backward()
        loss = loss.item() # torch to float
        ##print(loss)

        # ==== Unsupervised step ====
        if cons_ds or cons_gs or cons_mteacher:
            b1, _, _ = batch2torch(*next(bg_u))
            b2 = b1   # Kopie von x für Rauschen oder Veränderung

            # --> welche augmentation / noise ?
            if cons_ds:
                b2 = torch.flip(b2, [2, 3])
            if cons_gs:
                assert cons_gs != True, 'Use cons_gs as a coefficient before randn! Values like 1/8 are appropriate!'
                b2 = b2 + cons_gs * torch.randn_like(b2)

            # --> welches Netzwerk ?
            pp1, pv1 = net(b1)
            if net_teacher:
                pp2, pv2 = net_teacher(b2)
            else:
                pp2, pv2 = net(b2)
                
            if cons_ds:
                pp2 = torch.flip(pp2, [2, 3]) ## flip back
            semloss  = MSE(pp1.view(-1), pp2.view(-1)) + MSE(pv1, pv2)
            semloss.backward()
            semloss = semloss.item()
        else:
            semloss = 0
            
        if teacher_function:
            x1, _, _ = batch2torch(*next(bg_u))
            pp, pv = net(x1)
            pp_t, pv_t = teacher_function(x1)
            mtloss = MSE(pp.view(-1), pp_t.view(-1)) + MSE(pv, pv_t)
            mtloss = L1Loss(pp.view(-1), pp_t.view(-1)) + L1Loss(pv, pv_t)
            mtloss.backward()
            mtloss = mtloss.item()
        else:
            mtloss = 0
            

        # do the training step on the network
        optimizer.step()
        
        # update mean teacher if available
        if net_teacher:
            update_mt(net_teacher, net, tau)
            
        # Statistic
        losses += [loss]
        semlosses += [semloss]
        mtlosses += [mtloss]
        
        # Verbose output, measure minimum val loss
        if len(losses) == 20:
            # Val Loss
            net.eval()
            pred_val = [ net(x) for x in b_val ]
            net.train()
            vlosses = [ (LogLoss(pp.view(-1), tp.view(-1)) + MSE(pv, tv)).item() for ((pp, pv), tp, tv) in zip(pred_val, p_val, v_val)]
            vloss0, vloss1, vloss2 = vlosses
            if vloss1 < vloss_min:
                vlosses_min = vlosses
            vloss_min = min(vloss_min, vloss1)
            if verbose>=1:
                print('Loss=%f, SSLoss=%f, Val-Loss1=%f, Val-Loss2=%f'% (np.mean(losses), np.mean(semlosses), vloss1, vloss2))
                ##print('Loss=%f, SSLoss=%f, Val-Loss=%f±%f'% (np.mean(losses), np.mean(semlosses), np.mean(vlosses), np.std(vlosses)))
            losses = []
            losses2 = []

        if k % 200 == 0 and verbose>=2:
            board, value = collection[0]
            v_est = net.predict(encode_oh(board))

            show_board(v_est, cmap='cividis')
            plt.show()
            
    print('Val Losses Min:%f, %f, %f' % tuple(vlosses_min))
    print('Mean: %f, Std: %f' % (np.mean(vlosses_min), np.std(vlosses_min)))
    return net






# Funktion für das pitten zweier ai`s (Wettkampf mit n Runden)

def pit(ai1, ai2, n=20, nrnd=0):
    '''Pits two ai for n rounds. First ´nrnd´ half moves are chosen randomly.
    '''
    
    wins1 = 0
    wins2 = 0
    
    for k in range(n):

        if k%2 == 0:
            ai_red  = ai1
            ai_blue = ai2
        else:
            ai_red  = ai2
            ai_blue = ai1

        board = filledboard(bsize, nrnd)
        turn = (-1)**(board!=0).sum()

        while True:

            if turn>0:
                #'Reds turn'
                x, y = ai_red.findmove(board)
            else:
                #'Blues turn'
                x, y = ai_blue.findmove(board)

            board[x,y] = turn
            turn *= -1

            # If someone has won
            won = winner(board)
            if won!=0:
                break

        if k%2==1:
            won *= -1
        wins1 += int(+won/2+1/2)
        wins2 += int(-won/2+1/2)
        sigma = np.sqrt(wins1*wins2/(wins1+wins2))
        
        print(wins1, k+1)
        
    return wins1/n
        
        
        
    return wins1, wins2