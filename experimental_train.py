
import torch
from torch import nn

relu = torch.relu

triag = lambda x: relu(1-torch.abs(x)) # Dreiecksfunktion

def pins(x, n=5, vmin=-1, vmax=+1):
    'x: tensor bx1 von Werten'
    'result: tensor bxn mit Zugehörigkeit zu jeder Kategorie'
    
    dv = (vmax-vmin)/(n-1)
    vs = [ vmin + dv*k for k in range(n) ]
    l = [ triag((x-v0)/dv) for v0 in vs ]
    
    return torch.cat(l, -1)


def histogram_loss(v):
    p = pins(v, 11)
    hist = torch.mean(p, 0) # mean over first dimension to get a distribution of values
    m = torch.mean(hist)
    
    #print(hist)
    loss = torch.mean((hist-1)**2)
    #loss += (hist[1]-1)**2
    #loss += 2*(hist[2]-1)**2
    #loss += (hist[3]-1)**2
    
    return loss



def winner_loss(pv, tv):
    'pv und tv: tensoren bx1'
    'result: loss'
    b = len(tv)
    i, j = 0, 0
    loss = []
    while i<b and j<b:
        # Eintrag mit +1 suchen für i
        while tv[i,0] != +1:
            i+=1
            if i >= b:
                return torch.mean(torch.stack(loss))
        # in i ist jetzt ein index für ein gewinnenden zustand
        
        # Eintrag mit -1 suchen für j
        while tv[j,0] != -1:
            j+=1
            if j >= b:
                return torch.mean(torch.stack(loss))
        # in j ist jetzt ein index für ein verlierenden zustand
        
        # beide miteinander verknüpfen
        diff = pv[i]-pv[j] # soll positiv sein mit buffer
        loss += [ relu(diff + .3)**2 ]
        i, j = i+1, j+1
    return torch.mean(torch.stack(loss))
        
        