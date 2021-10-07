#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import pickle as pkl
from datetime import datetime

print('Starting')

model_fname = 'temp/net_temp.dat'
fnmask = 'temp/selfplay_%d.dat'

nparallel=10
nsearches=50
modulo=1

from hex_net13x13 import *

def save_records(game_records):
    for i in range(10000):
        fn = fnmask % i
        if not os.path.exists(fn):
            break
    with open(fn, 'wb') as f:
        pkl.dump(game_records, f)
    print(f'Saved {len(game_records)} boards to {fn}.')


# In[6]:


## Fake net
##torch.save(net.state_dict(), model_fname)


# In[3]:



def timenow():
    # current date and time
    now = datetime.now()
    return now.strftime("%H:%M:%S")

lastloaded = 0
def check_new_model(net):
    global lastloaded
    time = os.path.getmtime(model_fname)
    if time != lastloaded:
        net.load_state_dict(torch.load(model_fname))
        lastloaded = time
        print(f'Loaded new model parameters at time {timenow()}!')
        


# In[4]:

from tqdm import tqdm, trange

N = 13
bsize = 13

def selfplay_batched(ai, ngames=100000000000, verbose=0):
    
    bnum = ai.nparallel
    game_records = []
    
    ai.eta = 0.3

    # Iterator with tqdm??
    if verbose>=1:
        pbar = tqdm(total=ngames)
        
    newboard = lambda: filledboard(bsize, count=3)
    boards = [ newboard() for _ in range(bnum) ]
    turns = [ getturn(brd) for brd in boards ]
    records = [ [] for _ in range(bnum) ]
    
    
    completedgames = 0
    while completedgames < ngames:
        
        check_new_model(net)
        
        moves = ai.findmove(boards)
        for b in range(bnum):
            turn = turns[b]
            x, y = moves[b]
            records[b] += [(boards[b].copy(), (x, y), turn)] if turn > 0 else                           [(-boards[b].T.copy(), (y, x), turn)]
            boards[b][x, y] = turns[b]
            turns[b] *= -1
            
            won = winner(boards[b])
            if won:
                game_records += [ (b, m, t*won) for (b, m, t) in records[b][::modulo] ] # not all games
                
                # Flush if big enough
                if len(game_records) > 500:
                    save_records(game_records)
                    game_records = []
                
                completedgames += 1
                records[b] = []
                boards[b] = newboard()
                turns[b] = getturn(boards[b])
                ai.mcts[b].clear()
                if verbose>=1:
                    pbar.update(1)
    
    if verbose>=1:
        pbar.close()
        
    return game_records


# In[7]:




print('Starting Selfplay')
print('nparallel =', nparallel)
print('nsearches =', nsearches)

selfplay_batched(BatchMCTS(nparallel, nsearches, net=net), ngames=1000000, verbose=1)

