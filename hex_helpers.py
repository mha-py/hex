'''
Provides some helper functions like providing a half filled board, doing a rollout, etc.
(c) 23.8.2020 mha
'''

import numpy as np
from numba import jit, njit


    
def argmaxminrnd(v, maximize, rnd=True, thr=0.):
    '''Returns an index k for which v[k] max/min. If more than one index is
    max/minimizing, a random index of these is chosen.
    maximize: If True, finds the maximizing index, otherwise the minimizing index
    rnd: If False, this function will work like np.argmax/min
    thr: Small value that lowers the bound for a value to be considered as optimal.'''
    s = +1 if maximize else -1
    sv = s * np.array(v)
    if not rnd:
        return np.argmax(sv)
    else:
        available = (sv >= np.max(sv)-thr)
        inds = np.arange(len(sv))[available]
        return np.random.choice(inds)
    

def getturn(board):
    return (-1)**np.sum(board!=0)
    
    
def fillboard(board, count=None):
    '''Fills the board to the stone number of `count`. Count can be a ratio like 2/3. If count is `None`, the board will be completely filled.
    '''
    N = len(board)
    if count is None:  # None means completely filled
        count = N**2
    if count < 1:      # convert ratio to number
        count *= N**2
    board2 = board.copy().reshape(N**2)
    turn = (-1)**np.sum(board!=0)
    while sum(abs(board2)) < count:
        free = (board2==0) # boolean array
        k = np.random.choice(N**2, p=free/np.sum(free))
        board2[k] = turn
        turn *= -1
    return board2.reshape((N,N))


def filledboard(N, count=None, frame=0):
    '''Returns a filled board. Count: Number of stones filled in, can also be a ratio
    frame: Fills the outer fields to simulate a game of size `N-frame`
    '''
    if frame<=0:
        board = np.zeros((N,N), dtype='int8')
        return fillboard(board, count)
    else:
        board = np.zeros((N, N), dtype='int8')
        r, s = frame//2, (frame+1)//2
        board[:, :r] = +1
        board[:r, :] = -1
        board[:, -s:] = +1
        board[-s:, r:] = -1
        board[r:-s, r:-s] = filledboard(N-frame, count)
        return board
    
def deframe(board, frame):
    '''Removes a frame of the board (e. g. a filled board with a frame)
    '''
    r, s = frame//2, (frame+1)//2
    return board[r:-s, r:-s]
        

def randommove(board):
    '''Performs a random move on the board
    '''
    N = len(board)
    turn = -1 if (board<0).sum() < (board>0).sum() else +1
    board2 = board.copy().reshape(N**2)
    while True:
        k = np.random.randint(N**2)
        if board2[k] == 0: break
    board2[k] = turn
    return board2.reshape((N,N))
    

@jit(cache=True)
def choice(num, p):
    'Implements np.random.choice(num, p=p) which is otherwise not available in jit mode.'
    assert len(p)==num
    t = np.random.rand()
    pp = 0
    for i in range(num):
        pp += p[i] # probability summed up to i
        if t <= pp:
            break
    return i

@jit(cache=True)
def fillboard_fast(board, count=None):
    '''Fills the board to the stone number of `count`. Count can also be a ratio. If count is `None`, the board will be filled completely.
    '''
    N = len(board)
    if count is None:  # None means completely filled
        count = N**2
    if count < 1:      # convert ratio to number
        count *= N**2
    board2 = board.copy().reshape(N**2)
    turn = -1 if (board<0).sum() < (board>0).sum() else +1
    while np.sum(np.abs(board2)) < count:
        free = (board2==0) # boolean array
        k = choice(N**2, p=free/np.sum(free))
        board2[k] = turn
        turn *= -1
    return board2.reshape((N,N))

@jit(cache=True)
def filledboard_fast(N, count=None):
    '''Returns a filled board.
    '''
    board = np.zeros((N,N))
    return fillboard_fast(board, count)


#@jit(cache=True):
def getturn(board):
    return (-1)**(board!=0).sum()


