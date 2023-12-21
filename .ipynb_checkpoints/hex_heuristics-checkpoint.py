'''
Collection of heuristic estimations of positions
(c) 27.3.2021 mha
'''

from hex_helpers import *
from hex_pathfinding import *


'''
class heuristic_resistance:
    def __init__(self, N):
        self.N = N
        
    def resistance(self, board):
        'Given: board NxN'
        'Returns: Resistance of the network'
        r = 0*(board==+1) + 1*(board==0) + 9999*(board==-1)
        
        # Gleichungen für Knoten: N**2 + 2 (pro Feld plus Source u. Target)
        # Gleichung ist jeweils Sum(I) = 0
        # Gleichungen für Kanten: 2*(N-1)*N + (N-1)**2 + 2*N = (N-1)*(3N-1) + 2*N
        num1 = N**2 + 2
        num2 = (N-1)*(3N-1) + 2*N
        num = num1 + num2
        M, c = np.zeros((num, num)), np.zeros(num)
        
        k = 0
        for l in range(N**2):
            # Kanten (0,1) von (i,j)=divmod(k,N) aus
            i, j = divmod(l, N)
            ms = [ N*i + (j+1),  N*(i+1) + j,  N*(i-1) + (j+1) ]
            for m in ms:
                if not 0<=m<N**2: continue # second knot doesnt exist -> no connection to care about
                M[l, k] = +1
                M[l, m] = -1
                l += 1
            print(l)
            
'''



def heuristic_dijkstra(board):
    r = dijkstra_dist(board.T)  # how many field red has to take until closed path
    b = dijkstra_dist(-board)   # same for blue
    #assert r!=0 and b!=0
    #return b-r
    #return 1/r - 1/b
    return np.tanh((b-r)/3) # besser??

    

def rollout(board, n=5, bridges=False):
    '''Plays a board to the end using random rollout for n times. If bridges is true,
    bridge recognition is used.'''
    f = _rollout if not bridges else _rollout_bridge
    return np.mean([ f(board) for _ in range(n) ])

def rollout_prob(board, prob, n=5):
    '''Play a board to the end given a fieldwise probability for red to capture that field.'''
    return np.mean([ _rollout_prob(board, prob) for _ in range(n) ])
    
    

@njit(cache=True)
def _rollout_prob(board, prob):
    'Rollout with known probability'
    'prob: NxN field of probability for red to take that cell'
    board = board.copy()
    N = len(board)
    prob = prob.flatten()
    turn = (-1)**np.sum(board!=0)
    
    
    while np.sum(board==0) > 0:
        board2 = board.flatten()
        free = (board2==0)
        prob2 = free * (1e-8 + prob if turn > 0 else 1-prob)
        k = choice(N**2, p=prob2/np.sum(prob2))
        board[divmod(k,N)] = turn
        
        turn *= -1

    # Gewinner ermitteln
    if dfs_fast(board.T==1):
        winner = +turn
    else:
        winner = -turn
        
    return winner

    

    
@njit(cache=True)
def _rollout(board):
    'Rollout'
    '30 µs'
    board = board.copy()
    N = len(board)
    turn = (-1)**np.sum(board!=0)
    
    
    # Board füllen
    while np.sum(board==0) > 0:
        board2 = board.flatten()
        free = (board2==0) # boolean array
        k = choice(N**2, p=free/np.sum(free))
        i, j = divmod(k, N)
        board[i, j] = turn
        
        # nächster Zug
        turn *= -1

    # Gewinner ermitteln
    if dfs_fast(board.T==1):
        winner = +1
    else:
        winner = -1
        
    return winner



@njit(cache=True)
def _rollout_bridges(_board):
    'Rollout with recognition of bridge pattern'
    '33 µs bei'
    _N = len(_board)
    turn = (-1)**np.sum(_board!=0)
    
    # Rand ergänzen (macht brückenerkennung einfacher)
    N = _N+2
    board = np.zeros((N, N))
    board[:,0] = +1
    board[:,-1] = +1
    board[0,:] = -1
    board[-1,:] = -1
    board[1:N-1, 1:N-1] = _board
    
    
    # Board füllen
    while np.sum(board==0) > 0:
        board2 = board.flatten()
        free = (board2==0) # boolean array
        k = choice(N**2, p=free/np.sum(free))
        i, j = divmod(k, N)
        board[i, j] = turn
        
        # checken, ob brücke (zwischen (i,j) und (i2,j2) wenn von einer Farbe
        # und gleichzeitig (i3,j3) und (i4,j4) freie Felder sind)
        relatives = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        for k in range(6):
            di1, dj1 = relatives[k]
            di2, dj2 = relatives[(k+1)%6]
            i2, j2 = i + di1 + di2,  j + dj1 + dj2
            
            # prüfen, ob das zweite Feld im Spielfeld ist
            if i2<0 or i2>=N or j2<=-1 or j2>=N: # stein außerhalb des (erweiterten) brettes
                continue
            elif board[i2, j2] != turn: # stein im feld, aber falsche farbe
                continue
                
            i3, j3 = i+di1, j+dj1
            i4, j4 = i+di2, j+dj2
            if board[i3, j3] == 0 and board[i4, j4] == 0: # brückensituation
                #print(f'Bridge between {i, j} and {i2, j2} for player ({turn})')
                #show_board(board); plt.show()
                s = +1 if np.random.rand()<.5 else -1
                board[i3, j3] = +s*turn
                board[i4, j4] = -s*turn
                #show_board(board); plt.show()
        
        # nächster Zug
        turn *= -1

    # Gewinner ermitteln
    if dfs_fast(board.T==1):
        winner = +1
    else:
        winner = -1
        
    return winner


