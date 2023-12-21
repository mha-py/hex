'''
In this file a simple minmax and alphabeta ai is implemented
(c) 19.8.2020 mha
24.8.: Klasse geschrieben für MinMax mit Statistiken
'''

from numba import njit
import numpy as np

from hex_pathfinding import *
from hex_show import *
from hex_helpers import *



#### Helper / Wrapper functions ####

@njit(cache=True)
def _validmoves(board):
    N = len(board)
    return [ (i, j) for i in range(N) for j in range(N) if board[i,j] == 0 ]

@njit(cache=True)
def _makemove(board, move, turn):
    i, j = move
    board2 = board.copy()
    board2[i, j] = turn
    return board2

@njit(cache=True)
def _winner(board):
    won = winner_fast(board)
    if won == 0:
        return None
    progress = np.sum(np.abs(board)) / len(board)**2 # progress of the game
    won /= (1 + progress/10)
    return won


class HexGame:
    '''Class for the game Hex, providing game specific function which can be used
    by a generic AI'''
    
    @staticmethod
    def validmoves(board):
        return _validmoves(board)
    @staticmethod
    def makemove(board, move, turn):
        return _makemove(board, move, turn)
    @staticmethod
    def winner(board):
        return _winner(board)


    
#### Heuristiken für Minimax Algorithmus ####
    
from collections import defaultdict
from time import time


class HexHeuristicsDijkstra:
    def __init__(self):
        self.clear()
        self.bridges = False
        
    def clear(self):
        self.hash_v = dict()
        self.hash_rp = dict()
        self.hash_bp = dict()
        
    def getHeuristicValue(self, board):
        '''Returns the heuristic value of the board. Values will be hashed.'''
        s = board.tobytes()
        if s not in self.hash_v:
            self._createHash(board)
        return self.hash_v[s]
    
    def getSortedMoves(self, board, moves, red):
        '''Returns a list of moves which are heuristically sorted from best to worst'''
        onpath_r = self._getOnPath(board, red=True)
        onpath_b = self._getOnPath(board, red=False)
        moves1st = []
        moves2nd = []
        moves3rd = []
        for i,j in moves:
            if onpath_r[i,j] and onpath_b[i,j]:
                moves1st += [(i,j)]
            elif onpath_r[i,j] or onpath_b[i,j]:
                moves2nd += [(i,j)]
            else:
                moves3rd += [(i,j)]
        resmoves = moves1st + moves2nd + moves3rd
        
        return resmoves
        
    def _getOnPath(self, board, red):
        '''Returns the on_path array for the player'''
        s = board.tobytes()
        if s not in self.hash_v:
            self._createHash(board)
        if red:
            return self.hash_rp[s]
        else:
            return self.hash_bp[s]
        
    def _createHash(self, board):
        '''Calculates the heuristic value and the onpath information for the board and saves into the hash tables.'''
        s = board.tobytes()
        if self.bridges:
            raise NotImplemented
            #red_dist, red_onpath = dijkstra_dist_bridges_with_path(board.T)
            #blue_dist, blue_onpath = dijkstra_dist_bridges_with_path(-board)
        else:
            red_dist, red_onpath = dijkstra_dist_with_path(board.T)
            blue_dist, blue_onpath = dijkstra_dist_with_path(-board)
        value = self._heuristicvalue(red_dist, blue_dist)
        self.hash_v[s] = value
        self.hash_rp[s] = red_onpath
        self.hash_bp[s] = blue_onpath

    @staticmethod
    def _heuristicvalue(red_dist, blue_dist):
        '''Gives a heuristic for the boards value. A value near +1 means red is going to win, value near -1 means
        blue is going to win. This implementation counts the number of stones missing for a win and compares
        this number for both players.
        boardhash: A DijkstraHash Object which contains the dijkstra function outputs.
        extra: If True, a finer evaluation will be done where the number of good next moves will be taken into considerations
        (extra seems a bit suboptimal, moves become determined with, threshold at argmaxrnd doesnt compensate in a good way)'''
        rd = red_dist
        bd = blue_dist

        assert rd!=0 and bd!=0
        return 1./rd - 1./bd
    
    
class HexHeuristicsRollout:
    def __init__(self, n=20):
        '''Heuristic by performing rollouts
        n: Number of rollouts per board
        '''
        self.n = n
        
    def getHeuristicValue(self, board):
        '''Gives a heuristic for the boards value by doing random rollouts and averaging the number of wins and losses.
        '''
        return rollout(board, n=self.n)
    
    def getSortedMoves(self, board, moves, red):
        'Does nothing / not implemented'
        return moves
    
    
#### MiniMax-Klasse ####
    
class MiniMax:
    
    def __init__(self, game=None, heuristics=None, alphabeta=True, maxdepth=np.Inf, timelimit=np.Inf):
        '''MiniMax class for playing a two player game.
        heuristics: A class providing a heuristic board value and a heuristic sort for the moves (only if alphabeta enabled)
        '''
        if type(game) is type(None):
            self.game = HexGame()
        if type(heuristics) is type(None):
            self.heuristics = HexHeuristicsDijkstra()
        else:
            self.heuristics = heuristics
        self.maxdepth = maxdepth
        self.timelimit = timelimit
        self.use_alphabeta = alphabeta
        
    def clear(self):
        try: self.heuristics.clear()
        except: pass
        
    def findmove(self, board, maximize='auto', thr=0, verbose=0):
        '''Finds the ai`s move using maxmin algorithm.
        Board: The board for which the move shall be found.
        maximize: True if reds turn, false if blues turn
        alphabeta: If True it uses the alphabeta algorithm which includes pruning
        thr: Threshold for the board value for using suboptimal moves (allows diversity in play). Example thr=0.05'''
        
        if maximize=='auto':
            maximize = +1 if np.sum(board!=0)%2==0 else -1
        
        turn = +1 if maximize else -1
        function = self.alphabeta if self.use_alphabeta else self.minimax  # which function looks into the tree?
        
        # Abbreviation
        g = self.game
        
        # Check if game is over
        if g.winner(board):
            return None
        
        # Get valid moves and prepare list of move values
        moves = g.validmoves(board)
        boards = [ g.makemove(board, move, turn) for move in moves ]
        values = [ 0. for move in moves ]
        
        # Statistics
        self.count_pruned = defaultdict(lambda: 0)   # counts the number of pruned nodes for each tree depth
        self.count_searched = defaultdict(lambda: 0) # counts the number of not pruned nodes
        
        function(boards[0], not maximize, depth=1, maxdepth=1)  # compile everything
        
        # Timelimit for calculation
        t0 = time()
        self.tfinish = t0 + self.timelimit
        
        # if no timelimit, start with desired depth,
        # otherwise raise depth step by step
        if self.timelimit == np.Inf:
            mdepth = self.maxdepth  
        else:
            mdepth = 1
        
        k = 0
        while mdepth <= self.maxdepth and time() < self.tfinish:
                
            for i in range(len(moves)):
                values[i] = function(boards[i], not maximize, depth=1, maxdepth=mdepth)
                if not time() < self.tfinish:
                    if verbose >= 2:
                        print(f'Aborting at depth {mdepth} early after {time()-t0}s.')
                    break
            else:
                k = argmaxminrnd(values, maximize, rnd=True, thr=thr)
                fvalues = values.copy() # values with finished depth
                fmoves = moves
                if verbose >= 2:
                    print(f'Finished depth of {mdepth} after {time()-t0}s.')
                    
            mdepth += 1

        if verbose >= 1:
            print('Value:', fvalues[k]) # print values of a finished depth evaluation
        if verbose >= 4:
            boardvalues = 1.*np.zeros_like(board)
            for (i,j), v in zip(fmoves, fvalues):
                boardvalues[i,j] = turn*v/2 + 1/2
                boardvalues[i,j] = boardvalues[i,j]*0.9+0.1 # offset of 0.1 makes a difference for worst moves and unavailable moves
            show_board(boardvalues, cmap='gray')
        if verbose >= 3:
            d = 2
            print(f'Nodes pruned (d={d}): ', self.count_pruned[d])
            print(f'Nodes searched in (d={d}): ', self.count_searched[d])
            print(f'Nodes Pruned (d={d}):   %.1f%%' % (100*self.count_pruned[d]/(self.count_pruned[d]+self.count_searched[d]+1e-12)))
            ##print(f'Heuristic value hashed: %.1f%%' % (100*self.count_hashed/(self.count_hashed+self.count_calced)))  ## doesnt work atm
        if verbose >= 1:
            print()
            
        return moves[k]
    
    
    def minimax(self, board, maximize, depth, maxdepth):
        '''Evaluates the board using minimax algorithm.
        board: Board to be evaluated
        maximize: If this turns action is to be maximized or minimized
        depth: Current depth when calling this
        maxdepth: Maximum recursion depth'''
        g = self.game
        won = g.winner(board) # has anyone won?
        if won is not None:
            return won
        if depth >= maxdepth:
            val = self.heuristics.getHeuristicValue(board)
            return val
        
        turn = +1 if maximize else -1
        best = -np.Inf
        for move in g.validmoves(board):
            if time() > self.tfinish:
                break
            b = g.makemove(board, move, turn)
            value = self.minimax(b, not maximize, depth+1, maxdepth)
            best = max(best, turn*value)
        best = turn*best  # flip sign back
        return best
    
    
    def alphabeta(self, board, maximize, depth=0, maxdepth=np.Inf, alpha=-np.Inf, beta=np.Inf):
        '''Evaluates the board using minimax algorithm with alpha beta pruning.
        board: Board to be evaluated
        maximize: If this turns action is to be maximized or minimized
        depth: Current depth when calling this
        maxdepth: Maximum recursion depth
        alpha: maximal value which is relevant for this node and subnodes
        beta: minimal value (analog to alpha)'''
        g = self.game
        won = g.winner(board) # has anyone won?
        if won is not None:
            return won
        if depth >= maxdepth:
            val = self.heuristics.getHeuristicValue(board)
            return val
        moves = g.validmoves(board)
        moves = self.heuristics.getSortedMoves(board, moves, maximize)
        
        nsearched = 0
        npruned = 0
        
        if maximize:
            turn = +1
            for move in moves:
                if time() > self.tfinish:
                    break
                b = g.makemove(board, move, turn)
                value = self.alphabeta(b, not maximize, depth+1, maxdepth, alpha, beta)
                alpha = max(alpha, value)
                nsearched += 1
                if alpha >= beta:
                    npruned = len(moves) - nsearched
                    break
            self.count_pruned[depth] += npruned
            self.count_searched[depth] += nsearched
            return alpha
        else:
            turn = -1
            for move in moves:
                if time() > self.tfinish:
                    break
                b = g.makemove(board, move, turn)
                value = self.alphabeta(b, not maximize, depth+1, maxdepth, alpha, beta)
                beta = min(beta, value)
                nsearched += 1
                if alpha >= beta:
                    npruned = len(moves) - nsearched
                    break
            self.count_pruned[depth] += npruned
            self.count_searched[depth] += nsearched
            return beta
        
    