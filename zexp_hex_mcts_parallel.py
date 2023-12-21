'''
Class for Monte Carlo Tree Search
(c) 8.8.2020 mha
'''

#import threading
import multiprocessing


from hex_minmax import *
from hex_nnet import *
from hex_helpers import *

    
@njit(cache=True)
def _validmoves(board):
    N = len(board)
    return [ (i, j) for i in range(N) for j in range(N) if board[i,j] == 0 ]

@njit(cache=True)
def _makemove(board, move):
    i, j = move
    board2 = board.copy()
    board2[i, j] = _turn(board)
    return board2

@njit(cache=True)
def _winner(board):
    won = winner_fast(board)
    if won == 0:
        return None
    #progress = np.sum(np.abs(board)) / len(board)**2 # progress of the game
    #won /= (1 + progress/10)
    return won

@njit(cache=True)
def _turn(board):
    return 1 if np.sum(board!=0)%2==0 else -1


class HexGame:
    '''Class for the game Hex, providing game specific function which can be used
    by a generic AI'''

    @staticmethod
    def getValidActions(board):
        return _validmoves(board)
    @staticmethod
    def getNextState(board, move):
        return _makemove(board, move)
    @staticmethod
    def getEnded(board):
        return _winner(board)

    @staticmethod
    def getPlayerReward(reward, board):
        #turn = 1 if np.sum(board!=0)%2==0 else -1
        turn = _turn(board)
        return reward*turn
    
    @staticmethod
    def getTurn(board):
        return _turn(board)

    @staticmethod
    def getRepresentant(board):
        return board
    @staticmethod
    def getHashable(board):
        return board.tostring()





class HexPVDijkstra:
    'Get p and v by dijkstra path finding algorithm, instead of using a neural network.'
    dijkstraheuristics = HexHeuristicsDijkstra()
    def getPV(self, state, valid):
        red = (np.sum(state!=0)%2==0)
        parr = self.dijkstraheuristics._getOnPath(state, red=red)
        parr /= (1.*parr).sum()
        v = self.dijkstraheuristics.getHeuristicValue(state)
        p = np.asarray([ parr[i,j] for i, j in valid ])
        return p, v

class HexPVNNet:
    'Get p and v by neural network'
    def __init__(self, net):
        if net is None:
            net = 'net.dat'
        self.nnet = Net()
        self.nnet.load_state_dict(torch.load(net))
        self.nnet.eval()
    def getPV(self, state, valid):
        parr, v = self.nnet.predict(state)
        p = np.asarray([ parr[i,j] for i, j in valid ])
        return p, v




class HexPVNNetMulti:
    '''Class for evaluating states in the neural network by batching multiple of them and evaluating them as a batch.
    '''
    def __init__(self, net=None):
        if net is None:
            net = 'net.dat'
        if type(net) == str:
            self.nnet = Net()
            self.nnet.load_state_dict(torch.load(net))
        else:
            self.nnet = net
        self.nnet.eval()
        self.state_list = []
        self.valid_list = []
        #self.lock = threading.Lock()
        self.lock = multiprocessing.Lock()

    def getPV(self, state, valid):
        raise NotImplemented
        parr, v = self.nnet.predict(state)
        p = np.asarray([ parr[i,j] for i, j in valid ])
        return p, v

    def push(self, state, valid):
        'Gets the input of a worker and saves it. Returns the id of the worker to give him the right outputs'
        self.lock.acquire()
        self.done = False
        id = len(self.state_list)
        self.state_list += [state]
        self.valid_list += [valid]
        self.lock.release()
        return id

    def predictStack(self):
        'Performs the prediction of the NN with the pushed inputs. Stores them to be popped.'

        # Check if prediction is already done
        if self.done:
            return

        self.lock.acquire()
        # do the prediction
        xs = np.asarray(self.state_list)
        ys1, ys2 = self.nnet.predictStack(xs)

        # prepare output lists
        self.p_list = [ np.asarray([ parr[i,j] for i, j in valid ]) for parr, valid in zip(ys1, self.valid_list) ]
        self.v_list = ys2
        self.done = True

        # Clear entry lists
        self.state_list = []
        self.valid_list = []
        
        self.lock.release()

    def pop(self, id):
        return self.p_list[id], self.v_list[id]



@njit(cache=True)
def CPUCT(cpuct, p, sqrt_Ns, Na):
    # accelerated cpuct formula
    return cpuct * p * (sqrt_Ns + 1e-8) / (Na + 1)
    
    


class MCTS:
    'This class implements the Monte Carlo Tree Search. Everything is independent of a neural network.'

    def __init__(self, PV='NNet', net=None, game=None, timelimit=np.Inf, nsearches=np.Inf, params=None):
        if type(game) is type(None):
            self.game = HexGame()

        # Function for getting p and v (probability for best moves and board value)
        if PV == 'NNet': ## Neural Net
            #self.PV = HexPVNNet(net)
            self.PV = HexPVNNetMulti(net) ## Multi kann beides, ist eine obermenge von HexPVNNet()
            self.getPV = self.PV.getPV
        elif PV == 'Dijkstra': ## Dijkstra Heuristic
            self.PV = HexPVDijkstra()
            self.getPV = self.PV.getPV
        elif PV == 'ignore':
            pass
        else:
            raise NotImplemented

        self.timelimit = timelimit
        self.nsearches = nsearches

        # Q benutzen statt N für Zugentscheidung, dafür parameter negCP, der cpuct entspricht mit anderem Vorzeichen
        self.useQ = False # use Q for deciding the move instead of N
        self.negCP = 1    # uncertainty factor for finding moves
        
        # Mit Q rechnen statt V zwischen den Knoten, erlaubt andere Gewichtung der Unterknotenwerte
        self.notVbutQ = False # Formula for evaluating Q(node)
        self.part_for_Qown = 1
        self.weight_v_for_Qown = 1


        # Q default value (for N(s,a)=0)
        self.Q0 = 0
        self.cpuct = 4
        self.nvirtual = 1

        # Dirichlet Noise Parameters
        self.eta = 0 # 0.2  # nur bei selfplay
        self.alpha = 0.3
        self.discount = 0.999 # gamma, discount for reward

        # Override parameters if wanted
        if type(params) is not type(None):
            for attr, val in params.items():
                setattr(self, attr, val)

        self.clear()

    def clear(self):
        'Clears/Resets the tree'

        # Node info predicted by NN
        self.p = dict()                      # stores probabilities by Network for state s
        self.v = dict()                      # stores value of each state (either reward vector or number between -1 to +1)

        # Node info provided by game
        self.validactions = dict()                    # stores list of valid actions for state s
        self.r = dict()      # stores values/reward in case game has ended (either reward vector, or -1, 0, +1 for loss/draw/win)
        self.nextstate = dict()                 # which states follow from which (state, action) pairs

        # Collected by tree search
        self.N  = defaultdict(lambda: 0)      # stores number visits of each state
        self.Na = defaultdict(lambda: 0)     # stores number of choosing an action in a state
        self.W  = defaultdict(lambda: [])     # stores values collected
        self.Wa = defaultdict(lambda: [])     # stores values collected for an action in a state
        self.Qa = defaultdict(lambda: self.Q0)     # stores Action value of actions in a state (even needed?)
        self.Q  = dict()
        self.ahistory = defaultdict(lambda: []) # history of actions chosen
        self.turn = dict()
        
        self.Ds = dict() # depth of a state that occured

        self.states = dict() # states that occurred


    def perform_rollouts(self, board, verbose=0):
        '''Does the rollouts starting with the root node
        '''
        self.dirichlet = -1 # reset dirichlet
            
        # (1) Do the MC Tree Searches
        t0 = time()
        s = board.tostring()
        while self.N[s] < self.nsearches:
            #self.search(board, s)
            #self.search_iterative(board)
            self.search_split(board, nchosen=None)
            if time() > t0+self.timelimit:
                if verbose>0:
                    print(f'Finishing early after {self.N[s]} tree searches')
                break
        else:
            if verbose>0:
                print(f'Finished {self.nsearches} tree searches after {time()-t0}s')



    def findmove(self, board, tau=0, verbose=0):
        '''Finds the best move using a Monte Carlo Tree Search
        '''
        # (1) Do the MC Tree Searches
        self.perform_rollouts(board, verbose)

        # (2) Find the best move
        if tau is None:
            if np.sum(board!=0) <= 6: # in early game some randomness
                tau = 0.9**np.sum(board!=0)
            else:
                tau = 0.

        s = board.tostring()
        valid = self.validactions[s]
        Ns = np.array([ self.Na[s,a] for a in valid ])

        if self.useQ:
            Qs = np.array([ (self.Qa[s,a] if self.Qa[s,a] is not np.NaN else -1) for a in valid ])
            UC = 1./np.sqrt(Ns+1e-4)
            ##UC = 1./(Ns+1e-4)
            Ns = Qs - self.negCP * UC # simulate a N but actually taking Q for choosing the move
            assert tau==0, 'Not compatible with useQ mode!'

        ##print(f'tau = {tau}')
        if tau==0:
            k = argmaxminrnd(Ns, maximize=True, rnd=True)
        else:
            prob = 1.*Ns**(1/tau)
            prob /= prob.sum()
            k = np.random.choice(range(len(valid)), p=prob)
        move = valid[k]


        # (3) Verbose things
        if verbose>=3:
            print(self.Nsalist)
            arr = np.zeros_like(board).astype(float)
            for k, (i,j) in enumerate(valid):
                #arr[i,j] = self.Qlist[k]
                arr[i,j] = self.Nsalist[k] / np.max(self.Nsalist)
            show_board(arr, cmap='cividis')

        return move


    def getReward_Islfnd(self, state, s):
        '''Hands out the information: Reward of s, and if s is a leafnode.
        state: GameState object
        s: its hashable
        '''
        if s not in self.r:
            reward = self.game.getEnded(state)
            self.r[s] = reward
            self.turn[s] = self.game.getTurn(state)
            isleafnode = True
        else:
            reward = self.r[s]
            isleafnode = False
        return reward, isleafnode


    def getP_V_Valid(self, state, s):
        '''Hands out the information: p and v of neural network, and valid moves/actions.
        state: GameState object
        s: its hashable
        '''
        if s not in self.v:
            # First get valid actions
            valid = self.game.getValidActions(state)
            assert len(valid) > 0, 'There are no valid moves!'
            self.validactions[s] = valid
            # Then compute p and v
            p, v = self.getPV(state, valid)
            self.p[s] = p
            self.v[s] = v
        else:
            v = self.v[s]
            p = self.p[s]
            valid = self.validactions[s]
        # return v and validmoves
        return p, v, valid


   # def search(self, state, s=None, nchosen=None, depth=0):
        '''Performs a Monte Carlo Tree Search on the state s.
        state: State for which the best moves are searched
        s: The string representation of the state s (optional)
        nchosen: How often the caller has chosen this state to visit.
        depth: The depth of the search (if called by another node)'''

    '''
        if type(nchosen) == type(None):
            nchosen = np.Inf

        state_symrepr = self.game.getRepresentant(state) # state representant selected from symmetry set, "state modulo symmetry"
        s = self.game.getHashable(state_symrepr)         # hashable state representant selected from symmetry set

        self.states[s] = state_symrepr                   # save this state for training later

        # (1) Early returning
        # Check if game has ended
        reward, isleafnode = self.getReward_Islfnd(state, s)
        if type(reward) is not type(None):
            return reward

        # Next info we need is the neural network output and the valid actions
        p, v, valid = self.getP_V_Valid(state, s)

        # if leafnode then return
        if isleafnode:
            return v

        # If this node already was evaluated/searched several times but not from caller,
        # then return saved values, instead of creating new ones
        Ns = self.N[s]
        if nchosen < Ns:
            if self.notVbutQ:
                return self.Q[s]
            else:
                return self.W[s][nchosen]



        # (2) Go deeper into the tree
        sqrt_Ns = np.sqrt(Ns)
        if depth == 0 and self.eta>0:
            if self.dirichlet is -1:
                self.dirichlet = np.random.dirichlet(len(valid)*[self.alpha])
            p = (1-self.eta) * p + self.eta * self.dirichlet

        # (a) calculate U(s,a) for each action
        Ulist = []
        Nsalist = []
        Qlist = []
        for k, a in enumerate(valid):
            Nsa = self.Na[s,a]
            if Nsa > 0:
                Qvec = self.Qa[s, a]
                Q = self.game.getPlayerReward(Qvec, state)
            else:
                Q = self.Q0   # default value if a was never chosen in this state
            U = Q + self.cpuct * p[k] * (sqrt_Ns + 1e-8) / (Nsa+1)
            Ulist.append(U)
            Nsalist.append(Nsa)
            Qlist.append(Q)

        # (b) choose action with maximal U(s,a) value
        k = argmaxminrnd(Ulist, maximize=True, rnd=True)
        a = valid[k]

        if depth == 0:
            # Useful for debugging:
            self.Ulist = Ulist
            self.Nsalist = Nsalist
            self.Qlist = Qlist


        # Next the next state by performing the action
        Nsa = self.Na[s,a]
        nextstate = self.game.getNextState(state, a)


        if not self.notVbutQ:
            ### Standardvariante: v wird übergeben. Besser wäre: Knoten returnt seinen Wert, den er selbst berechnet.
            v = self.search(nextstate, nchosen=Nsa, depth=depth+1)

            # Accept the value and save into the node
            self.W[s].append(v)
            self.Wa[s,a].append(v)
            self.N[s] += 1
            self.Na[s,a] += 1
            self.Qa[s,a] = np.sum(self.Wa[s,a], axis=0) / self.Na[s,a]

            return v*self.discount


        else:
            ### Meine eigene Variante: Q wid vom Knoten berechnet und übergeben.
            #   Hier sollte dann auch der Part mit `return self.W[s][nchosen]` überdacht werden. --> Eigenes Q returnen und fertig.
            #   W und Wa existieren dann nicht mehr
            ##  Frage: An welcher Stelle Projiziere ich den Reward auf den Skalar für den Spieler?? Bei der Erstellung von U oder nahe dort.
            Q = self.search(nextstate, nchosen=Nsa, depth=depth+1)

            self.Qa[s,a] = Q
            self.N[s] += 1
            self.Na[s,a] += 1
            self.ahistory[s].append(a)

            self.Q[s] = self.formulatocalculateOwnQoutOfQa(state, s)

            return self.Q[s]*self.discount
    '''

    def formulatocalculateOwnQoutOfQa(self, state, s):
        '''Calculates the Q value of the node s based on its child nodes Q values.
        This formula weightens lastly visited nodes more than visited nodes in general (like in the original algorithm)'''
        weight_v = self.weight_v_for_Qown # weightening of v(s) for Q(s)
        part = self.part_for_Qown

        N = self.N[s]
        Npart = int(np.ceil(part*N))
        
        assert len(self.ahistory[s])==N, 'Inconsitentcy, len(ahistory) != N !!'

        Qown = weight_v * self.v[s] # this nodes v value is treated with a weight
        for a in self.ahistory[s][N-Npart:N]: # second half of the history of actions for weightening the nodes Q values
            Qown += self.Qa[s,a]
        Qown = Qown / (Npart + weight_v)
        return Qown

    def formulatocalculateOwnQoutOfQa_var2(self, state, s):
        'Formel für Q, die näher am Minimax Prinzip dran ist'
        alist = list({ a for a in self.ahistory[s] })
        k = np.argmax([ self.game.getPlayerReward(self.Qa[s,a], state) for a in alist ])
        a = alist[k]
        return self.Qa[s,a]










   # def search_iterative(self, state):
      #  '''Iterative version of the MC Tree Search algorithm
      #  This version ignores the option of `notVbutQ` but is identical in all other aspects.
      #  '''

    '''
        s_stack = []
        a_stack = []

        nchosen = np.Inf

        depth = -1
        while True:
            depth += 1

            state_symrepr = self.game.getRepresentant(state) # state representant selected from symmetry set, "state modulo symmetry"
            s = self.game.getHashable(state_symrepr)         # hashable state representant selected from symmetry set

            self.states[s] = state_symrepr                   # save this state for training later

            # (1) Early returning
            # Check if game has ended
            reward, isleafnode = self.getReward_Islfnd(state, s)
            if type(reward) is not type(None):
                v = reward
                break
                ## # return reward

            # Next info we need is the neural network output and the valid actions
            p, v, valid = self.getP_V_Valid(state, s)

            # if leafnode then return
            if isleafnode:
                ## # return v
                # v is already set
                break

            # If this node already was evaluated/searched several times but not from caller,
            # then return saved values, instead of creating new ones
            Ns = self.N[s]
            if nchosen < Ns:
                ## # return self.W[s][nchosen]
                ## v = self.W[s][nchosen]
                if self.notVbutQ:
                    v = self.Q[s]
                else:
                    v = self.W[s][nchosen]
                break


            # (2) Going deeper into the tree
            sqrt_Ns = np.sqrt(Ns)
            if depth == 0 and self.eta>0:
                if self.dirichlet is -1:
                    self.dirichlet = np.random.dirichlet(len(valid)*[self.alpha])
                p = (1-self.eta) * p + self.eta * self.dirichlet

            # (a) calculate U(s,a) for each action
            Ulist = []
            Nsalist = []
            Qlist = []
            for k, a in enumerate(valid):
                Nsa = self.Na[s,a]
                if Nsa > 0:
                    Qvec = self.Qa[s, a]
                    Q = self.game.getPlayerReward(Qvec, state)
                else:
                    Q = self.Q0   # default value if a was never chosen in this state
                U = Q + self.cpuct * p[k] * (sqrt_Ns + 1e-8) / (Nsa+1)
                Ulist.append(U)
                Nsalist.append(Nsa)
                Qlist.append(Q)

            # (b) choose action with maximal U(s,a) value
            k = argmaxminrnd(Ulist, maximize=True, rnd=True)
            a = valid[k]

            if depth == 0:
                # Useful for debugging:
                self.Ulist = Ulist
                self.Nsalist = Nsalist
                self.Qlist = Qlist


            # Next the next state by performing the action
            nchosen = self.Na[s,a]
            nextstate = self.game.getNextState(state, a)

            ## # v = self.search(nextstate, nchosen=Nsa, depth=depth+1)

            s_stack.append(s)
            a_stack.append(a)

            state = nextstate


        ## Standardvariante: v wird übergeben
        if not self.notVbutQ:
            
            # the former return value of the recursive self.search function is stored in `v`.
            while len(s_stack) > 0:

                s = s_stack.pop()
                a = a_stack.pop()

                # Accept the value and save into the node
                self.W[s].append(v)
                self.Wa[s,a].append(v)
                self.N[s] += 1
                self.Na[s,a] += 1
                self.Qa[s,a] = np.sum(self.Wa[s,a], axis=0) / self.Na[s,a]

                v *= self.discount

        ### Meine eigene Variante: Q wid vom Knoten berechnet und übergeben.
        #   Hier sollte dann auch der Part mit `return self.W[s][nchosen]` überdacht werden. --> Eigenes Q returnen und fertig.
        #   W und Wa existieren dann nicht mehr
        else:
            
            #print('WARNING, THIS MODE IS UNTESTED IN THIS FUNCTION!')
            
            Q = v # initiate Q with leaf nodes v
            
            while len(s_stack) > 0:

                s = s_stack.pop()
                a = a_stack.pop()

                self.Qa[s,a] = Q
                self.N[s] += 1
                self.Na[s,a] += 1
                self.ahistory[s].append(a)

                self.Q[s] = self.formulatocalculateOwnQoutOfQa(state, s)

                Q = self.Q[s]*self.discount
        '''



    def search_down(self, state):
        '''Split iterative version of the MC Tree Search algorithm.
        '''



        self.id = None
        s_stack = []
        a_stack = []
        v = None
        nchosen = np.Inf

        depth = -1
        while True:
            depth += 1

            state_symrepr = self.game.getRepresentant(state) # state representant selected from symmetry set, "state modulo symmetry"
            s = self.game.getHashable(state_symrepr)         # hashable state representant selected from symmetry set

            self.states[s] = state_symrepr                   # save this state for training later

            # (1) Early returning
            # Check if game has ended
            reward, _ = self.getReward_Islfnd(state, s)
            if type(reward) is not type(None):
                v = reward
                break
                ## # return reward

            # Next info we need is the neural network output and the valid actions

            # leaf node or not cant be decided with the above function (getreward_islfnd), as in
            # the case of a collision we still lack the neural network outputs
            isleafnode = (s not in self.v)

            # if leafnode then return
            if isleafnode:
                # Put state into the queue to be computed
                valid = self.game.getValidActions(state)
                self.validactions[s] = valid
                assert len(valid) > 0, 'No valid actions!'
                self.id = self.PV.push(state, valid)
                #self._temp_s = s
                s_stack.append(s)
                self.Ds[s] = depth # save depth of this state
                break
            else:
                p, v, valid = self.getP_V_Valid(state, s)

            # If this node already was evaluated/searched several times but not from caller,
            # then return saved values, instead of creating new ones
            if nchosen < len(self.W[s]):
                ## # return self.W[s][nchosen]
                v = self.W[s][nchosen]
                break


            # (2) Go deeper into the tree
            sqrt_Ns = np.sqrt(self.N[s])
            if depth == 0 and self.eta>0:
                if self.dirichlet is -1:
                    self.dirichlet = np.random.dirichlet(len(valid)*[self.alpha])
                p = (1-self.eta) * p + self.eta * self.dirichlet

            # (a) calculate U(s,a) for each action
            Ulist = []
            Nsalist = []
            Qlist = []
            '''
            for k, a in enumerate(valid):
                Nsa = self.Na[s,a]
                Qvec = self.Qa[s, a]
                Q = self.game.getPlayerReward(Qvec, state)
                #U = Q + self.cpuct * p[k] * (sqrt_Ns + 1e-8) / (Nsa+1)
                U = Q + self.cpuct_formula(p[k], sqrt_Ns, Nsa)
                Ulist.append(U)
                Nsalist.append(Nsa)
                Qlist.append(Q)'''
            
            #Qlist = [ self.game.getPlayerReward(self.Qa[s,a], state) for a in valid ]
            ###Ulist = [ Qlist[k] + self.cpuct * p[k] * (sqrt_Ns + 1e-8) / (self.Na[s,a] + 1) for k, a in enumerate(valid) ]
            #Ulist = [ Qlist[k] + CPUCT(self.cpuct, p[k], sqrt_Ns, self.Na[s,a]) for k, a in enumerate(valid) ]

            
            #Ulist = [ Qlist[k] + self.cpuct * p[k] * (sqrt_Ns + 1e-8) / (self.Na[s,a] + 1) for k, a in enumerate(valid) ]
            Ulist = [ self.Qa[s,a] + CPUCT(self.cpuct, p[k], sqrt_Ns, self.Na[s,a]) for k, a in enumerate(valid) ]
            
            
            # (b) choose action with maximal U(s,a) value
            #k = argmaxminrnd(Ulist, maximize=True, rnd=True)
            k = np.argmax(Ulist)
            a = valid[k]

            if depth == 0:
                # Useful for debugging:
                self.Ulist = Ulist
                self.Nsalist = Nsalist
                self.Qlist = Qlist


            # Next the next state by performing the action
            nchosen = self.Na[s,a]
            nextstate = self.game.getNextState(state, a)

            self.Na[s,a] += self.nvirtual # virtual loss of one.
            self.N[s] += self.nvirtual # virtual loss of one.

            ## # v = self.search(nextstate, nchosen=Nsa, depth=depth+1)

            s_stack.append(s)
            a_stack.append(a)

            state = nextstate

        # Store the stacks and the v for the search up function
        self.s_stack = s_stack
        self.a_stack = a_stack
        self.value = v


    def search_leaf(self):
        '''This subfunction does everything that happens for a leafnode in the nonsplit function `search`.
        '''

        if type(self.id) is not type(None):
            self.PV.predictStack()
            p, v = self.PV.pop(self.id)

            #s = self._temp_s
            s = self.s_stack.pop()

            self.p[s] = p
            self.v[s] = v
            self.value = v
            self.id = None


    def search_up(self):

        s_stack = self.s_stack
        a_stack = self.a_stack
        v = self.value
        
        ## Standardvariante: v wird übergeben
        if not self.notVbutQ:

            # the former return value of the recursive self.search function is stored in `v`.
            while len(s_stack) > 0:

                s = s_stack.pop()
                a = a_stack.pop()

                # Accept the value and save into the node
                self.W[s].append(v*self.turn[s])
                self.Wa[s,a].append(v*self.turn[s])
                self.N[s] += 1 - self.nvirtual
                self.Na[s,a] += 1 - self.nvirtual
                self.Qa[s,a] = np.sum(self.Wa[s,a], axis=0) / self.Na[s,a]

                v *= self.discount
            
            
        ### Meine eigene Variante: Q wid vom Knoten berechnet und übergeben.
        #   Hier sollte dann auch der Part mit `return self.W[s][nchosen]` überdacht werden. --> Eigenes Q returnen und fertig.
        #   W und Wa existieren dann nicht mehr
        else:
            raise "Not Implemented!"

                

    def search_split(self, state, nchosen=None):
        self.search_down(state)
        self.search_leaf()
        self.search_up()


class MultiMCTS(MCTS):
    '''Class for batched Monte Carlo Tree Search
    '''
    def __init__(self, net=None, nparallel=1, shared=True, timelimit=np.Inf, nsearches=np.Inf, params=None):
        assert shared, 'Not Implemented'
        super().__init__(net=net, timelimit=timelimit, nsearches=nsearches, params=params)
        #self.PV = HexPVNNetMulti() ## Multi kann beides, ist eine obermenge von HexPVNNet()

        self.s_stacks = [ [] for _ in range(nparallel) ]
        self.a_stacks = [ [] for _ in range(nparallel) ]
        self.values   = [ None for _ in range(nparallel) ]
        self.ids = [ None for _ in range(nparallel) ]
        self.nparallel = nparallel


    def activate_worker(self, k):
        '''Activates the worker k. The workers belief that some class variables are just for them, so this function
        manages to store their variables.'''
        self.s_stack = self.s_stacks[k]
        self.a_stack = self.a_stacks[k]
        self.value = self.values[k]
        self.id = self.ids[k]

    def deactivate_worker(self, k):
        '''The counterpart of activate_id.'''
        self.s_stacks[k] = self.s_stack
        self.a_stacks[k] = self.a_stack
        self.values[k] = self.value
        self.ids[k] = self.id


    def perform_rollouts(self, board, verbose=0):
        '''Does the rollouts starting with the root node
        '''
        # Dirichlet noise in case of selfplay
        if self.eta > 0:
            self.dirichlet = -1
            
        s = board.tostring()
        state = board
        t0 = time()
        N0 = self.N[s]
        while self.N[s] < self.nsearches:

            for k in range(self.nparallel):
                self.activate_worker(k)
                self.search_down(state)
                self.deactivate_worker(k)

            for k in range(self.nparallel):
                self.activate_worker(k)
                self.search_leaf()
                self.deactivate_worker(k)

            for k in range(self.nparallel):
                self.activate_worker(k)
                self.search_up()
                self.deactivate_worker(k)

            if time() > t0+self.timelimit:
                if verbose>0:
                    print(f'Finishing early after {self.N[s]-N0} tree searches')
                break
        else:
            if verbose>0:
                print(f'Finished {self.nsearches-N0} tree searches after {time()-t0}s')

        
        
class BatchMCTS:
    def __init__(self, nparallel, nsearches, net, params=None):
        '''Batch of MCTS move finders'''
        self.nsearches = nsearches
        self.nparallel = nparallel
        self.mcts = [ MCTS(nsearches=nsearches, PV='ignore')  for i in range(nparallel) ]
        self.PV = HexPVNNetMulti(net)
        # Netzwerke PV für alle mcts klassen gleich machen
        for m in self.mcts:
            m.PV = self.PV
            
        # set parameters for all mcts classes
        if params is not None:
            for m in self.mcts:
                for attr, val in params.items():
                    setattr(m, attr, val)
            
    # set parameter of children mcts
    def setparam(self, attr, val):
        for m in self.mcts:
            setattr(m, attr, val)
        
            
            
    def perform_rollouts(self, states):
        dirichlets = [ -1 for _ in states ]
        for _ in range(self.nsearches):
            
            for k in range(len(states)):
                self.mcts[k].dirichlet = dirichlets[k] # ?????????????????????????
                
            #thrds = [ threading.Thread(target=self.mcts[k].search_down, args=[state]) for k, state in enumerate(states) ]
            thrds = [ multiprocessing.Process(target=self.mcts[k].search_down, args=[state]) for k, state in enumerate(states) ]
            [ thr.start() for thr in thrds ]
            [ thr.join() for thr in thrds ]
            
            thrds = [ multiprocessing.Process(target=self.mcts[k].search_leaf, args=[]) for k, state in enumerate(states) ]
            [ thr.start() for thr in thrds ]
            [ thr.join() for thr in thrds ]
            
            thrds = [ multiprocessing.Process(target=self.mcts[k].search_up, args=[]) for k, state in enumerate(states) ]
            [ thr.start() for thr in thrds ]
            [ thr.join() for thr in thrds ]
            
            '''
            for k in range(len(states)):
                self.mcts[k].dirichlet = dirichlets[k]
                self.mcts[k].search_down(states[k])
                dirichlets[k] = self.mcts[k].dirichlet'''
            
            for k in range(len(states)):
                self.mcts[k].search_leaf()
            
            for k in range(len(states)):
                self.mcts[k].search_up()
                
            

    def findmove(self, states, tau=0, verbose=0):
        '''Finds the best move using a Monte Carlo Tree Search
        '''
        
        self.perform_rollouts(states)
        
        moves = []
        for k in range(len(states)):
            move = self.mcts[k].findmove(states[k]) # rollouts are already done and skipped in this function
            moves.append(move)
            
        return moves
    
    
    def clear(self):
        '''Clears all hashed state properties'''
        for mcts in self.mcts:
            mcts.clear()