'''
In this file are some function that allow path finding from left to right.
The game Hex uses them to check if a player has won.
23.8.2020 (c) mha
'''

import numpy as np
from collections import deque
from numba import jit, njit
from numba import int8
from heapq import heapify, heappush, heappop


class node:
    'Node class for book keeping of the path taken'
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent


def get_path(end_node):
    'Returns the path that is saved in `end_node` and it`s parent nodes.'
    path = []
    node = end_node
    while True:
        path.append(node.pos)
        if node.parent == None:
            return path
        else:
            node = node.parent


def bdfs(stones, path=False, bfs=True):
    '''BFS/DFS is a generic algorithm for path finding. Here it can be used to see if a board is won by a specific player.
    DFS is a bit faster, while BFS returns a nicer looking path.
    Time: ~124 µs
    stones: nxn-array of bool type, where the stones of the player are place.
    '''
    Nx, Ny = stones.shape
    assert Nx==Ny, 'Your board is not quadratic?'
    N = Nx

    ## for dfs use a list; for bfs use a deque.
    frontier = [ node((0, j)) for j in range(N) if stones[0, j]==1 ]
    if bfs:
        # Deque
        frontier = deque(frontier)
        front_pop = lambda: frontier.popleft()
        front_pop = frontier.popleft # falls fehler, diese zeile wieder löschen
    else:
        # List
        front_pop = lambda: frontier.pop()
        front_pop = frontier.pop # falls fehler, diese zeile wieder löschen
    explored = []

    while len(frontier) > 0:
        #nd = frontier.pop()
        nd = front_pop()
        (i, j) = nd.pos
        if i == N-1:
                return get_path(nd) if path else True
        for (di, dj) in [(1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0)]:
            if 0<=i+di<N and 0<=j+dj<N:
                if stones[i+di, j+dj] == 1:
                    if (i+di, j+dj) not in explored:
                        new_nd = node((i+di, j+dj), parent=nd)
                        frontier.append(new_nd)
        explored.append((i, j))
    return False


def winner(board):
    'Returns +1 if red has won, -1 if blue has won, and 0 if the game is not over yet.'
    if bdfs(board.T==1, path=False):
        return 1
    if bdfs(board==-1, path=False):
        return -1
    return 0


def winning_path(board):
    'Gives out the winning path for the one who won.'
    path = bdfs(board.T==1, path=True)
    if path:
        # Transpose it
        path = [ (j, i) for (i, j) in path ]
    if not path:
        path = bdfs(board==-1, path=True)
    return path




@njit(cache=True)
def dfs_fast(stones):
    '''The fast (jit) version of dfs for checking if bottom and top are connected. Here, the path cannot be returned.
    Time: ~5.4 µs'''
    Nx, Ny = stones.shape
    assert Nx==Ny, 'Your board is not quadratic?'
    N = Nx

    ## for dfs use a list; for bfs use a deque.
    frontier = [ (0, j) for j in range(N) if stones[0, j]==1 ]
    explored = []

    while len(frontier) > 0:
        (i, j) = frontier.pop()
        if i == N-1:
                return True
        for (di, dj) in [(1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0)]:
            if 0<=i+di<N and 0<=j+dj<N:
                if stones[i+di, j+dj] == 1:
                    #if (i+di, j+dj) not in explored:
                    for expld in explored:
                        if expld == (i+di, j+dj):
                            break
                    else:
                        frontier.append((i+di, j+dj))
        explored.append((i, j))
    return False




@njit(cache=True)
def winner_fast(board, player=0):
    '''The fast (jit) version of checking if a player has won. If player is non zero, it will only
    check the win for that player, i. e. after a move of player 1 (red) set player=+1.
    Returns +1 if red has won, -1 if blue has won, and 0 if the game is still not over.
    '''
    if player==+1 or player==0:
        if dfs_fast(board.T==1):
            return 1
    if player==-1 or player==0:
        if dfs_fast(board==-1):
            return -1
    return 0





@njit(cache=True)
def dijkstra_dist(board):
    '''Measures the distance from one boarder to the other. Own stones have cost 0, grey stones have cost 1 and enemys stones have cost 1000.
    This can be used as a heuristic for the ai (use the inverse value and take the difference of both players)
    Time: ~22µs'''
    N = len(board)
    b = np.ones((N+2, N))
    b[1:N+1] = board

    # Gegnerstein ist unpassierbar: kostet 1000
    # Leere Zelle muss noch besetzt werden: kostet 1
    # eingenommene Zelle: kostet 0
    v = (b==0) * 1 + (b==+1) * 0 + (b==-1) * 1000

    distances = np.Inf*np.ones((N+2, N))    # shortest distance to the node (i-1, j). (i==0 and i==N+1 are the border, 1<i<N+1 are the cells)
    distances[0, :] = 0.                    # distance is 0 for starting points
    inqueue = np.zeros((N+2, N))
    inqueue[0, :] = True

    while np.any(inqueue):
        notinqueue = 1-inqueue
        k = np.argmin(distances + np.Inf*notinqueue)
        i, j = k//N, k%N

        dist = distances[i,j]
        inqueue[i,j] = False

        for (di, dj) in [(1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0)]:
            if 1<=i+di<N+2 and 0<=j+dj<N:             # streng genommen wäre die untere Grenze 0, aber dort ist aber der Start
                d = dist + v[i,j]/2 + v[i+di,j+dj]/2  # Anwärter für die Distanz zu (i+di,j+dj)
                if d < distances[i+di, j+dj]:
                    distances[i+di, j+dj] = d         # Wert überschreiben, da Distanz nun geringer
                    inqueue[i+di, j+dj] = True        # Knoten in die Queue hinzufügen

    return np.min(distances[N+1,:])



@njit(cache=True)
def dijkstra_dist_with_path(board, freecells=True):
    '''Measures the distance from one boarder to the other.
    Additionally to the basic version of dijkstra, it returns all cells which are a part of one of the shortes paths.
    freecells: Only the cells which have to be taken are highlighted.
    Time: ~40µs
    '''
    N = len(board)
    b = np.ones((N+2, N))
    b[1:N+1] = board

    # Gegnerstein ist unpassierbar: kostet 1000
    # Leere Zelle muss noch besetzt werden: kostet 1
    # eingenommene Zelle: kostet 0
    v = (b==0) * 1 + (b==+1) * 0 + (b==-1) * 1000

    distances = np.Inf*np.ones((N+2, N))    # shortest distance to the node (i-1, j). (i==0 and i==N+1 are the border, 1<i<N+1 are the cells)
    distances[0, :] = 0.                    # distance is 0 for starting points
    inqueue = np.zeros((N+2, N))
    inqueue[0, :] = True
    ##reachedby = [ [ [ (0,0) for k in range(0) ] for j in range(N) ] for i in range(N+2) ]  # the preceders of (i,j) will be saves in
    reachedby = -np.ones((N+2, N, 6, 2), dtype=int8)

    while np.any(inqueue):
        k = np.argmin(distances + 1e5*(1-inqueue))

        i, j = k//N, k%N

        dist = distances[i,j]
        inqueue[i,j] = False

        for (di, dj) in [(1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0)]:
            if 1<=i+di<N+2 and 0<=j+dj<N:              # streng genommen wäre die untere Grenze für i die 0, aber dort ist aber der Start
                d = dist + v[i,j]/2 + v[i+di,j+dj]/2   # Anwärter für die Distanz zu (i+di,j+dj)
                if d < distances[i+di, j+dj]:
                    distances[i+di, j+dj] = d          # Wert überschreiben, da Distanz nun geringer
                    inqueue[i+di, j+dj] = True         # Knoten in die Queue hinzufügen
                    reachedby[i+di,j+dj,0] = (i,j)     # Dieser Knoten wurde (auf schnellstem Weg) nur von (i,j) aus erreicht.
                    reachedby[i+di,j+dj,1:] = -1       # ansonsten die Liste 'leeren'
                elif d == distances[i+di, j+dj]:
                    for l in range(6):                 # (ersten freien Platz der Liste finden)
                        if reachedby[i+di,j+dj,l,0] == -1:
                            break
                    reachedby[i+di,j+dj,l] = (i,j)    # Dieser Knoten wird auch durch (i,j) schnellstens erreicht.

    distance = np.min(distances[N+1,:])

    # Find  all cells which are on one of the shortest paths.
    # Uses dfs.
    on_path = np.zeros((N+2, N))
    frontier = [ (N+1, j) for j in range(N) ]
    visited = np.zeros((N+2, N))
    while len(frontier)>0:
        (i, j) = frontier.pop()
        if visited[i,j]:
            continue
        on_path[i,j] = True
        visited[i,j] = True
        parent_nodes = reachedby[i][j]
        for ii, jj in reachedby[i,j]:
            if ii < 0:
                break
            if visited[ii,jj]:
                continue
            frontier += [ (ii, jj) ]
        ##frontier.extend( [ (ii, jj) for (ii, jj) in parent_nodes if not visited[ii,jj] ] )

    # Cut the imaginary stones on the border off
    on_path = on_path[1:N+1]

    # If wanted, select the cells which are free
    if freecells:
        on_path *= (board == 0)

    return distance, on_path


@njit
def heuristic_value(board, extra=False):
    '''Gives a heuristic estimate value for the value of the board.
    A value near +1 means red is going to win, value near -1 means
    blue is going to win.
    This implementation counts the number of stones missing
    for a win and compares this number for both players.
    This algorithm can also handel position in which a player has already won,
    dfa_fast is much faster than this.
    Extra: Will use information about how many stones give advantage in a situation.
    Uses a more expensive algorithm (~2 times slower). Probably good for
    early stages of the game, since there are less ties between moves.'''

    if not extra:
        dist_red = dijkstra_dist(board.T)
        dist_blue = dijkstra_dist(-board)
    else:
        l, cells = dijkstra_dist_with_path(board.T)
        dist_red = l - np.sum(cells) / len(board)**2

        l, cells = dijkstra_dist_with_path(-board)
        dist_blue = l - np.sum(cells) / len(board)**2

    if dist_red==0:  return +1
    if dist_blue==0: return -1

    return 1./(dist_red) - 1./(dist_blue)


_bridgecost = 0.1

@njit(cache=True)
def dijkstra_dist_bridges(board):
    '''Version of dijkstra_dist which counts bridge type situations as half connected, i. e. distance of 0.5.'''
    N = len(board)
    b = np.ones((N+2, N))
    b[1:N+1] = board

    # Gegnerstein ist unpassierbar: kostet 1000
    # Leere Zelle muss noch besetzt werden: kostet 1
    # eingenommene Zelle: kostet 0
    v = (b==0) * 1 + (b==+1) * 0 + (b==-1) * 1000

    distances = np.Inf*np.ones((N+2, N))    # shortest distance to the node (i-1, j). (i==0 and i==N+1 are the border, 1<i<N+1 are the cells)
    distances[0, :] = 0.                    # distance is 0 for starting points
    inqueue = np.zeros((N+2, N))
    inqueue[0, :] = True

    while np.any(inqueue):
        notinqueue = 1-inqueue
        k = np.argmin(distances + np.Inf*notinqueue)
        i, j = k//N, k%N

        dist = distances[i,j]
        inqueue[i,j] = False

        neighbours = [(1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0)]
        for (di, dj) in neighbours:
            if 1<=i+di<N+2 and 0<=j+dj<N:             # streng genommen wäre die untere Grenze 0, aber dort ist aber der Start
                d = dist + v[i,j]/2 + v[i+di,j+dj]/2  # Anwärter für die Distanz zu (i+di,j+dj)
                if d < distances[i+di, j+dj]:
                    distances[i+di, j+dj] = d         # Wert überschreiben, da Distanz nun geringer
                    inqueue[i+di, j+dj] = True        # Knoten in die Queue hinzufügen

        # Brücke erkennen und ebenfalls erreichen
        if not b[i,j] == 1:
            continue
        for l in range(6):
            di1, dj1 = neighbours[l]
            di2, dj2 = neighbours[(l+1)%6]
            i2, j2 = i+di1+di2, j+dj1+dj2
            if 1<=i2<=N+2 and 0<=j2<N and b[i2,j2] == 1 \
                     and 1<=i+di1<=N+2 and 0<=j+dj1<N and b[i+di1,j+dj1]==0 \
                     and 1<=i+di2<=N+2 and 0<=j+dj2<N and b[i+di2,j+dj2]==0:
                d = dist + _bridgecost
                if d < distances[i2,j2]:
                    distances[i2,j2] = d
                    inqueue[i2,j2] = True

    return np.min(distances[N+1,:])




from numba import int8
@njit(cache=True)
def dijkstra_dist_bridges_with_path(board, freecells=True):
    '''Measures the distance from one boarder to the other.
    Additionally to the basic version of dijkstra, it returns all cells which are a part of one of the shortes paths.
    freecells: Only the cells which have to be taken are highlighted.
    Time: ~40µs
    '''
    N = len(board)
    b = np.ones((N+2, N))
    b[1:N+1] = board

    # Gegnerstein ist unpassierbar: kostet 1000
    # Leere Zelle muss noch besetzt werden: kostet 1
    # eingenommene Zelle: kostet 0
    v = (b==0) * 1 + (b==+1) * 0 + (b==-1) * 1000

    distances = np.Inf*np.ones((N+2, N))    # shortest distance to the node (i-1, j). (i==0 and i==N+1 are the border, 1<i<N+1 are the cells)
    distances[0, :] = 0.                    # distance is 0 for starting points
    inqueue = np.zeros((N+2, N))
    inqueue[0, :] = True
    ##reachedby = [ [ [ (0,0) for k in range(0) ] for j in range(N) ] for i in range(N+2) ]  # the preceders of (i,j) will be saves in
    reachedby = -np.ones((N+2, N, 12, 2), dtype=int8)

    while np.any(inqueue):
        k = np.argmin(distances + 1e5*(1-inqueue))

        i, j = k//N, k%N

        dist = distances[i,j]
        inqueue[i,j] = False

        neighbours = [(1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0)]
        for (di, dj) in neighbours:
            if 1<=i+di<N+2 and 0<=j+dj<N:              # streng genommen wäre die untere Grenze für i die 0, aber dort ist aber der Start
                d = dist + v[i,j]/2 + v[i+di,j+dj]/2   # Anwärter für die Distanz zu (i+di,j+dj)
                if d < distances[i+di, j+dj]:
                    distances[i+di, j+dj] = d          # Wert überschreiben, da Distanz nun geringer
                    inqueue[i+di, j+dj] = True         # Knoten in die Queue hinzufügen
                    reachedby[i+di,j+dj,0] = (i,j)     # Dieser Knoten wurde (auf schnellstem Weg) nur von (i,j) aus erreicht.
                    reachedby[i+di,j+dj,1:] = -1       # ansonsten die Liste 'leeren'
                elif d == distances[i+di, j+dj]:
                    for l in range(12):                # (ersten freien Platz der Liste finden)
                        if reachedby[i+di,j+dj,l,0] == -1:
                            break
                    reachedby[i+di,j+dj,l] = (i,j)    # Dieser Knoten wird auch durch (i,j) schnellstens erreicht.


        # Brücke erkennen und ebenfalls erreichen
        if not b[i,j] == 1:
            continue
        for l in range(6):
            di1, dj1 = neighbours[l]
            di2, dj2 = neighbours[(l+1)%6]
            i2, j2 = i+di1+di2, j+dj1+dj2
            if 1<=i2<=N+2 and 0<=j2<N and b[i2,j2] == 1 \
                     and 1<=i+di1<=N+2 and 0<=j+dj1<N and b[i+di1,j+dj1]==0 \
                     and 1<=i+di2<=N+2 and 0<=j+dj2<N and b[i+di2,j+dj2]==0:
                d = dist + _bridgecost
                if d < distances[i2,j2]:
                    distances[i2,j2] = d
                    inqueue[i2,j2] = True
                    reachedby[i2,j2,0] = (i,j)
                    reachedby[i2,j2,1:] = -1
                elif d == distances[i2,j2]:
                    for l in range(12):                # (ersten freien Platz der Liste finden)
                        if reachedby[i2,j2,l,0] == -1:
                            break
                    reachedby[i2,j2,l] = (i,j)         # Dieser Knoten wird auch durch (i,j) schnellstens erreicht.


    distance = np.min(distances[N+1,:])

    # Find  all cells which are on one of the shortest paths.
    # Uses dfs.
    on_path = np.zeros((N+2, N))
    frontier = [ (N+1, j) for j in range(N) ]
    visited = np.zeros((N+2, N))
    while len(frontier)>0:
        (i, j) = frontier.pop()
        if visited[i,j]:
            continue
        on_path[i,j] = True
        visited[i,j] = True
        parent_nodes = reachedby[i][j]
        for ii, jj in reachedby[i,j]:
            if ii < 0:
                break
            if visited[ii,jj]:
                continue
            frontier += [ (ii, jj) ]
        ##frontier.extend( [ (ii, jj) for (ii, jj) in parent_nodes if not visited[ii,jj] ] )

    # Cut the imaginary stones on the border off
    on_path = on_path[1:N+1]

    # If wanted, select the cells which are free
    if freecells:
        on_path *= (board == 0)

    return distance, on_path




@njit(cache=True)
def dijkstra_map(board):
    '25 µs für 13x13'
    N = len(board)

    # Gegnerstein ist unpassierbar: kostet 1000
    # Leere Zelle muss noch besetzt werden: kostet 1
    # eingenommene Zelle: kostet 0
    MAXCOST = 100000
    v = (board==0) * 1 + (board==+1) * 0 + (board==-1) * 2*MAXCOST

    distances = MAXCOST*np.ones((N, N))    # shortest distance to the node (i-1, j). (i==0 and i==N+1 are the border, 1<i<N+1 are the cells)
    distances[0, :] = v[0,:]/2            # distance is the step from the edge to the first stone for starting points

    queue = [ (v[0,j]/2, (0,j)) for j in range(N) ]
    heapify(queue)

    while len(queue)>0:
        dist, (i, j) = heappop(queue)
        for (di, dj) in [(1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0)]:
            if 1<=i+di<N and 0<=j+dj<N:             # streng genommen wäre die untere Grenze 0, aber dort ist aber der Start
                d = dist + v[i,j]/2 + v[i+di,j+dj]/2  # Anwärter für die Distanz zu (i+di,j+dj)
                if d < distances[i+di, j+dj] and d < MAXCOST:
                    distances[i+di, j+dj] = d         # Wert überschreiben, da Distanz nun geringer
                    heappush(queue, (d, (i+di,j+dj))) # Knoten in die Queue hinzufügen

    return distances # Erweiterung an den Seiten wieder wegnehmen
