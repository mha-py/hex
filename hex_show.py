'''
Functions for showing boards as a plot, as ascii, saving to a file, maybe even opengl output.
17.8.2020 (c) mha
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


##N = 6
##bsize = N



# color of red stones, blue stones and empty stones
c = {
      0: (.2, .2, .2),
     +1: (.6, .0, .0),
     -1: (.0, .05, .85)
}

# gap between two hexagons
GAP = 0.1

class _Vis_Consts:
    'Supplies us with constants which are relevant for plotting and GUI (but nothing else)'
    # Basechange between xy and ij coordinates
    ij2xy = np.array([
        [ 1, 0.5],
        [ 0, np.sqrt(3/4)],
    ])
    # Basechange back from xy to ij
    xy2ij = np.linalg.inv(ij2xy)

    # Shape of our hexagon
    thirty_degree = 2*np.pi/12
    sixty_degree = 2*np.pi/6
    gap = GAP
    hexag_factor = (1-GAP)/2/np.cos(thirty_degree) # the factor in the hexagon formula (to get a gap)
    hexagon = hexag_factor * np.asarray([(np.cos(alpha), np.sin(alpha)) for alpha in
                                   np.arange(thirty_degree, 2*np.pi+thirty_degree, sixty_degree)])

    # For detection if a point is on a hexagon we need the middle points on all lines of the hexagon
    middlepoints = (hexagon[1:] + hexagon[:-1]) / 2
    
    # later we will just take np.dot(dual_xy, ..) with a vector and if its abs is < 1 for all, we're in the hexagon.
    dual_xy = middlepoints / np.linalg.norm(middlepoints[0])
    
    @staticmethod
    def color(b):
        '''generalizes the color to float values (by interpolation)
        It maps -1 to blue, 0 to gray, +1 to red
        '''
        c0 = c[0]
        c1 = c[1] if b>0 else c[-1]
        t = abs(b)
        c0, c1 = np.asarray([c0, c1])
        return (1-t)*c0 + t*c1
    
    
_vis_consts = _Vis_Consts()

vis_consts = _vis_consts ####


def show_board(board, numbers=False, text=None, path=None, save=None, cmap=None):
    '''Shows the board on the scree via matplotlib.
    numbers: If True, the coordinates of each will be shown on the fields.
    save: If given, the board will be saved to the given path.
    cmap: A function that maps the numbers 0..1 to a color (optional)
    '''
    Nx, Ny = board.shape
    assert Nx == Ny, 'Your board is not quadratic??'
    N = Nx
    
    fig=plt.figure(figsize=(8,6), dpi=70)
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(left = 0)
    fig.subplots_adjust(right = 1)
    
    plt.axis('equal')
    plt.xlim([-1, N*1.5-0.5])
    plt.ylim([-1, N*np.sqrt(3/4)])
    plt.axis('off')
    
    # Black background
    patch = patches.Polygon(([-100,-100], [-100,100], [100,100], [100,-100]), facecolor='black', zorder=-100)
    plt.gca().add_patch(patch)
    
    # Set colorfunction
    if type(cmap) == type(None):
        cmap = _vis_consts.color # normal colors to plot the board: red/blue/gray
    elif type(cmap) == str:
        import matplotlib
        cmap = matplotlib.cm.get_cmap(cmap)
        board = board.astype(float)
    
    # Draw all the polygons
    for i in range(N):
        for j in range(N):
            ij = np.asarray((i,j))
            xy = _vis_consts.ij2xy @ ij
            x, y = xy
            patch = patches.Polygon(xy + _vis_consts.hexagon, facecolor=cmap(board[i,j]), zorder=-50)
            plt.gca().add_patch(patch)
            
            if text is not None:
                plt.text(x, y, text[i][j], color='white', horizontalalignment='center', verticalalignment='center')
            elif numbers:
                plt.text(x, y, '%d, %d' %(i,j), color='white', horizontalalignment='center', verticalalignment='center')
    
    # If a path is given, plot it also on the board
    if type(path) != type(None):
        show_path(path)
    
    # save if wished
    if save:
        plt.savefig(save)
    
def show_path(path):
    '''Adds a path to the plotted board, where path is a list of (i, j) - tuples.
    '''
    xs = []
    ys = []
    for (i, j) in path:
        ij = np.asarray((i,j))
        xy = _vis_consts.ij2xy @ ij
        xs.append(xy[0])
        ys.append(xy[1])
    plt.plot(xs, ys, c='orange', lw=6)
    
    
def print_board(board):
    '''Prints a simple ASCII representation of the board
    '''
    N = len(board)
    board = board.T
    char = ['·', 'r', 'b']
    print((N+1)*' ' + 2*N*'_')
    for i in range(N-1, -1, -1):
        line = ' '
        line += i*' ' + '/'
        for j in range(N):
            line += char[board[i,j]%3]
            if j<N-1: line += ' '
        line += '/'
        print(line)
    print((2*N+1)*'^')
    
    
def pos_on_board(x, y, N):
    '''For given coordinates x, y, this function returns the coordinates i, j of the cell we`re on.
    If x, y is on no cell, it returns None, None.'''
    pos_xy = np.asarray([x, y])
    pos_ij = _vis_consts.xy2ij @ pos_xy
    
    
    for i in range(N):
        if (i-pos_ij[0]) > 2: continue
        for j in range(N):
            if (j-pos_ij[1]) > 2: continue
            ij = 1.*np.array((i, j))
            xy = _vis_consts.ij2xy @ ij
            #if np.sum((pos_xy-xy)**2) <= hexag_factor:
            if np.sum((pos_xy-xy)**2) <= _vis_consts.hexag_factor**2:
                if np.all(np.abs(_vis_consts.dual_xy[:3] @ (pos_xy-xy)) < 0.5 - GAP/2):
                    return i, j
                else:
                    return None, None
    return None, None


def show_outerior(N):
    '''Shows outerior areas of the board (so that the players know which edges they got to connect).'''

    # Punkte direkt am Hexagonenfeld
    delta = _vis_consts.gap/np.cos(_vis_consts.thirty_degree)
    omikron = 1/2 # percentage of the most left point towards a complete halfstep
    xs = []
    ys = []
    for i in range(N):
        xs.append(-.5+i)
        ys.append(-np.tan(_vis_consts.thirty_degree)/2 +2*_vis_consts.gap/2*np.sqrt(1/3) - delta)
        xs.append(i)
        ys.append(-_vis_consts.hexag_factor - delta)

    # rechter Punkt, der an blau grenz und dem Spielfeld rechts unten
    x0 = N-1 + 1/4
    y0 = ys[-1] + 1/4*1/2

    # linkester Punkt
    x3 = xs[0]-omikron/2
    y3 = ys[0]-omikron/4

    # unten linker Punkt
    x2 = -1/2
    y2 = y3 - omikron/2*np.tan(_vis_consts.sixty_degree)

    # unten rechts Punkt
    ydel = y0-y2
    xdel = ydel/np.tan(_vis_consts.sixty_degree)
    x1 = x0+xdel
    y1 = y0-ydel

    xs += [x0, x1, x2, x3]
    ys += [y0, y1, y2, y3]
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    #plt.scatter(xs,ys, s=1, c='white')

    # Spiegeln Punktweise
    mx, my = _vis_consts.ij2xy @ np.asarray([(N-1)/2, (N-1)/2])
    xs2 = 2*mx - xs
    ys2 = 2*my - ys

    # Drehen und Spiegeln
    xs3 = xs - mx
    ys3 = ys - my
    c, s = np.sqrt(3)/2, 1/2 # sin and cos of 30°
    xs3, ys3 = np.array([[c, +s], [-s, c]]) @ np.asarray([xs3, ys3])
    ys3 = -ys3
    xs3, ys3 = np.array([[c, -s], [+s, c]]) @ np.asarray([xs3, ys3])
    xs3 = xs3 + mx
    ys3 = ys3 + my

    # Spiegeln Punktweise
    xs4 = 2*mx - xs3
    ys4 = 2*my - ys3

    patch = patches.Polygon(np.array((xs, ys)).T, facecolor=_vis_consts.color(+1), zorder=0)
    plt.gca().add_patch(patch)
    patch = patches.Polygon(np.array((xs2, ys2)).T, facecolor=_vis_consts.color(+1), zorder=0)
    plt.gca().add_patch(patch)
    patch = patches.Polygon(np.array((xs3, ys3)).T, facecolor=_vis_consts.color(-1), zorder=0)
    plt.gca().add_patch(patch)
    patch = patches.Polygon(np.array((xs4, ys4)).T, facecolor=_vis_consts.color(-1), zorder=0)
    plt.gca().add_patch(patch)