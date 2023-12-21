'''
Provides a function for rendering the board in the pyglet/opengl framework
17.8.2020 (c) mha
'''


import numpy as np
import pyglet
from pyglet.gl import *

from hex_show import *

import hex_show
_vis_consts = hex_show._vis_consts



def show_board_gl(board, mouse_pos=(-1, -1), numbers=False, path=None, wonby=0):
    '''Renders the board on the screen, usage is very similar to the matplotlib version.
    '''
    Nx, Ny = board.shape
    assert Nx == Ny, 'Your board is not quadratic??'
    N = Nx
    
    # Scaling between coordinate x/y and screen x/y
    _xy_sc = _xy_off = 50 / (N/8)
    
    if numbers:
        raise NotImplemented
    
    c = [(.2,.2,.2), (.6,.0,.0), (.0,.05,.85)]
    c = [np.array(cc) for cc in c]
    
    x_mouse = (mouse_pos[0]-_xy_off) / _xy_sc
    y_mouse = (mouse_pos[1]-_xy_off) / _xy_sc
    ij_mouse = pos_on_board(x_mouse, y_mouse, N)
    
    hexa = _xy_sc * _vis_consts.hexagon
    
    for i in range(N):
        for j in range(N):
            
            # Coordinate change and rescale
            ij = np.asarray((i,j))
            xy = _vis_consts.ij2xy @ ij
            xy = xy * _xy_sc + _xy_off
            x, y = xy
            
            glBegin(GL_POLYGON)
            
            clr = c[board[i,j]%3]
            if (i, j) == ij_mouse:
                clr = 0.6*clr
            glColor3d(*clr)
            
            for k in range(6):
                glVertex2f(hexa[k,0]+x, hexa[k,1]+y)
            glEnd()
            
            '''
            # Not implemented atm since is overwritten by the path
            if numbers:
                label = pyglet.text.Label('%d, %d' %(i,j),
                                          font_name='Verdana',
                                          font_size=12,
                                          x=x, y=y,
                                          anchor_x='center', anchor_y='center')
                label.draw()
            ''';
            
    show_outerior_gl(N)
    
    if path:
        show_path_gl(path, _xy_sc, _xy_off, extend=True, wonby=wonby)
        
              
                
def show_path_gl(path, _xy_sc, _xy_off, extend=True, wonby=0):
    '''Adds a path to the plotted board, where path is a list of (i, j) - tuples.
    Very similar to the matplotlib version.
    '''
    delta = 0.75
    orange = (1.0, 0.64, 0.0)
    
    
    if not path: return
    if extend and wonby: # extend path to outerior (schablone um die hexagons)
        if wonby > 0: path = [(j,i) for (i,j) in path]
        i, j = path[0]
        i += delta
        path = [(i,j)] + path
        i, j = path[-1]
        i -= delta
        path = path + [(i,j)]
        if wonby > 0: path = [(j,i) for (i,j) in path]
        
    glColor3d(*orange)
    glLineWidth(7)
    glBegin(GL_LINE_STRIP)
    for (i, j) in path:
        ij = np.asarray((i,j))
        xy = _vis_consts.ij2xy @ ij
        xy = xy * _xy_sc + _xy_off
        glVertex2f(xy[0], xy[1])
    glEnd()
    
    for (i, j) in path:
        ij = np.asarray((i,j))
        xy = _vis_consts.ij2xy @ ij
        xy = xy * _xy_sc + _xy_off
        glBegin(GL_TRIANGLE_FAN)
        for phi in np.linspace(0, 2*np.pi, 10):
            dx, dy = 3.5*np.cos(phi), 3.5*np.sin(phi)
            glVertex2f(xy[0]+dx, xy[1]+dy)
        glEnd()
              
            
def show_outerior_gl(N):
    '''Shows outerior areas of the board (so that the players know which edges they got to connect).'''
    # Punkte direkt am Hexagonenfeld
    delta = _vis_consts.gap/np.cos(_vis_consts.thirty_degree)
    omikron = 1/2 # percentage of the most left point towards a complete halfstep
    shading = 0.66 # darkening of the color
    
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
    xs1 = np.asarray(xs)
    ys1 = np.asarray(ys)
    
    # nur bei gl version: Skalieren auf Pixeleinheiten
    _xy_sc = _xy_off = 50 / (N/8)
    xs1 = xs1 * _xy_sc + _xy_off
    ys1 = ys1 * _xy_sc + _xy_off
    
    #patch = patches.Polygon(np.array((xs, ys)).T, facecolor=_vis_consts.color(+1), zorder=0)
    #plt.gca().add_patch(patch)

    # Spiegeln Punktweise
    mx, my = _vis_consts.ij2xy @ np.asarray([(N-1)/2, (N-1)/2]) * _xy_sc + _xy_off
    xs2 = 2*mx - xs1
    ys2 = 2*my - ys1

    # Drehen und Spiegeln
    xs3 = xs1 - mx
    ys3 = ys1 - my
    c, s = np.sqrt(3)/2, 1/2 # sin and cos of 30°
    xs3, ys3 = np.array([[c, +s], [-s, c]]) @ np.asarray([xs3, ys3])
    ys3 = -ys3
    xs3, ys3 = np.array([[c, -s], [+s, c]]) @ np.asarray([xs3, ys3])
    xs3 = xs3 + mx
    ys3 = ys3 + my

    # Spiegeln Punktweise
    xs4 = 2*mx - xs3
    ys4 = 2*my - ys3
    
    
    # triangulation des polygons (da gl nonconvexe polygone nicht handeln kann)
    triags = []
    for k in range(N-1):
        triags.append([2*k+1, 2*k+2, 2*k+3])
    triags.append([2*N+3, 0, 1])
    triags.append([2*N+3, 2*N+2, 1])
    triags.append([2*N+0, 2*N+1, 2*N-1])
    triags.append([2*N+2, 1, 2*N-1])
    triags.append([2*N+2, 2*N+1, 2*N-1])
    
    xs, ys = xs1, ys1
    glBegin(GL_TRIANGLES)
    glColor3d(*(_vis_consts.color(+1)*shading))
    for tr in triags:
        for i in tr:
            glVertex2f(xs[i], ys[i])
    glEnd()
    
    
    xs, ys = xs2, ys2
    glBegin(GL_TRIANGLES)
    glColor3d(*(_vis_consts.color(+1)*shading))
    for tr in triags:
        for i in tr:
            glVertex2f(xs[i], ys[i])
    glEnd()
    
    
    xs, ys = xs3, ys3
    glBegin(GL_TRIANGLES)
    glColor3d(*(_vis_consts.color(-1)*shading))
    for tr in triags:
        for i in tr:
            glVertex2f(xs[i], ys[i])
    glEnd()
    
    
    xs, ys = xs4, ys4
    glBegin(GL_TRIANGLES)
    glColor3d(*(_vis_consts.color(-1)*shading))
    for tr in triags:
        for i in tr:
            glVertex2f(xs[i], ys[i])
    glEnd()
    
    
    
    