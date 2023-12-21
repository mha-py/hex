#!/usr/bin/env python
# coding: utf-8

# ## Pyglet
# ist ein OpenGL Package. Tutorial [hier](https://greendalecs.wordpress.com/2012/04/21/3d-programming-in-python-part-1/) her.<br>
# Gutes PDF (OpenGL in Java) [hier](http://www.cs.cornell.edu/courses/cs4620/2011fa/lectures/practicum01.pdf)<br><br>

# In[1]:


import pyglet
from pyglet.gl import *
from pyglet.window import key
from pyglet.window import mouse

import numpy as np

from hex_show import *
from hex_show_gl import *
from hex_pathfinding import *



from time import time

bsize = 6

mouse_pos = np.zeros(2)

def reset():
    global board, turn, over, path, resettime
    # Initialization of the game
    board = np.zeros((bsize, bsize), 'int')
    turn = 1 # whos turn is it?
    over = False
    path = None # winning path if game ends
    resettime = None

reset()

# Allows anti aliasing
config = pyglet.gl.Config(sample_buffers=1, samples=8)
window = pyglet.window.Window(config=config, resizable=False) 


@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == mouse.LEFT:
        global board, turn, over, path, resettime
        
        N = len(board)
        # Scaling between coordinate x/y and screen x/y
        _xy_sc = _xy_off = 50 / (N/8)
        
        # Find cell the mouse clicked on
        x, y = (x-_xy_off)/_xy_sc, (y-_xy_off)/_xy_sc
        i, j = pos_on_board(x, y, bsize)
        if i is None:
            return
        
        # Check if game is over
        if over:
            return
        # Check if cell was already taken
        if board[i,j] != 0:
            return
        
        # Now take the cell
        board[i,j] = turn
        # Change current player
        turn *= -1
        if winner(board):
            over = True
            path = winning_path(board)
            resettime = time() + 3
        
        
@window.event
def on_mouse_motion(x, y, dx, dy):
    global mouse_pos
    
    # Scaling between coordinate x/y and screen x/y
    xy_sc = xy_off = 50 / (bsize/8)
    
    mouse_pos = (x-xy_off)/xy_sc, (y-xy_off)/xy_sc
    mouse_pos = x, y
        
        
@window.event
def on_draw():

    # Clear buffers (necessary?)
    glClear(GL_COLOR_BUFFER_BIT)
    #glEnable(GL_BLEND)                                                            
    #glEnable(GL_LINE_SMOOTH)
    #glHint (GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
    glColor3d(0.8, 0.0, 0.0)
    
    if resettime:
        if time() > resettime:
            reset()

    show_board_gl(board, mouse_pos, path=path)
    
    
pyglet.app.run()


# In[ ]:





# In[ ]:




