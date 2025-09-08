#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import vtk
from vtk.util.numpy_support import vtk_to_numpy

# vtk file prefix
prefix = "LBM_D2Q9_double_poiseuille_"

# interval between two files
delta_i = 1000

def getFilename(i):
    return prefix+"{:07}.vti".format(i)


xmin = 0.0
xmax = 1.0
nbx = 100
dx=(xmax-xmin)/nbx


x = np.linspace(xmin, xmax-dx, nbx) + dx/2

fig = plt.figure() # initialise la figure
line, = plt.plot([],[]) 
plt.xlim(xmin, xmax)
plt.ylim(-1e-5,8e-5)

# fonction à définir quand blit=True
# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i):
    index = delta_i*i

    print(index)
    
    # open vti file
    filename = getFilename(index)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    # retrieve data and dimensions
    im = reader.GetOutput()
    rows, cols, _ = im.GetDimensions()

    # extract "vx" component
    vx = vtk_to_numpy ( im.GetCellData().GetArray(2) )
    vx = vx.reshape(rows-1, cols-1)

    # extract a vertical column
    y = vx[:,(int)(rows/2)]

    line.set_data(x, y)
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=49, blit=True, interval=20, repeat=True)

plt.show()
