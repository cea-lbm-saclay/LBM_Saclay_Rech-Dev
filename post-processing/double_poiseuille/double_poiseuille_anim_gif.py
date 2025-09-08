#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# the following is adapted from the example
# https://matplotlib.org/examples/animation/subplots.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation

import vtk
from vtk.util.numpy_support import vtk_to_numpy

class DoublePoiseuilleAnimation(animation.TimedAnimation):
    def __init__(self):

        # vtk file prefix
        self.prefix = "LBM_D2Q9_double_poiseuille_"

        # interval between two files
        self.delta_i = 1000
        
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        self.xmin = 0
        self.xmax = 1
        self.nbx = 100
        self.dx = (self.xmax - self.xmin) / self.nbx
        
        self.x = np.linspace(self.xmin, self.xmax-self.dx, self.nbx) + self.dx/2

        ax1.set_xlabel('x')
        ax1.set_ylabel('vx')

        # numerical data
        self.line1 = Line2D([], [], color='black')

        # analytical solution
        self.line1s = Line2D([], [], color='red', linewidth=2)
        self.yp = self.double_poiseuille_analytic_solution()

        ax1.add_line(self.line1)
        ax1.add_line(self.line1s)
        ax1.set_xlim(self.xmin, self.xmax)
        ax1.set_ylim(0.0, 8e-5)
        #ax1.set_aspect('equal', 'datalim')

        animation.TimedAnimation.__init__(self, fig, interval=45, blit=True, repeat=False)

    def _draw_frame(self, framedata):

        # takes framedata modulo 50
        i = framedata - (framedata//50)*50

        print("i = {}".format(i))
        self.y = self.read_data(i)
        
        self.line1.set_data(self.x, self.y)
        self.line1s.set_data(self.x, self.yp)

        self._drawn_artists = [self.line1, self.line1s]

    def new_frame_seq(self):
        return iter(range(self.nbx))

    def getFilename(self,i):
        return self.prefix+"{:07}.vti".format(i)

    # read data
    def read_data(self,i):
        index = self.delta_i*i
        
        print(index)
        
        # open vti file
        filename = self.getFilename(index)
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(filename)
        reader.Update()
        
        # retrieve data and dimensions
        im = reader.GetOutput()
        rows, cols, _ = im.GetDimensions()
        
        # extract "vx" component (index2)
        vx = vtk_to_numpy ( im.GetCellData().GetArray(2) )
        vx = vx.reshape(rows-1, cols-1)
        
        # extract a vertical column
        y = vx[:,(int)(rows/2)]

        return y
    
    # double poiseuille analytical solution
    def double_poiseuille_analytic_solution(self):
        h = 0.5
        viscH = 0.35
        viscL = 0.07
        rap  = (viscH - viscL) / (viscH + viscL)
        
        uc = 5e-5
        G  = uc * (viscH + viscL) / h**2
        
        coef2 = G*h**2/(2*viscL)
        coef1 = G*h**2/(2*viscH)
        
        xx = self.x-h
        
        solution = (xx<=0)*( coef2*(-(xx/h)**2-(xx/h)*rap + 2*viscL/(viscH+viscL)) ) + (xx>0)*( coef1*(-(xx/h)**2-(xx/h)*rap + 2*viscH/(viscH+viscL)) )
        
        return solution
    
    def _init_draw(self):
        lines = [self.line1, self.line1s]
        for l in lines:
            l.set_data([], [])

ani = DoublePoiseuilleAnimation()
ani.save('./double_poiseuille_anim.gif', writer='imagemagick', fps=20)
plt.show()
