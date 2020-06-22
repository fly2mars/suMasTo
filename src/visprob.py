'''
Visualizing the TO problem.
input: class load or derived class for a TO problem.

'''

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as line
import matplotlib.lines as mlines

import suFuncStack 

from loads import Load, HalfBeam
from constraints import DensityConstraint

class VisProblem(object):
    def __init__(self):
        self.img = None
        self.load = None
        self.constraint = None
    def init(self, load, constraint):
        (nelx, nely) = load.shape()
         # mean density
        return np.ones((nely, nelx))*constraint.volume_frac()  
    
    def gen_fix_dof_coord(self):
        fix_dof = self.load.fixdofs()
        ndx = np.array(fix_dof) // 2
        x0 = ndx  // (self.load.nely + 1)
        y0 = self.load.nely - ndx % (self.load.nely + 1)
        return x0, y0
        
    def gen_force_arrow_coord(self):
        '''
        Compute arrow coordinate for force 
        '''
        arrow_len = 0.8
        dim = self.load.dim
        dof_idx = self.load.force()
        f_dof_idx = [i for i, v in enumerate(dof_idx) if v !=0]
        f_dof_idx = np.array(f_dof_idx)         
        dim = f_dof_idx % dim      # x direction: 0   y direction: 1
        ndx = f_dof_idx // 2
        x0 = ndx  // (self.load.nely + 1)
        y0 = self.load.nely - ndx % (self.load.nely + 1)
        # compute arrow start point        
        x1 = np.zeros(len(x0))
        y1 = np.zeros(len(x0))
        for i in range(len(x0)):
            f = dof_idx[f_dof_idx[i]]
            f = int(f/abs(f))
            if dim[i] == 0:  # x
                x1[i] = f * arrow_len
            else:
                y1[i] = f * arrow_len
        
        return x0, y0, x1, y1    # (x1,y1) -> (x0, y0)
       
    def draw(self, load, constraint):
        self.load = load
        self.constraint = constraint
        
        # Init design variables
        X = self.init(load, constraint)       
        # make a figure + axes
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        # make color map
        my_cmap = matplotlib.colors.ListedColormap(['w', 'r', 'g', 'b'])
        # set the 'bad' values (nan) to be white and transparent
        my_cmap.set_bad(color='w', alpha=0)
        # draw the grid
        (nelx, nely) = load.shape()
        for y in range(nely + 1):    
            plt.plot([0, nelx], [y,y], 'k-', lw=1, color='k')  
        for x in range(nelx + 1):
            plt.plot([x, x], [0,nely], 'k-', lw=1, color='k') 
           
        # marking the fix dof
        x0, y0 = self.gen_fix_dof_coord()
        for i in range(len(x0)):
            circle = plt.Circle((x0[i], y0[i]), 0.2, color='k')
            ax.add_patch(circle)
        
        # draw the load
        x0,y0, x1,y1 = self.gen_force_arrow_coord()
        for i in range(len(x0)):
            ax.arrow(x0[i], y0[i], x1[i], y1[i], head_width=0.2, head_length=0.1, fc='r', ec='r')

        ax.imshow(X, interpolation='none', cmap=my_cmap, extent=[-1, nelx+1, -1, nely+1], zorder=0)
        # turn off the axis labels
        ax.axis('off')
        plt.show()        


class VisMassTrend(object):
    '''
    
    '''
    def __init__(self, x_history, x, title="Mass trend", x_label = "Iteration", y_label="Mass"):
        self.x_history = x_history + [x] 
        self.X = []
        self.Y = []        
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
 
    def gen_mass_trend(self):   
        if len(self.x_history) == 0:
            return
        ori = np.sum(self.x_history[0])
        for i, v in enumerate(self.x_history):
            self.X.append(i)
            self.Y.append(np.sum(v) / ori)
            print("{} - {}".format(np.sum(v), ori))
       
        
    def draw(self):
        self.gen_mass_trend()
        plt.figure(2)
        plt.plot(self.X, self.Y, label="mass")
        plt.title("Mass of optimization domains")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.grid()   
        plt.show()
 
class VisConvergenceTrend(object):
    '''
    
    '''
    def __init__(self, x_history,title="Convergence trend", x_label = "Iteration", y_label="Change"):
        self.x_history = x_history
        self.X = []
        self.Y = []        
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
 
    def gen_trend(self):   
        if len(self.x_history) == 0:
            return
        covg = suFuncStack.convergence(5)
        for v in self.x_history:
            covg.add_data(v)
        X, Y = covg.get_data()   
        return X, Y
       
        
    def draw(self):
        X, Y = self.gen_trend()
        plt.figure(2)
        plt.plot(X, Y)#, label="Convergence trend")
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.grid()   
        plt.show()
        #print(Y)
        
if __name__ == "__main__":
    # input parameters
    nelx = 60
    nely = 20
    
    # constraints
    volfrac = 0.4
    xmin = 0.001
    xmax = 1.0    

    # loading/problem
    load = HalfBeam(nelx, nely)

    # constraints
    density_constraint = DensityConstraint(volume_frac = volfrac, density_min = xmin, density_max = xmax)

    # visualize problem
    vp = VisProblem()
    vp.draw(load, density_constraint)
    
    x = np.ones([3,3])
    X = []
    X.append(x)
    for i in range(10):
        x_i = np.random.random([3,3])
        X.append(x_i)
        
    covg = VisConvergenceTrend(X, x)
    covg.draw()

   