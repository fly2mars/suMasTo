'''
tester for topology optimization code
'''

import numpy as np
import math
import matplotlib.pyplot as plt

import visprob

from loads import HalfBeam
from constraints import DensityConstraint
from fesolvers import LilFESolver, CooFESolver
from env import Environment

if __name__ == "__main__":
    # material properties
    young = 1
    poisson = 0.6

    # constraints
    volfrac = 0.4
    xmin = 0.001
    xmax = 1.0

    # input parameters
    nelx = 90
    nely = 30

    penal = 3.0
    rmin = 5.4

    delta = 0.02
    loopy = 20#math.inf

    # loading/problem
    load = HalfBeam(nelx, nely)

    # constraints
    density_constraint = DensityConstraint(volume_frac = volfrac, density_min = xmin, density_max = xmax)

    # optimizer
    verbose = True
    fesolver = CooFESolver(verbose = verbose)
    optimizer = Environment(fesolver, young, poisson, verbose = verbose)

    # compute
    history = True
    x = optimizer.init(load, density_constraint)
    x, x_more = optimizer.evolve(load, density_constraint, x, penal, rmin, delta, loopy, history)

    if history:
        x_history = x_more
        loop = len(x_history)
    else:    
        loop = x_more
        x_history = None

    # save
    if x_history:
        # import scipy.misc
        # sequence = [scipy.misc.toimage(x, cmin=0, cmax=1) for x in x_history]
        import imageio
        x_history = [1 - v for v in x_history] 
        imageio.mimsave('r:/topopt.gif', x_history)
        covg = visprob.VisConvergenceTrend(x_history)
        covg.draw() 

    # plot
    plt.figure()
    plt.imshow(x, cmap=plt.cm.gray)
    plt.title(str(loop) + ' loops')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    
