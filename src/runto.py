'''
tester for topology optimization code
'''
import sys, argparse
import numpy as np
import math
import matplotlib.pyplot as plt

import visprob

from loads import HalfBeam
from constraints import DensityConstraint
from fesolvers import LilFESolver, CooFESolver
from env import Environment
from oc import Oc

if __name__ == "__main__":
    # material properties
    young = 1
    poisson = 0.6

    # constraints
    volfrac = 0.4
    xmin = 0.001
    xmax = 1.0

    # input parameters
    nelx = 180
    nely = 60

    penal = 3.0
    rmin = 5.4

    delta = 0.02
    loopy = 30#math.inf
    
    parser = argparse.ArgumentParser(description="Run a toplogy optimizer.")
    parser.add_argument('--optimizer', dest='optimizer', required=True)
   
    args = parser.parse_args()    

    # loading/problem
    load = HalfBeam(nelx, nely)

    # optimizer
    verbose = True
    fesolver = CooFESolver(verbose = verbose)
    
    optimizer = None
    density_constraint = None
    if args.optimizer:
        if str(args.optimizer) == 'mas':
            optimizer = Environment(fesolver, young, poisson, verbose = verbose)
            # constraints
            density_constraint = DensityConstraint(volume_frac = 1.0, density_min = xmin, density_max = xmax)              
        if str(args.optimizer) == 'oc':
            optimizer = Oc(fesolver, young, poisson, verbose = verbose)    
             # constraints
            density_constraint = DensityConstraint(volume_frac = volfrac, density_min = xmin, density_max = xmax)    
 
      
    # compute
    history = True
    x = optimizer.init(load, density_constraint)
    x, x_more = optimizer.run(load, density_constraint, x, penal, rmin, delta, loopy, history)

    if history:
        x_history = x_more
        loop = len(x_history)
    else:    
        loop = x_more
        x_history = None

    # save
    if x_history:        
        import imageio
        x_history = [1 - v for v in x_history] 
        imageio.mimsave('r:/topopt.gif', x_history)
        covg = visprob.VisConvergenceTrend(x_history)
        covg.draw() 

    # plot
    plt.figure()
    plt.imshow(1-x, cmap=plt.cm.gray)
    plt.title(str(loop) + ' loops')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    
