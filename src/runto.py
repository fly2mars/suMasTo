'''
tester for topology optimization code
'''
import sys, argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import time


import visprob

from loads import HalfBeam
from constraints import DensityConstraint
from fesolvers import LilFESolver, CooFESolver
## import different environment
import env
import env_ke 
import env_ke_parallel
from oc import Oc

if __name__ == "__main__":
    # time comsuming estimation
    time_used = []
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
    rmin = 7.5

    delta = 0.02
    loopy = 30   
    
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
            optimizer = env.Environment(fesolver, young, poisson, verbose = verbose)
            # constraints
            density_constraint = DensityConstraint(volume_frac = 1.0, density_min = xmin, density_max = xmax)              
        if str(args.optimizer) == 'mas_ke':
            optimizer = env_ke.Environment(fesolver, young, poisson, verbose = verbose)
            # constraints
            density_constraint = DensityConstraint(volume_frac = 1.0, density_min = xmin, density_max = xmax)                      
        if str(args.optimizer) == 'oc':            
            optimizer = Oc(fesolver, young, poisson, verbose = verbose)    
             # constraints
            density_constraint = DensityConstraint(volume_frac = volfrac, density_min = xmin, density_max = xmax)    
        if str(args.optimizer) == 'mas_ke_parallel':
            optimizer = env_ke_parallel.Environment(fesolver, young, poisson, verbose = verbose)
            # constraints
            density_constraint = DensityConstraint(volume_frac = 1.0, density_min = xmin, density_max = xmax)                              
    
    # statistic time
    start_time = time.perf_counter()
    # compute
    history = True
    x = optimizer.init(load, density_constraint)
    x, x_more = optimizer.run(load, density_constraint, x, penal, rmin, delta, loopy, history)
    end_time = time.perf_counter()
    print('Processed {} elements in {} seconds'.format(nelx*nely, end_time - start_time))      
    

    if history:
        x_history = x_more
        loop = len(x_history)
    else:    
        loop = x_more
        x_history = None

    
    # save
    if x_history:        
        import imageio
        import cv2
        #cv2.imwrite('r:/test.png', x_history[-1], )
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
    
    
