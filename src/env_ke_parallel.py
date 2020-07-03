'''
Class env_ke provide objects for buiding environments, which load and agent
topology optimization by MAS

2D only for now
'''
import time
import numpy as np
import math
from scipy.sparse import coo_matrix

import suFuncStack 
from suAI.misc import debug
from suAI.mas import agent
from suAI.misc import Qtrac
from suAI.ke.ke import KnowledgeEngineBase
from pyknow import *

import concurrent.futures

from random import choice
if __name__ == "__main__":
    from loads import HalfBeam
    from constraints import DensityConstraint
    from fesolvers import LilFESolver, CooFESolver   
    
    
    


####################
#   ActionPool     #   
####################   
class ActionPool(object):
    def __init__(self):
        pass
    def set_parent(self, parent):
        self.parent = parent
    def add_func(self, func):
        f = classmethod(func)
        setattr(self, 'ACT_'+func.__name__, func)
    def get_acts(self):
        func_list = dir(self)
        func_list = [i for i in func_list if i[:4]=="ACT_"]
        return func_list
    def run(self, method_name, pos):
        return getattr(self, "ACT_"+method_name)(pos)  
        
####################
#   Actions        #   
####################   
# action: to be added into ActionPool()
def increase_density(x):        
    move = 0.03    
    xnew = x + move 
    xnew = np.maximum(0, np.minimum(1.0, xnew))        
    return xnew
    
# action: to be added into ActionPool()
def decrease_density(x):    
    move = 0.03    
    xnew = x - move     
    xnew = np.maximum(0, np.minimum(1.0, xnew))    
    return xnew

##########
#   KE   #
##########
class ke_mas(KnowledgeEngineBase, KnowledgeEngine):
    '''    
    1. Hold an knowlege base (eg. rules and facts) for MAS based evolution
    2. Help agents to make a decision.
    '''
    def __init__(self):
        KnowledgeEngineBase.__init__(self)
        KnowledgeEngine.__init__(self)
    
    def reset(self):
        super().reset()
        self.answers = []
        
    @DefFacts()
    def __init_facts(self):
        yield(Fact(method='mas'))
        return
    
    ## dc > 0.4 -> 
    @Rule (Fact(dc = MATCH.dc),
           TEST(lambda dc: dc > 0.4)
          )
    def increase_density(self):
        self.answers.append('increase_density')
        
    @Rule (Fact(dc = MATCH.dc),
           TEST(lambda dc: dc <= 0.4)
           )
    def decrease_density(self):
        self.answers.append('decrease_density')
        
    def add_facts(self, facts):
        self.declare(*facts)
        
    
    def status(self):
        print(self.facts)


###################################################
#  Global variable for parallel computing         #
###################################################
env  = None
acts = None

acts = ActionPool()
acts.add_func(decrease_density)
acts.add_func(increase_density)     


def timer(func):
    def wraper(*args, **kargs):
        start_time = time.perf_counter()
        f = func(*args, **kargs)
        end_time = time.perf_counter()
        print('Done in {} seconds'.format(end_time - start_time))   
        return f
    return wraper

# parallel processing method  
def col_agents_act(col_idx, col_x, col_dc):
    '''
    Deal with each sub columns
    '''       
    ke = ke_mas()
    for i in range(len(col_x)):         
        f = Fact(dc = float(col_dc[i]))
        fs = []
        fs.append(f)
        ke.reset()
        ke.add_facts(fs)        
        ke.run()
        for func in ke.answers:
            col_x[i] = acts.run(func,col_x[i])        
    
    return col_idx, col_x

def update_parallel(env, constraint):
    #volfrac = constraint.volume_frac()
    #env.xmin = constraint.density_min()
    #env.xmax = constraint.density_max()   
    
    futures = set()         
    data = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for idx_col, col_x, col_dc in get_jobs(env):
            future = executor.submit(col_agents_act, idx_col, col_x, col_dc)            
            futures.add(future)
        data = wait_for(futures, env)
    
    return env.combine_data(data)

def get_jobs(env):
    dc_cols = env.divide_data(env.dc)
    x_cols = env.divide_data(env.x)
    for idx_col in range(len(dc_cols)):
        yield idx_col, x_cols[idx_col].flatten(),  dc_cols[idx_col].flatten()


def wait_for(futures, env):
    canceled = False    
    data = {}
    for future in concurrent.futures.as_completed(futures):
        err = future.exception()
        if err is None:
            result = future.result()
            #col_idx_arr += [result.col_idx]
            #ata += result.col_data
            data[result[0]] = result[1]
            #print(result)
            
        elif isinstance(err, TypeError):
            Qtrac.report(str(err), True)
        else:
            raise err # Unanticipated
    return data            

###################
#   Environment   #
###################  
class Environment(object):
    
    '''
    young: young's modulus
    poisson: poisson ratio
    '''
    def __init__(self, fesolver, young = 1, poisson = 0.3, verbose = False):
        self.convergence = suFuncStack.convergence(3)
        self.fesolver = fesolver
        self.young = young
        self.poisson = poisson
        self.dim = 2
        self.verbose = verbose
        self.x = []
        self.agts = {}
        self.func_pool = ActionPool()
        self.func_pool.set_parent(self)        
        
        # test action adding        
        self.func_pool.add_func(decrease_density)
        self.func_pool.add_func(increase_density)            
   
    def divide_data(self, m):
        '''
        dividing matrix into columns
        return list of np.ndarray
        '''
        nely, nelx = m.shape
        X = np.split(m, nelx, axis=1)
        return X
    
    def combine_data(self, data):
        '''
        combining columns into matrix
        notice the input col in data is 1d array
        '''
        if len(data) < 1:
            return []
        m = np.vstack(data[0])
      
        for i in range(1,len(data)):
            m = np.concatenate((m, np.vstack(data[i]) ), axis=1)
        
        return m
    
    def run(self, load, constraint, x, penal, rmin, delta, loopy, history = False):       
        loop = 0 # number of loop iterations
        change = 1.0 # maximum density change from prior iteration
        self.load = load
        self.constraint = constraint
        self.convergence.add_data(x)
        self.H, self.Hs = self.pre_filter(x, rmin)
        
        if history:
            x_history = [x.copy()]

        while (not self.convergence.is_convergence()) and (loop < loopy):
            loop = loop + 1
            x, u = self.iter(load, constraint, x, penal, rmin) 
            self.convergence.add_data(x)
            if self.verbose: print('iteration ', loop, ', change ', self.convergence.listy[-1], flush = True)
            if history: x_history.append(x.copy())
        
        if history:
            return x, x_history
        else:
            return x, loop
   
    # initialization
    def init(self, load, constraint):
        (nelx, nely) = load.shape()
        # mean density
        self.x = np.ones((nely, nelx))*constraint.volume_frac()      
        
        return self.x

    # iteration
    def iter(self, load, constraint, x, penal, rmin):

        xold = x.copy()

        # element stiffness matrix
        ke = self.lk(self.young, self.poisson)

        # displacement via finite element analysis
        u = self.fesolver.displace(load, x, ke, penal)

        # compliance and derivative
        c, dc = self.comp(load, x, u, ke, penal)

        # filter
        #dc = self.filt(x, rmin, dc)
        self.dc = self.fast_filt(x,dc,self.H, self.Hs)
        # update
        x = self.update(x, constraint)

        # how much has changed?
        change = np.amax(abs(x-xold))
        return x, u
    
    # compliance and its derivative
    def comp(self, load, x, u, ke, penal):
        c = 0
        dc = np.zeros(x.shape)

        nely, nelx = x.shape
        for ely in range(nely):
            for elx in range(nelx):
                ue = u[load.edofOld(elx, ely, nelx, nely)]
                ce = np.dot(ue.transpose(), np.dot(ke, ue))
                c = c + (x[ely,elx]**penal)*ce
                dc[ely,elx] = -penal*(x[ely,elx]**(penal-1))*ce

        return c, dc

    def pre_filter(self, x, rmin):
        rminf = round(rmin)
        nely, nelx = x.shape
        nfilter=int(nelx * nely * ((2 * rminf + 1)** 2)) 
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row=i*nely+j    #index order is the same as that for elements        
                for k in range(max(i-rminf, 0), min(i+rminf+1, nelx)):
                    for l in range(max(j-rminf, 0), min(j+rminf+1, nely)):                    
                        col = k*nely+l  #index order is the same as that for elements
                        weight = max(0, rmin - np.sqrt((i-k)**2+(j-l)**2));
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = weight
                        cc += 1
        H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).todense()
        Hs = H.sum(1)    
        
        return H, Hs
    
    def fast_filt(self,x, dc, H, Hs):   
        nely,nelx = x.shape
        s = nely*nelx
        x_col = x.flatten('F').reshape([s,1])         
        dc_col = dc.flatten('F').reshape([s,1])
        
        xdc = x_col * dc_col       
        dcf = np.dot(H,xdc)/np.multiply(Hs, x_col)
        dc = dcf.reshape([nely,nelx], order='F')
        return dc
    
    # filter
    def filt(self, x, rmin, dc):
        rminf = round(rmin)

        dcn = np.zeros(x.shape)
        nely, nelx = x.shape

        for i in range(nelx):
            for j in range(nely):
                sum = 0.0
                for k in range(max(i-rminf, 0), min(i+rminf+1, nelx)):
                    for l in range(max(j-rminf, 0), min(j+rminf+1, nely)):
                        weight = max(0, rmin - math.sqrt((i-k)**2+(j-l)**2));
                        sum = sum + weight;
                        dcn[j,i] = dcn[j,i] + weight*x[l,k]*dc[l,k];
            
                dcn[j,i] = dcn[j,i]/(x[j,i]*sum);

        return dcn
          
    
    def update(self, x, constraint):
        self.x = x        
        m = np.abs(self.dc)
        m = m / np.max(m)
        self.dc = np.asarray(m)
        
        x = update_parallel(self, constraint)       
        
        return x
        
   
    # element (local) stiffness matrix
    def lk(self, young, poisson):
        e = young
        nu = poisson
        k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        ke = e/(1-nu**2)* \
            np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                       [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                       [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                       [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                       [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                       [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                       [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                       [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);

        return ke




    
if __name__ == "__main__":    
    
    # loading/problem
    nelx = 15
    nely = 5
    penal = 3.0
    rmin = 7.5

    delta = 0.02
    loopy = 10#math.inf    
    load = HalfBeam(nelx, nely)

    # parallel optimizer testing
    verbose = True
    fesolver = CooFESolver(verbose = verbose)
    
    optimizer = None
    density_constraint = None   
    
    # material properties
    young = 1
    poisson = 0.6    
        
    optimizer = Environment(fesolver, young, poisson, verbose = verbose)
    # constraints
    density_constraint = DensityConstraint(volume_frac = 1.0, density_min = 0, density_max = 1)     
    
    # statistic time
    start_time = time.perf_counter()
    # compute
    history = True
    x = optimizer.init(load, density_constraint)
    x, x_more = optimizer.run(load, density_constraint, x, penal, rmin, delta, loopy, history)
    end_time = time.perf_counter()
    print('Processed {} elements in {} seconds'.format(nelx*nely, end_time - start_time))          
        
       
    
    
    
