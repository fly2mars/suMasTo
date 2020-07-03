'''
topology optimization

2D only for now
'''

import numpy as np
import math
from scipy.sparse import coo_matrix

import suFuncStack 
from suAI.misc import debug
from suAI.mas import agent

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
    def run(self, method_name, agt):
        getattr(self, "ACT_"+method_name)(agt)  


# test: to be added into ActionPool()
def test_update(agt):
    if agt.pos in agt.env.boundary:
        return
    nely, nelx = agt.env.x.shape
    (ely, elx) = agt.pos
 
    # ugly hardwired constants to fix later
    xmin = agt.env.constraint.density_min()
    xmax = agt.env.constraint.density_max()  

    x = agt.env.x[ely, elx]
    dc = agt.env.dc[ely,elx]
    move = 0.03 * xmax
    
    # if dc > 2/3 neighbour
    xnew = agt.env.x[ely,elx]
    if dc > 0.4:
        xnew = xnew + move 
    else:
        xnew = xnew - move
    xnew = np.maximum(0, np.minimum(1.0, xnew))
    agt.env.x[ely,elx] = xnew
      
    
class AgentSIMP(agent.Agent):
    def __init__(self):
        super().__init__()
        self.x = 0
        self.dc = 0
    def sense(self, name):
        pass
    def make_decision(self):
        return ["test_update"]
    def act(self):
        re = self.make_decision()
        for i in re:
            self.acts.run(i,self)

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
        self.func_pool.add_func(test_update)
    
    def bind(self, kb):
        nely, nelx = self.x.shape 
        self.agts = {}
        for ely in range(nely):
            for elx in range(nelx):
                a = AgentSIMP()                
                a.set_acts(self.func_pool)   # dynamic define act 
                a.set_knowledge_engine(kb)
                a.set_environment(self)
                a.bind_pos((ely,elx))               
                self.agts[(ely,elx)] = a   
                
        return
    
    def update(self, x, dc):
        self.x = x
        
        m = np.abs(dc)
        m = m / np.max(m)  
             
        self.dc = m
    # topology optimization
    def run(self, load, constraint, x, penal, rmin, delta, loopy, history = False):
        # debug
        ugif = debug.MakeUFieldGif(load.nelx, load.nely, load.alldofs())
        self.boundary = load.boundary_ele()

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
            ## debug
            #debug.save_img(1.0 -x_history[-1], "r:/output%d.png" % loop)
            #debug.show_matrix(self.dc, True)
            #debug
            #im = debug.show_displacement_field(u, load.alldofs(), load.nelx, load.nely)
            #ugif.add_data(u.copy())
            
        # done
        #print(self.convergence.listy)
        #print('Saving gif for visulizing the u field ...')
        #ugif.save_gif("r:/u.gif")
        
        if history:
            return x, x_history
        else:
            return x, loop
   
    # initialization
    def init(self, load, constraint):
        (nelx, nely) = load.shape()
        # mean density
        self.x = np.ones((nely, nelx))*constraint.volume_frac()
        # set up and binding agents with environment & KE
        self.bind(None)
        
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
        dc = self.fast_filt(x,dc,self.H, self.Hs)
        # update
        x = self.update_agent(constraint, x, dc)

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
                ue = u[load.edofNode(elx, ely, nelx, nely)]
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
    
    # optimality criteria update
    def update_oc(self, constraint, x, dc):
        volfrac = constraint.volume_frac()
        xmin = constraint.density_min()
        xmax = constraint.density_max()

        # ugly hardwired constants to fix later
        move = 0.2 * xmax
        l1 = 0
        l2 = 100000
        lt = 1e-4

        nely, nelx = x.shape
        while (l2-l1 > lt):
            lmid = 0.5*(l2+l1)
            xnew = np.multiply(x, np.sqrt(-dc/lmid))

            x_below = np.maximum(xmin, x - move)
            x_above = np.minimum(xmax, x + move)
            xnew = np.maximum(x_below, np.minimum(x_above, xnew));

            if (np.sum(xnew) - volfrac*nelx*nely) > 0:
                l1 = lmid
            else:
                l2 = lmid

        return xnew    

    # update by multiple agents
    def update_agent(self, constraint, x, dc, kb=""):
        volfrac = constraint.volume_frac()
        xmin = constraint.density_min()
        xmax = constraint.density_max()
        
        self.update(x,dc)
    
        # ugly hardwired constants to fix later
        move = 0.2 * xmax
        l1 = 0
        l2 = 100000
        lt = 1e-4            
        
        ## evolve
        for a in self.agts.values():
            a.act()  # (1)sense (2)decite (3)act        
        xnew = self.x
        return xnew

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
    pass