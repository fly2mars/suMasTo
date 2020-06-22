'''

'''
import numpy as np
from pathos.multiprocessing import ProcessPool
import concurrent.futures
import multiprocessing
import time
import inspect
from pyknow import *


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
        getattr(self, "ACT_"+method_name)(pos)  
        
####################
#   Actions        #   
####################   
# action: to be added into ActionPool()
def increase_density(pos):
    global env        
    (ely,elx) = pos
   
    xmax = env.constraint.density_max()  
    move = 0.03 #* xmax
        
    xnew = env.x[ely,elx]
    xnew = xnew + move 
    xnew = np.maximum(0, np.minimum(1.0, xnew))
    env.x[ely,elx] = xnew

####################
#   Agent          #   
####################   
class Agent():
    def __init__(self):
        self.pos = []
    def bind_pos(pos):
        self.pos = pos
    def act(self):
        self.run(1)
        
    def run(self, dc):        
        self.dc = dc[self.pos]
        self.dc = self.dc + 0.1  
        
####################
#   Environment    #   
####################         
class Environment(object):  
    def __init__(self, nelx, nely):        
        self.x = np.random.random([nelx, nely])
        self.agts = {}                
    
    def bind(self, kb):
        nely, nelx = self.x.shape 
        self.agts = {}
        for ely in range(nely):
            for elx in range(nelx):
                a = Agent() 
                a.bind_pos((ely,elx))               
                self.agts[(ely,elx)] =  a
        return
    
    def build_paralle_data(self):
        pass
 
    # topology optimization
    def run(self, x, loopy):               
        loop = 0
        while (loop < loopy):
            loop = loop + 1
            x = self.iter(x) 
        return x
    # initialization
    def init(self):                   
        return self.x

    # iteration
    def iter(self, x):

        xold = x.copy()
        dc = np.random.random(self.x.shape[0] * self.x.shape[1])
        # update
        x = self.update_agent(x, dc)
        # how much has changed?
        change = np.amax(abs(x-xold))
        return x
    
    # update by multiple agents
    def update_agent(self, x, dc, ke=""):
        
        xmin = 0
        xmax = 1   
        
        ## parallel evolve
        for a in self.agts.values():
            a.act()  # (1)sense (2)decite (3)act        
        xnew = self.x
        return xnew

########################################
#   Paralle example code               # 
########################################  
    
def timer(func):
    def wraper(*args, **kargs):
        start_time = time.perf_counter()
        f = func(*args, **kargs)
        end_time = time.perf_counter()
        print('Done in {} seconds'.format(end_time - start_time))   
        return f
    return wraper

def run_once(pos, dc=1):
    d = dc
    d = dc + 0.1
    a = 0
    f = Fact(dc = float(dc))
    print(f)
    for i in range(1000):
        a = a + i*i
    return a
    
def run_once_zip(param):
    p1 = param[0]
    p2 = param[1]
    f = Fact(dc = float(p1))
    print(f)
    a = 0
    for i in range(1000):
        a = a + i*i
    return a
        
@timer 
def run_process_by_pathos_blockingmap(pos_arr, dc_arr):
    this_function_name = inspect.currentframe().f_code.co_name
    print("Begin  {}...".format (this_function_name))
    pool = ProcessPool(nodes=8)
    # do a blocking map on the chosen function
    pool.map(run_once, pos_arr, dc_arr)
    #print(pool.map(run_once, pos_arr, dc_arr))   
     

@timer 
def run_process_by_pathos_nonblockingmap(pos_arr, dc_arr):
    this_function_name = inspect.currentframe().f_code.co_name
    print("Begin {}...".format (this_function_name))
    pool = ProcessPool(nodes=8)
    # do a non-blocking map, then extract the results from the iterator
    results = pool.imap(run_once, pos_arr, dc_arr) 
    list(results)
    #print("...")
    #print(list(results))      
    
  
    
@timer 
def run_process(pos_arr, dc_arr):
    this_function_name = inspect.currentframe().f_code.co_name
    print("Begin  {}...".format (this_function_name))
    for i in range(len(pos_arr)):
        run_once(pos_arr[i], dc_arr[i])

@timer
def run_process_futures(pos_arr, dc_arr):
    this_function_name = inspect.currentframe().f_code.co_name
    print("Begin  {}...".format (this_function_name))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        to_do = []
        for pos in pos_arr:
            future = executor.submit(run_once, pos)
            to_do.append(future)
            
        for future in concurrent.futures.as_completed(to_do):
            future.result()    
            
@timer
def run_process_pool(pos_arr, dc_arr):
    this_function_name = inspect.currentframe().f_code.co_name
    print("Begin  {}...".format (this_function_name))
    with multiprocessing.Pool(processes = 8) as pool:         # start cpu_num worker processes
        result = pool.apply_async(run_once, pos_arr)              # evaluate "download_one" asynchronously in a single process
        pool.map(run_once, pos_arr)      
        

@timer
def run_process_pool_zip(param):
    this_function_name = inspect.currentframe().f_code.co_name
    print("Begin  {}...".format (this_function_name))
    with multiprocessing.Pool(processes = 8) as pool:         # start cpu_num worker processes
        result = pool.apply_async(run_once_zip, param)              # evaluate "run_once_zip" asynchronously in a single process
        pool.map(run_once_zip, param)          
########################################
#   Paralle for agent update           #
########################################     
##########
#   KE   #
##########
class RobotCrossStreet(KnowledgeEngine):
    @Rule(Fact(color='green'))
    def green_light(self):
        print("Walk")

    @Rule(Fact(color='red'))
    def red_light(self):
        print("Don't walk")
   
ke =    RobotCrossStreet()

def run_agents_update(param):
    p1 = param[0]
    p2 = param[1]
    f = Fact(dc = float(p2))
    ke.reset()
    ke.run()
    
    pass

@timer
def run_agents_update_parallel(param):
    this_function_name = inspect.currentframe().f_code.co_name
    print("Begin  {}...".format (this_function_name))
    with multiprocessing.Pool() as pool:         # start cpu_num worker processes
        result = pool.apply_async(run_agents_update, param)              # evaluate "download_one" asynchronously in a single process
        pool.map(run_agents_update, param)      
    pass

if __name__ == '__main__':
    env = Environment(15, 5)
    num = 100
    pos_arr = range(num)
    dc_arr = np.random.random([num])    
    
    ## normal
    #run_process(pos_arr, dc_arr)
    ## pathos
    #run_process_by_pathos_blockingmap (pos_arr, dc_arr)
    #run_process_by_pathos_nonblockingmap(pos_arr, dc_arr)
    ## futures
    #run_process_futures(pos_arr, dc_arr)
    ## multiprocessing
    #run_process_pool_zip(zip(pos_arr, dc_arr))
    #print(pos_arr)
    
    optimizer = Environment(100,100)
    x = optimizer.init()
    x = optimizer.run(x, 10)
    run_agents_update_parallel(zip(pos_arr,dc_arr))
    
    