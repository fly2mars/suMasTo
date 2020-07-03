'''
Test multiprocess access matrix and return matrix
'''
from multiprocessing import Pool
from multiprocessing import Process
import concurrent.futures

import os
import time
import inspect
import numpy as np

import Qtrac
from pyknow import *
from random import choice

def timer(func):
    def wraper(*args, **kargs):
        start_time = time.perf_counter()
        f = func(*args, **kargs)
        end_time = time.perf_counter()
        print('Done in {} seconds'.format(end_time - start_time))   
        return f
    return wraper

class Light(Fact):
    """Info about the traffic light."""
    pass

class RobotCrossStreet(KnowledgeEngine):
    @Rule(Light(color='green'))
    def green_light(self):
        #print("Walk")
        pass

    @Rule(Light(color='red'))
    def red_light(self):
        #print("Don't walk")
        pass

    @Rule(AS.light << Light(color=L('yellow') | L('blinking-yellow')))
    def cautious(self, light):
        #print("Be cautious because light is", light["color"])
        pass
        
engine = RobotCrossStreet()        

class EnvSim():
    def __init__(self, nelx, nely):
        self.x = np.ones([nely,nelx])
        self.dc = np.ones([nely,nelx])
        
    def serialize(self):
        pass
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
    
    def run(self):
        self.update_parallel(self)
    @staticmethod
    @timer
    def update_parallel(env):    
        this_function_name = inspect.currentframe().f_code.co_name
        print("Begin  {}...".format (this_function_name))
        print(env.x)
        futures = set()         
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for idx_col, col_x, col_dc in get_jobs(env):
                future = executor.submit(f, idx_col, col_x, col_dc)
                futures.add(future)
            data = wait_for(futures, env)
            print(env.combine_data(data))
            return     
            
  



def info():
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
        print('process id:', os.getpid())
        
def f(col_idx, col_x, col_dc):
    '''
    Deal with each sub sequence
    '''   
    
    for i in range(len(col_x)):
        engine.reset()
        engine.declare(Light(color=choice(['green', 'yellow', 'blinking-yellow', 'red'])))
        engine.run()        
        col_x[i] = col_x[i] + 1
        col_dc[i] = col_dc[i] + 1
    
    return col_idx, col_x


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
            
        elif isinstance(err):
            Qtrac.report(str(err), True)
        else:
            raise err # Unanticipated
    return data

@timer 
def mp_futures(env):    
    this_function_name = inspect.currentframe().f_code.co_name
    print("Begin  {}...".format (this_function_name))
    print(env.x)
    futures = set()         
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for idx_col, col_x, col_dc in get_jobs(env):
            future = executor.submit(f, idx_col, col_x, col_dc)
            futures.add(future)
        data = wait_for(futures, env)
        print(env.combine_data(data))
        return 

@timer
def bench_mark(env):
    this_function_name = inspect.currentframe().f_code.co_name
    print("Begin  {}...".format (this_function_name))    
    for y in range(env.x.shape[0]):
        for x in range(env.x.shape[1]):
            engine.reset()
            engine.declare(Light(color=choice(['green', 'yellow', 'blinking-yellow', 'red'])))
            engine.run()    
            env.x[y][x] = env.x[y][x] + 1
    


if __name__ == '__main__':
    nelx = 150
    nely = 50
    env = EnvSim(nelx, nely)
    #env.update_parallel(env)
    env.run()
    bench_mark(env)
    