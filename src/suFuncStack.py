################################################
#Some class for computing time serias.
#  eg. convergence.
################################################
import collections
import numpy as np
import copy


class data_stack():
    '''
    A stack for first in first out, which always hold max_len items
    example:
           ds = data_stack(5)
           ds.add(1.0)
           ....
           ds.mean()
    '''
    def __init__(self,max_len):
        self.max_len = max_len
        self.data = collections.deque()
    def __getitem__(self,index):
        return self.data[index]
    
    def __setitem__(self,index,value):
        self.data[index] = value    
    def len(self):
        return len(self.data)
    def add_data(self,d):        
        if(len(self.data) == self.max_len):
            self.data.popleft()        
        self.data.append(copy.deepcopy(d))        
    def mean(self):
        s = 0
        if self.len() > 0:
            s = self.data[0] * 0
        else:
            return 0
        for v in self.data:
            s += v          
        return s / self.len()
    def flush(self):
        print(self.data)
        
class convergence():
    '''
    return 2D points(conv, iter) for showing convergence trend.
    @max_len is the length of a time window.
    '''
    def __init__(self, max_len):
        self.diffs = data_stack(max_len)
        self.data  = data_stack(2)
        self.listx = []
        self.listy = []
    def add_data(self, y):
        '''
        total_max is the max sum of input data, eg. for np.ones([3,3]), the sum_total = 9.
        '''                      
        self.data.add_data(y)
        if self.data.len() == 1:
            self.full_max = np.shape(y)[0] * np.shape(y)[1]
        if self.data.len() == 2:
            congv = self.get_change()
            self.diffs.add_data(congv)
            self.listy.append(self.diffs.mean()) 
    
    def is_convergence(self):
        mean = self.diffs.mean()
        print(mean)
        if mean < 0.00001:
            return True
        return False
    
    def get_data(self):
        self.listx = [i for i in range(1, len(self.listy)+1)]
        return self.listx, self.listy
    
    # compare difference between two matrix
    def get_change(self): 
        change = np.sum( abs(self.data[1] - self.data[0]) )
        return change / self.full_max

if __name__ == "__main__":
    d = data_stack(3)
    for i in range(10):
        d.add_data(i)
        
        d.flush()
        print(d.mean())
        
