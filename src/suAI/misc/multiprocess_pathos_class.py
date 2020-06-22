from pathos.multiprocessing import ProcessingPool as Pool
from pyknow import *

class myClass:
    def __init__(self):
        pass

    def square(self, x):
        f = Fact(dc = 1)
        return x*x

    def run(self, inList):
        pool = Pool(8).map
        result = pool(self.square, inList)
        return result

if __name__== '__main__' :
    m = myClass()
    #print(m.run(range(10)))
    #m.square(3)