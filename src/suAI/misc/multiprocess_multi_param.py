import multiprocessing
import time

def add(tuple_of_numbers):
    x = tuple_of_numbers[0]
    y = tuple_of_numbers[1]
    return x+y
if __name__ == '__main__':
    p = multiprocessing.Pool(4)
    inX = range(10)
    inY = range(10)
    print(p.map(add, zip(inX, inY)))