import concurrent.futures
import requests
import time
import os
import multiprocessing

def timer(func):
    def wraper(*args, **kargs):
        start_time = time.perf_counter()
        f = func(*args, **kargs)
        end_time = time.perf_counter()
        print('Download {} sites in {} seconds'.format(len(args[0]), end_time - start_time))   
        return f
    return wraper
        
def download_one(url):
    resp = requests.get(url)
    print('Read {} from {}'.format(len(resp.content), url))

def agent_act(agt):
    resp = requests.get(agt.url)
    print('Read {} from {}'.format(len(resp.content), url))
    
class Agent(object):
    def __init__(self):
        self.url = ""
    def set_url(self, url):
        self.url = url
        

def download_one(agt):
    '''
    Test if class object can be used
    '''
    resp = requests.get(agt.url)
    #print('Read {} from {}'.format(len(resp.content), url))
    
@timer
def download_all(sites):
    for s in sites:
        download_one(s)
@timer        
def download_all_by_thread(sites, thread_num=5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        executor.map(download_one, sites)
        
@timer
def download_all_by_futures(sites):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        to_do = []
        for site in sites:
            future = executor.submit(download_one, site)
            to_do.append(future)
            
        for future in concurrent.futures.as_completed(to_do):
            future.result()

@timer
def download_all_by_pool(sites, cpu_num=5):
    with multiprocessing.Pool(processes = cpu_num) as pool:         # start cpu_num worker processes
        result = pool.apply_async(download_one, sites)              # evaluate "download_one" asynchronously in a single process
        pool.map(download_one, sites)
 
 
@timer
def download_all_by_future_map(sites):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(download_one, sites)
        
def agents_by_futures(agents):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        to_do = []
        for a in list(agents):
            future = executor.submit(agent_act, a)
            to_do.append(future)
            
        for future in concurrent.futures.as_completed(to_do):
            future.result()   

                
def main():
    cpu_cores = 0
    if cpu_cores == 0:  # use all processor cores
        cpu_cores = multiprocessing.cpu_count()
        print("Using {} cpu cores.".format(cpu_cores))
        os.putenv('OMP_NUM_THREADS', str(cpu_cores))    
    sites = [
        'https://www.baidu.com/s?wd=Arts',
        'https://www.baidu.com/s?wd=History',
        'https://www.baidu.com/s?wd=Society',
        'https://www.baidu.com/s?wd=Biography',
        'https://www.baidu.com/s?wd=Mathematics',
        'https://www.baidu.com/s?wd=Technology',
        'https://www.baidu.com/s?wd=Geography',
        'https://www.baidu.com/s?wd=Science',
        'https://www.baidu.com/s?wd=Computer_science',
        'https://www.baidu.com/s?wd=Python_(programming_language)',
        'https://www.baidu.com/s?wd=Java_(programming_language)',
        'https://www.baidu.com/s?wd=PHP',
        'https://www.baidu.com/s?wd=Node.js',
        'https://www.baidu.com/s?wd=The_C_Programming_Language',
        'https://www.baidu.com/s?wd=Go_(programming_language)'
    ]
    
    #init a agent list in agts.values()
    agts = {}
    for i, url in enumerate(sites):
        agt = Agent()
        #agt.set_url(url)
        agts[i] = agt
        
        
    
    #start_time = time.perf_counter()
    #print("Begin download one by one.")
    #download_all(sites)
    #print("Begin download by thread.")
    #download_all_by_thread(sites, 4)
    print("Begin download by future.")
    download_all_by_futures(sites)
    #print("Begin download by multiprocess.Pool")
    #download_all_by_pool(sites, cpu_cores)
    print("Begin download by future map directly")
    download_all_by_future_map(sites)
    #end_time = time.perf_counter()
    #print('Download {} sites in {} seconds'.format(len(sites), end_time - start_time))
    #agents_by_futures(agts.values())

if __name__ == '__main__':
    main()
