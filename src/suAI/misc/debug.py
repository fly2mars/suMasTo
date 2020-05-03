import imageio
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

class MakeUFieldGif(object):
    def __init__(self, nelx, nely, all_dofs):
        '''
        nelx and nely are the element number on x and y direction.
        pos_map is the mapping relation between visulization data and the node or element. 
        e.g. allDofs
        All the mapping reconstruction will be done in the embed process function.
        '''
        self.bufs = []  
        self.nelx = nelx
        self.nely = nely
        self.all_dofs = all_dofs
        self.fig= plt.figure(2)
          
        ndx = np.array(self.all_dofs) // 2
        ndx = np.array([idx for i, idx in enumerate(ndx) if i%2 ==0])
        self.X = ndx  // (nely + 1)
        self.Y = nely - ndx % (nely + 1)          
        
    def add_data(self, u):        
        self.bufs.append(u)
        
    def show_field(self, idx = 0):
        
        ax = plt.gca()
        u = self.bufs[idx]
        
        U = np.array([ux for i,ux in enumerate(u) if i % 2 == 0])
        V = np.array([uy for i,uy in enumerate(u) if i % 2 == 1]) 
        
        ax.axis('off')
        plt.quiver(self.X,self.Y,U,V, color='red')

        # Used to return the plot as an image rray
        self.fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        print(self.bufs[idx])
        return image        
        
        
    def save_gif(self,file_path, fps = 10):        
        len_data = len(self.bufs)
        imageio.mimsave(file_path, [self.show_field(i) for i in range(len_data)], fps=fps)
        plt.close()
        

def save_img(img, str_path):
    print(img)
    imageio.imsave(str_path, img)
    
def show_matrix(m, reverse=False):     
    min_val, max_val = 0, 1
    m = np.abs(m)
    m = m / np.max(m)  
    if reverse:
        m = 1 - m
    
    plt.figure(11)  
    plt.imshow(m, cmap=plt.cm.gray)
    
    plt.xticks([])
    plt.yticks([])    
    plt.show()

def show_displacement_field(u, dofs, nelx, nely):
    '''
    m: u_x,u_y list
    node all dof list
    Because the unit is 1, so the x index and y index of a node can also be the (x,y) of a node.
    '''
    ndx = np.array(dofs) // 2
    ndx = np.array([idx for i, idx in enumerate(ndx) if i%2 ==0])
    X = ndx  // (nely + 1)
    Y = nely - ndx % (nely + 1)    
    
    U = np.array([ux for i,ux in enumerate(u) if i % 2 == 0])
    V = np.array([uy for i,uy in enumerate(u) if i % 2 == 1])
    
    ax = plt.gca()
    ax.axis('off')
    plt.quiver(X,Y,U,V, color='red')
    im = plt.show()    
    
    return im 
            
if __name__ == '__main__':
    
    #a = np.random.random([100,100])
    #show_matrix(a)
    us = np.loadtxt("r:/u.txt")
    
    us = list(us[:20])
    nelx = 90
    nely = 30    
    alldofs = [x for x in range(2*(nely+1)*(nelx+1))]    
    gif = MakeUFieldGif(nelx, nely, alldofs)
    for u in us:
        gif.add_data(u)
    gif.save_gif("r:/t.gif",2)