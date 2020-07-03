'''
loading scenario aka boundary condition
density at elements, force/displacement at nodes (element corners)
dofs are for displacements
'''

import numpy as np

class Load(object):
    def __init__(self, nelx, nely):
        self.nelx = nelx
        self.nely = nely
        self.dim = 2

    '''
    different convention from numpy array shape: x/y versus row/col
    '''
    def shape(self):
        return (self.nelx, self.nely)

    # compute 1D index from 2D position for node (boundary of element)
    def node(self, elx, ely, nelx, nely):
        return (nely+1)*elx + ely; 

    # compute the 4 boundary nodes of an element
    def nodes(self, elx, ely, nelx, nely):
        n1 = self.node(elx,     ely,     nelx, nely) 
        n2 = self.node(elx + 1, ely,     nelx, nely) 
        n3 = self.node(elx + 1, ely + 1, nelx, nely) 
        n4 = self.node(elx,     ely + 1, nelx, nely)
        return n1, n2, n3, n4

                    
    # edof
    def edofNode(self, elx, ely, nelx, nely):
        n1, n2, n3, n4 = self.nodes(elx, ely, nelx, nely)
        return np.array([self.dim*n1,self.dim*n1+1, self.dim*n2,self.dim*n2+1, self.dim*n3,self.dim*n3+1, self.dim*n4,self.dim*n4+1])

    # edof that returns an array
    def edof(self, nelx, nely):
        """
        Generates an array with the position of the nodes of each element in
        the global stiffness matrix

        The following nodal defenitions are used:
        _________________
        >>>
                       |                                       |
        --   2*el+2*elx , 2*el+2*elx+1 ---- 2*el+2*elx+2*nely+2 , 2*el+2*elx+2*nely+3 --
                       |                                       |
                       |                                       |
                       |                                       |
                       |                                       |
                       |                                       |
                       |                                       |
                       |                                       |
        -- 2*el+2*elx+1 , 2*el+2*elx+2 ---- 2*el+2*elx+2*nely+4 , 2*el+2*elx+2*nely+5 --
                       |                                       |
        """
        # Creating list with element numbers
        elx = np.repeat(range(nelx), nely).reshape((nelx*nely, 1))  # x position of element
        ely = np.tile(range(nely), nelx).reshape((nelx*nely, 1))  # y position of element

        n1, n2, n3, n4 = self.nodes(elx, ely, nelx, nely)
        edof = np.array([self.dim*n1,self.dim*n1+1, self.dim*n2,self.dim*n2+1,
                         self.dim*n3,self.dim*n3+1, self.dim*n4,self.dim*n4+1])
        edof = edof.T[0]

        x_list = np.repeat(edof, 8)  # flat list pointer of each node in an element
        y_list = np.tile(edof, 8).flatten()  # flat list pointer of each node in element
        return edof, x_list, y_list

    def force(self):
        return np.zeros(self.dim*(self.nely+1)*(self.nelx+1))

    def alldofs(self):
        return [x for x in range(self.dim*(self.nely+1)*(self.nelx+1))]

    def fixdofs(self):
        return []

    def freedofs(self):
        return self.alldofs()    
    
    def u_matrix(self, u):
        '''
        u is a dim*(nely+1)*(nelx+1)... list
        u_matrix convert gen
        '''
        pass

# example loading scenario, half mbb-beam
class HalfBeam(Load):
    def __init__(self, nelx, nely):
        super().__init__(nelx, nely)
        
    def setforce(self, f, col, row, v, ori='y'):
        if ori == 'x':
            f[(self.nely+1)*2*col + row*2  ]= v
        if ori == 'y':
            f[(self.nely+1)*2*col + row*2 + 1 ]= v
        
    def force(self):
        f = super().force()
        # downward force at the upper left corner
        #f[self.dim-1] = -1.0
        #f[(self.nely+1)*self.nelx*2+1]= -1.0
        row = 0
        col = 40
        self.setforce(f, col, row, -1, 'y')   
        
        col = 50
        self.setforce(f, col, row, -1, 'y')         
        return f

    def fixdofs(self):
        # left side fixed to a wall, lower right corner fixed to a point
        #return ([x for x in range(0, self.dim*(self.nely+1), self.dim)] + [self.dim*(self.nelx+1)*(self.nely+1)-1])
        # lower left and lower right fixed to a point`
        return [self.dim*(self.nely+1)-1] +  [self.dim*(self.nelx+1)*(self.nely+1)-1]

    def freedofs(self):
        return list(set(self.alldofs()) - set(self.fixdofs()))
    
    def boundary_ele(self):
        B = []
        nely = self.nely
        nelx = self.nelx
        for y in range(nely):
            for x in range(nelx):
                if min(y,x) == 0 or y == nely-1 or x == nelx-1:
                    B.append((y,x))
        return B    

# example of load, establish structure from image.
import cv2
class ImageTypeLoad(Load):
    def __init__(self, image_path):
        nelx = 30
        nely = 10
        try:
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
            nely, nelx, channel = im.shape
            ret, thresh = cv2.threshold(im, 127, 255, 1)                          
            
        except Exception as e:
            print(str(e))
        finally:
            super().__init__(nelx, nely)  
    
        
    def force(self):
        f = super().force()        
        row = 0
        col = 40
        self.setforce(f, col, row, -1, 'y')   
        return f

    def fixdofs(self):
        # left side fixed to a wall, lower right corner fixed to a point
        #return ([x for x in range(0, self.dim*(self.nely+1), self.dim)] + [self.dim*(self.nelx+1)*(self.nely+1)-1])
        # lower left and lower right fixed to a point`
        return [self.dim*self.nely-1] +  [self.dim*(self.nelx+1)*(self.nely+1)-1]

    def freedofs(self):
        return list(set(self.alldofs()) - set(self.fixdofs()))    
    
    def get_outside_ele(self):
        B = []
        nely = self.nely
        nelx = self.nelx
        for y in range(nely):
            for x in range(nelx):
                if self.thresh[nely, nelx] == 0:
                    B.append((y,x))
        return B        
    
if __name__=='__main__':
    
    nelx = 40
    nely = 10
    load = HalfBeam(nelx, nely)
    print("node[0,0]'s dof index list: {}".format(load.nodes(0,0,nelx,nely)))
    print("edof:")
    print(load.edof(nelx, nely)[0])
    print(load.edof(nelx, nely)[1])
    print(load.edof(nelx, nely)[2])
    print("force:")
    print(load.force())
    print("dof index:")
    print(load.alldofs())
    print("fix dof")
    print(load.fixdofs())
    print(load.edofNode(0,0,nelx, nely))
    print(load.boundary_ele())
    
    
    
