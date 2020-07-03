from loads import *
# example of load, establish structure from image.
# blue: force Y
# red: force X
# grey: fix
import cv2
class ImageTypeLoad(Load):
    def __init__(self, image_path):
        nelx = 30
        nely = 10        
        try:
            self.im = cv2.imread(image_path, cv2.IMREAD_COLOR)
            nely, nelx, channel = self.im.shape
            grey = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
            ret, self.thresh = cv2.threshold(grey, 20, 255, cv2.THRESH_BINARY)  
            print((nely, nelx, channel))
            cv2.imwrite('r:/color.png', self.im)
            cv2.imwrite('r:/thresh.png', self.thresh)
            cv2.imwrite('r:/grey.png', grey)    
            im, contours, hiearchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.im, contours, -1, (0,0,255))
            cv2.imwrite('r:/contour.png', self.im)    
            
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
        BG = []
        nely = self.nely
        nelx = self.nelx
        for y in range(nely):
            for x in range(nelx):
                if self.thresh[nely, nelx] == 0:
                    BG.append((y,x))
        return BG   
    
    def get_force_pos(self):
        F={}       # F index for different DOFs
        LABEL={}
        ft = ['X_LEFT','X_RIGHT', 'Y_UP', 'Y_DOWN']
        LABEL[ft[0]] = (0,255,255)
        LABLE[ft[1]] = (255,255,0)
        LABLE[ft[2]] = (255,0,0)        
        LABLE[ft[3]] = (0, 0, 255)        
        for t in ft:
            F[t]=set()
            
        nely = self.nely
        nelx = self.nelx
        
        repeat_counter = {}
        
        for y in range(nely):
            for x in range(nelx):
                if self.im[nely, nelx] == FORCE_Y_DOWN:
                    edofNode = self.edofNode(x, y, nelx, nely)
                    
                    for n in edofNode:
                        if repeat_counter.get(n) == None:
                            repeat_counter[n] = 0
                        else:
                            repeat_counter[n] += 1
        
        new_repeat_counter =  { key:value for (key,value) in repeat_counter.items() if value < 3}
        return B           
    def get_fixdof(self):
        B = []
        nely = self.nely
        nelx = self.nelx
        FIX = (127,127,127)
        for y in range(nely):
            for x in range(nelx):
                if self.thresh[nely, nelx] == 0:
                    B.append((y,x))
        return B   
            
            
if __name__ == '__main__':
    load = ImageTypeLoad('C:/git/suData/doc/script/ruleBasedTO/suTOMAS/models/images/bunny-s38.png')
    