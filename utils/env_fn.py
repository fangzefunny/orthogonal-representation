import numpy as np 
from scipy.stats import multivariate_normal

def get_gauss_blob():

    # define some variables
    vec = np.linspace(.2, .8, 5)
    x = np.linspace(.2, .8, 5)
    y = np.linspace(.2, .8, 5)
    x, y = np.meshgrid(x, y)
        
    # the generative function 
    def gauss_2d(x_mu=.0, y_mu=.0, xy_sigma=.1, n=20):
        xx,yy = np.meshgrid(np.linspace(0, 1, n),np.linspace(0, 1, n))
        gausspdf = multivariate_normal([x_mu,y_mu],[[xy_sigma,0],[0,xy_sigma]])
        x_in = np.empty(xx.shape + (2,))
        x_in[:, :, 0] = xx; x_in[:, :, 1] = yy
        return gausspdf.pdf(x_in)

    stim = []
    for mu_y in vec:   
        for mu_x in vec:    
            # Generate the Gaussian blob
            z = gauss_2d(x_mu=mu_x, y_mu=mu_y, xy_sigma=.08, n=5)
            z = z / z.max()
            stim.append(z.reshape([-1, 5, 5]))
    
    return np.vstack(stim) 

class gauss_blob_task:
    name = 'guass blob'
    nS = 25
    nC = 2
    nD = 2
    nF = 5

    def __init__(self):
        self._init_C()
        self._init_S()
        self._init_A()
        self._init_P()
        self._init_R()

    def _init_C(self):
        self.C = [0, 1]
    
    def _init_S(self):
        self.S = get_gauss_blob()
        
    def _init_A(self):
        pass 
        
    def _init_P(self):
        pass

    def _init_R(self):
        def R(c, s):
            r = [-2, -1, 0, 1, 2]
            R_0 = np.tile(r, 5)
            R_1 = np.repeat(list(reversed(r)), 5)
            return np.vstack([R_0, R_1])[c, s]
        self.R = R
     
    def instan(self):

        x, y = [], [] 
        for c in self.C:
            S = np.hstack([self.S.reshape([-1, self.nF**2]), 
                           np.tile(np.eye(self.nC)[c], [self.nF**2, 1])])
            x.append(S) 
            y.append([self.R(c, s) for s in range(self.nS)])
        x = np.vstack(x)
        x[:, :25] = x[:, :25] / np.linalg.norm(x[:, :25])
        x[:, 25:] = x[:, 25:] / np.linalg.norm(x[:, 25:])
        y = np.hstack(y)

        return x, y 