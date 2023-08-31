import numpy as np 


class MLP():
    ''' implements simple feedforward MLP with a single hidden layer 
        of relu nonlinearities. trained with SGD on MSE loss 
        INPUTS:
        n_in = number of input units 
        n_ctx = number of context units 
        n_hidden = number of hidden units 
        n_out = number of output units 
        lrate = SGD learning rate 
        scale_whxs = weight scale for input-to-hidden weights 
        scale_whxc = weight scale for context-to-hidden weights 
        scale_wyh = weight scale for output weights 
    '''

    def __init__(self, n_in, n_ctx, n_hidden, n_out, lrate,  scale_whxs, scale_whxc, scale_wyh):
        # set parameters
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.eta = lrate

        # initialise weights
        self.w_hxs = scale_whxs*np.random.randn(self.n_hidden, self.n_in)
        self.w_hxc = scale_whxc*np.random.randn(self.n_hidden, self.n_ctx)
        self.w_yh = scale_wyh*np.random.randn(self.n_out, self.n_hidden)
        self.b_yh = np.repeat(0.1, n_out)
        self.b_hx = np.repeat(0.1, n_hidden)

    def fprop(self, x_stim, x_ctx, y):
        self.h_in = self.w_hxs.dot(
            x_stim) + self.w_hxc.dot(x_ctx) + self.b_hx[:, np.newaxis]
        self.h_out = self.relu(self.h_in)
        self.y_ = self.w_yh.dot(self.h_out)+self.b_yh
        self.l = self.loss(y, self.y_)

    def bprop(self, x_stim, x_ctx, y):
        # partial derivatives
        dl_dy = self.deriv_loss(self.y_, y)
        dy_dh = self.w_yh
        dy_dw = self.h_out
        dho_dhi = self.deriv_relu(self.h_in)
        dhi_dws = x_stim
        dhi_dwc = x_ctx
        # backward pass:
        self.dl_dwyh = dl_dy.dot(dy_dw.T)
        self.dl_dbyh = dl_dy

        self.dl_dwhxs = (dl_dy.T.dot(dy_dh)*dho_dhi.T).T.dot(dhi_dws.T)
        self.dl_dwhxc = (dl_dy.T.dot(dy_dh)*dho_dhi.T).T.dot(dhi_dwc.T)
        self.dl_dbhx = (dl_dy.T.dot(dy_dh)*dho_dhi.T).T

    def update(self):
        # weight updates
        self.w_yh = self.w_yh - self.eta*self.dl_dwyh
        self.b_yh = self.b_yh - self.eta*np.sum(self.dl_dbyh, axis=1)

        self.w_hxs = self.w_hxs - self.eta*self.dl_dwhxs
        self.w_hxc = self.w_hxc - self.eta*self.dl_dwhxc
        self.b_hx = self.b_hx - self.eta*np.sum(self.dl_dbhx, axis=1)

    def train(self, x_stim, x_ctx, y):
        self.fprop(x_stim, x_ctx, y)
        self.bprop(x_stim, x_ctx, y)
        self.update()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def deriv_sigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def relu(self, x):
        return x*(x > 0)

    def deriv_relu(self, x):
        return (x > 0).astype('double')

    def loss(self, y_, y):
        return .5*np.linalg.norm(y_-y, 2)**2

    def deriv_loss(self, y_, y):
        return (y_-y)