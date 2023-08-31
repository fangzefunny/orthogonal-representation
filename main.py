import pickle
import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
import torch.optim as optim

from tqdm import tqdm

from utils.env_fn import gauss_blob_task
from utils.model import MLP

# --------------------------- #
#            MODEL            #
# --------------------------- #

class net(nn.Module):
    def __init__(self, sig_w=.1, sig_c=.5):
        super().__init__()
        self.fc1 = nn.Linear(27, 100)
        self.fc2 = nn.Linear(100, 1)
        with torch.no_grad():
            w_fc1 = torch.hstack([sig_w*torch.randn([100, 25]), 
                                  sig_w*torch.randn([100, 2])])
            #w_fc1 = torch.FloatTensor(w1)
            self.fc1.weight.copy_(w_fc1)
            self.fc1.bias.fill_(0)
            self.fc2.weight.copy_(.1*torch.randn([1, 100]))
            self.fc2.bias.fill_(0)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# --------------------------- #
#            TRAIN            #
# --------------------------- #

def train(sig_w, sig_c, n_run=30, verbose=False):

    # save for analysis 
    num_epochs = 10000
    w1_lst       = []
    w2_lst       = []
    w1_init_lst  = []
    w2_init_lst  = []
    params       = []
    clf_accs     = np.empty([n_run, int(num_epochs/100)])

    for i in tqdm(range(n_run)):

        # get input and output data 
        x, y = gauss_blob_task().instan()
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y.reshape([-1, 1]))

        # prepare model        
        model = net(sig_w=sig_w, sig_c=sig_c)
        w1_init = model.fc1.weight.data.numpy().reshape([-1]).copy()
        w2_init = model.fc2.weight.data.numpy().reshape([-1]).copy()
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.SGD(model.parameters(), lr=.001, weight_decay=0)  # Stochastic Gradient Descent
        j = 0
        
        for epoch in range(num_epochs):

            ind = torch.randperm(x.size(0))
            
            # Forward pass
            y_hat = model.forward(x[ind])
            loss = criterion(y_hat, y[ind])
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get data
            if (epoch) % 100 == 0:
                # get accuracy
                idx = (y[ind].data.numpy().reshape([-1]) != 0)
                acc = np.mean((y_hat.data.numpy().reshape([-1])[idx]>0) 
                            == (y[ind].data.numpy().reshape([-1])[idx]>0))
                clf_accs[i, j] = acc
                # add a number 
                j += 1

                # Print loss every 1000 epochs
                if verbose:
                    if (epoch) % 100 == 0:
                        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {acc:.2f}")
    
        w1_new = model.fc1.weight.data.numpy()
        w2_new = model.fc2.weight.data.numpy()

        # save to
        w1_lst.append(w1_new)
        w2_lst.append(w2_new)
        w1_init_lst.append(w1_init)
        w2_init_lst.append(w2_init)

        net_param = {
            'w1': model.fc1.weight.data,
            'b1': model.fc1.bias.data,
            'w2': model.fc2.weight.data,
            'b2': model.fc2.bias.data,
        }
        params.append(net_param)
        
    return w1_lst, w2_lst, w1_init_lst, w2_init_lst, clf_accs, params

def train_mlp(sig_w, sig_c, n_run=10, verbose=False):

    # save for analysis 
    num_epochs = 10000
    w1_lst       = []
    w2_lst       = []
    w1_init_lst  = []
    w2_init_lst  = []
    params       = []
    clf_accs     = np.empty([n_run, int(num_epochs/100)])

    for i in tqdm(range(n_run)):

        # get input and output data 
        x, y = gauss_blob_task().instan()

        # prepare model        
        model = MLP(25, 2, 100, 1, .005, sig_w, sig_c, .1)
        w1_init = model.w_hxs.copy()
        w2_init = model.w_yh.copy()
        j = 0 

        for epoch in range(num_epochs):

            ind = np.random.choice(x.shape[0], size=x.shape[0])
            
            # Forward pass
            model.train(x[ind][:, :25].T, x[ind][:, 25:].T, y[ind].reshape([1, -1]))
            loss = model.l
            #model.bprop(x[ind][:, :25].T, x[ind][:, 25:].T, y[ind].reshape([1, -1]))
        
            # get data
            if (epoch) % 100 == 0:
                # get accuracy
                idx = (y[ind].reshape([-1]) != 0)
                acc = np.mean((model.y_.reshape([-1])[idx]>0) 
                              == (y[ind].reshape([-1])[idx]>0))
                clf_accs[i, j] = acc
                # add a number 
                j += 1

                # Print loss every 1000 epochs
                if verbose:
                    if (epoch) % 1000 == 0:
                        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss:.4f}, Acc: {acc:.2f}")
            
        w1_new =  model.w_hxs.copy()   
        w2_new =  model.w_yh.copy() 

        w1_lst.append(w1_new)
        w2_lst.append(w2_new)
        w1_init_lst.append(w1_init)
        w2_init_lst.append(w2_init)

        net_param = {
            'w1': np.hstack([model.w_hxs.copy(), model.w_hxc.copy()]),
            'b1': model.b_hx.copy(),
            'w2': model.w_yh.copy(),
            'b2': model.b_yh.copy(),
        }
        params.append(net_param)
        
    return w1_lst, w2_lst, w1_init_lst, w2_init_lst, clf_accs, params


def save_weight():
    sig_w_lst=[.01, .4,   3]
    sig_c_lst=[ .5, .5, 1.5]
    w_dict = {}
    for sig_w, sig_c in zip(sig_w_lst, sig_c_lst):
        print(f'Training sig={sig_w}')
        w1_lst, w2_lst, delta_w1_lst, delta_w2_lst, clf_accs, \
            net_param = train_mlp(sig_w=sig_w, sig_c=sig_c, n_run=15, verbose=False)
        w_dict[sig_w] = {
            'w1': w1_lst.copy(),
            'w2': w2_lst.copy(),
            'w1_init': delta_w1_lst.copy(),
            'w2_init': delta_w2_lst.copy(),
            'clf_acc': clf_accs.copy(),
            'param': net_param,
        }
    # save weight 
    with open('data/weights_lr=.005.pkl', 'wb')as handle:
        pickle.dump(w_dict, handle)

if __name__ == '__main__':

    save_weight()