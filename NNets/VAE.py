import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#pytorch
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
#Differential equation package for pytorch
from torchdiffeq import odeint

class Encoder(nn.Module):
    ''' This the encoder part of VAE

            Args:
        input_dim: A integer indicating the size of input.
        hidden_dim: A integer indicating the size of hidden dimension.
        rnn_layers: number of recurrent layers for the GRU decoder
        z_dim: A integer indicating the latent dimension.

    '''
    def __init__(self, input_dim, hidden_dim, z_dim, rnn_layers=0):
        super().__init__()

        self.rnn_layers = rnn_layers
        self.z_dim = z_dim

        #Arquitechture when including the rnn
        if rnn_layers > 0:
            self.rnn = nn.GRU(input_dim, hidden_dim, rnn_layers, batch_first=True)
            self.r2lin = nn.Linear(hidden_dim, z_dim * 2)

        #Simple 1-hiddden unit for vanilla VAE
        self.linear = nn.Linear(input_dim, hidden_dim)#(rnn_enc_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)
    

    def forward(self, x):
        if self.rnn_layers > 0:
            output, _ = self.rnn(x.unsqueeze(0).float())
            # take last step trough the dense layer
            hid = output[:, -1, :]
            #Hidden layer for mu and var
            out_enc = self.r2lin(hid)
            z_mu, z_var = out_enc[:, :self.z_dim], out_enc[:, self.z_dim:]

        else:
            # x is of shape [batch_size, input_dim]
            hidden = F.relu(self.linear(x.float()))
            # hidden is of shape [batch_size, hidden_dim]
            z_mu = self.mu(hidden)
            # z_mu is of shape [batch_size, latent_dim]
            z_var = self.var(hidden)
            # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var 
        
class Decoder(nn.Module):
    ''' This the decoder part of VAE

        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension.
            rnn_layers: number of recurrent layers for the GRU decoder.
    '''
    def __init__(self, z_dim, hidden_dim, output_dim, rnn_layers=0):       
        super().__init__()

        self.rnn_layers = rnn_layers
        self.act = nn.Tanh()
        #Architecture for the rnn version
        if self.rnn_layers > 0:
            self.rnn = nn.GRU(z_dim, hidden_dim, rnn_layers, batch_first=True)
            self.h1 = nn.Linear(hidden_dim, hidden_dim - 5)
            self.h2 = nn.Linear(hidden_dim - 5, output_dim)

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):

        if self.rnn_layers > 0:
            output, _ = self.rnn(z)
            hid = self.h1(output)
            hid = self.act(hid)
            predicted = self.h2(hid)
        else:
            # z is of shape [batch_size, latent_dim]
            hidden = F.relu(self.linear(z))
            # hidden is of shape [batch_size, hidden_dim]
            predicted = self.out(hidden)
            # predicted is of shape [batch_size, output_dim]
        return predicted



class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self, enc, dec, device, ode=None):
        super().__init__()

        self.enc = enc
        self.dec = dec

        self.device = device

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        z_std = torch.exp(z_var / 2)
        eps = torch.randn_like(z_std).to(self.device)
        x_sample = eps.mul(z_std).add_(z_mu)
        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var
        
        
''' From the "Reduced-order Model for Fluid Flows via Neural Ordinary Differential Equations" by Rojas, et.al.
'''
class LatentOdeF(nn.Module):
    """ ODE-NN: takes the value z at the current time step and outputs the
        gradient dz/dt """

    def __init__(self, layers):
        super(LatentOdeF, self).__init__()

        self.act = nn.Tanh()
        self.layers = layers

        # Feedforward architecture
        arch = []
        for ind_layer in range(len(self.layers) - 2):
            layer = nn.Linear(self.layers[ind_layer], self.layers[ind_layer + 1])
            #Glorot initialization of tensor 
            nn.init.xavier_uniform_(layer.weight)
            arch.append(layer)

        layer = nn.Linear(self.layers[-2], self.layers[-1])
        nn.init.xavier_uniform_(layer.weight)
        layer.weight.data.fill_(0)
        arch.append(layer)

        self.linear_layers = nn.ModuleList(arch)
        self.nfe = 0

    def forward(self, t, z):

        self.nfe += 1

        #All, but last, layers have Tanh activation function
        for ind in range(len(self.layers) - 2):
            z = self.act(self.linear_layers[ind](z))

        # last layer has identity activation (i.e linear)
        grad = self.linear_layers[-1](z)
        return grad

#Re-defining VAE because it's slightly different
class VAE_NODE(nn.Module):
        ''' This the VAE function for the NODE method, which takes a encoder, decoder.
            Args:
            enc : Encoder function
            dec : Decoder function
            ode : ODE block function
        '''
        def __init__(self, enc, dec, node, device):
            super().__init__()

            self.enc = enc
            self.dec = dec
            self.node = node
            self.device = device

        def forward(self, x):
            # encode
            z_mu, z_var = self.enc(x)

            # sample from the distribution having latent parameters z_mu, z_var
            # reparameterize
            z_std = torch.exp(z_var / 2)
            eps = torch.randn_like(z_std).to(self.device)
            z0 = eps.mul(z_std).add_(z_mu)
            #node latent space evolution
            ts_ode = np.linspace(0, 1, x.shape[0])
            ts_ode = torch.from_numpy(ts_ode).float().to(self.device) 
            #Solve system of diff equations using the ODE block, z0 initial values, and time points td_ode
            zt = odeint(self.node, z0, ts_ode, method='rk4').permute(1, 0, 2)
            # decode
            predicted = self.dec(zt)
            return predicted, z_mu, z_var
        
def decode_redz(x, model, red_dim, device):
    ''' Decodes a reduced representation of the latent space
    '''
    red_i, red_f = red_dim
    x = x.to(device)


    def weight(dev, var):
        return model.state_dict()[f"{dev}.{var}.weight"]
    def bias(dev, var):
        return model.state_dict()[f"{dev}.{var}.bias"]

    def linear(weight, bias, xi):
        xi = xi.float()
        wgt = weight.T
        b = bias.unsqueeze(0)
        return torch.mm(xi,wgt).add_(b)

    #encode
    x_enc = F.relu(linear(weight('enc','linear'),bias('enc','linear'),x))
    red_mu = linear(weight('enc','mu')[red_i:red_f],bias('enc','mu')[red_i:red_f],x_enc)
    red_var = linear(weight('enc','var')[red_i:red_f],bias('enc','var')[red_i:red_f],x_enc)
    
    red_var = red_var.to(device)
    red_mu = red_mu.to(device)
    #reparametrize
    red_std = torch.exp(red_var / 2)
    eps = torch.randn_like(red_std).to(device)
    red_z =  eps.mul(red_std).add_(red_mu)

    # decode
    x_dec = F.relu(linear(weight('dec','linear')[:,red_i:red_f],bias('dec','linear'),red_z))
    x_out = linear(weight('dec','out'),bias('dec','out'),x_dec)

    return x_out