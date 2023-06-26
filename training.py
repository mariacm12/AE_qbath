import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy

#pytorch
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

class Data(Dataset):
  def __init__(self,t_series,fourier=False):
      self.data=t_series
  def __getitem__(self,idx):
      sample = self.data[idx]
      return sample
  def __len__(self):
      return len(self.data)
  

class LaggedDataset(Dataset): 
    ''' From "Time-lagged autoencoders: Deep learning of slow collectivevariables for molecular kinetics" by Wehmeyer et.al.

    Dataset for wrapping time-lagged data from a single stored time series.
    Each sample will contain the data_tensor at index t and the (not explicitly
    stored) target_tensor via data_tensor at index t+lag. We need this for
    training the time-lagged autoencoder and TICA.
    Arguments:
        data_tensor (Tensor): contains time series data
        lag (int): specifies the lag in time steps
    '''
    def __init__(self, data_tensor, lag=1):
        assert data_tensor.size(0) > lag, 'you need more samples than lag'
        assert lag >= 0, 'you need a non-negative lagtime'
        self.data_tensor = data_tensor
        self.lag = lag
    def __getitem__(self, index):
        return self.data_tensor[index], self.data_tensor[index + self.lag]
    def __len__(self):
        return self.data_tensor.size(0) - self.lag

def define_lagloaders(samples, lag, batch=4):
    ''' Take the time lag as input
    '''
    t_samples = samples.shape[0]

    X = torch.from_numpy(samples)
    tr, v, te = 0.7, 0.20, 0.10
    trwindow = int(tr*t_samples)
    vwindow = int(v*t_samples) + trwindow
    tewindow = t_samples - int(te*t_samples)

    X_train = X[:trwindow]
    X_val = X[:vwindow+1]
    X_test = X[tewindow:]

    traindata = LaggedDataset(X_train, lag)
    valdata = LaggedDataset(X_val,lag)
    testdata = LaggedDataset(X_test,lag)

    # Build dataloader 
    batchsize = batch
    train_loader = DataLoader(dataset=traindata,batch_size=batchsize,shuffle=False)
    val_loader = DataLoader(dataset=valdata,batch_size=batchsize,shuffle=False)
    test_loader = DataLoader(dataset=testdata,batch_size=batchsize,shuffle=False)

    return train_loader, val_loader, test_loader

def loop(model, loader, epoch, loss_fn, optimizer, device, beta=0.0001, evaluation=False):
    '''
        Train/test your VAE model
    '''
    
    if evaluation:
        model.eval()
        mode = "eval"
    else:
        model.train()
        mode = 'train'
    batch_losses = []
        
    tqdm_data = tqdm(loader, position=0, leave=True, desc='{} (epoch #{})'.format(mode, epoch))
    for data in tqdm_data:
        x = data.to(device)
        recon_batch, mu, std = model(x)
        loss_recon, loss_kl = loss_fn(recon_batch, x, mu, std)
        loss = loss_recon + beta * loss_kl
        if not evaluation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_losses.append(loss.item())

        postfix = ['recon loss={:.3f}'.format(loss_recon.item()) ,
                   'KL loss={:.3f}'.format(loss_kl.item()) ,
                   'total loss={:.3f}'.format(loss.item()) , 
                   'avg. loss={:.3f}'.format(np.array(batch_losses).mean())]
        
        tqdm_data.set_postfix_str(' '.join(postfix))
    
    return np.array(batch_losses).mean()

def test_loop_vae(model, loader, epoch, loss_fn, device, beta=0.0001):
    '''
       Testing model
    '''
    model.eval()
    mode = "eval"

    batch_losses = []
    samples = loader.dataset.__getitem__(0).shape[0]
    all_preds = torch.empty(0,samples,dtype=float)
    all_trues = torch.empty(0,samples,dtype=float)
        
    tqdm_data = tqdm(loader, position=0, leave=True, desc='{} (epoch #{})'.format(mode, epoch))
    for data in tqdm_data:
        x = data
        x = x.to(device)

        recon_batch, mu, std = model(x)
        loss_recon, loss_kl = loss_fn(recon_batch, x, mu, std, mse=True)
        loss = loss_recon + beta * loss_kl

        batch_losses.append(loss.item())
        all_preds = torch.cat((all_preds.to(device),recon_batch),0)
        all_trues = torch.cat((all_trues.to(device),x),0)
    
    return all_preds, all_trues, np.array(batch_losses).mean()

def loop_TLA(model, loader, epoch, loss_fn, optimizer, device, beta=0.0001, evaluation=False):
    '''
        Train/test VAE model
    '''
    
    if evaluation:
        model.eval()
        mode = "eval"
    else:
        model.train()
        mode = 'train'

    batch_losses = []
        
    tqdm_data = tqdm(loader, position=0, leave=True, desc='{} (epoch #{})'.format(mode, epoch))
    for data in tqdm_data:
        x, y = data
        x = x.to(device)
        y = y.to(device)

        recon_batch, mu, std = model(x)
        loss_recon, loss_kl = loss_fn(recon_batch, y, mu, std, mse=False)
        loss = loss_recon + beta * loss_kl
        if not evaluation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_losses.append(loss.item())

        postfix = ['recon loss={:.3f}'.format(loss_recon.item()) ,
                   'KL loss={:.3f}'.format(loss_kl.item()) ,
                   'total loss={:.3f}'.format(loss.item()) , 
                   'avg. loss={:.3f}'.format(np.array(batch_losses).mean())]
        
        tqdm_data.set_postfix_str(' '.join(postfix))
    
    return np.array(batch_losses).mean()

def test_loopTLA(model, loader, epoch, loss_fn, device, beta=0.0001):
    '''
       Testing model
    '''
    model.eval()
    mode = "eval"
    batch_losses = []
    samples = loader.dataset.__getitem__(0)[0].shape[0]
    all_preds = torch.empty(0,samples,dtype=float)
        
    tqdm_data = tqdm(loader, position=0, leave=True, desc='{} (epoch #{})'.format(mode, epoch))
    for data in tqdm_data:
        x, y = data
        x = x.to(device)
        y = y.to(device)

        recon_batch, mu, std = model(x)
        loss_recon, loss_kl = loss_fn(recon_batch, y, mu, std, mse=False)
        loss = loss_recon + beta * loss_kl

        batch_losses.append(loss.item())
        all_preds = torch.cat((all_preds.to(device),recon_batch.squeeze(0)),0)
    
    return all_preds, np.array(batch_losses).mean()

def hyperopt(loop, vae, loaders, enc, dec, param, input_dim, z_dim, loss_fn, device, n_epochs=3, ode=None):
    
    '''Trains the nn for given number of epochs, for each hyperparam combination.
        The combination resulting in the smallest total lost (for the last epoch) is selected.
        Args:
        param: Dictionary containing parameters to optimize 
        loop: Function that trains the model
        n_epochs: Number of epochs each combination is trained in 
    '''
    enc_hids = param['enc_dim'] #assume dimensions for encoding/decoding are the same
    rnn_layers = param['rnn_layers']
    beta_loss = param['beta']
    lr_loss = param['lr']

    train_loader, val_loader = loaders

    all_loss = []
    combs = []
    for i, ehid in enumerate(enc_hids):
        for j, rnn in enumerate(rnn_layers):
            for k, beta in enumerate(beta_loss):
                for l, lr in enumerate(lr_loss):

                    encoder = enc(input_dim=input_dim,hidden_dim=ehid, z_dim=z_dim, rnn_layers=rnn)
                    decoder = dec(z_dim=z_dim, hidden_dim=ehid, output_dim=input_dim, rnn_layers=rnn)
                    model = vae(encoder, decoder, ode)
                    model.to(device)
                    optimizer = optim.Adam(model.parameters(),lr=lr)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                                    factor=0.5, patience=5, verbose=True)
                    
                    print('hidden dim = {}, rnn_layers = {}, beta = {} and lr = {}'.format(ehid,rnn,beta,lr))

                    #Running for n_epochs
                    for epoch in range(0, n_epochs):    
                        train_loss = loop(model, train_loader, epoch, loss_fn, optimizer, beta)
                        val_loss = loop(model, val_loader, epoch, loss_fn,optimizer, beta, evaluation=True)
                        scheduler.step(val_loss)
                        if epoch == 0:
                            best_loss = train_loss.item()
                        else:
                            if train_loss.item() < best_loss:
                                best_loss = train_loss.item()

                    all_loss.append(val_loss)
                    combs.append([i,j,k,l])
            
    best_idx = np.argmin(np.array(all_loss))
    best_par = combs[best_idx]
          
    best_ehid = param['enc_dim'][int(best_par[0])]
    best_rnn = param['rnn_layers'][int(best_par[1])]
    best_beta = param['beta'][int(best_par[2])]
    best_lr = param['lr'][int(best_par[3])]

    return best_ehid, best_rnn, best_beta, best_lr, all_loss[best_idx]
def BCELoss(recon_x, x, mu, var, mse=True):
    #BCELoss because the input/target aren't categorical
    #L_recon = torch.nn.BCELoss()(recon_x.float(),x.float()) #Change to MSE
    if mse:
        L_recon = (recon_x.float() - x.float()).pow(2).mean()
    else: #RMSE error for the TLA implementation
        L_recon = abs(recon_x.float() - x.float()).mean()
    lreg = 1/2* torch.sum(torch.exp(var)+torch.square(mu)-var-1,dim=1)
    L_kl = torch.mean(lreg,dim=0) # KL loss
    return abs(L_recon), L_kl
