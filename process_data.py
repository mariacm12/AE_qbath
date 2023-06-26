import os
import numpy as np
import pandas as pd
import random

#plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import LogNorm, Normalize


#Fourier Transform of time domain data
def fourier_data(time_data, npoints):
    tot_points = npoints+1
    ft = np.fft.fft(time_data,axis=-1)[1:]
    freq = np.fft.fftfreq(tot_points)
    return ft,freq

def make_ensenble(sample,k,etotal):
  ensem = []
  for en in range(etotal):
    nsamples = sample.shape[0]
    idx = np.random.randint(nsamples, size=k)
    e_choice = sample[idx]
    avg_ens = np.mean(e_choice,axis=0)
    ensem.append(avg_ens)

  return np.array(ensem)

def power_recon(all_ps, npoints):
    re_ft = all_ps[:,:npoints+1]
    im_ft = all_ps[:,npoints+1:]
    ps = re_ft + im_ft*1j

    #Inverse fourier transform
    t_recon = np.fft.ifft(ps,axis=-1)

    return t_recon

def plotAE(pred_data, true_data, lossTrain, lossVal, testLoader, filename):
    '''
    Args:
      pred_data: the model prediction (from validation)
      true_data: the label to compare to
      lossTrain, loss_val: the arrays with train,val losses
    '''

    ###Plot to compare time-series
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(18, 15)
    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.20, wspace=0.15)

    test_len = len(testLoader)
    random_ts = random.choices( range( test_len ), k=4)
    for idx, k in enumerate(random_ts):

        #Back to time domain
        ps_orig = true_data[k].unsqueeze(0).detach().cpu().numpy()
        ps_recon = pred_data[k].unsqueeze(0).detach().cpu().numpy()
        
        ts_orig = power_recon(ps_orig).squeeze()
        ts_recon = power_recon(ps_recon).squeeze()

        ax = fig.add_subplot(3, 2, idx + 1)
        ax.plot(ts_orig, color='r', linewidth=2., alpha=1, label='true')
        ax.plot(ts_recon, 'k--', linewidth=2., label='VAE')

        ax.set_ylabel('$E_{%d}$' % (k + 1), rotation=0, size=20, labelpad=10)
        
        ax.legend(bbox_to_anchor=(0.7, 0.95), loc='upper left', borderaxespad=0., fontsize=20)

        ax.set_xlabel(r'$t(fs)$', size=20)

        plt.setp(ax.spines.values(), linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        ax.set_xlim(0,2000)
        ax.set_ylim(0,1)
        
    plt.savefig("%s.eps" % (filename+'_comp'),bbox_inches='tight')
    ##plot the train/val error
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Epoch",size=20)
    ax.set_ylabel("Loss",size=20)
    ax.plot(lossVal,label='Validation')
    ax.plot(lossTrain,label='Training')
    plt.setp(ax.spines.values(), linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    ax.legend(bbox_to_anchor=(0.68, 0.98), loc='upper left', borderaxespad=0., fontsize=12)

    plt.savefig("%s.eps" % (filename+'_loss'),bbox_inches='tight')

def plotTLA(pred_data, true_data, lossTrain, lossVal, filename):
    '''
    Args:
      pred_data: the model prediction (from validation)
      true_data: the label to compare to
      lossTrainm loss_val: the arrays with train,val losses
      itr: epoch 
      train_winf: train window
    '''

    ###Plot to compare time-series
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(18, 15)

    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.20, wspace=0.15)

    # vector to define time axis in plots
    t_steps_t = np.arange(0, true_data.shape[0])
    t_steps_p = np.arange(0, pred_data.shape[0])

    true_data = true_data.detach().cpu().numpy()
    pred_data = pred_data.detach().cpu().numpy()

    for k in range(4):

        ax = fig.add_subplot(3, 2, k + 1)
        ax.plot(t_steps_t, true_data[:, k], color='r', linewidth=2., alpha=1, label='true')
        ax.plot(t_steps_p, pred_data[:, k], 'k--', linewidth=2., label='NODE')

        ax.set_ylabel('$E_{%d}$' % (k + 1), rotation=0, size=20, labelpad=10)
        
        ax.legend(bbox_to_anchor=(0.7, 0.95), loc='upper left', borderaxespad=0., fontsize=20)

        ax.set_xlabel(r'$t$', size=20)

        plt.setp(ax.spines.values(), linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        ax.set_xlim(0,len(t_steps_t))
        ax.set_ylim(-0.01,1)
        

    plt.savefig("%s.eps" % (filename+'_TLA_comp'),bbox_inches='tight')
    ##plot the train/val error
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Epoch",size=20)
    ax.set_ylabel("Loss",size=20)
    ax.plot(lossVal,label='Validation')
    ax.plot(lossTrain,label='Training')
    plt.setp(ax.spines.values(), linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    ax.legend(bbox_to_anchor=(0.68, 0.98), loc='upper left', borderaxespad=0., fontsize=12)

    plt.savefig("%s.eps" % (filename+'_TLA_loss'),bbox_inches='tight')

def plotTLA(pred_data, true_data, lossTrain, lossVal, filename):
    '''
    Args:
      pred_data: the model prediction (from validation)
      true_data: the label to compare to
      lossTrainm loss_val: the arrays with train,val losses
      itr: epoch 
      train_winf: train window
    '''

    ###Plot to compare time-series
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(18, 15)

    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.20, wspace=0.15)

    # vector to define time axis in plots
    t_steps_t = np.arange(0, true_data.shape[0])
    t_steps_p = np.arange(0, pred_data.shape[0])

    true_data = true_data.detach().cpu().numpy()
    pred_data = pred_data.detach().cpu().numpy()

    for k in range(4):

        ax = fig.add_subplot(3, 2, k + 1)
        ax.plot(t_steps_t, true_data[:, k], color='r', linewidth=2., alpha=1, label='true')
        ax.plot(t_steps_p, pred_data[:, k], 'k--', linewidth=2., label='NODE')

        ax.set_ylabel('$E_{%d}$' % (k + 1), rotation=0, size=20, labelpad=10)
        
        ax.legend(bbox_to_anchor=(0.7, 0.95), loc='upper left', borderaxespad=0., fontsize=20)

        ax.set_xlabel(r'$t$', size=20)

        plt.setp(ax.spines.values(), linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        ax.set_xlim(0,len(t_steps_t))
        ax.set_ylim(-0.01,1)
        

    plt.savefig("%s.eps" % (filename+'_TLA_comp'),bbox_inches='tight')
    ##plot the train/val error
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Epoch",size=20)
    ax.set_ylabel("Loss",size=20)
    ax.plot(lossVal,label='Validation')
    ax.plot(lossTrain,label='Training')
    plt.setp(ax.spines.values(), linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    ax.legend(bbox_to_anchor=(0.68, 0.98), loc='upper left', borderaxespad=0., fontsize=12)

    plt.savefig("%s.eps" % (filename+'_TLA_loss'),bbox_inches='tight')