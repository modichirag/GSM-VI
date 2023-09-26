###Functions for diagnostic plots

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def corner(samples, savepath="./tmp/", savename=None, save=True, maxdims=10):
    '''Make corner plot for the distribution from samples
    '''
    D = samples.shape[1]
    D = min(D, maxdims)
    
    fig, ax = plt.subplots(D, D, figsize=(3*D, 2*D), sharex='col')

    for i in range(D):
        for j in range(D):
            if i==j:
                ax[i, j].hist(samples[:, i])
                ax[i, j].set_title('W[{}]'.format(i))
            elif i>j:
                ax[i, j].plot(samples[:, j], samples[:, i], '.')
            else:
                ax[i, j].axis('off')

    plt.tight_layout()

    if save:
        if savename is None: savename='corner'
        plt.savefig(savepath + savename)
        plt.close()
    else: return fig, ax



def compare_hist(samples, ref_samples, nbins=20, lbls="", savepath="./tmp/", savename=None, maxdims=10, suptitle=None):
    '''Compare histogram of samples with reference sample
    '''
    D = min(samples.shape[1], ref_samples.shape[1])
    D = min(D, maxdims)
    if (lbls == "") & (type(samples) == list): lbls = [""]*len(samples)

    fig, ax = plt.subplots(1, D, figsize=(3*D, 3))

    for ii in range(D):
        ax[ii].hist(ref_samples[:, ii], bins=nbins, density=True, alpha=1, lw=2, histtype='step', color='k', label='Ref');
        
        if type(samples) == list: 
            for j, ss in enumerate(samples):
                ax[ii].hist(ss[:, ii], bins=nbins, density=True, alpha=0.5, label=lbls[j]);
        else:
            ax[ii].hist(samples[:, ii], bins=nbins, density=True, alpha=0.5, label=lbls);

    ax[0].legend()
    plt.suptitle(suptitle)
    plt.tight_layout()

    if savename is None: savename='compare_hist'
    plt.savefig(savepath + savename)
    plt.close()
    #return fig, ax


def plot_mode(mode, ref_samples, nbins=20, lbls="", savepath="./tmp/", savename=None, suptitle=None):
    '''Compare mode  with reference sample
    '''
    print("Saving mode")
    if len(mode.shape) > 1: mode = mode[0]
    D = ref_samples.shape[1]
    if (lbls == "") & (type(ref_samples) == list): lbls = [""]*len(ref_samples)
    
    fig, ax = plt.subplots(1, D, figsize=(3*D, 3))

    for ii in range(D):
        ax[ii].hist(ref_samples[:, ii], bins=nbins, density=True, alpha=1, label='Ref');
        ax[ii].axvline(mode[ii], color='k')

    ax[0].legend()
    plt.suptitle(suptitle)
    plt.tight_layout()

    if savename is None: savename='mode'
    plt.savefig(savepath + savename + ".png")
    plt.close()


def plot_polyak_losses(losses, elbos, epss, savepath="./tmp/", savename=None, suptitle=None, skip=100):

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    
    skip = int(min(skip, elbos.size/10))
    if (elbos[skip:] > 0).sum() > elbos.size/2:
        ax[0].plot(elbos[skip:])
        ax[0].semilogy()
        ax[0].set_title('ELBO')
    else:
        ax[0].plot(-elbos[skip:])
        ax[0].semilogy()
        ax[0].set_title('-ELBO')
    #
    ax[1].plot(epss[skip:])
    ax[1].set_yscale('symlog', linthresh=1e-5)
    ax[1].set_title('step size')
    #
    ax[2].plot(abs(losses[skip:]))
    ax[2].semilogy()
    ax[2].set_title('abs(Loss)')
    
    for axis in ax: axis.grid(which='both', lw=0.5)
    plt.suptitle(suptitle)
    if savename is None: savename='losses'
    plt.savefig(savepath + savename)
    plt.close()
    #return fig, ax
    
def plot_bbvi_losses(elbos, losses=None, savepath="./tmp/", savename=None, suptitle=None, skip=100):

    skip = int(min(skip, elbos.size/10))
    if losses is None:
        if (elbos[skip:] > 0).sum() > elbos.size/2:
            plt.plot(elbos[skip:])
            plt.semilogy()
            plt.title('ELBO')
        else:
            plt.plot(-elbos[skip:])
            plt.semilogy()
            plt.title('-ELBO')
        #
        plt.grid(which='both', lw=0.5)
        plt.suptitle(suptitle)
        if savename is None: savename='losses'
        plt.savefig(savepath + savename)
        plt.close()

    else:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))

        if (elbos[skip:] > 0).sum() > elbos.size/2:
            ax[0].plot(elbos[skip:])
            ax[0].semilogy()
            ax[0].set_title('ELBO')
        else:
            ax[0].plot(-elbos[skip:])
            ax[0].semilogy()
            ax[0].set_title('-ELBO')
        #
        if (losses[skip:] > 0).sum() > losses.size/2:
            ax[1].plot(losses[skip:])
            ax[1].semilogy()
            ax[1].set_title('Loss')
        else:
            ax[1].plot(-losses[skip:])
            ax[1].semilogy()
            ax[1].set_title('-Loss')

        for axis in ax: axis.grid(which='both', lw=0.5)
        plt.suptitle(suptitle)
        if savename is None: savename='losses'
        plt.savefig(savepath + savename)
        plt.close()

