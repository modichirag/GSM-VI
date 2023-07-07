import numpy as np
import matplotlib.pyplot as plt


def plot_loss(x, y, savepath, fname, logit=True):

    x, y = np.array(x), np.array(y)
    plt.figure()
    plt.plot(x, y)
    plt.grid()
    if logit :
        plt.plot(x, -y, '--')
        plt.loglog()
    plt.savefig(f'{savepath}/{fname}.png')
    plt.close()



def corner(samples, savepath="./tmp/", savename='corner', save=True, maxdims=10):
    '''Make corner plot for the distribution from samples
    '''
    D = min(samples.shape[1], maxdims)
    
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
        plt.savefig(savepath + savename)
        plt.close()
    else: return fig, ax



def compare_hist(samples, ref_samples=None, nbins=20, lbls="", savepath="./tmp/", savename='hist', maxdims=10, suptitle=None):
    '''Compare histogram of samples with reference sample
    '''
    D = min(samples.shape[1], maxdims)
    if ref_samples is not None: D = min(D, ref_samples.shape[1])
    
    if (lbls == "") & (type(samples) == list): lbls = [""]*len(samples)

    fig, ax = plt.subplots(1, D, figsize=(3*D, 3))

    for ii in range(D):
        if ref_samples is not None:
            ax[ii].hist(ref_samples[:, ii], bins=nbins, density=True, alpha=1, lw=2, histtype='step', color='k', label='Ref');
        
        if type(samples) == list: 
            for j, ss in enumerate(samples):
                ax[ii].hist(ss[:, ii], bins=nbins, density=True, alpha=0.5, label=lbls[j]);
        else:
            ax[ii].hist(samples[:, ii], bins=nbins, density=True, alpha=0.5, label=lbls);

    ax[0].legend()
    plt.suptitle(suptitle)
    plt.tight_layout()

    plt.savefig(savepath + savename)
    plt.close()
    
