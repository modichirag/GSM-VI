import numpy as np
import sys, os, time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#
sys.path.append('../src/')
import gsm, bbvi
import gaussianq
import diagnostics as dg
import divergences as divs
import folder_path
#



##################################################################
# GSM
def run_gsm(args, model, x0, modelpath, callback, samples):
    '''This function sets up the path for GSM, trains the variational distribution, and saves it and evolution of different metrics
    '''
    D = args.D
    folderpath = folder_path.frg_path_gsm(args)
    savepath = f"{modelpath}/{folderpath}/S{args.seed}/"
    os.makedirs(savepath, exist_ok=True)
    print(f"\n#####\nResults for GSM will be saved in folder\n{savepath}\n#####\n")
    suptitle = "-".join(savepath.split('/')[-3:-1])
    #print('\nSuptitle : ', suptitle)

    def train(qdist, model, niter=1000, batch_size=4, nprint=10, dtype=tf.float32, callback=None, verbose=True, samples=None):

        qdivs, fdivs, counts = [], [], []
        gradcount = 0
        for epoch in range(niter + 1):

            x = list(qdist.sample(batch_size))
            _ = gsm.gaussian_update_batch(x, model, qdist)
            if np.linalg.eigvals(qdist.cov).min() < 0:
                print("ERROR : NEGATIVE EIGENVALUES IN COVARIANCE")
                break
            gradcount += batch_size
            counts.append(gradcount)
            qdiv = divs.qdivergence(qdist, model)        
            qdivs.append(qdiv)
            fdiv = divs.fkl_divergence(qdist, model, samples=samples)
            fdivs.append(fdiv)

            if (epoch %(niter//nprint) == 0) & verbose: 
                print("Loss at epoch %d is"%epoch, qdiv[0])
                if callback is not None: 
                    callback(qdist, [np.array(qdivs)], epoch, savepath, suptitle)

        return qdist, np.array(qdivs), np.array(counts), np.array(fdivs)


    ### Setup VI
    print()
    print("Start GSM")
    qdist = gaussianq.FR_Gaussian_cov(D, mu=tf.constant(x0[0]), scale=args.scale)
    #print('log prob : ', qdist.log_prob(np.random.random(D).reshape(1, D).astype(np.float32)))

    #Train
    qdist, qdivs, counts, fdivs = train(qdist, model,
                                        batch_size=args.batch, 
                                        niter=args.niter, 
                                        callback=callback,
                                        samples=samples,
                                        )

    #Save diagnostics
    print("number of gradient calls in GSM : ", counts[-1])
    dg.plot_bbvi_losses(qdivs[..., 0], qdivs[..., 0], savepath, suptitle=suptitle)
    np.save(savepath + 'qdivs', qdivs)
    np.save(savepath + 'fdivs', fdivs)
    np.save(savepath + 'counts', counts)

    qsamples = qdist.sample(1000)
    qsamples = qsamples.numpy()
    np.save(savepath + 'mufit', qdist.loc.numpy())
    np.save(savepath + 'covfit', qdist.cov.numpy())
    idx = list(np.arange(D)[:int(min(D, 10))])
    dg.compare_hist(qsamples[:, idx], samples.numpy()[:, idx], savepath=savepath, savename='hist', suptitle=suptitle)
    return qsamples, qdivs, fdivs, counts



##################################################################
# BBVI
def run_bbvi(args, model, x0, modelpath, callback, samples):
#def run_bbvi():
    '''This function sets up the path for BBVI, trains the variational distribution, and saves it and evolution of different metrics
    '''

    D = args.D
    folderpath = folder_path.frg_path_bbvi(args)
    savepath = f"{modelpath}/{folderpath}/S{args.seed}/"
    os.makedirs(savepath, exist_ok=True)
    print(f"\n#####\nResults for BBVI will be saved in folder\n{savepath}\n#####\n")
    suptitle =  "-".join(savepath.split('/')[-3:-1])


    def train(qdist, model, lr=1e-3, batch_size=32, niter=1001, nprint=None, verbose=True, callback=None, samples=None):

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elbos, losses, qdivs, counts, fdivs = [], [], [], [], []
        gradcount = 0
        for epoch in range(niter+1):

            counts.append(gradcount) 
            qdiv = divs.qdivergence(qdist, model, nsamples=samples.shape[0])
            qdivs.append(qdiv)
            if samples is not None:
                fdiv = divs.fkl_divergence(qdist, model, samples=samples)
                fdivs.append(fdiv)
            else:
                fdivs = None

            elbo, loss, grads = bbvi.bbvi_elbo(qdist, model, batch=tf.constant(batch_size))
            elbo = elbo.numpy()
            if np.isnan(elbo):
                print("NaNs!!! :: ", epoch, elbo)
                break

            elbos.append(elbo)
            losses.append(loss)
            gradcount += batch_size

            opt.apply_gradients(zip(grads, qdist.trainable_variables))
            if (epoch %nprint == 0) & verbose: 
                print("Elbo at epoch %d is"%epoch, elbo)
                if callback is not None: callback(qdist, [np.array(elbos)], epoch, savepath, suptitle)

        return qdist, np.array(losses), np.array(elbos), np.array(qdivs), np.array(counts), np.array(fdivs)


    ### Setup VI
    print()
    print("Start BBVI")
    qdist = gaussianq.FR_Gaussian(D, mu=tf.constant(x0[0]), scale=args.scale)
    #print('log prob : ', qdist.log_prob(np.random.random(D).reshape(1, D).astype(np.float32)))


    qdist, losses, elbos, qdivs, counts, fdivs = train(qdist, model,
                                                            batch_size=args.batch, 
                                                            lr=args.lr, 
                                                            niter=args.niter,
                                                            nprint=args.niter//10,
                                                            callback=callback,
                                                            samples=samples)
    #
    print(losses.shape, elbos.shape, qdivs.shape, counts.shape)
    print("number of gradient calls in BBVI : ", counts[-1])
    dg.plot_bbvi_losses(elbos, qdivs, savepath, suptitle=suptitle)
    np.save(savepath + 'elbo', elbos)
    np.save(savepath + 'qdivs', qdivs)
    np.save(savepath + 'losses', losses)
    np.save(savepath + 'counts', counts)
    np.save(savepath + 'fdivs', fdivs)


    qsamples = qdist.sample(1000)
    qsamples = qsamples.numpy()
    idx = list(np.arange(D)[:int(min(D, 10))])
    dg.compare_hist(qsamples[:, idx], samples.numpy()[:, idx], savepath=savepath, savename='hist', suptitle=suptitle)
    np.save(savepath + 'mufit', qdist.loc.numpy())
    np.save(savepath + 'scalefit', qdist.scale.numpy())
    return qsamples, qdivs, fdivs, counts


##################################################################
# 
def make_plot(args, samples, gsmrun, bbvirun, modelpath):

    ##################
    # make plot of comparing the FKL-divergence
    mode = 'fdivs'

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharex=True, sharey=True)
    plt.title(f'D={args.D}', fontsize=10)
    if mode == 'qdivs':
        plt.plot(gsmrun[3], gsmrun[1][:, 0], label='GSM-VI')
        plt.plot(bbvirun[3], bbvirun[1][:, 0], label='BBVI')
    else:
        plt.plot(gsmrun[3], -gsmrun[2][:, 0], label='GSM-VI')
        plt.plot(bbvirun[3], -bbvirun[2][:, 0], label='BBVI')

    plt.loglog()
    plt.grid(which='both', lw=0.2)
    plt.xlabel('# gradient evaluations')
    if mode == 'qdivs': 
        #axis.set_ylim(1e-6, 1e4)    
        plt.ylabel(r'$\sum_{x \sim q} \log q(x) - \log p(x)$')
    if mode == 'fdivs': 
        #axis.set_ylim(1e-6, 1e6)    
        plt.ylabel(r'$\sum_{x \sim p} \log p(x) - \log q(x)$')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'./{modelpath}/{mode}.png')
    print(f"\nFigure comparing evolution of FKL for GSM and BBVI is saved at \n{modelpath}/{mode}.png")
    plt.close()
    
    ##################
    # make plot of comparing the histograms
    idx = min(args.D, 5)
    fig, ax = plt.subplots(1, idx, figsize=(3*idx, 3))
    
    for i in range(idx):
        ax[i].hist(samples.numpy()[:, i], histtype='step', color='k', lw=2, density=True, label='Target')
        ax[i].hist(gsmrun[0][:, i], density=True, alpha=0.7, label='GSM-VI')
        ax[i].hist(bbvirun[0][:, i], density=True, alpha=0.7, label='BBVI-VI')
        
    ax[0].legend()
    plt.title(f'D={args.D}', fontsize=10)
    plt.savefig(f'./{modelpath}/hist.png')
    print(f"\nFigure comparing histograms of marginal posteriors is saved at  \n{modelpath}/hist.png")
    plt.close()


