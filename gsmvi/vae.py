"""
Follow tutorial on autoencoders with JAX from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.html and adapt to VAE.
"""

## Standard libraries
import os
import json
import math
import numpy as np
from scipy import spatial

## Import for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from matplotlib.colors import to_rgb
import matplotlib

## Progress bar
from tqdm.auto import tqdm



## JAX
import jax
import jax.numpy as jnp
from jax import random
import jax.scipy.stats as stats
import optax


## Flax (neural networks with JAX)
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints


class Encoder(nn.Module):
    c_hid : int
    latent_dim : int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 32x32 -> 16x16
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 16x16 => 8x8
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x)  # 8x8 => 4x4
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = nn.Dense(features=self.latent_dim)(x)
        return x


class Decoder(nn.Module):
    c_out : int
    c_hid : int
    latent_dim : int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*16*self.c_hid)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], 4, 4, -1)
        x = nn.ConvTranspose(features=2*self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_out, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.tanh(x)
        return x


class GenerateCallback:
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs

    def log_generations(self, model, state, logger, epoch):
        if epoch % self.every_n_epochs == 0:
            reconst_imgs = model.apply({'params': state.params}, self.input_imgs)
            reconst_imgs = jax.device_get(reconst_imgs)

            # Plot and add to tensorboard
            imgs = np.stack([self.input_imgs, reconst_imgs], axis=1).reshape(-1, *self.input_imgs.shape[1:])
            imgs = jax_to_torch(imgs)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(-1,1))
            logger.add_image("Reconstructions", grid, global_step=epoch)


def ELBO_eval(x, z, mu_x, mu_z, sigma_x=0.1, sigma_z=1):
    """
    Evalute the integrand in the ELBO.
    x: image data.
    z: latent variable, taken to have a standard normal prior.
    mu_x: likelihood mean (outputed by likelihood network).
    sigma_x: likelihood scale (outputed by likelihood network).
    mu_z: encoder mean (outputed by inference network).
    sigma_z: encoder scale (outputed by inference network)
    """
    log_like = stats.norm.logpdf(x, loc=mu_x, scale=sigma_x).sum()
    log_prior = stats.norm.logpdf(z, loc=0, scale=1).sum()
    log_q = stats.norm.logpdf(z, loc=mu_z, scale=sigma_z).sum()
    return log_like + log_prior - log_q


class VAE(nn.Module):
    c_hid: int
    latent_dim: int

    def setup(self):
        self.encoder = Encoder(c_hid=self.c_hid, latent_dim=self.latent_dim)
        self.decoder = Decoder(c_hid=self.c_hid, latent_dim=self.latent_dim, c_out=3)

    def __call__(self, rng, mc_sim, x, z=None):
        # Returns a Monte Carlo estimator of the NEGATIVE ELBO, using mc_sim
        # samples and the output of the decoder (zhat) and the decoder (xhat).
        # Expect x as an input (image data).
        # By default z is None, in which case it's generated randomly from
        # the variational distribution q(z | x), centered as zhat.
        ELBO_sum = 0
        for i in range(mc_sim):
            zhat = self.encoder(x)
            if z is None:
                eps = jax.random.normal(key=rng+i, shape=zhat.shape)
                z = zhat + eps
            xhat = self.decoder(z)
            ELBO_sum += ELBO_eval(x, z, mu_x=xhat, mu_z=zhat)

        return - ELBO_sum / mc_sim, xhat, zhat


class TrainerModule:

    def __init__(self, c_hid, latent_dim, lr=1e-3, seed=1954, mc_sim=1):
        super().__init__()
        self.c_hid = c_hid
        self.latent_dim = latent_dim
        self.lr = lr
        self.seed = seed
        self.mc_sim = mc_sim

        # Create empty inference and likelihood models
        self.model = VAE(c_hid=self.c_hid, latent_dim=self.latent_dim)

        # Prepare logging
        self.exmp_imgs = next(iter(val_loader))[0][:8]
        self.log_dir = os.path.join(CHECKPOINT_PATH, f'cifar10_{self.latent_dim}')
        self.generate_callback = GenerateCallback(self.exmp_imgs, every_n_epochs=50)
        self.logger = SummaryWriter(log_dir=self.log_dir)

        # Create jitted training and eval functions
        self.create_functions()

        # Initialize model
        self.init_model()

        # Store loss function during training
        self.loss = []

    def neg_ELBO(self, params, rng, imgs):
        negative_ELBO, _, _ = self.model.apply({'params': params}, rng, self.mc_sim, imgs)
        return negative_ELBO

    def create_functions(self):
        # Training function
        def train_step(state, batch, rng):
            imgs, _ = batch
            loss_fn = lambda params: self.neg_ELBO(params, rng, imgs)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)  # optimizer update step
            return state, loss
        self.train_step = jax.jit(train_step)

        # Eval function
        def eval_step(state, batch, rng):
            imgs, _ = batch
            return self.neg_ELBO(state.params, rng, imgs)
        self.eval_step = jax.jit(eval_step)

    def init_model(self):
        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        params = self.model.init(init_rng, init_rng, self.mc_sim, self.exmp_imgs)['params']

        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            warmup_steps=100,
            decay_steps=500*len(train_loader),
            end_value=1e-5
        )

        optimizer = optax.chain(
            optax.adam(lr_schedule)
        )

        # Initialize training set
        self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                                    params=params,
                                                    tx=optimizer)
    def train_model(self, init_rng, num_epochs=500):
        # Train model for defined number of epochs
        best_eval = 1e6
        running_rng = init_rng
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            running_rng += 33
            self.train_epoch(epoch=epoch_idx, rng=running_rng)
            if epoch_idx % 1 == 0:
                running_rng += 11
                eval_loss = self.eval_model(val_loader, running_rng)
                self.loss.append(eval_loss)

    def train_epoch(self, epoch, rng):
        # Train model for one epoch and log avg loss
        losses = []
        for batch in train_loader:
            self.state, loss = self.train_step(self.state, batch, rng)
            losses.append(loss)

        losses_np = np.stack(jax.device_get(losses))
        avg_loss = losses_np.mean()
        self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)

    def eval_model(self, data_loader, rng):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            loss = self.eval_step(self.state, batch, rng)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss

    def save_model(self, step=0):
        # Save current model at certain iterations
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params,
                            prefix=f'cifar10_{self.latent_dim}_', step=step)

    def load_model(self):
        params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params,
                                                prefix=f'cifar10_{self.latent_dim}_')

def train_cifar(latent_dim, rng):
    trainer = TrainerModule(c_hid=32, latent_dim=latent_dim, lr=1e-4)
    trainer.train_model(num_epochs=100, init_rng=rng)
    test_loss = trainer.eval_model(test_loader, rng + 43)

    # Store model with the trainer parameters
    trainer.model_bd = trainer.model.bind({'params': trainer.state.params})
    return trainer, test_loss


def visualize_reconstructions(trainer, input_imgs, rng):
    # Reconstruct imgs using likelihood mean
    _, reconst_imgs, _ = trainer.model_bd(rng, mc_sim=1, x=input_imgs)
    imgs = np.stack([input_imgs, reconst_imgs], axis=1).reshape(-1, *reconst_imgs.shape[1:])

    # Plotting
    imgs = jax_to_torch(imgs)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1,1))
    grid = grid.permute(1, 2, 0)

    plt.figure(figsize=(7,4.5))
    plt.title(f"Reconstructed from {trainer.latent_dim} latents")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

