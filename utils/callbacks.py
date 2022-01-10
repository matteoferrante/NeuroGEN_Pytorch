import os

from pytorch_lightning.callbacks import Callback
import numpy as np
import wandb
from imutils import build_montages
import torch
import matplotlib.pyplot as plt
from os.path import join as opj

def SaveModel(model,outdir):
    os.makedirs(outdir,exist_ok=True)
    torch.save(model.state_dict(), opj(outdir,"model.ckp"))


def visualize(x, y, n=4):
    fig, axs = plt.subplots(2, n, figsize=(10, 4))
    print(x.shape, y.shape)
    for i in range(n):
        axs[0, i].imshow(x[i])
        axs[1, i].imshow(y[i])

    plt.axis('off')
    plt.show()


def WandbImagesVAE(model,val_imgs,show=False):

    target_shape=model.input_dim
    val_imgs = val_imgs.to(device=model.device)

    x_recon = model(val_imgs)

    x_recon = x_recon[:100]  ## use more than 100 in bS
    images = x_recon.cpu().detach().permute(0, 2, 3, 1).numpy()*255
    if target_shape[0] == 1:
        images = np.repeat(images, 3, axis=-1)

    vis = build_montages(images, (target_shape[1], target_shape[-1]), (10, 10))[0]

    log = {f"image": wandb.Image(vis)}
    wandb.log(log)

    if show:
        visualize(val_imgs.cpu().permute(0,2,3,1).numpy(),images/255.)

    ## just sampling


    z = torch.randn(100, model.latent_dim).to(device=model.device)

    x_sampled = model.decoder(z)

    images = x_sampled.cpu().detach().permute(0, 2, 3, 1).numpy()*255

    if target_shape[0] == 1:
        images = np.repeat(images, 3, axis=-1)
    vis = build_montages(images, (target_shape[1], target_shape[-1]), (10, 10))[0]

    log = {f"image_sampled": wandb.Image(vis)}
    wandb.log(log)




def WandbImagesVQVAE(model,val_imgs,show=False):

    target_shape=model.input_dim
    val_imgs = val_imgs.to(device=model.device)

    loss, x_recon, perplexity = model(val_imgs)

    x_recon = x_recon[:100]  ## use more than 100 in bS
    images = x_recon.cpu().detach().permute(0, 2, 3, 1).numpy()*255
    if target_shape[0] == 1:
        images = np.repeat(images, 3, axis=-1)

    vis = build_montages(images, (target_shape[1], target_shape[-1]), (10, 10))[0]

    log = {f"image": wandb.Image(vis)}
    wandb.log(log)

    if show:
        visualize(val_imgs.cpu().permute(0,2,3,1).numpy(),images/255.)










def WandbImagesGAN(model,show=False):
    target_shape = model.target_shape
    z = torch.randn(100, model.latent_dim).to(device=model.device)
    x_sampled = model.generator(z)

    images = x_sampled.cpu().detach().permute(0, 2, 3, 1).numpy()*255

    if show:
        visualize(images/255.,images/255.)

    if target_shape[0] == 1:
        images = np.repeat(images, 3, axis=-1)
    vis = build_montages(images, (target_shape[1], target_shape[-1]), (10, 10))[0]

    log = {f"image_sampled": wandb.Image(vis)}
    wandb.log(log)
