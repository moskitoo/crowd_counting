import argparse
import logging
import os
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import typer
from crowd_dataset import prepare_dataloaders, set_seed
from ddpm import Diffusion
from model import UNet
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


set_seed()


def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis("off")
    if path is not None:
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    plt.close()


def create_result_folders(experiment_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs(os.path.join("models", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("reports", experiment_name), exist_ok=True)


def train(
    T=1000,
    cfg=True,
    # img_height=560,
    # img_width=870,
    img_height=128,
    img_width=128,
    input_channels=1,
    channels=32,
    time_dim=256,
    batch_size=1,
    lr=1e-3,
    num_epochs=30,
    experiment_name="DDPM-cfg",
    show=False,
    device="cpu",
    kernel_size=3,
    sigma=1.0,
    verbose=False,
):
    create_result_folders(experiment_name)
    train_loader, _, _ = prepare_dataloaders(
        batch_size=batch_size, kernel_size=kernel_size, sigma=sigma, img_height=img_height, img_width=img_width
    )

    num_classes = 5 if cfg else None

    model = UNet(
        img_height=img_height,
        img_width=img_width,
        c_in=input_channels,
        c_out=input_channels,
        num_classes=num_classes,
        time_dim=time_dim,
        channels=channels,
        device=device,
        verbose=verbose,
    ).to(device)

    diff_type = "DDPM-cFg" if cfg else "DDPM"
    diffusion = Diffusion(
        img_height=img_height,
        img_width=img_width,
        T=T,
        beta_start=1e-4,
        beta_end=0.02,
        diff_type=diff_type,
        device=device,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    logger = SummaryWriter(os.path.join("runs", experiment_name))
    l = len(train_loader)

    min_train_loss = 1e10
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader)
        epoch_loss = 0
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # if diff_type == "DDPM-cFg":
            #     # one-hot encode labels for classifier-free guidance
            #     labels = labels.to(device)
            #     labels = F.one_hot(labels, num_classes=num_classes).float()
            # else:
            #     labels = None

            # Train a diffusion model with classifier-free guidance
            # Do not forget randomly discard labels
            p_uncod = 0.1

            if np.random.rand() < p_uncod:
                images = None

            t = diffusion.sample_timesteps(labels.shape[0]).to(device)  # line 3 from the Training algorithm
            x_t, noise = diffusion.q_sample(
                labels, t
            )  # inject noise to the images (forward process), HINT: use q_sample
            predicted_noise = model(x_t, t, images)
            loss = mse(predicted_noise, noise)  # loss between noise and predicted noise

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        epoch_loss /= l
        if epoch_loss <= min_train_loss:
            torch.save(
                model.state_dict(),
                os.path.join("models", experiment_name, f"model.pth"),
            )
            min_train_loss = epoch_loss

        if epoch % 5 == 0:
            if diffusion.diff_type == "DDPM-cFg":
                # to be optimized
                dataset = train_loader.dataset
                random_idx = random.randint(0, len(dataset) - 1)
                image, label = dataset[random_idx]
                image = image.to(device)
                label = label.to(device)
                image = image.unsqueeze(0)
                # images, labels = next(iter(train_loader)).to(device)
                # y = torch.tensor([np.random.randint(0, 5)], device=device)
                title = f"epoch_{epoch}_sample"
            else:
                y = None
                title = f"Epoch {epoch}"

            sampled_images = diffusion.p_sample_loop(model, batch_size=labels.shape[0], y=image)
            save_images(
                images=sampled_images,
                path=os.path.join("reports", experiment_name, f"{epoch}.jpg"),
                show=show,
                title=title,
            )
            save_images(
                images=label,
                path=os.path.join("reports", experiment_name, f"gt_{epoch}.jpg"),
                show=show,
                title=title,
            )


def run_experiment(cfg: DictConfig):
    verbose: bool = cfg.verbose
    free_guidance: bool = cfg.get("free_guidance", True)
    num_epochs: int = cfg.get("num_epochs", 30)
    batch_size: int = cfg.get("batch_size", 1)
    kernel_size: int = cfg.get("kernel_size", 3)
    sigma: float = cfg.get("sigma", 1.0)
    exp_name: str = cfg.get("exp_name", "DDPM-cfg")
    img_height: int = cfg.get("img_height", 128)
    img_width: int = cfg.get("img_width", 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model will run on {device}, classifier-free guidance: {free_guidance} \n")

    if not free_guidance:
        print("To train a classifier-free guidance model, activate the flag by running the script as follows>")
        print("python ddm_train.py --cfg \n")

    set_seed()
    train(
        cfg=free_guidance,
        num_epochs=num_epochs,
        experiment_name=exp_name,
        device=device,
        batch_size=batch_size,
        kernel_size=kernel_size,
        sigma=sigma,
        verbose=verbose,
        img_height=img_height,
        img_width=img_width,
    )


def main(config_name: str = "exp_1.yaml"):
    """Run training with configuration from specified YAML file."""
    from hydra.compose import compose
    from hydra.initialize import initialize
    
    with initialize(version_base=None, config_path="configs"):
        # Load the config from the specified YAML file
        cfg = compose(config_name=config_name)
        run_experiment(cfg)


if __name__ == "__main__":
    typer.run(main)
