import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import numpy as np

from dataset.crowd_dataset import CrowdDataset


SEED = 1
CLASS_LABELS = ['human', 'non-human', 'food', 'spell', 'side-facing']
train_size = 40
val_size = 0
test_size = 0
DATASET_SIZE = train_size + val_size + test_size

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True


def prepare_dataloaders(batch_size=100, val_batch_size=32, kernel_size=3, sigma=1.0, img_height=600, img_width=872):
    images_transform = transforms.Compose([
        transforms.Resize(size=(img_height, img_width)),
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,)),  # range [-1,1]
    ])

    labels_transform = transforms.Compose([
        transforms.Resize(size=(img_height, img_width)),
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
    ])
    
    train_dataset = CrowdDataset(images_transform=images_transform, labels_transform=labels_transform)

    # train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)
    val_loader = None
    test_loader = None

    return train_loader, val_loader, test_loader


def show(imgs, title=None, fig_titles=None, save_path=None): 

    if fig_titles is not None:
        assert len(imgs) == len(fig_titles)

    fig, axs = plt.subplots(1, ncols=len(imgs), figsize=(15, 5))
    for i, img in enumerate(imgs):
        axs[i].imshow(img)
        axs[i].axis('off')
        if fig_titles is not None:
            axs[i].set_title(fig_titles[i], fontweight='bold')

    if title is not None:
        plt.suptitle(title, fontweight='bold')
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()
