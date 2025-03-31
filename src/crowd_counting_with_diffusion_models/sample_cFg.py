import matplotlib.pyplot as plt

# torch
import torch
import torch.nn.functional as F

# custom imports
from ddpm import Diffusion
from model import UNet
from dataset.helpers import *
from crowd_counting_with_diffusion_models.utils.util import show, set_seed, CLASS_LABELS

def show_n_forward(imgs, title=None, fig_titles=None, save_path=None): 

    num_cols = len(fig_titles) if fig_titles is not None else 1
    num_rows = len(imgs) // num_cols
    fig_width = 15  # Adjust the figure width as needed
    fig_height = 2 * num_rows  # Adjust the figure height as needed
    fig, axs = plt.subplots(num_rows, ncols=num_cols, figsize=(fig_width, fig_height))

    for i in range(num_rows):
        for j in range(len(fig_titles)):
            idx = j * (num_rows) + i
            img = imgs[idx]
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            if fig_titles is not None:
                axs[i, j].set_title(f"class: {i} t: {fig_titles[j]}")

    if title is not None:
        plt.suptitle(title)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed()

# Load model
ddpm_cFg = Diffusion(img_size=16, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cFg', device=device)

unet_ddpm_cFg = UNet(num_classes=5, device=device)
unet_ddpm_cFg.eval()
unet_ddpm_cFg.to(device)
# unet_ddpm_cFg.load_state_dict(torch.load('weights/DDPM-cfg/model.pth', map_location=device))
unet_ddpm_cFg.load_state_dict(torch.load('/zhome/d4/a/214319/advanced_dl_in_cv/week_5/ex6/jobs/weights/DDPM-cfg/model.pth', map_location=device))

INTERMEDIATES_LABELS = [0, 50, 100, 150, 200, 300, 499]
# Sample
y = torch.tensor([0,1,2,3,4], device=device) 
y = F.one_hot(y, num_classes=5).float()
x_new, intermediates = ddpm_cFg.p_sample_loop(unet_ddpm_cFg, 5, y=y, timesteps_to_save=INTERMEDIATES_LABELS)
imgs = [im_normalize(tens2image(x_gen.cpu())) for x_gen in x_new]
show(imgs, fig_titles=CLASS_LABELS, title='classifier FREE guidance', save_path='assets/cFg_samples.png')

print(len(intermediates))
print(len(intermediates[0]))
imgs = [im_normalize(tens2image(x_gen.cpu())) for x_gen in intermediates]
# show(imgs, fig_titles=INTERMEDIATES_LABELS, title='Classifier Guidance intermediates', save_path='assets/cg_samples_10_intermediates.png')
show_n_forward(imgs, fig_titles=INTERMEDIATES_LABELS, save_path='assets/cFg_samples_500_150epochs_intermediates.png')
