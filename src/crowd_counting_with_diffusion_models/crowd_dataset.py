import os
import random

import numpy as np
import scipy
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import re

import matplotlib.pyplot as plt
import cv2

# from .helpers import set_seed

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class CrowdDataset(Dataset):
    def __init__(self, 
            images_transform,
            labels_transform,
            images_dir_path ='/zhome/d4/a/214319/adlcv_project/data/part_A_final/train_data/images', 
            labels_dir_path ='/zhome/d4/a/214319/adlcv_project/data/part_A_final/train_data/ground_truth', 
            num_samples=40,
            seed=1,
        ):

        self.images_paths = [path for path in os.scandir(images_dir_path)]
        self.labels_paths = [path for path in os.scandir(labels_dir_path)]

        self.images_paths = sorted(self.images_paths, key=lambda path: int(re.search(r'\d+', path.name).group()))
        self.labels_paths = sorted(self.labels_paths, key=lambda path: int(re.search(r'\d+', path.name).group()))

        # Reduce dataset size
        if num_samples:
            set_seed(seed=seed)
            sampled_indeces = random.sample(range(len(self.images_paths)), num_samples)
            self.images_paths = [os.path.join(images_dir_path, self.images_paths[i]) for i in sampled_indeces]
            self.labels_paths = [os.path.join(labels_dir_path, self.labels_paths[i]) for i in sampled_indeces]

        self.images_transform = images_transform
        self.labels_transform = labels_transform
       
                
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = Image.open(image_path)
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        label_path = self.labels_paths[idx]
        mat = scipy.io.loadmat(label_path)
        label_points = mat["image_info"][0, 0][0, 0][0].astype(np.int32)
        # width, height, _ = image.shape
        width, height = image.size
        label = np.zeros((height, width))
        label_points[:, 1] = np.clip(label_points[:, 1], 0, height - 1)
        label_points[:, 0] = np.clip(label_points[:, 0], 0, width - 1)
        label[label_points[:, 1], label_points[:, 0]] = 1

        if self.images_transform:
            image = self.images_transform(image)

        # raw_count = label.sum()

        if self.labels_transform:
            label = self.labels_transform(Image.fromarray(label))
            # Extract the tensor data
            # label_blurred = label.numpy()
            
            # # Normalize to preserve people count
            # if raw_count > 0:  # Avoid division by zero
            #     current_sum = label_blurred.sum()
            #     label_blurred = label_blurred * (raw_count / current_sum)
            #     label = torch.from_numpy(label_blurred)
        
        return image, label


# # transform = transforms.Compose([
# #         transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
# #         transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
# # ])

# kernel_size = 3
# sigma = 1.0

# # labels_transform = transforms.Compose([
# #         transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
# #         transforms.GaussianBlur(kernel_size=kernel_size),
# # ])

# img_height = 256
# img_width = 256

# images_transform = transforms.Compose([
#     transforms.Resize(size=(img_height, img_width)),
#     transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
#     transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
# ])

# labels_transform = transforms.Compose([
#     transforms.Resize(size=(img_height, img_width)),
#     transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
#     transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
# ])


# train_dataset = CrowdDataset(images_transform=images_transform, labels_transform=labels_transform)
# print(train_dataset)

# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# image, label = next(iter(train_loader))

# label_np = label.numpy()

# sample_label = np.einsum('bchw->hw', label_np)


# plt.imsave(f"sample_density_map_{kernel_size}.png", sample_label)

# estimated_crowd = sample_label.sum()
# print(estimated_crowd)


# # Convert to BGR for OpenCV
# # image_bgr = cv2.cvtColor(image[0].numpy(), cv2.COLOR_RGB2BGR)
# image_bgr = cv2.cvtColor(image[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)



# # Get coordinates where label is 1
# points = np.column_stack(np.where(label[0].numpy() == 1))

# # Draw small red circles at those points
# for y, x in points:
#     cv2.circle(image_bgr, (x, y), radius=3, color=(0, 0, 255), thickness=-1)  # Red circles

# # Convert back to RGB for displaying with Matplotlib
# image_with_circles = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# image_with_circles = image_with_circles.astype(np.float32) / 255.0

# plt.imsave("output_image.png", image_with_circles)



