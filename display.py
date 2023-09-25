import torch
from loops import generate_noise
from config import device, z_dimensions, mapping_network, generator

import torchvision

import matplotlib.pyplot as plot

from pathlib import Path

load_model_directory = Path("./models/stylegan_WGAN_GP/128")
target_epoch = 47
number_images = 32
resolution = 128
alpha = 1

mapping_network.load_state_dict(torch.load(Path.joinpath(load_model_directory, f"mapping_network_{target_epoch}.pth")))
generator.load_state_dict(torch.load(Path.joinpath(load_model_directory, f"generator_{target_epoch}.pth")))

sample_noise = generate_noise(number_images, z_dimensions, device)
sample_w = mapping_network(sample_noise)
sample_images = generator(sample_w, resolution, alpha)
sample_images = (sample_images + 1) / 2

grid = torchvision.utils.make_grid(sample_images, 8)

plot.imshow(grid.permute(1, 2, 0).cpu())
plot.show()