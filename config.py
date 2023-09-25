import torch
from stylegan import MappingNetwork, Generator, Discriminator

from pathlib import Path

batch_size = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda": raise Exception("Please enable CUDA.")

z_dimensions = 384
w_dimensions = 384

###################

batches_per_epoch = 800
epochs_per_double = 48
current_epoch = 0
current_doubles = 0
maximum_doubles = 3
alpha_recovery_epochs = epochs_per_double // 2

load_models_from_epoch = None
load_model_directory = Path("./models/stylegan/xxx")
save_model_base_directory = Path("./models/stylegan")

###################

mapping_network = MappingNetwork(z_dimensions, w_dimensions).to(device)
generator = Generator(w_dimensions, image_resolution=16 * 2**maximum_doubles, starting_channels=448).to(device)
discriminator = Discriminator(image_resolution=16 * 2**maximum_doubles, max_channels=448).to(device)