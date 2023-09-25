import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, Compose, Normalize, InterpolationMode

import os
import time
from pathlib import Path

from ffhq_dataset import FFHQ
from loops import generate_noise, train_generator, train_discriminator
from config import batch_size, device, z_dimensions, mapping_network, generator, discriminator
from config import batches_per_epoch, epochs_per_double, current_epoch, load_models_from_epoch
from config import current_doubles, maximum_doubles, alpha_recovery_epochs
from config import load_model_directory, save_model_base_directory


def load_epoch(mapping_network, generator, discriminator, epoch, model_directory):
    mapping_network.load_state_dict(torch.load(Path.joinpath(model_directory, f"mapping_network_{epoch}.pth")))
    generator.load_state_dict(torch.load(Path.joinpath(model_directory, f"generator_{epoch}.pth")))
    discriminator.load_state_dict(torch.load(Path.joinpath(model_directory, f"discriminator_{epoch}.pth")))

def save_epoch(mapping_network, generator, discriminator, epoch, model_directory):
    torch.save(mapping_network.state_dict(), Path.joinpath(model_directory, f"mapping_network_{epoch}.pth"))
    torch.save(generator.state_dict(), Path.joinpath(model_directory, f"generator_{epoch}.pth"))
    torch.save(discriminator.state_dict(), Path.joinpath(model_directory, f"discriminator_{epoch}.pth"))

# Turns images in range -1 -> 1 into 0 -> 1
def adjust_images(images):
    return (images + 1) / 2

resize_128 = Resize((128, 128), InterpolationMode.NEAREST_EXACT, antialias=True)
def generate_adjusted_sample_images_128(mapping_network, generator, sample_noise, resolution, alpha):
    sample_w = mapping_network(sample_noise)
    sample_images = generator(sample_w, resolution, alpha)
    adjusted_sample_images = adjust_images(sample_images)
    return resize_128(adjusted_sample_images)



# ToTensor, flip, and normalize to range -1 -> 1
transformations = Compose([ ToTensor(), Resize((128, 128), antialias=True), RandomHorizontalFlip(), Normalize(0.5, 0.5) ])
ffhq_dataset = FFHQ(transform=transformations)
ffhq_dataloader = DataLoader(ffhq_dataset, batch_size, shuffle=True, drop_last=True)

if load_models_from_epoch != None:
    load_epoch(mapping_network, generator, discriminator, load_models_from_epoch, load_model_directory)

generator_optimizer = torch.optim.RMSprop([
    { "params": mapping_network.parameters(), "lr": 0.00001 },
    { "params": generator.parameters() }
], lr=0.001)
discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.001)
penalty_factor = 10

if current_doubles == 0:
    alpha = 1
else:
    alpha = current_epoch / alpha_recovery_epochs
    if(alpha > 0.99999): alpha = 1
alpha_difference = 1 / (alpha_recovery_epochs * batches_per_epoch)

sample_noise = generate_noise(batch_size, z_dimensions, device)

while current_doubles <= maximum_doubles:

    current_resolution = 16 * 2**current_doubles
    resizer = Resize((current_resolution, current_resolution), antialias=True)
    save_model_directory = Path.joinpath(save_model_base_directory, str(current_resolution))
    os.makedirs(save_model_directory, exist_ok=True)

    writer = SummaryWriter("./runs/STYLEGAN/STYLEGAN_" + str(current_resolution) + "_" + str(int(time.time())))

    while current_epoch < epochs_per_double or current_doubles == maximum_doubles: # Let run at final resolution
        print(f"Epoch {current_epoch}")

        track_generator_losses = []
        track_real_batch_scores = []
        track_fake_batch_scores = []
        track_discriminator_losses = []
        
        mapping_network.train()
        generator.train()
        discriminator.train()
        for i, real_batch in enumerate(ffhq_dataloader):
            if(i == batches_per_epoch): break

            real_batch = real_batch.to(device)
            real_batch = resizer(real_batch)

            real_batch_scores, fake_batch_scores, discriminator_loss = train_discriminator(mapping_network, generator, discriminator, real_batch, discriminator_optimizer, penalty_factor, batch_size, current_resolution, alpha)
            generator_loss = train_generator(mapping_network, generator, discriminator, generator_optimizer, batch_size, current_resolution, alpha)
            
            track_real_batch_scores.append(real_batch_scores)
            track_fake_batch_scores.append(fake_batch_scores)
            track_discriminator_losses.append(discriminator_loss)
            track_generator_losses.append(generator_loss)
            
            del discriminator_loss
            del generator_loss
            
            if(alpha >= 0.99999): alpha = 1
            else: alpha += alpha_difference
        mapping_network.eval()
        generator.eval()
        discriminator.eval()
        
        mean_real_batch_scores = torch.tensor(track_real_batch_scores).mean().item()
        mean_fake_batch_scores = torch.tensor(track_fake_batch_scores).mean().item()
        mean_discriminator_loss = torch.tensor(track_discriminator_losses).mean().item()
        mean_generator_loss = torch.tensor(track_generator_losses).mean().item()
        
        print(f"Mean real batch scores: {mean_real_batch_scores}")
        print(f"Mean fake batch scores: {mean_fake_batch_scores}")
        print(f"Mean discriminator loss: {mean_discriminator_loss}")
        print(f"Mean generator loss: {mean_generator_loss}")
        
        writer.add_scalar("Loss/discriminator", mean_discriminator_loss, current_epoch)
        writer.add_scalar("Loss/generator", mean_generator_loss, current_epoch)
        
        sample_images = generate_adjusted_sample_images_128(mapping_network, generator, sample_noise, current_resolution, alpha)
        grid = torchvision.utils.make_grid(sample_images, 8)
        writer.add_image("Images", grid, current_epoch)

        save_epoch(mapping_network, generator, discriminator, current_epoch, save_model_directory)
        
        del mean_real_batch_scores
        del mean_fake_batch_scores
        del mean_discriminator_loss
        del mean_generator_loss
        del sample_images
        del grid

        current_epoch += 1
    
    alpha = 0
    current_epoch = 0
    current_doubles += 1