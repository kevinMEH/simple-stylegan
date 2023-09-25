import torch
from torch.autograd import grad
from config import device, w_dimensions, z_dimensions

def generate_noise(batch_size, noise_channels, device):
    return torch.randn([batch_size, noise_channels], device=device)

def r1_penalty(real_batch_predictions, real_batch):
    gradients = grad(outputs=real_batch_predictions.sum(), inputs=real_batch, create_graph=True)[0]
    return gradients.pow(2).sum([1, 2, 3]).mean()

def gradient_penalty(discriminator, real_batch, fake_batch, batch_size, resolution, alpha):
    ratio = torch.rand([batch_size, 1, 1, 1], device=device)
    interpolated = torch.lerp(real_batch, fake_batch, ratio)
    
    interpolated_predictions = discriminator(interpolated, resolution, alpha)
    gradients = grad(
        interpolated_predictions, interpolated,
        torch.ones_like(interpolated_predictions),
        create_graph=True
    )[0]
    
    return ((gradients.norm(2, dim=[1, 2, 3]) - 1) ** 2).mean()

loss_function = torch.nn.BCEWithLogitsLoss()

# real_batch should be resized to match resolution before inputting.
def train_discriminator(mapping_network, generator, discriminator, real_batch, optimizer, penalty_factor, batch_size, resolution, alpha):
    
    real_batch.requires_grad_(True)
    
    real_batch_scores = discriminator(real_batch, resolution, alpha)
    
    # Mixing regularization
    fake_batch_w1 = mapping_network(generate_noise(batch_size, z_dimensions, device))
    fake_batch_w2 = mapping_network(generate_noise(batch_size, z_dimensions, device))
    mixing_indices = torch.floor(torch.rand([batch_size]) * w_dimensions) # Generate mixing indices
    mixing_indices = torch.round(torch.rand([batch_size])) * mixing_indices # 50% of indices mix, other 50% don't
    mixing_indices = (z_dimensions - mixing_indices).to(torch.int) # Make fake_batch_w1 primary and convert to int
    fake_batch_w = torch.empty_like(fake_batch_w1)
    for i, index in enumerate(mixing_indices):
        fake_batch_w[i, :index] = fake_batch_w1[i, :index]
        fake_batch_w[i, index:] = fake_batch_w2[i, index:]

    fake_batch = generator(fake_batch_w, resolution, alpha)
    fake_batch_scores = discriminator(fake_batch, resolution, alpha)
    
    # penalty = gradient_penalty(discriminator, real_batch, fake_batch, batch_size, resolution, alpha)
    # discriminator_loss = fake_batch_scores.mean() - real_batch_scores.mean() + penalty_factor * penalty + 0.001 * torch.mean(real_batch_scores ** 2)
    penalty = r1_penalty(real_batch_scores, real_batch)
    real_batch.requires_grad_(False)
    discriminator_loss = (
        loss_function(fake_batch_scores, torch.zeros_like(fake_batch_scores))
        + loss_function(real_batch_scores, torch.ones_like(real_batch_scores))
        + penalty_factor * penalty
    )

    optimizer.zero_grad(True)
    discriminator_loss.backward()
    optimizer.step()
    
    return real_batch_scores.detach().mean(), fake_batch_scores.detach().mean(), discriminator_loss.detach()

def train_generator(mapping_network, generator, discriminator, optimizer, batch_size, resolution, alpha):
    # Mixing regularization
    fake_batch_w1 = mapping_network(generate_noise(batch_size, z_dimensions, device))
    fake_batch_w2 = mapping_network(generate_noise(batch_size, z_dimensions, device))
    mixing_indices = torch.floor(torch.rand([batch_size]) * w_dimensions) # Generate mixing indices
    mixing_indices = torch.round(torch.rand([batch_size])) * mixing_indices # 50% of indices mix, other 50% don't
    mixing_indices = (z_dimensions - mixing_indices).to(torch.int) # Make fake_batch_w1 primary and convert to int
    fake_batch_w = torch.empty_like(fake_batch_w1)
    for i, index in enumerate(mixing_indices):
        fake_batch_w[i, :index] = fake_batch_w1[i, :index]
        fake_batch_w[i, index:] = fake_batch_w2[i, index:]

    fake_batch = generator(fake_batch_w, resolution, alpha)
    fake_batch_scores = discriminator(fake_batch, resolution, alpha)
    
    # generator_loss = -fake_batch_scores.mean()
    generator_loss = loss_function(fake_batch_scores, torch.ones_like(fake_batch_scores))

    optimizer.zero_grad(True)
    generator_loss.backward()
    optimizer.step()
    
    return generator_loss.detach()