import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

class CLinear(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linear = nn.Linear(input, output)
        self.scale = (2 / input)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)
    def forward(self, x):
        return self.linear(x * self.scale) + self.bias * self.scale

class CConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.scale = (2 / (input_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(-1, 1, 1) * self.scale

class MappingNetwork(nn.Module):
    def __init__(self, z_dimensions, w_dimensions, rms_norm_epsilon = 1e-6):
        super().__init__()
        self.rms_norm_epsilon = rms_norm_epsilon
        self.mapping = nn.Sequential(
            CLinear(z_dimensions, z_dimensions),
            nn.LeakyReLU(0.2),
            CLinear(z_dimensions, z_dimensions),
            nn.LeakyReLU(0.2),
            CLinear(z_dimensions, z_dimensions),
            nn.LeakyReLU(0.2),
            CLinear(z_dimensions, z_dimensions),
            nn.LeakyReLU(0.2),
            CLinear(z_dimensions, z_dimensions),
            nn.LeakyReLU(0.2),
            CLinear(z_dimensions, z_dimensions),
            nn.LeakyReLU(0.2),
            CLinear(z_dimensions, z_dimensions),
            nn.LeakyReLU(0.2),
            CLinear(z_dimensions, w_dimensions),
        )
    
    def forward(self, z):
        # RMS Norm
        z = z * (torch.mean(z**2, dim=1, keepdim=True) + self.rms_norm_epsilon).rsqrt()
        return self.mapping(z)

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, w_dimensions, channels):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_network = CLinear(w_dimensions, channels)
        self.style_shift_network = CLinear(w_dimensions, channels)
    
    def forward(self, x_and_w): # Accept x_and_w tuple for nn.Sequential compatibility
        x, w = x_and_w
        x = self.instance_norm(x)
        style_scale = self.style_scale_network(w).unsqueeze(2).unsqueeze(3)
        style_shift = self.style_shift_network(w).unsqueeze(2).unsqueeze(3)
        return (style_scale * x + style_shift, w) # Return both for next layer

class InjectNoise(nn.Module):
    def __init__(self, channels, image_dimensions):
        super().__init__()
        self.image_dimensions = image_dimensions
        self.register_parameter("scale", nn.Parameter(torch.zeros([channels, 1, 1])))
    
    def forward(self, x_and_w): # Accept x_and_w tuple for nn.Sequential compatibility
        x, w = x_and_w
        noise = torch.randn([x.shape[0], 1, self.image_dimensions, self.image_dimensions], device=x.device)
        return (x + self.scale * noise, w) # Returns both for next layer

class LeakyReLUFirst(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
    def forward(self, x_and_w):
        x, w = x_and_w
        return (self.leaky_relu(x), w)

class ConvNoiseNorm(nn.Module):
    def __init__(self, in_channels, out_channels, image_dimensions, w_dimensions):
        super().__init__()
        self.conv = CConv2d(in_channels, out_channels, 3, 1, 1)
        self.noise_relu_norm = nn.Sequential(
            InjectNoise(out_channels, image_dimensions),
            LeakyReLUFirst(0.2),
            AdaptiveInstanceNorm(w_dimensions, out_channels),
        )
    
    def forward(self, x_and_w): # Accept x_and_w tuple for nn.Sequential compatibility
        x, w = x_and_w
        x = self.conv(x)
        return self.noise_relu_norm((x, w)) # Return both for next layer


class Generator(nn.Module):
    # We don't include the mapping network here as we are going to perform mixing regularization
    def __init__(self, w_dimensions, image_resolution, image_channels=3, starting_dimensions=4, starting_channels=384):
        super().__init__()
        self.starting_dimension_log2 = int(log2(starting_dimensions))

        self.doubles_required = int(log2(image_resolution)) - self.starting_dimension_log2
        self.register_parameter("starting_constant", nn.Parameter(torch.ones([starting_channels, starting_dimensions, starting_dimensions])))
        
        channels = [ min(64 * 2**i, starting_channels) for i in range(self.doubles_required) ]
        channels.reverse() # From starting_channels -> 64 channels
        self.initial = nn.Sequential(
            InjectNoise(starting_channels, starting_dimensions),
            LeakyReLUFirst(0.2),
            AdaptiveInstanceNorm(w_dimensions, starting_channels),
            ConvNoiseNorm(starting_channels, starting_channels, starting_dimensions, w_dimensions)
        )
        
        self.post_upsample_list = nn.ModuleList()
        self.rgb_converter_list = nn.ModuleList()
        last_channel = starting_channels
        current_dimension = starting_dimensions
        for channel in channels:
            current_dimension = current_dimension * 2
            self.post_upsample_list.append(nn.Sequential(
                ConvNoiseNorm(last_channel, channel, current_dimension, w_dimensions),
                ConvNoiseNorm(channel, channel, current_dimension, w_dimensions),
            ))
            self.rgb_converter_list.append(nn.Sequential(
                CConv2d(channel, image_channels, 1),
                nn.Tanh()
            ))
            last_channel = channel
    
    def forward(self, w, resolution, alpha): # Starts at 16
        resolution_log2 = int(log2(resolution))
        current_doubles_required = resolution_log2 - self.starting_dimension_log2

        x = self.starting_constant.repeat([w.shape[0], 1, 1, 1])
        x, w = self.initial((x, w))
        # Go through doubles but manually do last one for interpolation
        for doubling in self.post_upsample_list[:current_doubles_required - 1]:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, w = doubling((x, w))
        
        upscaled_x = F.interpolate(x, scale_factor=2, mode="bilinear")
        upscaled_x_rgb = self.rgb_converter_list[current_doubles_required - 2](upscaled_x)

        x, w = self.post_upsample_list[current_doubles_required - 1]((upscaled_x, w))
        x_rgb = self.rgb_converter_list[current_doubles_required - 1](x)

        # After we double resolution, for the first few iterations we will
        # interpolate the upscaled resolution of the image processed through the
        # old generator with the new double resolution generated image. Low
        # alpha = new image less weight, high alpha = new image more weight.
        return torch.lerp(x_rgb, upscaled_x_rgb, 1 - alpha)


class Discriminator(nn.Module):
    def __init__(self, image_resolution, image_channels=3, max_channels=384):
        super().__init__()
        self.image_resolution_log2 = int(log2(image_resolution))
        channels = [ min(64 * 2**i, max_channels) for i in range(self.image_resolution_log2) ]

        # Always the first layer, converts from RGB to next layer channels
        self.rgb_converter_list = nn.ModuleList()
        self.pre_downsample_list = nn.ModuleList()
        last_channel = channels[0]
        # Stop at 2^2, or 4x4 because that is our final convolutions
        for channel in channels[:0 - 2]: # High resolution channels -> Low resolution channels
            self.rgb_converter_list.append(nn.Sequential(
                CConv2d(image_channels, last_channel, 3, 1, 1),
                nn.LeakyReLU(0.2),
            ))
            self.pre_downsample_list.append(
                nn.Sequential(
                    CConv2d(last_channel, channel, 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    CConv2d(channel, channel, 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            )
            last_channel = channel
        # In official implementations unncessary 0 padding is introduced in the
        # convolutional layer which maintains 4x4 size with 0s on the sides, but
        # adds unnecessary parameters into fully connected layers (they are
        # constant 0s. Why pass 0s into linear layers?). This is now removed.
        self.output = nn.Sequential(
            # Conv might not even be necessary, maybe remove and use 4x4 linear
            CConv2d(last_channel, last_channel, 3),
            nn.Flatten(),
            CLinear(last_channel * 2 * 2, last_channel),
            nn.LeakyReLU(0.2),
            CLinear(last_channel, 1)
        )
    
    def forward(self, x, resolution, alpha): # Resolution start at 16
        resolution_log2 = int(log2(resolution))
        start_index = self.image_resolution_log2 - resolution_log2
        
        half_x = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        half_x = self.rgb_converter_list[start_index + 1](half_x)

        # Manually conv down once to interpolate before going through conv down
        # loop.
        x = self.rgb_converter_list[start_index](x)
        x = self.pre_downsample_list[start_index](x)
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        
        # After we double resolution, for the first few iterations we will
        # interpolate the half resolution of the image processed through the
        # previous discriminator pipeline with the full resolution image
        # processed through the new discriminator pipeline with the new front
        # blocks. Low alpha = new blocks results less weight, high alpha = new
        # block results greater weight.
        x = torch.lerp(x, half_x, 1 - alpha)

        # Proceed through conv down loop as normal
        for pre_downsample in self.pre_downsample_list[start_index + 1:]:
            x = pre_downsample(x)
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        return self.output(x)