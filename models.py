import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-Attention Layer
# Corrected SelfAttention Layer
import torch
import torch.nn as nn
import torch.nn.functional as F

# Corrected SelfAttention Layer with Downsampling
class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(SelfAttention, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Downsample input to reduce memory usage
        down_height, down_width = height // self.reduction_ratio, width // self.reduction_ratio
        x_down = F.interpolate(x, size=(down_height, down_width))

        # Compute query, key, and value
        query = self.query(x_down).view(batch_size, -1, down_height * down_width).permute(0, 2, 1)
        key = self.key(x_down).view(batch_size, -1, down_height * down_width)
        value = self.value(x_down).view(batch_size, -1, down_height * down_width)

        # Attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Compute weighted value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, down_height, down_width)

        # Upsample back to original resolution
        out = F.interpolate(out, size=(height, width))

        # Apply residual connection
        out = self.gamma * out + x
        return out

class EfficientSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(EfficientSelfAttention, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Keep channels consistent
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Reduce spatial dimensions before computing attention
        down_height, down_width = height // self.reduction_ratio, width // self.reduction_ratio
        x_down = F.adaptive_avg_pool2d(x, (down_height, down_width))

        # Query, Key, and Value projections
        query = self.query(x_down).view(batch_size, -1, down_height * down_width).permute(0, 2, 1)
        key = self.key(x_down).view(batch_size, -1, down_height * down_width)
        value = self.value(x_down).view(batch_size, -1, down_height * down_width)

        # Attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Compute weighted value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, down_height, down_width)

        # Upsample and residual connection
        out = F.interpolate(out, size=(height, width))
        out = self.gamma * out + x  # Residual connection
        return out




# U-Net Block
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.attention = EfficientSelfAttention(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.attention(x)
        return x

# U-Net with Self-Attention
class unet(nn.Module):
    def __init__(self, in_channels=5, out_channels=10):
        super(unet, self).__init__()

        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)

        # Bottleneck
        self.bottleneck = UNetBlock(256, 512)

        # Decoder
        self.dec3 = UNetBlock(512 + 256, 256)
        self.dec2 = UNetBlock(256 + 128, 128)
        self.dec1 = UNetBlock(128 + 64, 64)

        # Final Output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))

        # Decoder
        dec3 = self.dec3(torch.cat([F.interpolate(bottleneck, scale_factor=2), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, scale_factor=2), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, scale_factor=2), enc1], dim=1))

        # Final Output
        output = self.final_conv(dec1)
        return output