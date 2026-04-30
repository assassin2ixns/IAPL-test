import math

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma, device, dtype):
    sigma = max(float(sigma), 1e-6)
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _depthwise_conv(images, kernel):
    channels = images.shape[1]
    kernel = kernel.to(device=images.device, dtype=images.dtype)
    kernel = kernel.expand(channels, 1, kernel.shape[-2], kernel.shape[-1])
    return F.conv2d(images, kernel, padding=0, groups=channels)


def _gaussian_blur(images, sigma=1.0):
    if sigma <= 0:
        return images

    kernel = _gaussian_kernel1d(sigma, images.device, images.dtype)
    radius = kernel.numel() // 2
    kernel_h = kernel.view(1, 1, -1, 1)
    kernel_w = kernel.view(1, 1, 1, -1)

    x = F.pad(images, (0, 0, radius, radius), mode="reflect")
    x = _depthwise_conv(x, kernel_h)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = _depthwise_conv(x, kernel_w)
    return x


def build_highpass_view(images, sigma=1.0):
    return images - _gaussian_blur(images, sigma=sigma)


def build_laplacian_view(images):
    kernel = images.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).view(1, 1, 3, 3)
    x = F.pad(images, (1, 1, 1, 1), mode="reflect")
    return _depthwise_conv(x, kernel)


def build_chroma_residual_view(images):
    return images - images.mean(dim=1, keepdim=True)


def build_blur_view(images, sigma=1.0):
    return _gaussian_blur(images, sigma=sigma)


def build_downup_view(images, scale=0.5):
    _, _, height, width = images.shape
    down_h = max(int(round(height * scale)), 1)
    down_w = max(int(round(width * scale)), 1)
    x = F.interpolate(images, size=(down_h, down_w), mode="bilinear", align_corners=False)
    return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)


def build_lowpass_view(images):
    return build_blur_view(images, sigma=2.0)


def build_artifact_view(images, mode="canonical", training=True):
    if mode == "canonical" or not training:
        return build_highpass_view(images, sigma=1.0)
    if mode != "random_family":
        raise ValueError(f"Unsupported artifact view mode: {mode}")

    family = int(torch.randint(0, 3, (1,), device=images.device).item())
    if family == 0:
        return build_highpass_view(images, sigma=1.0)
    if family == 1:
        return build_laplacian_view(images)
    return build_chroma_residual_view(images)


def build_structure_view(images, mode="canonical", training=True):
    if mode == "canonical" or not training:
        return build_blur_view(images, sigma=1.0)
    if mode != "random_family":
        raise ValueError(f"Unsupported structure view mode: {mode}")

    family = int(torch.randint(0, 3, (1,), device=images.device).item())
    if family == 0:
        return build_blur_view(images, sigma=1.0)
    if family == 1:
        return build_downup_view(images, scale=0.5)
    return build_lowpass_view(images)


def build_tile_views(images, grid=2):
    grid = int(grid)
    if grid <= 1:
        return images, 1

    batch, channels, height, width = images.shape
    tiles = []
    for row in range(grid):
        h0 = row * height // grid
        h1 = (row + 1) * height // grid
        for col in range(grid):
            w0 = col * width // grid
            w1 = (col + 1) * width // grid
            tile = images[:, :, h0:h1, w0:w1]
            tile = F.interpolate(tile, size=(height, width), mode="bilinear", align_corners=False)
            tiles.append(tile)

    tile_count = grid * grid
    tile_views = torch.stack(tiles, dim=1).reshape(batch * tile_count, channels, height, width)
    return tile_views, tile_count
