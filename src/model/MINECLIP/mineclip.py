"""
Adapted from OpenAI CLIP implementation: https://github.com/openai/CLIP
"""
from collections import OrderedDict
from transformers import CLIPImageProcessor
import numpy as np
import torch
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self._resolution = resolution
        self._patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(
            scale * torch.randn((resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.repeat((B, 1, 1)), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.projection is not None:
            x = x @ self.projection

        return x

    def forward_patch_features(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.repeat((B, 1, 1)), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x[:, 1:, :]


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize((n_px, n_px), interpolation=BICUBIC),
            # CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def load(encoder_ckpt_path, device):
    clip_config = {
            "embed_dim": 512,
            "image_resolution": 224,
            "vision_layers": 12,
            "vision_patch_size": 16,
            "vision_width": 768,
        }

    model = VisionTransformer(
            resolution=clip_config['image_resolution'],
            patch_size=clip_config['vision_patch_size'],
            width=clip_config['vision_width'],
            layers=clip_config['vision_layers'],
            heads=clip_config['vision_width'] // 16,
            output_dim=clip_config['embed_dim'],
        )
    state_dict = torch.load(encoder_ckpt_path)
    model.load_state_dict(state_dict, strict = False)


    return model, _transform(clip_config['image_resolution'])