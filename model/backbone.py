import timm
from torch import nn
import torch

""" Layer/Module Helpers
Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__( self, in_channels: int, out_channels: int):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride= 1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride= 1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.gelu1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = y + x
        y = self.gelu2(y)
        return y


class VisionTransformerExtractor(nn.Module):
    def __init__(self, name='vit_base_patch32_384', img_size=(512, 1024), patch_size=32, return_intermediate_states=True, in_chans=3, num_classes=0):
        super().__init__()
        self.model = timm.create_model(name, num_classes=0, pretrained=False)
        self.model.img_size = 1024
        self.patch_size = patch_size
        self.model.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=self.model.embed_dim)
        num_patches = self.model.patch_embed.num_patches
        print('num_patches', num_patches)

        self.model.cls_token = nn.Parameter(torch.zeros(1, 1, self.model.embed_dim))
        self.model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.model.embed_dim))
        self.return_intermediate_states = return_intermediate_states
        if self.return_intermediate_states:
            self.conv = nn.Conv2d(in_channels =len(self.model.blocks) * self.model.embed_dim, out_channels=512, kernel_size=1)
        else:
            self.conv = nn.Conv2d(in_channels= self.model.embed_dim, out_channels=512, kernel_size=1)
        self.residual_block = Block(512, 512)

    def forward(self, x):
        _, _, h, w = img.shape
        B = x.shape[0]
        x = self.model.patch_embed(x)

        cls_tokens = self.model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)
        y = []
        for blk in self.model.blocks:
            x = blk(x)
            if self.return_intermediate_states:
                y.append(x)

        if self.return_intermediate_states:
            x = torch.cat(y, -1)
            b, _, c = x.shape
            x = torch.reshape(x[:, 1:, :], (b, h // self.patch_size, w // self.patch_size, c)).permute(0, 3, 1, 2)
        else:
            x = self.model.norm(x)
            b, _, c = x.shape
            x = torch.reshape(x[:, 1:, :], (b, h // self.patch_size, w // self.patch_size, c)).permute(0, 3, 1, 2)

        x = self.conv(x)
        x = self.residual_block(x)

        return x



if __name__ == '__main__':
    vit = VisionTransformerExtractor()
    img = torch.randn(2, 3, 512, 1024)
    print(vit)
    print(vit(img).shape)