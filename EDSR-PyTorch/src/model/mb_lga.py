import torch
import torch.nn as nn
from timm.layers import trunc_normal_

def make_model(args, parent=False):
    return MB_LGAN(args)

class ConvBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    def forward(self, x, h, w):
        B, L, C = x.shape
        x = x.view(B, h, w, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        return x.flatten(2).permute(0, 2, 1)

class AttentionBranch(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.window_size = window_size

    def forward(self, x, h, w):
        B, L, C = x.shape
        x = x.view(B, h, w, C)
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[1], x.shape[2]
        x = x.view(B, Hp//self.window_size, self.window_size, Wp//self.window_size, self.window_size, C)
        x = x.permute(0,1,3,2,4,5).reshape(-1, self.window_size**2, C)
        x = self.attn(self.norm(x), self.norm(x), self.norm(x))[0]
        x = x.view(B, Hp//self.window_size, Wp//self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0,1,3,2,4,5).reshape(B, Hp, Wp, C)

        if pad_h or pad_w:
            x = x[:, :h, :w, :]

        return x.reshape(B, h*w, C)

class MultiBranchBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.local_branch = ConvBranch(dim)
        self.global_branch = AttentionBranch(dim, num_heads, window_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x_input = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        B, C, H, W = x.shape

        x = self.shallow_conv(x)
        h, w = H, W

        x_flat = x.flatten(2).permute(0, 2, 1)
        for blk in self.blocks:
            x_flat = blk(x_flat)  # h, w 인자를 제거하고 블록 내부에서 자동계산

        x = x_flat.permute(0, 2, 1).view(B, self.embed_dim, h, w)
        x = self.conv_after_body(x) + x
        x = self.upsample(x)

        return x + x_input

class MB_LGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scale = args.scale[0] if isinstance(args.scale, list) else args.scale
        self.embed_dim = 64
        self.shallow_conv = nn.Conv2d(3, self.embed_dim, 3, 1, 1)

        self.blocks = nn.ModuleList([
            MultiBranchBlock(self.embed_dim) for _ in range(8)
        ])

        self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim * (self.scale ** 2), 3, 1, 1),
            nn.PixelShuffle(self.scale),
            nn.Conv2d(self.embed_dim, 3, 3, 1, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_input = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        B, C, H, W = x.shape

        x = self.shallow_conv(x)
        h, w = H, W

        x_flat = x.flatten(2).permute(0, 2, 1)
        for blk in self.blocks:
            x_flat = blk(x_flat, h, w)

        x = x_flat.permute(0, 2, 1).view(B, self.embed_dim, h, w)
        x = self.conv_after_body(x) + x
        x = self.upsample(x)

        return x + x_input