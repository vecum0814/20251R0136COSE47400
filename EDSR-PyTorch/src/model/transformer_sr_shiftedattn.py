import torch
import torch.nn as nn
from timm.layers import trunc_normal_

def make_model(args, parent=False):
    return TransformerSR(args)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p, p).permute(0, 2, 1, 3, 4).reshape(B, -1, C * p * p)
        x = self.proj(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=16, out_chans=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(embed_dim, out_chans * patch_size * patch_size)

    def forward(self, x, h, w):
        B, L, C = x.shape
        x = self.proj(x)
        x = x.reshape(B, h, w, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, -1, h * self.patch_size, w * self.patch_size)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.window_size = window_size

    def forward(self, x, h, w, shift=False):
        B, L, C = x.shape
        x = x.view(B, h, w, C)

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0,0,0,pad_w,0,pad_h))

        Hp, Wp = x.shape[1], x.shape[2]

        # Shift window
        if shift:
            x = torch.roll(x, shifts=(-self.window_size//2, -self.window_size//2), dims=(1,2))

        x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        x = x.permute(0,1,3,2,4,5).reshape(-1, self.window_size*self.window_size, C)

        x_res = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0] + x
        x_res = self.mlp(self.norm2(x_res)) + x_res

        x_res = x_res.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        x_res = x_res.permute(0,1,3,2,4,5).reshape(B, Hp, Wp, C)

        # Reverse shift window
        if shift:
            x_res = torch.roll(x_res, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))

        if pad_h > 0 or pad_w > 0:
            x_res = x_res[:, :h, :w, :]

        return x_res.reshape(B, h*w, C)

class TransformerSR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scale = args.scale[0] if isinstance(args.scale, list) else args.scale
        self.embed_dim = 64
        self.patch_embed = PatchEmbed(patch_size=4, in_chans=3, embed_dim=self.embed_dim)

        # Stage 1: Local Transformer blocks
        self.stage1 = nn.ModuleList([
            TransformerBlock(dim=self.embed_dim, num_heads=4, window_size=8) for _ in range(4)
        ])

        # Patch merging (Stage 2)
        self.merge = nn.Linear(4 * self.embed_dim, 2 * self.embed_dim)

        # Stage 2: Global Transformer blocks
        self.stage2 = nn.ModuleList([
            TransformerBlock(dim=2 * self.embed_dim, num_heads=4, window_size=8) for _ in range(2)
        ])

        # Patch expansion
        self.expand = nn.Linear(2 * self.embed_dim, 4 * self.embed_dim)

        # Stage 3: Post-processing blocks
        self.stage3 = nn.ModuleList([
            TransformerBlock(dim=self.embed_dim, num_heads=4, window_size=8)
        ])

        # Output
        self.patch_unembed = PatchUnEmbed(patch_size=16, out_chans=3, embed_dim=self.embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # 패딩을 추가하여 H, W가 4의 배수가 되게 설정
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4

        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x_input = x.clone()  # bicubic skip 연결을 위한 복사본 유지
        _, _, H_pad, W_pad = x.shape

        h, w = H_pad // 4, W_pad // 4
        x = self.patch_embed(x)

        # Stage 1
        for blk in self.stage1:
            x = blk(x, h, w)

        # Merge tokens
        x_merge = x.view(B, h, w, -1)

        # merge 시 크기 맞추기 위한 padding
        pad_h2 = h % 2
        pad_w2 = w % 2
        if pad_h2 or pad_w2:
            x_merge = nn.functional.pad(x_merge, (0, 0, 0, pad_w2, 0, pad_h2))

        h_merge, w_merge = x_merge.shape[1] // 2, x_merge.shape[2] // 2
        x_merge = torch.cat([
            x_merge[:, 0::2, 0::2, :], x_merge[:, 1::2, 0::2, :],
            x_merge[:, 0::2, 1::2, :], x_merge[:, 1::2, 1::2, :]
        ], dim=-1).view(B, -1, 4 * self.embed_dim)
        x_merge = self.merge(x_merge)

        for blk in self.stage2:
            x_merge = blk(x_merge, h_merge, w_merge)

        # Expand tokens
        x_expand = self.expand(x_merge).view(B, h_merge, w_merge, 2, 2, self.embed_dim)
        x_expand = x_expand.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, h_merge * 2, w_merge * 2, self.embed_dim)
        x_expand = x_expand.view(B, -1, self.embed_dim)

        # Stage 3
        for blk in self.stage3:
            x_expand = blk(x_expand, h_merge * 2, w_merge * 2)

        out = self.patch_unembed(x_expand, h_merge * 2, w_merge * 2)

        # Global skip connection
        upsampled_input = nn.functional.interpolate(x_input, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # 추가된 패딩 부분 제거
        final_out = out[:, :, :H * self.scale, :W * self.scale]

        return final_out + upsampled_input[:, :, :H * self.scale, :W * self.scale]


