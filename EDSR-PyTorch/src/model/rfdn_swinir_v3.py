import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock
from model import common
import model.block as B

def make_model(args, parent=False):
    return RFDN_SwinIR_PreRFDB(args)

class RFDN_SwinIR_PreRFDB(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RFDN_SwinIR_PreRFDB, self).__init__()
        n_colors = args.n_colors
        scale = args.scale
        scale = scale[0] if isinstance(scale, (list, tuple)) else scale
        n_feats = args.n_feats if hasattr(args, 'n_feats') else 48

        # Initial feature extraction
        self.fea_conv = conv(n_colors, n_feats, kernel_size=3)

        # Swin Transformer blocks (before each RFDB)
        self.Swin1 = SwinTransformerBlock(n_feats, (24,24), 6, 8, shift_size=0, mlp_ratio=4.0)
        self.Swin2 = SwinTransformerBlock(n_feats, (24,24), 6, 8, shift_size=0, mlp_ratio=4.0)
        self.Swin3 = SwinTransformerBlock(n_feats, (24,24), 6, 8, shift_size=0, mlp_ratio=4.0)
        self.Swin4 = SwinTransformerBlock(n_feats, (24,24), 6, 8, shift_size=0, mlp_ratio=4.0)

        # RFDB blocks
        self.RFDB1 = B.RFDB(in_channels=n_feats)
        self.RFDB2 = B.RFDB(in_channels=n_feats)
        self.RFDB3 = B.RFDB(in_channels=n_feats)
        self.RFDB4 = B.RFDB(in_channels=n_feats)

        # Fusion layer (1x1 conv)
        self.fusion_conv = conv(n_feats * 4, n_feats, kernel_size=1)

        # Global Residual Convolution
        self.global_res_conv = conv(n_feats, n_feats, kernel_size=3)

        # Upsampler & Output conv
        self.upsampler = common.Upsampler(conv, scale, n_feats, act=False)
        self.conv_last = conv(n_feats, n_colors, kernel_size=3)

    def forward(self, x):
        fea = self.fea_conv(x)

        # Swin → RFDB sequentially
        out1_s = self.apply_swin(fea, self.Swin1)
        out1 = self.RFDB1(out1_s)

        out2_s = self.apply_swin(out1, self.Swin2)
        out2 = self.RFDB2(out2_s)

        out3_s = self.apply_swin(out2, self.Swin3)
        out3 = self.RFDB3(out3_s)

        out4_s = self.apply_swin(out3, self.Swin4)
        out4 = self.RFDB4(out4_s)

        # Concatenate RFDB outputs
        out_cat = torch.cat([out1, out2, out3, out4], dim=1)
        out_fused = self.fusion_conv(out_cat)

        # Global residual connection
        out_res = self.global_res_conv(out_fused) + fea

        # Upsampling & final conv
        out_hr = self.upsampler(out_res)
        out = self.conv_last(out_hr)

        return out

    def apply_swin(self, x, swin_block):
        B, C, H, W = x.shape
        x_permuted = x.permute(0, 2, 3, 1)  # (B,H,W,C)
        x_swin = swin_block(x_permuted)
        x_swin = x_swin.permute(0, 3, 1, 2)
        return x_swin + x  # Residual connection
