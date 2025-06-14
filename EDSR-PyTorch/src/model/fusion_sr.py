import torch
import torch.nn as nn
import torch.nn.functional as F

from .rfdn import RFDN                     
from .transformer_sr import TransformerSR
from timm.layers import trunc_normal_

def make_model(args, parent=False):
    return FusionSR(
        rfdn_ckpt=args.rfdn_ckpt,
        transformerSR_ckpt=args.transformerSR_ckpt,
        transformerSR_args=args,           
        fusion_ch=args.fusion_ch
    )

def safe_load(path):
    obj = torch.load(path, map_location='cpu')
    for key in ('params', 'model', 'state_dict'):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
            break
    return obj



def load_rfdn(ckpt_path):
    model = RFDN()
    model.load_state_dict(safe_load(ckpt_path), strict=False)
    #model.eval()
    model.train()
    #for params in model.parameters():
    #    params.requires_grad = False

    return model


def load_transformerSR(ckpt_path, args):
    model = TransformerSR(args)
    model.load_state_dict(safe_load(ckpt_path), strict=False)
    #model.eval()
    model.train()
    #for params in model.parameters():
    #    params.requires_grad = False
            
    return model

def rfdn_forward_features(model, x):
    features = model.fea_conv(x)
    
    block1 = model.B1(features)
    block2 = model.B2(block1)
    block3 = model.B3(block2)
    block4 = model.B4(block3)
    
    out_lr = model.LR_conv(model.c(torch.cat([block1, block2, block3, block4], dim=1))) + features
    return out_lr # [B, 50, H/4, W/4]

def transformerSR_forward_features(model, x):
    return model.forward_features(x) # [B, 64, H/4, W/4]



class FusionSR(nn.Module):
    def __init__(self, rfdn_ckpt, transformerSR_ckpt, transformerSR_args, fusion_ch=50):
        super().__init__()
        
        self.cnn_model = load_rfdn(rfdn_ckpt)
        self.transformer_model = load_transformerSR(transformerSR_ckpt, transformerSR_args)
        self.fusion_layer = nn.Conv2d(50 + 64, fusion_ch, 1) # 1x1 Conv
        self.relu = nn.ReLU(inplace=True)
        self.decoder = self.cnn_model.upsampler
        
    #@torch.no_grad()
    def _extract(self, x):
        f_cnn  = rfdn_forward_features(self.cnn_model, x)      # (B,50,H,W)
        f_tr   = transformerSR_forward_features(self.transformer_model, x)  # (B,64,H/4,W/4)

        # (H/4,W/4) → (H,W)
        if f_tr.shape[-2:] != f_cnn.shape[-2:]:
            f_tr = F.interpolate(f_tr, size=f_cnn.shape[-2:], mode='bilinear', align_corners=False)

        return f_cnn, f_tr
    
    def forward(self, x):
        features_cnn, features_transformer = self._extract(x)
        features = torch.cat([features_cnn, features_transformer], dim=1)
        features = self.relu(self.fusion_layer(features))
        
        return self.decoder(features)
        
        
    


