import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer


class SparseSwin(nn.Module):
    def __init__(self, num_classes=100, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super(SparseSwin, self).__init__()
        self.swin = SwinTransformer(
            img_size=32,
            patch_size=2,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=96,
            depths=depths,
            num_heads=num_heads,
            window_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )

    def forward(self, x):
        return self.swin(x)


def swin1():
    return SparseSwin()

def swin2():
    return SparseSwin(depths=[2, 2, 4, 2],
                      num_heads=[3, 6, 12, 24])

def swin3():
    return SparseSwin(depths=[2, 2],
                      num_heads=[12, 12])

def swin4():
    return SparseSwin(depths=[2, 2],
                      num_heads=[12, 24])

def swin5():
    return SparseSwin(depths=[2, 2],
                      num_heads=[20, 100])

def swin6():
    return SparseSwin(depths=[3, 3],
                      num_heads=[12, 24])