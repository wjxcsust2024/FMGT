from norms import *
import copy
from deform_conv_v3 import *
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from common import default_conv, ResBlock
from Transformer import Block
from wavelet_module import Wavelet_block as WMamba


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Embed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        # _, _, H, W = x.shape
        if self.norm is not None:
            x = self.norm(x)
        return x


class Merge(nn.Module):   #缩小
    def __init__(self, dim, h, w):
        super(Merge, self).__init__()
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=2, stride=2, padding=0)
        self.h = h
        self.dim = dim
        self.w = w
        self.norm = nn.BatchNorm2d(dim*2)
        self.c_1 = nn.Sequential(nn.Conv2d(dim*2, dim*2, 3, 1, 1),
                                 nn.ReLU())      #3x3

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)
        x = self.norm(self.conv(x))
        x = self.c_1(x)
        return x.reshape(B, self.dim*2, -1).permute(0, 2, 1)

class Expand(nn.Module):  #放大
    def __init__(self, dim, h):
        super(Expand, self).__init__()
        self.dim = dim
        self.h = h
        self.conv = nn.ConvTranspose2d(self.dim, self.dim//2, 2, stride=2)
        self.c_1 = nn.Sequential(nn.Conv2d(dim // 2, dim // 2, 3, 1, 1),
                                 nn.ReLU())  #3x3

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.h, self.h)
        x = self.conv(x)
        x = self.c_1(x)
        return x.reshape(B, self.dim//2, -1).permute(0, 2, 1)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = Embed(512)

        self.l1 = nn.Sequential(Block(96, heads=8, mlp_ratio=4),
                                Block(96, heads=8, mlp_ratio=4))

        self.l2 = nn.Sequential(Block(192, heads=8, mlp_ratio=4),
                                Block(192, heads=8, mlp_ratio=4))

        self.l3 = nn.Sequential(Block(384, heads=8, mlp_ratio=4),
                                Block(384, heads=8, mlp_ratio=4))

        self.l4 = nn.Sequential(Block(768, heads=8, mlp_ratio=4),
                                Block(768, heads=8, mlp_ratio=4),
                                Block(768, heads=8, mlp_ratio=4),
                                Block(768, heads=8, mlp_ratio=4))

        self.m1 = Merge(96, 128, 128)
        self.m2 = Merge(192, 64, 64)
        self.m3 = Merge(384, 32, 32)

        self.p3 = Expand(768, 16)
        self.p2 = Expand(384, 32)
        self.p1 = Expand(192, 64)

        self.d3 = nn.Sequential(Block(384, heads=8, mlp_ratio=4),
                                Block(384, heads=8, mlp_ratio=4))

        self.d2 = nn.Sequential(Block(192, heads=8, mlp_ratio=4),
                                Block(192, heads=8, mlp_ratio=4))

        self.d1 = nn.Sequential(Block(96, heads=8, mlp_ratio=4),
                                Block(96, heads=8, mlp_ratio=4))

        self.dbm3 = WMamba(384)
        self.dbm2 = WMamba(192)
        self.dbm1 = WMamba(96)

        self.up = nn.PixelShuffle(4)
        self.seg = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.embed(x)  # torch.Size([1, 16384, 96])
        x1 = self.l1(x)  # torch.Size([1, 16384, 96])
        #
        x = self.m1(x1)  # torch.Size([1, 4096, 192])
        x2 = self.l2(x)  # torch.Size([1, 4096, 192])
        #
        x = self.m2(x2)  # torch.Size([1, 1024, 384])
        x3 = self.l3(x)  # torch.Size([1, 1024, 384])
        #
        x = self.m3(x3)  # torch.Size([1, 256, 768])
        x4 = self.l4(x)  # torch.Size([1, 256, 768])
        #
        x = self.p3(x4)  # torch.Size([1, 1024, 384])
        x3_temp = self.dbm3(x3+x)
        x = self.d3(x3_temp)  # torch.Size([1, 1024, 384])
        #
        x = self.p2(x)  # torch.Size([1, 4096, 192])
        x2_temp = self.dbm2(x+x2)
        x = self.d2(x2_temp)
        #
        x = self.p1(x)  # torch.Size([1, 16384, 96])
        x1_temp = self.dbm1(x+x1)
        x = self.d1(x1_temp)  # 128x128
        x = self.up(x.permute(0, 2, 1).reshape(B, 96, 128, 128))  # torch.Size([1, 6, 512, 512])
        x = self.seg(x)
        return x


if __name__ == '__main__':
    x = torch.rand(2, 3, 512, 512).cuda()
    # y = torch.rand(1, 1024, 384).cuda()
    # dbm = DeBlock(384).cuda()
    part = Net().cuda()
    out = part(x)
    print(out.shape)
    # out = dbm(y)
    # print(out.shape)