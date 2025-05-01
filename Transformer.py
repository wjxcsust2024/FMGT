# --------------------------------------------------------
# Slide-Transformer: Hierarchical Vision Transformer with Local Self-Attention
# Originally written by Xuran Pan
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import torch.nn.functional as F
import torch
from ecb import SeqConv3x3
import torch.nn as nn
from fusion import CGAFusion
import einops
from timm.models.layers import trunc_normal_

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        y1 = self.attention(x)
        #y2 = self.attention(x)
        out = (x * y1)
        return out

class GatedAttentionUnit(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.w1 = nn.Sequential(
            DepthWiseConv3d(dim_in=in_c, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

        self.w2 = nn.Sequential(
            DepthWiseConv3d(in_c, kernel_size=kernel_size + 2, padding=(kernel_size + 2) // 2),
            nn.GELU()
        )
        self.wo = nn.Sequential(
            DepthWiseConv3d(in_c, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU()
        )
        self.cw = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        x1, x2 = self.w1(x), self.w2(x)
        out = self.wo(x1 * x2) + self.cw(x)
        return out

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Sequential(nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels),
                                 nn.GELU())
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, kernel_size=3, padding=1, stride=1, dilation=1):
            super().__init__()

            self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                                   stride=stride, dilation=dilation, groups=dim_in)
            self.norm_layer = nn.GroupNorm(4, dim_in)
            self.conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, reduction_ratio=2, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        inner_dim = max(16, hidden_features // reduction_ratio)
        self.gau = GatedAttentionUnit(hidden_features, hidden_features, 3)
        self.conv1 = nn.Sequential(nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize // 2, stride=1,
                      groups=hidden_features), nn.GELU(), nn.BatchNorm2d(hidden_features))
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
            nn.Conv2d(hidden_features, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, hidden_features, kernel_size=1),
            nn.BatchNorm2d(hidden_features), )
        self.multi_conv1 = MultiScaleDWConv(hidden_features)
        self.conv = DepthWiseConv2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', hidden_features, hidden_features, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', hidden_features, hidden_features, -1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(self.fc1(x))
        x = self.act(x)
        x1 = self.gau(x)
        x = self.proj(x1 + self.conv1x1_sbx(x))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.reshape(B, C, -1).permute(0, 2, 1)

class Attention(nn.Module):

    def __init__(self, dim, num_heads, ka, qkv_bias=True, order=3, s=1.0, qk_scale=None, attn_drop=0.,proj_drop=0., dim_reduction=4, rpb=True, padding_mode='zeros', share_dwc_kernel=True, share_qkv=False):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if share_qkv:
            self.qkv_scale = 1
        else:
            self.qkv_scale = 3
        self.rpb = rpb
        self.share_dwc_kernel = share_dwc_kernel
        self.padding_mode = padding_mode
        self.share_qkv = share_qkv
        self.ka = ka
        self.dim_reduction = dim_reduction
        self.qkv = nn.Linear(dim, dim * self.qkv_scale//self.dim_reduction, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim//self.dim_reduction, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #self.fuse = CGAFusion(dim//self.dim_reduction)
        self.ca = ChannelAttention(dim, 10)
        self.relu = nn.ReLU()
        self.attn_dim = self.ka * self.ka * self.num_heads
        self.order = order
        self.dims = [self.attn_dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(self.attn_dim, 2 * self.attn_dim, 1)
        self.dwconv = nn.Sequential(nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=(7 - 1) // 2, bias=True,
                                    groups=sum(self.dims)), nn.ReLU(),
                                    nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=5, padding=(5 - 1) // 2, bias=True,
                                    groups=sum(self.dims)), nn.ReLU(),
                                    nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=3, padding=(3 - 1) // 2,bias=True,
                                              groups=sum(self.dims)),
                                    )
        self.proj_out = nn.Conv2d(self.attn_dim, self.attn_dim, 1)
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s

        self.dep_conv = nn.Sequential(nn.Conv2d(dim//self.dim_reduction//self.num_heads, dim//self.dim_reduction//self.num_heads, kernel_size=self.ka, bias=True, groups=dim//self.dim_reduction//self.num_heads, padding=self.ka//2, padding_mode=padding_mode),
                                      nn.Conv2d(dim // self.dim_reduction // self.num_heads, self.ka * self.ka * dim //self.dim_reduction // self.num_heads, kernel_size=3, bias=True, groups=1, padding=1, padding_mode=padding_mode))
        self.dep_conv1 = nn.Sequential(nn.Conv2d(dim//self.dim_reduction//self.num_heads, dim//self.dim_reduction//self.num_heads, kernel_size=self.ka, bias=True, groups=dim//self.dim_reduction//self.num_heads, padding=self.ka//2, padding_mode=padding_mode),
                                       nn.Conv2d(dim // self.dim_reduction // self.num_heads, self.ka * self.ka * dim //self.dim_reduction // self.num_heads, kernel_size=3, bias=True, groups=1, padding=1, padding_mode=padding_mode))
        if not share_dwc_kernel:
            self.dep_conv2 = nn.Sequential(nn.Conv2d(dim//self.dim_reduction//self.num_heads, dim//self.dim_reduction//self.num_heads, kernel_size=self.ka, bias=True, groups=dim//self.dim_reduction//self.num_heads, padding=self.ka//2, padding_mode=padding_mode),
                                       nn.Conv2d(dim // self.dim_reduction // self.num_heads, self.ka * self.ka * dim //self.dim_reduction // self.num_heads, kernel_size=5, bias=True, groups=1, padding=2, padding_mode=padding_mode))


        self.reset_parameters()

        # define a parameter table of relative position bias
        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.num_heads, 1, self.ka*self.ka, 1, 1))
            trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=3)

    def reset_parameters(self):
        # shift initialization for group convolution
        kernel = torch.zeros(self.ka*self.ka, self.ka, self.ka)
        for i in range(self.ka*self.ka):
            kernel[i, i//self.ka, i%self.ka] = 1.
        kernel = kernel.unsqueeze(1).repeat(self.dim//self.dim_reduction//self.num_heads, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)
        # x = einops.rearrange(x, 'b c h w -> b h w c')
        # B, H, W, C = x.shape
        qkv = self.qkv(x)

        f_conv = qkv.permute(0, 3, 1, 2).reshape(B * self.num_heads, self.qkv_scale * C // self.dim_reduction // self.num_heads, H, W)

        if self.qkv_scale == 3:
            q = (f_conv[:, :C // self.dim_reduction // self.num_heads, :, :] * self.scale).reshape(B, self.num_heads,
                                                                                                   C // self.dim_reduction // self.num_heads,
                                                                                                   1, H, W)
            k = f_conv[:, C // self.dim_reduction // self.num_heads:2 * C // self.dim_reduction // self.num_heads, :,
                :]  # B*self.nhead, C//self.nhead, H, W
            v = f_conv[:, 2 * C // self.dim_reduction // self.num_heads:, :, :]
        elif self.qkv_scale == 1:
            q = (f_conv * self.scale).reshape(B, self.num_heads, C // self.dim_reduction // self.num_heads, 1, H, W)
            k = v = f_conv

        if self.share_dwc_kernel:
            k = (self.dep_conv(k) + self.dep_conv1(k)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)
            v = (self.dep_conv(v) + self.dep_conv1(v)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)
        else:
            k = (self.dep_conv(k) + self.dep_conv1(k)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)
            v = (self.dep_conv(v) + self.dep_conv2(v)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)

        if self.rpb:
            k = k + self.relative_position_bias_table
        attn_1 = (q * k).sum(2, keepdim=True)  # B, self.nhead, 1, k^2, H, W
        attn_1 = self.softmax(attn_1)
        attn_1 = self.attn_drop(attn_1)
        attn = attn_1.reshape(B, -1, H, W)

        fused_x = self.proj_in(attn)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        attn = pwa * dw_list[0]

        for i in range(self.order - 1):
            attn = self.pws[i](attn) * dw_list[i + 1]

        attn = self.proj_out(attn).reshape(B, self.num_heads, -1, self.ka * self.ka, H, W)

        x = (attn * v+attn_1 * v).sum(3).reshape(B, C // self.dim_reduction, H, W).permute(0, 2, 3, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, N, C)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, num_feat, layerscale_value=2e-4, heads=8, mlp_ratio=4, drop=0., act_layer=nn.GELU):
        super(Block, self).__init__()
        self.drop_path = nn.Dropout(drop) if drop > 0. else nn.Identity()
        self.norm = LayerNorm(num_feat, eps=1e-6)
        self.attn = Attention(num_feat, heads, 3)  #之前为3，现在改成5
        self.mlp = MLP(in_features=num_feat, hidden_features=int(num_feat * mlp_ratio), act_layer=nn.Hardswish,
                       drop=drop, ksize=3)
        self.gamma = nn.Parameter(layerscale_value * torch.ones((num_feat)), requires_grad=True)


    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        res = x
        x = x + self.drop_path(self.attn(self.norm(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm(x), H, W))
        x = x + (res * self.gamma)
        return x

if __name__ == '__main__':
    x = torch.rand(2, 16384, 96).cuda()
    # y = torch.rand(1, 128, 128, 96).cuda()
    # model = CAB(96, 3, 16).cuda()
    model = Block(96, heads=8, mlp_ratio=4).cuda()
    # model = WindowAttention(96, (4, 4), 8).cuda()
    # out = model(x)
    # out = window_partition(y, 4).reshape(1024, 16, 96)
    out = model(x)
    print(out.shape)
