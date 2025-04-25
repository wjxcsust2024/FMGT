import torch
from thop import profile, clever_format
from Net3 import Net

#flops * 2, macs直接算
net = Net()
input = torch.randn(1, 3, 512, 512)
flops, params = profile(net, (input,))
flops, params = clever_format([flops, params], "%.3f")
print('flops: ', flops, 'params: ', params)

# flops:  96.615G params:  47.709M