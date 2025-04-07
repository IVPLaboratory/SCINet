import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn import Parameter
import numpy as np
import torch.nn as nn
from PIL import Image

ori_3_3 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride = 1, padding = 1, bias=True)
ori_1_1 = nn.Conv2d(in_channels=10, out_channels=6, kernel_size= 1, stride=1, padding=0, bias=True)

weight_3_3_ = ori_3_3.weight.data.clone()
weight_1_1_ = ori_1_1.weight.data.clone()

bias_3_3_ = ori_3_3.bias.data.clone()
bias_1_1 = ori_1_1.bias.data.clone()


reweight_3_3 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=True)
print(reweight_3_3.weight.data.shape)

for i in range(weight_1_1_.shape[0]):

    v = weight_1_1_[i, ...].data.shape
    t = weight_1_1_[i, ...].unsqueeze(1).data.shape
    reweight_3_3.weight.data[i,...] = torch.sum(weight_3_3_ * weight_1_1_[i,...].unsqueeze(1), dim=0)
    p = reweight_3_3.weight.data[i, ...].data.shape

    reweight_3_3.bias.data[i] = bias_1_1[i] + \
                                    torch.sum(bias_3_3_ * weight_1_1_[i,...].squeeze(1).squeeze(1))

model = nn.Sequential(ori_3_3, ori_1_1)

x = torch.tensor(np.array(Image.open('cat.jpg'))).unsqueeze(0).permute(0,3,1,2).float()
out = model(x)
out2 = reweight_3_3(x)

print(torch.sum(torch.abs(out - out2))) # tensor(5.4640, grad_fn=<SumBackward0>)