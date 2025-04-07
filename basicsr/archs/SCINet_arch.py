'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''
import cv2
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn import Parameter
import numpy as np
from basicsr.archs.dbb_transforms import *
class RDG_cdc(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG_cdc, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        "循环4个CD-RDB"
        for i in range(n_RDB):
            # if i ==0:
            #     RDBs.append(RDB_cdc(G0, C, G))
            # else:
            #     RDBs.append(RDB(G0, C, G))
            "RDB_cdc: Central Difference Residual Dense Block(CD-RDB)"
            "RDB_cdc(64, 6, 32)"
            "每一个CD-RDB由一个CD-Conv和五个普通Conv组成"
            RDBs.append(RDB_cdc(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        "Conv-1"
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(G0 * 2, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x, immediate_fea):
        buffer = x
        temp = []
        immediate_fea = immediate_fea
        for i in range(self.n_RDB):
            # if i == 1:
            #     buffer = self.RDB[i](buffer)
            #     buffer = torch.cat([buffer, immediate_fea[0]], dim=1)
            #     buffer = self.conv2(buffer)
            #     temp.append(buffer)
            # elif i == 3:
            #     buffer = self.RDB[i](buffer)
            #     buffer = torch.cat([buffer, immediate_fea[1]], dim=1)
            #     buffer = self.conv2(buffer)
            #     temp.append(buffer)
            # elif i == 5:
            #     buffer = self.RDB[i](buffer)
            #     buffer = torch.cat([buffer, immediate_fea[2]], dim=1)
            #     buffer = self.conv2(buffer)
            #     temp.append(buffer)
            # else:
            #     buffer = self.RDB[i](buffer)
            #     temp.append(buffer)
            if i == 1:
                buffer = self.RDB[i](buffer)
                buffer = torch.cat([buffer, immediate_fea[0]], dim=1)
                buffer = self.conv2(buffer)
                temp.append(buffer)
            elif i == 3:
                buffer = self.RDB[i](buffer)
                buffer = torch.cat([buffer, immediate_fea[1]], dim=1)
                buffer = self.conv2(buffer)
                temp.append(buffer)
            elif i == 5:
                buffer = self.RDB[i](buffer)
                buffer = torch.cat([buffer, immediate_fea[2]], dim=1)
                buffer = self.conv2(buffer)
                temp.append(buffer)
            else:
                buffer = self.RDB[i](buffer)
                temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out
"间隔使用"
# class RDB_cdc(nn.Module):
#     "RDB_cdc(64, 6, 32)"
#     def __init__(self, G0, C, G):
#         super(RDB_cdc, self).__init__()
#         convs = []
#         "一共六个卷积，第一个是CD-Conv,剩下是普通的Conv"
#         for i in range(C):
#             if i == 0 or i == 2 or i == 4:
#                 "OneConv_cdc: Central Difference Convolution (CD-Conv) "
#                 convs.append(OneConv_Hori_Veri_Cross(G0+i*G, G))
#             else:
#                 convs.append(OneConv_Diag_Cross(G0+i*G, G))
#         self.conv = nn.Sequential(*convs)
#         self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
#
#     def forward(self, x):
#         out = self.conv(x)
#         lff = self.LFF(out)
#         return lff + x

class RDB_cdc(nn.Module):
    "RDB_cdc(64, 6, 32)"
    def __init__(self, G0, C, G):
        super(RDB_cdc, self).__init__()
        convs = []
        "一共六个卷积，第一个是CD-Conv,剩下是普通的Conv"
        for i in range(C):
            if i == 0:
                "OneConv_cdc: Central Difference Convolution (CD-Conv) "
                convs.append(define_bolck(G0 + i * G, G))
            else:
                convs.append(DBB_bolck(G0 + i * G, G))
                # convs.append(OneConv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x

class define_bolck(nn.Module):
    def __init__(self, G0, G):
        super(define_bolck, self).__init__()
        self.conv_hori_veri = OneConv_Hori_Veri_Cross(G0, G)
        self.conv_Diag = OneConv_Diag_Cross(G, G)
        # self.changechannel = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1)
        self.changechannel = nn.Conv2d(G0, G, kernel_size=1, stride=1, padding=0)
    def forward(self, input):
        hori_veri = self.conv_hori_veri(input)
        output = self.conv_Diag(hori_veri)
        input_residual = self.changechannel(input)
        output = input_residual + output
        temp = torch.cat((input, output), dim=1)
        return temp

class DBB_bolck(nn.Module):
    def __init__(self, G0, G):
        super(DBB_bolck, self).__init__()
        self.some_dbb = DiverseBranchBlock(in_channels=G0, out_channels=G, kernel_size=3, stride=1,
                    padding=1, dilation=1, groups=1,  deploy = DEPLOY_FLAG, nonlinear=nn.ReLU())
    def forward(self, input):
        out = self.some_dbb(input)
        temp = torch.cat((input, out), dim=1)
        return temp


class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.8):

        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        "tensor_zeros: torch.Size([32, 64, 1])"
        "self.conv.weight[:, :, :, 2]: torch.Size([32, 64, 1])"
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()

        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                 self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4], tensor_zeros),2)
        "conv_weight: torch.Size([32, 64, 9])"
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        "conv_weight: torch.Size([32, 64, 3, 3])"
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.8):
        super(Conv2d_Diag_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        "x: torch.Size([4, 32, 32, 32])"
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1], tensor_zeros,
                                 self.conv.weight[:, :, :, 2], tensor_zeros, self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        "conv_weight: torch.Size([32, 96, 3, 3])"
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

        return out_normal - self.theta * out_diff

class OneConv_Hori_Veri_Cross(nn.Module):
    def __init__(self, G0, G):
        super(OneConv_Hori_Veri_Cross, self).__init__()
        "Conv2d_cd:Central difference convolution (CD-Conv)"
        self.conv = Conv2d_Hori_Veri_Cross(G0, G, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return output

class OneConv_Diag_Cross(nn.Module):
    def __init__(self, G0, G):
        super(OneConv_Diag_Cross, self).__init__()
        "Conv2d_cd:Central difference convolution (CD-Conv)"
        self.conv = Conv2d_Diag_Cross(G0, G, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return output
"间隔使用"
# class OneConv_Hori_Veri_Cross(nn.Module):
#     def __init__(self, G0, G):
#         super(OneConv_Hori_Veri_Cross, self).__init__()
#         "Conv2d_cd:Central difference convolution (CD-Conv)"
#         self.conv = Conv2d_Hori_Veri_Cross(G0, G, stride=1, padding=1, bias=True)
#         self.relu = nn.LeakyReLU(0.1, inplace=True)
#     def forward(self, x):
#         output = self.relu(self.conv(x))
#         return torch.cat((x, output), dim=1)

# class OneConv_Diag_Cross(nn.Module):
#     def __init__(self, G0, G):
#         super(OneConv_Diag_Cross, self).__init__()
#         "Conv2d_cd:Central difference convolution (CD-Conv)"
#         self.conv = Conv2d_Diag_Cross(G0, G, stride=1, padding=1, bias=True)
#         self.relu = nn.LeakyReLU(0.1, inplace=True)
#     def forward(self, x):
#         output = self.relu(self.conv(x))
#         return torch.cat((x, output), dim=1)
#
class OneConv_cdc(nn.Module):
    def __init__(self, G0, G):
        super(OneConv_cdc, self).__init__()
        "Conv2d_cd:Central difference convolution (CD-Conv)"
        self.conv = Conv2d_cd(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)

class OneConv(nn.Module):
    def __init__(self, G0, G):
        super(OneConv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                   padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se

class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)

class DiverseBranchBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False):
        super(DiverseBranchBlock, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:
            "Conv-3和BN"
            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

            "Conv-1和AVG"
            "---no pooling---"
            # self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                "Conv-1"
                # self.dbb_avg.add_module('conv',
                #                         nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                #                                   stride=1, padding=0, groups=groups, bias=False))
                # "BN"
                # self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                # "AVG"
                # self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                "Conv-1和BN"
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            # else:
                # self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            "AVGBN"
            # self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))


            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels   # For mobilenet, it is better to have 2X internal channels
            "Conv-1,BN,Conv-3,BN"
            self.dbb_1x1_kxk = nn.Sequential()
            "Conv-1"
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                          kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            "BN"
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            "Conv-3"
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                            kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            "BN"
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        "Conv-3和BN的融合"
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)
        "k_origin:torch.Size([64, 64, 3, 3])"

        "Conv-1和BN的融合"
        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
            "k_1x1:torch.Size([64, 64, 3, 3])"
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight

        "融合Conv1-BN-Conv3-BN"
        "融合Conv1-BN to Conv1"
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        "融合Conv3-BN to Conv3"
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        "融合Conv1-Conv3 to Conv3"
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)
        "---pooling---"
        # "融合Conv1-BN-AVG-BN"
        # k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        # "融合AVG-BN to Conv-3"
        # k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
        #                                                    self.dbb_avg.avgbn)
        # if hasattr(self.dbb_avg, 'conv'):
        #     '融合Conv1-BN to Conv1'
        #     k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
        #     "融合Conv1-Conv3 to Conv3"
        #     k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
        #                                                           b_1x1_avg_second, groups=self.groups)
        # else:
        #     k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        # return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
        #                          (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))
        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged))
    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels,
                                     out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        "---no pooling---"
        # self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        # out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            "self.conv.weight: torch.Size([32, 64, 3, 3])"
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            "kernel_diff1: torch.Size([32, 64])"
            kernel_diff1 = self.conv.weight.sum(2).sum(2)
            "kernel_diff2: torch.Size([32, 64, 1, 1])"
            kernel_diff2 = kernel_diff1[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff2, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=bias,
                padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class BSConvS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                 padding_mode="zeros", p=0.25, min_mid_channels=4, with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        assert 0.0 <= p <= 1.0
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise 1
        self.pw1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # pointwise 2
        self.add_module("pw2", torch.nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        fea = self.pw1(x)
        fea = self.pw2(fea)
        fea = self.dw(fea)
        return fea

    def _reg_loss(self):
        W = self[0].weight[:, :, 0, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p="fro")


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)

        return input * m


class ESDB(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=0.25):
        super(ESDB, self).__init__()
        kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}

        self.dc = self.distilled_channels = in_channels // 2
        "最后残差连接用"
        self.rc = self.remaining_channels = in_channels

        "1 Conv-1:(64,32)"
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        "1 BSRB:(64,64)"
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)

        "2 Conv-1:(64,32)"
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        "2 BSRB:(64,64)"
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        "3 Conv-1:(64,32)"
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        "3 BSRB:(64,64)"
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        "BSConv:(64,32)"
        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        "Conv-1:(128,64)"
        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)

        self.esa = ESA(in_channels, conv)
        self.cca = CCALayer(in_channels)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)
        out_fused = self.cca(out_fused)
        return out_fused + input


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x

class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x

@ARCH_REGISTRY.register()
class SCINet(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=4,
                 conv='BSConvU', upsampler='pixelshuffledirect', p=0.25):
        super(SCINet, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'BSConvS':
            kwargs = {'p': p}
        print(conv)
        if conv == 'DepthWiseConv':
            self.conv = DepthWiseConv
        elif conv == 'BSConvU':
            self.conv = BSConvU
        elif conv == 'BSConvS':
            self.conv = BSConvS
        else:
            self.conv = nn.Conv2d
        self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)

        self.B1 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B2 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B3 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B4 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B5 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B6 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B7 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B8 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.deep_conv = RDG_cdc(G0=64, C=6, G=32, n_RDB=6)

        self.f_block = RRDB(num_feat * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                            norm_type=None, act_type='leakyrelu', mode='CNA')
        self.f_concat = conv_block(num_feat * 2, num_feat, kernel_size=3, norm_type=None, act_type=None)

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)


        self.HP_branch1 = Parameter(torch.zeros([1, 1]))
        self.HP_branch2 = Parameter(torch.zeros([1, 1]))

        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        "SR branch"
        "通道维度对输入进行拼接"
        input = torch.cat([input, input, input, input], dim=1)
        "shallow feature extraction"
        out_fea = self.fea_conv(input)

        "8个ESDB"
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)
        immediate_fea= []
        immediate_fea.append(out_B2)
        immediate_fea.append(out_B4)
        immediate_fea.append(out_B6)
        "特征拼接"
        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        "Conv-1"
        out_RB = self.c1(trunk)

        if out_RB.shape[1] == 64:
            for i in range(64):
                huatu = out_RB[:, i, :, :].squeeze().cpu().numpy()
                huatu = huatu[:128, :160]
                huatu = cv2.resize(huatu, (640, 512))
                plt.imsave(
                    f'/data/dl/SCINet-main/duibitu_qian/{i}.bmp',
                    huatu, cmap='jet')


        "Contrast branch"
        out_GB = self.deep_conv(out_fea, immediate_fea)
        out_GB =  out_GB + out_fea

        if out_GB.shape[1] == 64:
            for i in range(64):
                huatu2 = out_GB[:, i, :, :].squeeze().cpu().numpy()
                huatu2 = huatu2[:128, :160]
                huatu2 = cv2.resize(huatu2, (640, 512))
                plt.imsave(
                    f'/data/dl/SCINet-main/duibitu_qian2/{i}.bmp',
                    huatu2, cmap='jet')


        "[4,128,128,160]"
        x_f_cat = torch.cat([out_RB, out_GB], dim=1)
        if x_f_cat.shape[1] == 128:
            for i in range(128):
                huatu3 = x_f_cat[:, i, :, :].squeeze().cpu().numpy()
                huatu3 = huatu3[:128, :160]
                huatu3 = cv2.resize(huatu3, (640, 512))
                plt.imsave(
                    f'/data/dl/SCINet-main/duibitu_qian3/{i}.bmp',
                    huatu3, cmap='jet')

        "[4,128,128,160]"
        x_f_cat = self.f_block(x_f_cat)
        "[4,64,32,32]"
        x_out = self.f_concat(x_f_cat)

        x_out = self.GELU(x_out)
        if x_out.shape[1] == 64:
            for i in range(64):
                huatu4 = out_GB[:, i, :, :].squeeze().cpu().numpy()
                huatu4 = huatu4[:128, :160]
                huatu4 = cv2.resize(huatu4, (640, 512))
                plt.imsave(
                    f'/data/dl/SCINet-main/duibitu_qian4/{i}.bmp',
                    huatu4, cmap='jet')

        "BSConv,residual connection"
        out_lr = self.c2(x_out) + out_fea

        "上采样"
        output = self.upsampler(out_lr)

        return output

def switch_deploy_flag(deploy):
    global DEPLOY_FLAG
    DEPLOY_FLAG = deploy
    print('deploy flag: ', DEPLOY_FLAG)