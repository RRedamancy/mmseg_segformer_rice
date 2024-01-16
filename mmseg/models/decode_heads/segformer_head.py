# Modified from
# https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
#
# This work is licensed under the NVIDIA Source Code License.
#
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator
# Augmentation (ADA)
#
#  1. Definitions
#  "Licensor" means any person or entity that distributes its Work.
#  "Software" means the original work of authorship made available under
# this License.
#  "Work" means the Software and any additions to or derivative works of
# the Software that are made available under this License.
#  The terms "reproduce," "reproduction," "derivative works," and
# "distribution" have the meaning as provided under U.S. copyright law;
# provided, however, that for the purposes of this License, derivative
# works shall not include works that remain separable from, or merely
# link (or bind by name) to the interfaces of, the Work.
#  Works, including the Software, are "made available" under this License
# by including in or with the Work either (a) a copyright notice
# referencing the applicability of this License to the Work, or (b) a
# copy of this License.
#  2. License Grants
#      2.1 Copyright Grant. Subject to the terms and conditions of this
#     License, each Licensor grants to you a perpetual, worldwide,
#     non-exclusive, royalty-free, copyright license to reproduce,
#     prepare derivative works of, publicly display, publicly perform,
#     sublicense and distribute its Work and any resulting derivative
#     works in any form.
#  3. Limitations
#      3.1 Redistribution. You may reproduce or distribute the Work only
#     if (a) you do so under this License, (b) you include a complete
#     copy of this License with your distribution, and (c) you retain
#     without modification any copyright, patent, trademark, or
#     attribution notices that are present in the Work.
#      3.2 Derivative Works. You may specify that additional or different
#     terms apply to the use, reproduction, and distribution of your
#     derivative works of the Work ("Your Terms") only if (a) Your Terms
#     provide that the use limitation in Section 3.3 applies to your
#     derivative works, and (b) you identify the specific derivative
#     works that are subject to Your Terms. Notwithstanding Your Terms,
#     this License (including the redistribution requirements in Section
#     3.1) will continue to apply to the Work itself.
#      3.3 Use Limitation. The Work and any derivative works thereof only
#     may be used or intended for use non-commercially. Notwithstanding
#     the foregoing, NVIDIA and its affiliates may use the Work and any
#     derivative works commercially. As used herein, "non-commercially"
#     means for research or evaluation purposes only.
#      3.4 Patent Claims. If you bring or threaten to bring a patent claim
#     against any Licensor (including any claim, cross-claim or
#     counterclaim in a lawsuit) to enforce any patents that you allege
#     are infringed by any Work, then your rights under this License from
#     such Licensor (including the grant in Section 2.1) will terminate
#     immediately.
#      3.5 Trademarks. This License does not grant any rights to use any
#     Licensor’s or its affiliates’ names, logos, or trademarks, except
#     as necessary to reproduce the notices described in this License.
#      3.6 Termination. If you violate any term of this License, then your
#     rights under this License (including the grant in Section 2.1) will
#     terminate immediately.
#  4. Disclaimer of Warranty.
#  THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
# NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER
# THIS LICENSE.
#  5. Limitation of Liability.
#  EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL
# THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE
# SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
# OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
# (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION,
# LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER
# COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGES.

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

# fuhongtao
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from torch.nn.modules.utils import _pair
import math
from mmcv.ops.modulated_deform_conv import modulated_deform_conv2d
# from .dydeform_cuda import DyDeform

# fuhongtao
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.ops.carafe import carafe
from mmcv.cnn import normal_init

class GGSUp(nn.Module):
    def __init__(self):
        super(GGSUp, self).__init__()
        self.kernel_size = 3
        center = torch.from_numpy(np.float32(
                np.array([-1 / 4, 1 / 4, -1 / 4, 1 / 4, -1 / 4, -1 / 4, 1 / 4, 1 / 4]))).view(1, 8, 1, 1)
        self.register_buffer('center', center)
        neighbor = torch.from_numpy(np.float32(
                np.array([-1., 0., 1., -1., 0., 1., -1., 0., 1., -1., -1., -1., 0., 0., 0., 1., 1., 1.]))
            ).view(1, 18, 1, 1)
        self.register_buffer('neighbor', neighbor)
        self.offset = nn.Conv2d(1, 2, kernel_size=5, padding=2)
        normal_init(self.offset, std=0.001)

    def get_kernel(self, offset):
        B, C, H, W = offset.shape
        center = F.pixel_shuffle(F.interpolate(self.center, size=[H // 2, W // 2]), upscale_factor=2)
        shift = center + offset
        neighbor = F.interpolate(self.neighbor, size=[H, W]).view(1, 2, 9, H, W)
        # kernels = 1 / torch.sqrt(torch.sum((shift.unsqueeze(2) - neighbor) ** 2, dim=1))
        kernels = 1 / (torch.sum((shift.unsqueeze(2) - neighbor) ** 2, dim=1) + 0.2)
        return kernels

    def forward(self, x):
        B, C, H, W = x.shape
        mean = torch.mean(x, dim=1, keepdim=True)
        offset = torch.tanh(self.offset(F.interpolate(mean, scale_factor=2))) / 4
        kernels = self.get_kernel(offset)
        grad = F.unfold(mean, kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2).view(B, self.kernel_size ** 2, H, W) - mean
        grad = 1 / (grad ** 2 + 1)
        kernels = F.softmax(F.interpolate(grad, scale_factor=2) * kernels, dim=1)
        return carafe(x, kernels, self.kernel_size, 1, 2)

class GGUp(nn.Module):
    def __init__(self, kernel_size=3):
        super(GGUp, self).__init__()
        self.kernel_size = kernel_size
        kernel = self.get_kernel()
        self.register_buffer('kernel', kernel)

    def get_kernel(self):
        coord = [[self.kernel_size // 2 - 1 / 4, self.kernel_size // 2 - 1 / 4],
                 [self.kernel_size // 2 + 1 / 4, self.kernel_size // 2 - 1 / 4],
                 [self.kernel_size // 2 - 1 / 4, self.kernel_size // 2 + 1 / 4],
                 [self.kernel_size // 2 + 1 / 4, self.kernel_size // 2 + 1 / 4]]
        kernel = torch.zeros(4, self.kernel_size ** 2, requires_grad=False)
        for n in range(4):
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    kernel[n, self.kernel_size * i + j] = 1 / math.sqrt((coord[n][0] - j) ** 2 + (coord[n][1] - i) ** 2)
        return kernel.transpose(0, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        mean = torch.mean(x, dim=1, keepdim=True)
        grad = F.unfold(mean, kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2).view(B, self.kernel_size ** 2, H, W) - mean
        grad = 1 / (grad ** 2 + 1).unsqueeze(2)
        kernels = self.kernel.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        kernels = F.softmax(F.pixel_shuffle(grad * kernels, upscale_factor=2).squeeze(2), dim=1)
        return carafe(x, kernels, self.kernel_size, 1, 2)

class GG(nn.Module):
    def __init__(self, scale=2, kernel_size=3):
        super().__init__()
        assert isinstance(scale, int) and scale >= 2, \
            'scale must be integers and greater than 2'
        assert isinstance(kernel_size, int) and kernel_size >= 3 and kernel_size % 2 == 1, \
            'kernel size must be odd integers and greater than 3'
        self.scale = scale
        self.kernel_size = kernel_size
        kernel = self.get_kernel()
        self.register_buffer('kernel', kernel)

    def get_kernel(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        center = torch.stack(torch.meshgrid([h, h])).view(2, 1, self.scale ** 2)
        h = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        neighbor = torch.stack(torch.meshgrid([h, h])).view(2, self.kernel_size ** 2, 1)
        # kernel = 1 / torch.sqrt(torch.sum((center - neighbor) ** 2, dim=0))   # the old version of GG,
        # does not support for 3X upscaling.
        kernel = 1 / (torch.sum((center - neighbor) ** 2, dim=0) + 0.5)
        return kernel

    def forward(self, x):
        B, C, H, W = x.shape
        mean = torch.mean(x, dim=1, keepdim=True)
        grad = F.unfold(mean, kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2).view(B, self.kernel_size ** 2, H, W) - mean
        grad = 1 / (grad ** 2 + 1).unsqueeze(2)
        kernels = self.kernel.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        kernels = F.softmax(F.pixel_shuffle(grad * kernels, upscale_factor=self.scale).squeeze(2), dim=1)
        return carafe(x, kernels, self.kernel_size, 1, self.scale)

class GGS(nn.Module):
    def __init__(self, scale=2, kernel_size=3):
        super().__init__()
        assert isinstance(scale, int) and scale >= 2, \
            'scale must be integers and greater than 2'
        assert isinstance(kernel_size, int) and kernel_size >= 3 and kernel_size % 2 == 1, \
            'kernel size must be odd integers and greater than 3'
        self.scale = scale
        self.kernel_size = kernel_size
        h = torch.arange((-scale + 1) / 2, (scale - 1) / 2 + 1) / scale
        center = torch.stack(torch.meshgrid([h, h])).view(1, 2 * scale ** 2, 1, 1)
        self.register_buffer('center', center)
        h = torch.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
        neighbor = torch.stack(torch.meshgrid([h, h])).view(1, 2 * kernel_size ** 2, 1, 1)
        self.register_buffer('neighbor', neighbor)
        self.offset = nn.Conv2d(1, 2, kernel_size=2 * scale + 1, padding=scale)
        normal_init(self.offset, std=0.001)

    def get_kernel(self, offset):
        B, C, H, W = offset.shape
        center = F.pixel_shuffle(F.interpolate(
            self.center, size=[H // self.scale, W // self.scale]), upscale_factor=self.scale)
        shift = center + offset
        neighbor = F.interpolate(self.neighbor, size=[H, W]).view(1, 2, self.kernel_size ** 2, H, W)
        kernels = 1 / (torch.sum((shift.unsqueeze(2) - neighbor) ** 2, dim=1) + 0.5)   # 20240115  lr=0.00001,  0.2-->0.5  0.5 hao
        return kernels

    def forward(self, x):
        B, C, H, W = x.shape
        mean = torch.mean(x, dim=1, keepdim=True)
        offset = torch.tanh(self.offset(F.interpolate(mean, scale_factor=self.scale))) / (2 * self.scale)
        kernels = self.get_kernel(offset)
        grad = F.unfold(mean, kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2).view(B, self.kernel_size ** 2, H, W) - mean
        grad = 1 / (grad ** 2 + 1)
        kernels = F.softmax(F.interpolate(grad, scale_factor=self.scale) * kernels, dim=1)
        return carafe(x, kernels, self.kernel_size, 1, self.scale)

class DCNv2(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
            deformable_groups=1,
    ):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == mask.shape[1]
        return modulated_deform_conv2d(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            1,
            self.deformable_groups,
        )


class DCN(DCNv2):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1,
                 extra_offset_mask=False, ):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)

        self.extra_offset_mask = extra_offset_mask
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride,
                                          padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        if self.extra_offset_mask:
            out = self.conv_offset_mask(input[1])
            input = input[0]
        else:
            out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        # mask = torch.sigmoid(mask)
        mask = F.hardsigmoid(mask)
        return modulated_deform_conv2d(input, offset, mask, self.weight, self.bias, self.stride,
                                       self.padding, self.dilation, 1, self.deformable_groups, )


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128):
        super(FeatureAlign_V2, self).__init__()
        self.operator = DCN(in_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,)
        # self.operator = DyDeform(in_nc, scale=1, num_point=9, groups=8, kernel_grid=True)

    def forward(self, feat_l, feat_s):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            # feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
            feat_up = F.interpolate(feat_s, HW, mode='nearest')
        else:
            feat_up = feat_s
        feat_align = self.operator(feat_up)
        # return feat_align + feat_l
        # return feat_align * 0.3
        return feat_align




# @HEADS.register_module()
# class SegformerHead(BaseDecodeHead):
#     """The all mlp Head of segformer.
#
#     This head is the implementation of
#     `Segformer <https://arxiv.org/abs/2105.15203>` _.
#
#     Args:
#         interpolate_mode: The interpolate mode of MLP head upsample operation.
#             Default: 'bilinear'.
#     """
#
#     def __init__(self, interpolate_mode='bilinear', **kwargs):
#         super().__init__(input_transform='multiple_select', **kwargs)
#
#         self.interpolate_mode = interpolate_mode
#         num_inputs = len(self.in_channels)
#
#         assert num_inputs == len(self.in_index)
#
#         self.convs = nn.ModuleList()
#         for i in range(num_inputs):
#             self.convs.append(
#                 ConvModule(
#                     in_channels=self.in_channels[i],
#                     out_channels=self.channels,
#                     kernel_size=1,
#                     stride=1,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg))
#
#         self.fusion_conv = ConvModule(
#             in_channels=self.channels * num_inputs,
#             out_channels=self.channels,
#             kernel_size=1,
#             norm_cfg=self.norm_cfg)
#         # self.upfusion = []
#         # self.upfusion.append(FeatureAlign_V2(in_nc=256, out_nc=256))
#         # self.upfusion.append(FeatureAlign_V2(in_nc=256, out_nc=256))
#         # self.upfusion.append(FeatureAlign_V2(in_nc=256, out_nc=256))
#         # self.upfusion.append(FeatureAlign_V2(in_nc=256, out_nc=256))
#         self.operator = nn.ModuleList()
#         self.operator.append(DCN(256, 256, 3, stride=1, padding=1, dilation=1, deformable_groups=8, ))
#         self.operator.append(DCN(256, 256, 3, stride=1, padding=1, dilation=1, deformable_groups=8, ))
#         self.operator.append(DCN(256, 256, 3, stride=1, padding=1, dilation=1, deformable_groups=8, ))
#         self.operator.append(DCN(256, 256, 3, stride=1, padding=1, dilation=1, deformable_groups=8, ))
#
#     def forward(self, inputs):
#         # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
#         inputs = self._transform_inputs(inputs)
#         outs = []
#         for idx in range(len(inputs)):
#             x = inputs[idx]
#             conv = self.convs[idx]
#             up = self.operator[idx]
#             x = resize(
#                     input=conv(x),
#                     size=inputs[0].shape[2:],
#                     mode=self.interpolate_mode,
#                     align_corners=self.align_corners)
#             outs.append(
#                 up(x))
#
#         out = self.fusion_conv(torch.cat(outs, dim=1))
#
#         out = self.cls_seg(out)
#
#         return out

# @HEADS.register_module()
# class SegformerHead(BaseDecodeHead):
#     """The all mlp Head of segformer.
#
#     This head is the implementation of
#     `Segformer <https://arxiv.org/abs/2105.15203>` _.
#
#     Args:
#         interpolate_mode: The interpolate mode of MLP head upsample operation.
#             Default: 'bilinear'.
#     """
#
#     def __init__(self, interpolate_mode='bilinear', **kwargs):
#         super().__init__(input_transform='multiple_select', **kwargs)
#
#         self.interpolate_mode = interpolate_mode
#         num_inputs = len(self.in_channels)
#
#         assert num_inputs == len(self.in_index)
#
#         self.convs = nn.ModuleList()
#         for i in range(num_inputs):
#             self.convs.append(
#                 ConvModule(
#                     in_channels=self.in_channels[i],
#                     out_channels=self.channels,
#                     kernel_size=1,
#                     stride=1,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg))
#
#         self.fusion_conv = ConvModule(
#             in_channels=self.channels * num_inputs,
#             out_channels=self.channels,
#             kernel_size=1,
#             norm_cfg=self.norm_cfg)
#
#     def forward(self, inputs):
#         # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
#         inputs = self._transform_inputs(inputs)
#         outs = []
#         for idx in range(len(inputs)):
#             x = inputs[idx]
#             conv = self.convs[idx]
#             outs.append(
#                 resize(
#                     input=conv(x),
#                     size=inputs[0].shape[2:],
#                     mode=self.interpolate_mode,
#                     align_corners=self.align_corners))
#
#         out = self.fusion_conv(torch.cat(outs, dim=1))
#
#         out = self.cls_seg(out)
#
#         return out

@HEADS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.up1 = GGS()
        self.up2 = GGS()
        self.up3 = GGS()
        self.up4 = GGS()
        self.up5 = GGS()
        self.up6 = GGS()

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []

        up1 = self.up1
        up2 = self.up2
        up3 = self.up3
        up4 = self.up4
        up5 = self.up5
        up6 = self.up6

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            if idx == 0:
                outs.append(conv(x))
            elif idx == 1:
                outs.append(up1(conv(x)))
            elif idx == 2:
                outs.append(up2(up3(conv(x))))
            elif idx == 3:
                outs.append(up4(up5(up6(conv(x)))))
            else:
                outs.append(
                    resize(
                        input=conv(x),
                        size=inputs[0].shape[2:],
                        mode=self.interpolate_mode,
                        align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out