import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.nn.modules.utils import _single, _pair, _triple


class CatCondBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_classes, affine=False, track_running_stats=True, is_node_zero=False):
        super().__init__()
        self.num_features = num_features
        self.is_node_zero = is_node_zero
        self.bn = nn.BatchNorm1d(num_features, affine=affine, track_running_stats=track_running_stats)

        self.embed = nn.Embedding(num_classes, num_features * 2)
        # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        # Initialise bias at 0
        self.embed.weight.data[:, num_features:].zero_()
        if is_node_zero:
            self.f_embed = nn.Embedding(1, num_features * 2)
            # Initialise scale at N(1, 0.02)
            self.f_embed.weight.data[:, :num_features].normal_(1, 0.02)
            # Initialise bias at 0
            self.f_embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y, first_layer=False):
        out = self.bn(x)
        # gamma, beta = self.embed(y).squeeze(1).chunk(2, 1)
        if first_layer and self.is_node_zero:
            gamma, beta = self.f_embed(y).chunk(2, -1)
        else:
            gamma, beta = self.embed(y).chunk(2, -1)

        out = gamma.view(-1, self.num_features, 1) * out + beta.view(-1, self.num_features, 1)
        return out


class CatCondBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, affine=False, track_running_stats=True, is_node_zero=False):
        super().__init__()
        self.num_features = num_features
        self.is_node_zero = is_node_zero
        self.bn = nn.BatchNorm2d(num_features, affine=affine, track_running_stats=track_running_stats)

        self.embed = nn.Embedding(num_classes, num_features * 2)
        # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        # Initialise bias at 0
        self.embed.weight.data[:, num_features:].zero_()
        if is_node_zero:
            self.f_embed = nn.Embedding(1, num_features * 2)
            # Initialise scale at N(1, 0.02)
            self.f_embed.weight.data[:, :num_features].normal_(1, 0.02)
            # Initialise bias at 0
            self.f_embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y, first_layer=False):
        out = self.bn(x)
        # gamma, beta = self.embed(y).squeeze(1).chunk(2, 1)
        if first_layer and self.is_node_zero:
            gamma, beta = self.f_embed(y).chunk(2, -1)
        else:
            gamma, beta = self.embed(y).chunk(2, -1)

        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class CatCondConv1d(nn.Conv1d):
    def __init__(self, num_classes, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

        self.num_classes = num_classes

        self.conv = nn.ModuleList([nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                             padding, dilation, groups, bias, padding_mode)
                                   for _ in range(num_classes)])

    def forward(self, x, y):
        tmp = [self.conv[n](x) for n in range(self.num_classes)]
        out = sum([(tmp[n] * (y == n).type(tmp[n].dtype).to(tmp[n].device)[:, None, None])
                   for n in range(self.num_classes)])
        return out


class CatCondConvTrans2d(nn.ConvTranspose2d):
    def __init__(self, num_classes, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, output_padding, groups, bias,
                         dilation, padding_mode)

        self.num_classes = num_classes

        self.conv = nn.ModuleList([nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                                      padding, output_padding, groups, bias,
                                                      dilation, padding_mode) for _ in range(num_classes)])

    def forward(self, x, y):
        tmp = [self.conv[n](x) for n in range(self.num_classes)]
        out = sum([(tmp[n] * (y == n).type(tmp[n].dtype).to(tmp[n].device)[:, None, None, None])
                   for n in range(self.num_classes)])
        return out
