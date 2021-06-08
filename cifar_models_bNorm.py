from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import math
import numpy as np

import torchvision
from scipy import stats
from torch.nn import functional as F
from src.utils import *
import src.losses as losses
import torch.nn.functional as F
from torch import nn
import torch.nn.init as nninit

import math
# from layers.categorical_batch_norm import CategoricalBatchNorm
from layers.spectral_norm import *



def avg_pool2d(x):
    '''Twice differentiable implementation of 2x2 average pooling.'''
    return (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4

def BasicDisEncNeuro(input,output,filter_size,stride,padding,bias): 
    return nn.Sequential(nn.Conv2d(input, output, filter_size , stride, padding, bias = bias),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2, inplace=True))


class Encoder(nn.Module):
    def __init__(self,ngpu,ndf, nz=64,nc = 3):
        super(Encoder,self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.main = nn.Sequential(
            BasicDisEncNeuro(nc,ndf*2,4,2,1,bias=False),
            BasicDisEncNeuro(ndf*2,ndf*4,4,2,1,bias = False),
            BasicDisEncNeuro(ndf*4,ndf*8,4,2,1,bias = False),
            nn.Conv2d(ndf*8,self.nz,4,1,0,bias=False),
            )
    def forward(self,input):
        x = self.main(input)
        x = x.view(-1, self.nz)
        return x
    

class DiscriminatorBlock(nn.Module):
    '''ResNet-style block for the discriminator model.'''

    def __init__(self, in_chans, out_chans, downsample=False, first=False):
        super().__init__()

        self.downsample = downsample
        self.first = first

        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.downsample:
            shortcut = avg_pool2d(x)
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        if not self.first:
            x = nn.functional.relu(x, inplace=False)
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)
        if self.downsample:
            x = avg_pool2d(x)

        return x + shortcut


class Res_Discriminator(nn.Module):
    '''The discriminator (aka critic) model.'''

    def __init__(self,channel, ch):
        super().__init__()

        feats = ch
        self.block1 = DiscriminatorBlock(channel, feats, downsample=True, first=True)
        self.block2 = DiscriminatorBlock(feats, feats, downsample=True)
        self.block3 = DiscriminatorBlock(feats, feats, downsample=False)
        self.block4 = DiscriminatorBlock(feats, feats, downsample=False)
        self.output_linear = nn.Linear(128, 1)

        # Apply Xavier initialisation to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.block1.conv1 else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

    def forward(self, *inputs):
        x = inputs[0]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.relu(x, inplace=False)
        x = x.mean(-1, keepdim=False).mean(-1, keepdim=False)
        x = x.view(-1, 128)
        x = self.output_linear(x)

        return x
    def feature(self,*inputs):
        x = inputs[0]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.relu(x, inplace=False)
        x = x.mean(-1, keepdim=False).mean(-1, keepdim=False)
        x = x.view(-1, 128)

        return x
    
    
    

class Block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None,
                 kernel_size=3, stride=1, padding=1, optimized=False, spectral_norm=1):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.optimized = optimized
        self.hidden_channels = out_channels if not hidden_channels else hidden_channels

        self.conv1 = Conv2d(self.in_channels, self.hidden_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, spectral_norm_pi=spectral_norm)
        self.conv2 = Conv2d(self.hidden_channels, self.out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, spectral_norm_pi=spectral_norm)
        self.s_conv = None
        torch.nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        if self.in_channels != self.out_channels or optimized:
            self.s_conv = Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0,
                                 spectral_norm_pi=spectral_norm)
            torch.nn.init.xavier_uniform_(self.s_conv.weight.data, 1.)

        self.activate = torch.nn.ReLU()

    def residual(self, input):
        x = self.conv1(input)
        x = self.activate(x)
        x = self.conv2(x)
        if self.optimized:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def shortcut(self, input):
        x = input
        if self.optimized:
            x = torch.nn.functional.avg_pool2d(x, 2)
        if self.s_conv:
            x = self.s_conv(x)
        return x


    def forward(self, input):
        x = self.residual(input)
        x_r = self.shortcut(input)
        return x + x_r


class Gblock(Block):

    def __init__(self, in_channels, out_channels, hidden_channels=None, num_categories=None,
                 kernel_size=3, stride=1, padding=1, upsample=True):
        super(Gblock, self).__init__(in_channels, out_channels, hidden_channels, kernel_size, stride, padding,
                                     upsample, spectral_norm=0)
        self.upsample = upsample
        self.num_categories = num_categories

        self.bn1 = self.batch_norm(self.in_channels)
        self.bn2 = self.batch_norm(self.hidden_channels)
        if upsample:
            # self.up = torch.nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
            self.up = lambda a: torch.nn.functional.interpolate(a, scale_factor=2)
        else:
            self.up = lambda a: None

    def batch_norm(self, num_features):
        return torch.nn.BatchNorm2d(num_features) if not self.num_categories \
            else CategoricalBatchNorm(num_features, self.num_categories)

    def residual(self, input, y=None):
        x = input
        x = self.bn1(x, y) if self.num_categories else self.bn1(x)
        x = self.activate(x)
        if self.upsample:
            x = self.up(x)
            # output_size = list(input.size())
            # output_size[-1] = output_size[-1] * 2
            # output_size[-2] = output_size[-2] * 2
            # x = self.up(x, output_size=output_size)
        x = self.conv1(x)
        x = self.bn2(x, y) if self.num_categories else self.bn2(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x

    def shortcut(self, input):
        x = input
        if self.upsample:
            x = self.up(x)
        if self.s_conv:
            x = self.s_conv(x)
        return x

    def forward(self, input, y=None):
        x = self.residual(input, y)
        x_r = self.shortcut(input)
        return x + x_r


class Dblock(Block):

    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, stride=1, padding=1,
                 downsample=False, spectral_norm=1):
        super(Dblock, self).__init__(in_channels, out_channels, hidden_channels, kernel_size, stride, padding,
                                     downsample, spectral_norm)
        self.downsample = downsample

    def residual(self, input):
        x = self.activate(input)
        x = self.conv1(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.downsample:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def shortcut(self, input):
        x = input
        if self.s_conv:
            x = self.s_conv(x)
        if self.downsample:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def forward(self, input):
        x = self.residual(input)
        x_r = self.shortcut(input)
        return x + x_r


class BaseGenerator(torch.nn.Module):

    def __init__(self, z_dim, ch, d_ch=None, n_categories=None, bottom_width=4):
        super(BaseGenerator, self).__init__()
        self.z_dim = z_dim
        self.ch = ch
        self.d_ch = d_ch if d_ch else ch
        self.n_categories = n_categories
        self.bottom_width = bottom_width
        self.dense = torch.nn.Linear(self.z_dim, self.bottom_width * self.bottom_width * self.d_ch)
        torch.nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        self.blocks = torch.nn.ModuleList()
        self.final = self.final_block()

    def final_block(self):
        conv = torch.nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(conv.weight.data, 1.)
        final_ = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.ch),
            torch.nn.ReLU(),
            conv,
            torch.nn.Tanh()
        )
        return final_


    def forward(self, input, y=None):
        x = self.dense(input)
        x = x.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        for block in self.blocks:
            x = block(x, y)
        x = self.final(x)
        return x



class ResnetGenerator32(BaseGenerator):

    def __init__(self, ch=256, z_dim=128, n_categories=None, bottom_width=4):
        super(ResnetGenerator32, self).__init__(z_dim, ch, ch, n_categories, bottom_width)
        self.blocks.append(Gblock(self.ch, self.ch, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch, self.ch, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch, self.ch, upsample=True, num_categories=self.n_categories))


class BaseDiscriminator(torch.nn.Module):

    def __init__(self, in_ch, out_ch=None, n_categories=0, l_bias=True, spectral_norm=1,stack = 3):
        super(BaseDiscriminator, self).__init__()
        self.activate = torch.nn.ReLU()
        self.ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        self.n_categories = n_categories
        self.blocks = torch.nn.ModuleList([Block(stack, self.ch, optimized=True, spectral_norm=spectral_norm)])
        self.l = Linear(self.out_ch, 1, l_bias, spectral_norm_pi=spectral_norm)
        torch.nn.init.xavier_uniform_(self.l.weight.data, 1.)
        if n_categories > 0:
            self.l_y = Embedding(n_categories, self.out_ch, spectral_norm_pi=spectral_norm)
            torch.nn.init.xavier_uniform_(self.l_y.weight.data, 1.)

    def forward(self, input, y=None):
        x = input
        for block in self.blocks:
            x = block(x)
        x = self.activate(x)
        x = torch.sum(x, (2, 3))
        output = self.l(x)
        if y is not None:
            w_y = self.l_y(y)
            output += torch.sum(w_y*x, dim=1, keepdim=True)
        return output
    
    def feature(self, input):
        batch_size = input.shape[0]
        x = input
        for block in self.blocks:
            x = block(x)
        x = self.activate(x)
        x = torch.sum(x, (2, 3))
        feature = x.view(batch_size,-1)
        return feature


class ResnetDiscriminator32(BaseDiscriminator):

    def __init__(self, ch=128, n_categories=0, spectral_norm=1,stack = 3):
        super(ResnetDiscriminator32, self).__init__(ch, ch, n_categories, l_bias=False, spectral_norm=spectral_norm,stack = stack)
        self.blocks.append(Dblock(self.ch, self.ch, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch, self.ch, downsample=False, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch, self.ch, downsample=False, spectral_norm=spectral_norm))


