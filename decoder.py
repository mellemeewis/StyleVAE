from torch import nn
import torch.nn.functional as F
import torch.distributions as ds

import torch
from torch.autograd import Variable
from torchvision.transforms import CenterCrop, ToTensor, Compose, Lambda, Resize, Grayscale, Pad, RandomHorizontalFlip
from torchvision.datasets import coco
from torchvision import utils

from torch.nn.functional import binary_cross_entropy, relu, nll_loss, cross_entropy, softmax
from torch.nn import Embedding, Conv2d, Sequential, BatchNorm2d, ReLU, MSELoss
from torch.optim import Adam

import util
class StyleDecoder(nn.Module):

    def __init__(self, out_size, channels, zchannels, zs=256, k=3, mapping=3, batch_norm=False, dropouts=None):
        super().__init__()

        self.out_size = out_size

        c, h, w = self.out_size
        self.channels = channels
        c1, c2, c3, c4, c5 = self.channels
        z0, z1, z2, z3, z4, z5 = zchannels

        # resnet blocks
        self.block5 = util.Block(c5, c4, kernel_size=k, batch_norm=batch_norm)
        self.block4 = util.Block(c4, c3, kernel_size=k, batch_norm=batch_norm)
        self.block3 = util.Block(c3, c2, kernel_size=k, batch_norm=batch_norm)
        self.block2 = util.Block(c2, c1, kernel_size=k, batch_norm=batch_norm)
        self.block1 = util.Block(c1, c,  kernel_size=k, batch_norm=batch_norm)

        # affine mappings from latent space sample
            
        self.affine5 = nn.Linear(zs, 2 * util.prod((c5, h//32, w//32)))
        self.affine4 = nn.Linear(zs, 2 * util.prod((c4, h//16, w//16)))
        self.affine3 = nn.Linear(zs, 2 * util.prod((c3, h//8, w//8)))
        self.affine2 = nn.Linear(zs, 2 * util.prod((c2, h//4, w//4)))
        self.affine1 = nn.Linear(zs, 2 * util.prod((c1, h//2, w//2)))
        self.affine0 = nn.Linear(zs, 2 * util.prod(out_size))

        # 1x1 convolution from "noise space" sample
        self.tonoise5 = nn.Conv2d(z5, c5, kernel_size=1, padding=0)
        self.tonoise4 = nn.Conv2d(z4, c4, kernel_size=1, padding=0)
        self.tonoise3 = nn.Conv2d(z3, c3, kernel_size=1, padding=0)
        self.tonoise2 = nn.Conv2d(z2, c2, kernel_size=1, padding=0)
        self.tonoise1 = nn.Conv2d(z1, c1, kernel_size=1, padding=0)
        self.tonoise0 = nn.Conv2d(z0, c,  kernel_size=1, padding=0)

        self.conv0 = nn.Conv2d(c, 2*c, kernel_size=1)

        m = []
        for _ in range(mapping):
            m.append(nn.Linear(zs, zs))
            m.append(nn.ReLU())
        self.mapping = nn.Sequential(*m)

        self.dropouts = dropouts

        # constant, learnable input
        self.x5 = nn.Parameter(torch.randn(1, c5, h//32, w//32))
        self.x4 = nn.Parameter(torch.randn(1, c4, h//16, w//16))
        self.x3 = nn.Parameter(torch.randn(1, c3, h//8, w//8))
        self.x2 = nn.Parameter(torch.randn(1, c2, h//4, w//4))
        self.x1 = nn.Parameter(torch.randn(1, c1, h//2, w//2))

    def forward(self, z, n0, n1, n2, n3, n4, n5):
        """
        z, n0 are never none all others can be, depending on the depth
        :param z:
        :param n0:
        :param n1:
        :param n2:
        :param n3:
        :param n4:
        :param n5:
        :return:
        """

        x0 = x1 = x2 = x3 = x4 = x5 = None

        c, h, w = self.out_size
        c1, c2, c3, c4, c5 = self.channels

        if self.dropouts is not None:
            dz, d0, d1, d2, d3, d4, d5 = self.dropouts
            z = F.dropout(z, p=dz, training=True)
            if n0 is not None: n0 = F.dropout(n0, p=d0, training=True)
            if n1 is not None: n1 = F.dropout(n1, p=d1, training=True)
            if n2 is not None: n2 = F.dropout(n2, p=d2, training=True)
            if n3 is not None: n3 = F.dropout(n3, p=d3, training=True)
            if n4 is not None: n4 = F.dropout(n4, p=d4, training=True)
            if n5 is not None: n5 = F.dropout(n5, p=d5, training=True)

        z = self.mapping(z)

        if n5 is not None:
            x5 = self.x5 + self.tonoise5(n5)
            z5 = self.affine5(z).view(-1, 2 * c5, h//32, w//32)

            x5 = util.adain(z5, x5)

        if n4 is not None:
            if x5 is None:
                x5 = self.x5

            x4 = F.upsample(self.block5(x5), scale_factor=2)
            x4 = x4 + self.tonoise4(n4)
            z4 = self.affine4(z).view(-1, 2 * c4, h//16, w//16)
            x4 = util.adain(z4, x4)

        if n3 is not None:
            if x4 is None:
                x4 = self.x4

            x3 = F.upsample(self.block4(x4), scale_factor=2)
            x3 = x3 + self.tonoise3(n3)
            z3 = self.affine3(z).view(-1, 2 * c3, h//8, w//8)
            x3 = util.adain(z3, x3)

        if n2 is not None:
            if x3 is None:
                x3 = self.x3

            x2 = F.upsample(self.block3(x3), scale_factor=2)
            x2 = x2 + self.tonoise2(n2)
            z2 = self.affine2(z).view(-1, 2 * c2, h//4, w//4)
            x2 = util.adain(z2, x2)

        if n1 is not None:
            if x2 is None:
                x2 = self.x2

            x1 = F.upsample(self.block2(x2), scale_factor=2)
            x1 = x1 + self.tonoise1(n1)
            z1 = self.affine1(z).view(-1, 2 * c1, h//2, w//2)
            x1 = util.adain(z1, x1)

        if n0 is not None:
            if x1 is None:
                x1 = self.x1

            x0 = F.upsample(self.block1(x1), scale_factor=2)
            x0 = x0 + self.tonoise0(n0)
            z0 = self.affine0(z).view(-1, 2 * c, h, w)
            x0 = util.adain(z0, x0)

        return self.conv0(x0)

class StyleDecoder2(nn.Module):

    def __init__(self, out_size, channels, zchannels, zs=256, k=3, mapping=3, batch_norm=False, dropouts=None):
        super().__init__()

        self.out_size = out_size

        c, h, w = self.out_size
        self.channels = channels
        c1, c2, c3, c4, c5 = self.channels
        z0, z1, z2, z3, z4, z5 = zchannels

        # resnet blocks
        self.block5 = util.Block(c5, c4, kernel_size=k, batch_norm=batch_norm)
        self.block4 = util.Block(c4, c3, kernel_size=k, batch_norm=batch_norm)
        self.block3 = util.Block(c3, c2, kernel_size=k, batch_norm=batch_norm)
        self.block2 = util.Block(c2, c1, kernel_size=k, batch_norm=batch_norm)
        self.block1 = util.Block(c1, c,  kernel_size=k, batch_norm=batch_norm)

        # affine mappings from latent space sample
            
        self.affine5 = nn.Linear(zs, 2 * util.prod((c5, h//32, w//32)))
        self.affine4 = nn.Linear(zs, 2 * util.prod((c4, h//16, w//16)))
        self.affine3 = nn.Linear(zs, 2 * util.prod((c3, h//8, w//8)))
        self.affine2 = nn.Linear(zs, 2 * util.prod((c2, h//4, w//4)))
        self.affine1 = nn.Linear(zs, 2 * util.prod((c1, h//2, w//2)))
        self.affine0 = nn.Linear(zs, 2 * util.prod(out_size))

        # 1x1 convolution from "noise space" sample
        self.tonoise5 = nn.Conv2d(z5, c5, kernel_size=1, padding=0)
        self.tonoise4 = nn.Conv2d(z4, c4, kernel_size=1, padding=0)
        self.tonoise3 = nn.Conv2d(z3, c3, kernel_size=1, padding=0)
        self.tonoise2 = nn.Conv2d(z2, c2, kernel_size=1, padding=0)
        self.tonoise1 = nn.Conv2d(z1, c1, kernel_size=1, padding=0)
        self.tonoise0 = nn.Conv2d(z0, c,  kernel_size=1, padding=0)

        self.conv0 = nn.Conv2d(c, 2*c, kernel_size=1)

        m = []
        for _ in range(mapping):
            m.append(nn.Linear(zs, zs))
            m.append(nn.ReLU())
        self.mapping = nn.Sequential(*m)

        self.dropouts = dropouts

        # constant, learnable input
        self.x5 = nn.Parameter(torch.randn(1, c5, h//32, w//32))
        self.x4 = nn.Parameter(torch.randn(1, c4, h//16, w//16))
        self.x3 = nn.Parameter(torch.randn(1, c3, h//8, w//8))
        self.x2 = nn.Parameter(torch.randn(1, c2, h//4, w//4))
        self.x1 = nn.Parameter(torch.randn(1, c1, h//2, w//2))

    def forward(self, z, depth): #, n0, n1, n2, n3, n4, n5
        """
        z, n0 are never none all others can be, depending on the depth
        :param z:
        :param n0:
        :param n1:
        :param n2:
        :param n3:
        :param n4:
        :param n5:
        :return:
        """

        x0 = x1 = x2 = x3 = x4 = x5 = None

        c, h, w = self.out_size
        c1, c2, c3, c4, c5 = self.channels

        # if self.dropouts is not None:
        #     dz, d0, d1, d2, d3, d4, d5 = self.dropouts
        #     z = F.dropout(z, p=dz, training=True)
        #     if n0 is not None: n0 = F.dropout(n0, p=d0, training=True)
        #     if n1 is not None: n1 = F.dropout(n1, p=d1, training=True)
        #     if n2 is not None: n2 = F.dropout(n2, p=d2, training=True)
        #     if n3 is not None: n3 = F.dropout(n3, p=d3, training=True)
        #     if n4 is not None: n4 = F.dropout(n4, p=d4, training=True)
        #     if n5 is not None: n5 = F.dropout(n5, p=d5, training=True)

        z = self.mapping(z)

        if depth == 5:
            x5 = self.x5
            z5 = self.affine5(z).view(-1, 2 * c5, h//32, w//32)
            print(z5.size(), x5.size())

            x5 = util.adain(z5, x5)

        if depth == 4:
            if x5 is None:
                x5 = self.x5

            x4 = F.upsample(self.block5(x5), scale_factor=2)
            z4 = self.affine4(z).view(-1, 2 * c4, h//16, w//16)
            x4 = util.adain(z4, x4)

        if depth == 3:
            if x4 is None:
                x4 = self.x4

            x3 = F.upsample(self.block4(x4), scale_factor=2)
            z3 = self.affine3(z).view(-1, 2 * c3, h//8, w//8)
            x3 = util.adain(z3, x3)

        if depth == 2:
            if x3 is None:
                x3 = self.x3

            x2 = F.upsample(self.block3(x3), scale_factor=2)
            z2 = self.affine2(z).view(-1, 2 * c2, h//4, w//4)
            x2 = util.adain(z2, x2)

        if depth == 1:
            if x2 is None:
                x2 = self.x2

            x1 = F.upsample(self.block2(x2), scale_factor=2)
            z1 = self.affine1(z).view(-1, 2 * c1, h//2, w//2)
            x1 = util.adain(z1, x1)

        if depth == 0:
            if x1 is None:
                x1 = self.x1

            x0 = F.upsample(self.block1(x1), scale_factor=2)
            z0 = self.affine0(z).view(-1, 2 * c, h, w)
            x0 = util.adain(z0, x0)

        return self.conv0(x0)