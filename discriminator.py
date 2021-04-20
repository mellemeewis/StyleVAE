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

class Discriminator(nn.Module):
    def __init__(self, in_size, dchannels):
        super(Discriminator, self).__init__()
        C, H, W = in_size
        d1, d2, d3, d4 = dchannels
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # nn.Conv2d(C, ndf, 4, 2, 1, bias=False),
            nn.Conv2d(C, d1, 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.Conv2d(d1, d2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(d2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(d2, d3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d3),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(d3, d4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(d4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
