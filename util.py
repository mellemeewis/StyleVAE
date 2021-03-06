import os, sys, math


import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):

    def __init__(self, in_channels, channels, num_convs = 3, kernel_size = 3, batch_norm=False, use_weight=True, use_res=True, deconv=False):
        super().__init__()

        layers = []
        self.use_weight = use_weight
        self.use_res = use_res

        padding = int(math.floor(kernel_size / 2))

        self.upchannels = nn.Conv2d(in_channels, channels, kernel_size=1)

        for i in range(num_convs):
            if deconv:
                layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm))
            else:
                layers.append(nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm))

            if batch_norm:
                layers.append(nn.BatchNorm2d(channels))

            layers.append(nn.ReLU())

        self.seq = nn.Sequential(*layers)

        if use_weight:
            self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x):

        x = self.upchannels(x)

        out = self.seq(x)

        if not self.use_res:
            return out

        if not self.use_weight:
            return out + x

        return out + self.weight * x


def prod(xs):
    res = 1

    for x in xs:
        res *= x

    return res

def adain(y, x):
    """
    Adaptive instance normalization
    :param y: Parameters for the normalization
    :param x: Input to normalize
    :return:
    """
    b, c, h, w = y.size()

    ys = y[:, :c//2, :, :]
    yb = y[:, c//2:, :, :]

    x = F.instance_norm(x)

    return (ys + 1.) * x + yb


def kl_loss_image(z):
    if z is None:
        return 0.0

    b, c, h, w = z.size()

    mean = z[:, :c//2, :, :].view(b, -1)
    sig = z[:, c//2:, :, :].view(b, -1)

    kl = 0.5 * torch.sum(sig.exp() - sig + mean.pow(2) - 1, dim=1)
    kl = torch.clamp(kl, min=0.00000000000001, max=10000000000000)

    assert kl.size() == (b,)

    return kl


def kl_loss(zmean, zlsig):

    b, l = zmean.size()

    kl = 0.5 * torch.sum(zlsig.exp() - zlsig + zmean.pow(2) - 1, dim=1)

    kl = torch.clamp(kl, min=0.00000000000001, max=10000000000000)
    assert kl.size() == (b,)


    return kl


def normal_lt_loss(output, target):
    if output == None or target == None:
        return torch.tensor([0.])
    b, l = output.size()


    means = torch.sigmoid(output[:,  :l//2])
    vars  = torch.sigmoid(output[:,  l//2:])

    means=torch.clamp(means, min=0.00000000000001) 
    vars=torch.clamp(vars, min=0.00000000000001)

    return vars.log() + (1.0/(2.0 * vars.pow(2.0))) * (target - means).pow(2.0)

def sample(zmean, zlsig, eps=None):
    b, l = zmean.size()

    if eps is None:
        eps = torch.randn(b, l)
        if zmean.is_cuda:
            eps = eps.cuda()
        eps = Variable(eps)

    return zmean + eps * (zlsig * 0.5).exp()


def sample_image(z, eps=None):

    if z  is None:
        return None

    b, c, h, w = z.size()

    mean = z[:, :c//2, :, :].view(b, -1)
    sig = z[:, c//2:, :, :].view(b, -1)

    if eps is None:
        eps = torch.randn(b, c//2, h, w).view(b, -1)
        if z.is_cuda:
            eps = eps.cuda()
        eps = Variable(eps)

    sample = mean + eps * (sig * 0.5).exp()

    return sample.view(b, c//2, h, w)

def latent_sample(b, zsize, outsize, depth, zchannels, dev):
    """
    Samples latents from the normal distribution.
    :param b:
    :param zsize:
    :param outsize:
    :param depth:
    :param zchannels:
    :param dev:
    :return:
    """

    c, h, w = outsize
    zc0, zc1, zc2, zc3, zc4, zc5 = zchannels
    n = [None] * 6

    z = torch.randn(b, zsize, device=dev)

    n[0] = torch.randn(b, zc0, h, w, device=dev)

    if depth >=1:
        n[1] = torch.randn(b, zc1, h // 2, w // 2, device=dev)

    if depth >= 2:
        n[2] = torch.randn(b, zc2, h // 4, w // 4, device=dev)

    if depth >= 3:
        n[3] = torch.randn(b, zc3, h // 8, w // 8, device=dev)

    if depth >= 4:
        n[4] = torch.randn(b, zc4, h // 16, w // 16, device=dev)

    if depth >= 5:
        n[5] = torch.randn(b, zc5, h // 32, w // 32, device=dev)

    return z, n


def batchedn(input, model, batch_size, cuda=torch.cuda.is_available()):
    """
    Performs inference in batches. Input and output are non-variable, non-gpu tensors.
    :param input: A tuple
    :param model:
    :param batch_size:
    :param cuda:
    :return:
    """
    n = input[0].size(0)

    out_batches = []

    for fr in range(0, n, batch_size):
        to = min(n, fr + batch_size)

        batches = []
        for e in input:
            if e is not None:
                batch = e[fr:to]

                if cuda:
                    batch = batch.cuda()

                batches.append(batch)
            else:
                batches.append(None)

        out_batches.append(model(*batches).cpu().data)

        del batches

    return torch.cat(out_batches, dim=0)

def readn(loader, n, cls=False, maxval=None):
    """
    Reads from the loader to fill a large batch of size n
    :param loader: a dataloader
    :param n:
    :return:
    """

    batches = []
    total = 0
    for input in loader:
        batch = input[0] if not cls else input[1]

        if cls:
            batch = one_hot(batch, maxval)

        total += batch.size(0)
        batches.append(batch)

        if total > n:
            break

    result = torch.cat(batches, dim=0)

    return result[:n]


def nbatched(input, model, batch_size, cuda=torch.cuda.is_available(), **kwargs):
    """
    Performs inference in batches. Input and output are non-variable, non-gpu tensors.
    :param input:
    :param model:
    :param batch_size:
    :param cuda:
    :return:
    """
    n = input.size(0)

    out_batches = []

    for fr in range(0, n, batch_size):
        to = min(n, fr + batch_size)

        batch = input[fr:to]
        if cuda:
            batch = batch.cuda()

        outputs = model(batch, **kwargs)

        if fr == 0:
            for _ in range(len(outputs)):
                out_batches.append([])

        for i in range(len(outputs)):
            out_batches[i].append(None if outputs[i] is None else outputs[i].cpu().data)

        del outputs

    res = []
    for batches in out_batches:
        res.append(None if none(batches) else torch.cat(batches, dim=0))

    return res

def none(lst):
    for l in lst:
        if l is None:
            return True

    return False