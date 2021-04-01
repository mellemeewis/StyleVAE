import os, tqdm, random, pickle, sys

import torch
import torchvision

from torch import nn
import torch.nn.functional as F
import torch.distributions as ds

from torch.autograd import Variable
from torchvision.transforms import CenterCrop, ToTensor, Compose, Lambda, Resize, Grayscale, Pad, RandomHorizontalFlip
from torchvision.datasets import coco
from torchvision import utils

from torch.nn.functional import binary_cross_entropy, relu, nll_loss, cross_entropy, softmax
from torch.nn import Embedding, Conv2d, Sequential, BatchNorm2d, ReLU, MSELoss
from torch.optim import Adam

# import nltk

from argparse import ArgumentParser

from collections import defaultdict, Counter, OrderedDict

import util#, models
from models.alexnet import AlexNet
from models.densenet import DenseNet
from data import return_data
import slack_util

from tensorboardX import SummaryWriter

# from layers import PlainMaskedConv2d, MaskedConv2d

SEEDFRAC = 2
DV = 'cuda' if torch.cuda.is_available() else 'cpu'

# def standard(b, c, h, w):
#     mean = torch.zeros(b, c, h, w)
#     sig  = torch.ones(b, c, h, w)

#     res = torch.cat([mean, sig], dim=1)

#     if torch.cuda.is_available():
#         res = res.cuda()
#     return res




class StyleEncoder(nn.Module):

    def __init__(self, in_size, channels, zchannels, zs=256, k=3, unmapping=3, batch_norm=False):
        super().__init__()

        c, h, w = in_size
        c1, c2, c3, c4, c5 = channels
        z0, z1, z2, z3, z4, z5 = zchannels

        # resnet blocks
        self.block1 = util.Block(c,  c1, kernel_size=k, batch_norm=batch_norm)
        self.block2 = util.Block(c1, c2, kernel_size=k, batch_norm=batch_norm)
        self.block3 = util.Block(c2, c3, kernel_size=k, batch_norm=batch_norm)
        self.block4 = util.Block(c3, c4, kernel_size=k, batch_norm=batch_norm)
        self.block5 = util.Block(c4, c5, kernel_size=k, batch_norm=batch_norm)

        # affine mappings to distribution on latent space
        self.affine0 = nn.Linear(util.prod(in_size), 2 * zs)
        self.affine1 = nn.Linear(util.prod((c1, h//2, w//2)), 2 * zs)
        self.affine2 = nn.Linear(util.prod((c2, h//4, w//4)), 2 * zs)
        self.affine3 = nn.Linear(util.prod((c3, h//8, w//8)), 2 * zs)
        self.affine4 = nn.Linear(util.prod((c4, h//16, w//16)), 2 * zs)
        self.affine5 = nn.Linear(util.prod((c5, h//32, w//32)), 2 * zs)

        self.affinez = nn.Linear(12 * zs, 2 * zs)

        # 1x1 convolution to distribution on "noise space"
        # (mean and sigma)
        self.tonoise0 = nn.Conv2d(c,  z0*2, kernel_size=1, padding=0)
        self.tonoise1 = nn.Conv2d(c1, z1*2, kernel_size=1, padding=0)
        self.tonoise2 = nn.Conv2d(c2, z2*2, kernel_size=1, padding=0)
        self.tonoise3 = nn.Conv2d(c3, z3*2, kernel_size=1, padding=0)
        self.tonoise4 = nn.Conv2d(c4, z4*2, kernel_size=1, padding=0)
        self.tonoise5 = nn.Conv2d(c5, z5*2, kernel_size=1, padding=0)

        um = []
        for _ in range(unmapping):
            um.append(nn.ReLU())
            um.append(nn.Linear(zs*2, zs*2))
        self.unmapping = nn.Sequential(*um)

    def forward(self, x0, depth):

        b = x0.size(0)

        n0 = n1 = n2 = n3 = n4 = n5 = None

        z0 = self.affine0(x0.view(b, -1))
        n0 = self.tonoise0(x0)

        if depth <= 0:
            z = self.unmapping(z0)
            return z, n0, n1, n2, n3, n4, n5

        x1 = F.avg_pool2d(self.block1(x0), 2)
        z1 = self.affine1(x1.view(b, -1))
        n1 = self.tonoise1(x1)

        if depth <= 1:
            z = self.unmapping(z0 + z1)
            return z, n0, n1, n2, n3, n4, n5

        x2 = F.avg_pool2d(self.block2(x1), 2)
        z2 = self.affine2(x2.view(b, -1))
        n2 = self.tonoise2(x2)

        if depth <= 2:
            z = self.unmapping(z0 + z1 + z2)
            return z, n0, n1, n2, n3, n4, n5

        x3 = F.avg_pool2d(self.block3(x2), 2)
        z3 = self.affine3(x3.view(b, -1))
        n3 = self.tonoise3(x3)

        if depth <= 3:
            z = self.unmapping(z0 + z1 + z2 + z3)
            return z, n0, n1, n2, n3, n4, n5

        x4 = F.avg_pool2d(self.block4(x3), 2)
        z4 = self.affine4(x4.view(b, -1))
        n4 = self.tonoise4(x4)

        if depth <= 4:
            z = self.unmapping(z0 + z1 + z2 + z3 + z4)
            return z, n0, n1, n2, n3, n4, n5

        x5 = F.avg_pool2d(self.block5(x4), 2)
        z5 = self.affine5(x5.view(b, -1))
        n5 = self.tonoise5(x5)

        z = self.unmapping(z0 + z1 + z2 + z3 + z4 + z5)
        return z, n0, n1, n2, n3, n4, n5

        # combine the z vectors
        # zbatch = torch.cat([
        #     z0[:, :, None],
        #     z1[:, :, None],
        #     z2[:, :, None],
        #     z3[:, :, None],
        #     z4[:, :, None],
        #     z5[:, :, None]], dim=2)
        #
        # z = self.affinez(zbatch.view(b, -1))
        # z = z)

        return z, n0, n1, n2, n3, n4, n5

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

        return torch.sigmoid(self.conv0(x0))

def go(arg):

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    C, H, W, trainset, trainloader, testset, testloader = return_data(arg.task, arg.data_dir, arg.batch_size)

    zs = arg.latent_size

    encoder = StyleEncoder((C, H, W), arg.channels, arg.zchannels, zs=zs, k=arg.kernel_size, unmapping=arg.mapping_layers, batch_norm=arg.batch_norm)
    decoder = StyleDecoder((C, H, W), arg.channels, arg.zchannels, zs=zs, k=arg.kernel_size, mapping=arg.mapping_layers, batch_norm=arg.batch_norm, dropouts=arg.dropouts)
    

    if arg.perceptual_loss:
        if arg.perceptual_loss == 'AlexNet':
            perceptual_loss_model = AlexNet()
            checkpoint = torch.load('saved_models/alexnet.pth.tar')
        elif arg.perceptual_loss == 'DenseNet':
            perceptual_loss_model = DenseNet()
            checkpoint = torch.load('saved_models/densenet.pth.tar')

        else:
            raise Exception('Model for perceptual_loss {} not recognized.'.format(arg.perceptual_loss))


        new_state_dict = {key.replace('module.', ''): checkpoint['state_dict'][key] for key in checkpoint['state_dict'].keys()}
        perceptual_loss_model.load_state_dict(new_state_dict)
        perceptual_loss_model.eval()
        print(f"{arg.perceptual_loss} loaded")

    # optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=arg.lr)
    opte = Adam(list(encoder.parameters()), lr=arg.lr)
    optd = Adam(list(decoder.parameters()), lr=arg.lr)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

        if arg.perceptual_loss:
            perceptual_loss_model.cuda()

    instances_seen = 0
    for depth in range(6):

        print(f'starting depth {depth}, for {arg.epochs[depth]} epochs')
        for epoch in range(arg.epochs[depth]):

            epoch_loss = [0,0,0]

            # Train
            err_tr = []
            encoder.train(True)
            decoder.train(True)
            # for i, (input, _) in enumerate(trainloader):

            for i, (input, _) in enumerate(tqdm.tqdm(trainloader)):
                if arg.limit is not None and i * arg.batch_size > arg.limit:
                    break

                # Prepare the input
                b, c, w, h = input.size()
                if torch.cuda.is_available():
                    input = input.cuda()

                # -- encoding
                z, n0, n1, n2, n3, n4, n5 = encoder(input, depth)

                # -- compute KL losses

                # zkl  = util.kl_loss(z[:, :zs], z[:, zs:])
                # n0kl = util.kl_loss_image(n0)
                # n1kl = util.kl_loss_image(n1)
                # n2kl = util.kl_loss_image(n2)
                # n3kl = util.kl_loss_image(n3)
                # n4kl = util.kl_loss_image(n4)
                # n5kl = util.kl_loss_image(n5)

                # -- take samples
                zsample  = util.sample(z[:, :zs], z[:, zs:])
                n0sample = util.sample_image(n0)
                n1sample = util.sample_image(n1)
                n2sample = util.sample_image(n2)
                n3sample = util.sample_image(n3)
                n4sample = util.sample_image(n4)
                n5sample = util.sample_image(n5)

                # -- decoding
                xout = decoder(zsample, n0sample, n1sample, n2sample, n3sample, n4sample, n5sample)




                perceptual_loss = 0
                if arg.perceptual_loss:
                    with torch.no_grad():
                        perceptual_input = perceptual_loss_model(input)
                        perceptual_output = perceptual_loss_model(xout)
                        perceptual_loss = F.mse_loss(perceptual_input, perceptual_output, reduction='none').view(b, -1).sum(dim=1)



                assert torch.isinf(xout).sum() == 0
                assert torch.isnan(xout).sum() == 0

                rec_loss = util.normal_im(xout, input).view(b, c*h*w).sum(dim=1)

                # rec_loss = F.binary_cross_entropy(xout, input, reduction='none').view(b, -1).sum(dim=1)

                assert torch.isnan(rec_loss).sum() == 0
                assert torch.isinf(rec_loss).sum() == 0

                br, bz, b0, b1, b2, b3, b4, b5 = arg.betas

                # dense_loss = 0
                # kl_loss = bz * zkl + b0 * n0kl + b1 * n1kl + b2 * n2kl + b3 * n3kl + b4 * n4kl + b5 * n5kl
                # assert torch.isnan(kl_loss).sum() == 0
                # assert torch.isinf(kl_loss).sum() == 0

                loss = rec_loss
                # loss = br * rec_loss + kl_loss

                assert torch.isnan(loss).sum() == 0
                assert torch.isinf(loss).sum() == 0

                loss = loss.mean(dim=0)
                with torch.no_grad():
                    epoch_loss[0] += rec_loss.mean(dim=0).item()
                    # epoch_loss[1] += kl_loss.mean(dim=0).item()

                loss.backward()
                optd.step()
                optd.zero_grad()

                for i in range(arg.encoder_update_per_iteration):

                    zrand, (n0rand, n1rand, n2rand, n3rand, n4rand, n5rand) = util.latent_sample(b,\
                            zsize=arg.latent_size, outsize=(C, H, W), zchannels=arg.zchannels, \
                            dev='cuda', depth=depth)

                    with torch.no_grad():
                        i = decoder(zrand, n0rand, n1rand, n2rand, n3rand, n4rand, n5rand)


                    isample = util.sample_image(i)

                    iz, in0, in1, in2, in3, in4, in5 = encoder(isample, depth)

                    iz_loss = util.normal_lt_loss(iz, zrand).mean()
                    in0_loss = util.normal_lt_loss(torch.flatten(in0, start_dim=1), torch.flatten(n0rand, start_dim=1)).mean()
                    i_loss = iz_loss + in0_loss 
                    if depth >0:
                        in1_loss = util.normal_lt_loss(torch.flatten(in1, start_dim=1), torch.flatten(n1rand, start_dim=1)).mean()
                        i_loss += in1_loss
                    if depth > 1:
                        in2_loss = util.normal_lt_loss(torch.flatten(in2, start_dim=1), torch.flatten(n2rand, start_dim=1)).mean()
                        i_loss += in2_loss
                    if depth > 2:
                        in3_loss = util.normal_lt_loss(torch.flatten(in3, start_dim=1), torch.flatten(n3rand, start_dim=1)).mean()
                        i_loss += in3_loss
                    if depth > 3:
                        in4_loss = util.normal_lt_loss(torch.flatten(in4, start_dim=1), torch.flatten(n4rand, start_dim=1)).mean()
                        i_loss += in4_loss
                    if depth > 4:
                        in5_loss = util.normal_lt_loss(torch.flatten(in5, start_dim=1), torch.flatten(n5rand, start_dim=1)).mean()
                        i_loss += in5_loss
                    # iz_loss = F.kl_div(iz, zrand, size_average=None, reduce=None, reduction='mean', log_target=False)
                    # in0_loss = F.kl_div(in0, n0rand, size_average=None, reduce=None, reduction='mean', log_target=False)
                    # i_loss = iz_loss + in0_loss 
                    # if depth >0:
                    #     in1_loss = in0_loss = F.kl_div(in1, n1rand, size_average=None, reduce=None, reduction='mean', log_target=False)
                    #     i_loss += in1_loss
                    # if depth > 1:
                    #     in2_loss = in0_loss = F.kl_div(in2, n2rand, size_average=None, reduce=None, reduction='mean', log_target=False)
                    #     i_loss += in2_loss
                    # if depth > 2:
                    #     in3_loss = in0_loss = F.kl_div(in3, n3rand, size_average=None, reduce=None, reduction='mean', log_target=False)
                    #     i_loss += in3_loss
                    # if depth > 3:
                    #     in4_loss = in0_loss = F.kl_div(in4, n4rand, size_average=None, reduce=None, reduction='mean', log_target=False)
                    #     i_loss += in4_loss
                    # if depth > 4:
                    #     in5_loss = in0_loss = F.kl_div(in5, n5rand, size_average=None, reduce=None, reduction='mean', log_target=False)
                    #     i_loss += in5_loss


                    # F.kl_div(iz, zrand, size_average=None, reduce=None, reduction='mean', log_target=False)




                    # loss = br * rec_loss + bz * zkl + b0 * n0kl + b1 * n1kl + b2 * n2kl + b3 * n3kl + b4 * n4kl + b5 * n5kl
                    # loss = loss.mean(dim=0)
                    # loss = br * i_loss + bz * zkl + b0 * n0kl + b1 * n1kl + b2 * n2kl + b3 * n3kl + b4 * n4kl + b5 * n5kl
                    epoch_loss[2] += i_loss.mean(dim=0).item()
                    loss = i_loss.mean(dim=0)

                    # i_loss = iz_loss.mean(dim=0)
                    # print(i_loss)

                    # if i%720 == 0:
                    #     print('PER: ', perceptual_loss)
                    #     print('REC: ', rec_loss)
                    #     print("Z KL: ", zkl)
                    #     print('NO-N5 KL: ', n0kl, n1kl, n2kl, n3kl, n4kl, n5kl)
                    #     print('MEAN: ', loss)

                    instances_seen += input.size(0)

                    # tbw.add_scalar('style-vae/zkl-loss', float(zkl.data.mean(dim=0).item()), instances_seen)
                    # tbw.add_scalar('style-vae/n0kl-loss', float(n0kl.data.mean(dim=0).item()), instances_seen)
                    # tbw.add_scalar('style-vae/n1kl-loss', float(n1kl.data.mean(dim=0).item()), instances_seen)
                    # tbw.add_scalar('style-vae/n2kl-loss', float(n2kl.data.mean(dim=0).item()), instances_seen)
                    # tbw.add_scalar('style-vae/n3kl-loss', float(n3kl.data.mean(dim=0).item()), instances_seen)
                    # tbw.add_scalar('style-vae/n4kl-loss', float(n4kl.data.mean(dim=0).item()), instances_seen)
                    # tbw.add_scalar('style-vae/n5kl-loss', float(n5kl.data.mean(dim=0).item()), instances_seen)
                    # tbw.add_scalar('style-vae/rec-loss', float(rec_loss.data.mean(dim=0).item()), instances_seen)
                    # tbw.add_scalar('style-vae/total-loss', float(loss.data.item()), instances_seen)

                    # Backward pass

                    assert torch.isnan(loss).sum() == 0
                    assert torch.isinf(loss).sum() == 0

                    loss.backward()
                    opte.step()
                    opte.zero_grad()


                # optimizer.zero_grad()

                # loss.backward()

                # optimizer.step()
            print([int(e) for e in epoch_loss])

            if arg.epochs[depth] <= arg.np or epoch % (arg.epochs[depth]//arg.np) == 0 or epoch == arg.epochs[depth] - 1:
                with torch.no_grad():
                    err_te = []
                    encoder.train(False)
                    decoder.train(False)

                    if not arg.skip_test:
                        # for i, (input, _) in enumerate(testloader):
                        for i, (input, _) in enumerate(tqdm.tqdm(testloader)):
                            if arg.limit is not None and i * arg.batch_size > arg.limit:
                                break

                            if torch.cuda.is_available():
                                input = input.cuda()

                            b = input.size(0)

                            # -- encoding
                            z, n0, n1, n2, n3, n4, n5 = encoder(input, depth)

                            # -- compute KL losses

                            # zkl  = util.kl_loss(z[:, :zs], z[:, zs:])
                            # n0kl = util.kl_loss_image(n0)
                            # n1kl = util.kl_loss_image(n1)
                            # n2kl = util.kl_loss_image(n2)
                            # n3kl = util.kl_loss_image(n3)
                            # n4kl = util.kl_loss_image(n4)
                            # n5kl = util.kl_loss_image(n5)

                            # -- take samples
                            zsample  = util.sample(z[:, :zs], z[:, zs:])
                            n0sample = util.sample_image(n0)
                            n1sample = util.sample_image(n1)
                            n2sample = util.sample_image(n2)
                            n3sample = util.sample_image(n3)
                            n4sample = util.sample_image(n4)
                            n5sample = util.sample_image(n5)

                            # -- decoding
                            xout = decoder(zsample, n0sample, n1sample, n2sample, n3sample, n4sample, n5sample)

                            perceptual_loss = 0
                            if arg.perceptual_loss:
                                perceptual_input = perceptual_loss_model(input)
                                perceptual_output = perceptual_loss_model(xout)
                                perceptual_loss = F.mse_loss(perceptual_input, perceptual_output, reduction='none').view(b, -1).sum(dim=1)


                            # m = ds.Normal(xout[:, :C, :, :], xout[:, C:, :, :])
                            # rec_loss = -m.log_prob(target).sum(dim=1).sum(dim=1).sum(dim=1)

                            rec_loss = util.normal_im(xout, input).view(b, c*h*w).sum(dim=1)
                            loss = rec_loss.mean(dim=0)

                            err_te.append(loss.data.item())

                        tbw.add_scalar('pixel-models/test-loss', sum(err_te)/len(err_te), epoch)
                        print('epoch={:02}; test loss: {:.3f}'.format(
                            epoch, sum(err_te)/len(err_te)))

                    # take some samples

                    # sample 6x12 images
                    b = 6*12

                    zrand, (n0rand, n1rand, n2rand, n3rand, n4rand, n5rand) = util.latent_sample(b,\
                        zsize=arg.latent_size, outsize=(C, H, W), zchannels=arg.zchannels, \
                        dev='cuda', depth=depth)

                    sample = util.batchedn((zrand, n0rand, n1rand, n2rand, n3rand, n4rand, n5rand), decoder, batch_size=8).clamp(0, 1)[:, :C, :, :]

                    # reconstruct 6x12 images from the testset
                    input = util.readn(testloader, n=6*12)
                    if torch.cuda.is_available():
                        input = input.cuda()

                    # -- encoding
                    z, n0, n1, n2, n3, n4, n5 = util.nbatched(input, encoder, batch_size=32, depth=depth)

                    # -- take samples
                    zsample = util.sample(z[:, :zs], z[:, zs:])
                    n0sample = util.sample_image(n0)
                    n1sample = util.sample_image(n1)
                    n2sample = util.sample_image(n2)
                    n3sample = util.sample_image(n3)
                    n4sample = util.sample_image(n4)
                    n5sample = util.sample_image(n5)

                    # -- decoding
                    xout = util.batchedn((zsample, n0sample, n1sample, n2sample, n3sample, n4sample, n5sample), decoder, batch_size=4).clamp(0, 1)[:, :C, :, :]
                    # -- mix the latent vector with random noise

                    mixout = util.batchedn((zsample, n0rand, n1rand, n2rand, n3rand, n4rand, n5rand), decoder, batch_size=4).clamp(0, 1)[:, :C, :, :]

                    # -- mix a random vector with the sample noise
                    mixout2 = util.batchedn((zrand, n0sample, n1sample, n2sample, n3sample, n4sample, n5sample), decoder, batch_size=4).clamp(0, 1)[:, :C, :, :]



                    # xout = torch.sigmoid(xout)
                    # mixout = torch.sigmoid(mixout)
                    # mixout2 = torch.sigmoid(mixout2)
                    # sample = torch.sigmoid(sample)


                    images = torch.cat([input.cpu()[:24,:,:], xout[:24,:,:], mixout[:24,:,:], mixout2[:24,:,:], sample[:24,:,:],
                                        input.cpu()[24:48,:,:], xout[24:48,:,:], mixout[24:48,:,:], mixout2[24:48,:,:], sample[24:48,:,:],
                                        input.cpu()[48:,:,:], xout[48:,:,:], mixout[48:,:,:], mixout2[48:,:,:], sample[48:,:,:]], dim=0)

                    # sys.exit()
                    utils.save_image(images, f'images.{depth}.{epoch}.png', nrow=24, padding=2)

                    slack_util.send_message(f'Epoch {epoch} Depth {depth} Finished\n options: {arg}')
                    slack_util.send_image(f'images.{depth}.{epoch}.png', f'Depth {depth, }Epoch: {epoch}')
                    # utils.save_image(input.cpu(), f'images_input.{depth}.{epoch}.png', nrow=3, padding=2)
                    # utils.save_image(xout, f'images_xout_recon.{depth}.{epoch}.png', nrow=3, padding=2)
                    # utils.save_image(mixout, f'images_mixout_lv_rn.{depth}.{epoch}.png', nrow=3, padding=2)
                    # utils.save_image(mixout2, f'images_mixout2_rv_sn.{depth}.{epoch}.png', nrow=3, padding=2)
                    # utils.save_image(sample, f'images_sample_total_random.{depth}.{epoch}.png', nrow=3, padding=2)



if __name__ == "__main__":
    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task: [mnist, cifar10].",
                        default='mnist', type=str)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Epoch schedule per depth.",
                        nargs=6,
                        default=[1, 2, 3, 6, 12, 12],
                        type=int)

    parser.add_argument("-c", "--channels",
                        dest="channels",
                        help="Number of channels per block (list of 5 integers).",
                        nargs=5,
                        default=[32, 64, 128, 256, 512],
                        type=int)

    parser.add_argument("--zchannels",
                        dest="zchannels",
                        help="Number of channels per noise input.",
                        nargs=6,
                        default=[1, 2, 4, 8, 16, 32],
                        type=int)

    parser.add_argument("--skip-test",
                        dest="skip_test",
                        help="Skips evaluation on the test set (but still takes a sample).",
                        action='store_true')

    parser.add_argument("--batch-norm",
                        dest="batch_norm",
                        help="Adds batch normalization after each block.",
                        action='store_true')

    parser.add_argument("--evaluate-every",
                        dest="eval_every",
                        help="Run an exaluation/sample every n epochs.",
                        default=1, type=int)

    parser.add_argument("-k", "--kernel_size",
                        dest="kernel_size",
                        help="Size of convolution kernel",
                        default=3, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Size of the batches.",
                        default=32, type=int)

    parser.add_argument("-z", "--latent-size",
                        dest="latent_size",
                        help="Size of latent space.",
                        default=128, type=int)

    parser.add_argument('--betas',
                        dest='betas',
                        help="Scaling parameters of the kl losses. The first two are for reconstruction loss and the z parameter, the rest are for the noise parameters in order. Provide exactly 7 floats.",
                        nargs=8,
                        type=float,
                        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    parser.add_argument('--dropouts',
                        dest='dropouts',
                        help="Dropout parameters for the various decoder inputs.",
                        nargs=7,
                        type=float,
                        default=None)

    parser.add_argument("--limit",
                        dest="limit",
                        help="Limit on the number of instances seen per epoch (for debugging).",
                        default=None, type=int)

    parser.add_argument("--mapping-layers",
                        dest="mapping_layers",
                        help="Number of layers mapping from and to the distribution on z.",
                        default=3, type=int)

    parser.add_argument("--numplots",
                        dest="np",
                        help="Number of plots per depth.",
                        default=8, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate.",
                        default=0.001, type=float)

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/style', type=str)

    parser.add_argument("-PL", "--perceptual-loss",
                        dest="perceptual_loss",
                        help="Use perceptual/feature loss. Options: DenseNet, AlexNet. Default: None",
                        default=None, type=str)

    parser.add_argument("-EU", "--encoder-update-per-iteration",
                        dest="encoder_update_per_iteration",
                        help="Amount of times the encoder is updated each iteration. (sleep phase).",
                        default=1, type=int)


    options = parser.parse_args()

    print('OPTIONS', options)

    slack_util.send_message(f"Run Started.\nOPTIONS:\n{options}")
    go(options)
    print('Finished succesfully')