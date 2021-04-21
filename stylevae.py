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
from encoder import StyleEncoder, StyleEncoder2
from decoder import StyleDecoder, StyleDecoder2
from discriminator import Discriminator
import slack_util

from tensorboardX import SummaryWriter
# from layers import PlainMaskedConv2d, MaskedConv2d

SEEDFRAC = 2
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def go(arg):

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    br, bz, b0, b1, b2, b3, b4, b5, bg = arg.betas
    bsz, bs0, bs1, bs2, bs3, bs4, bs5 = arg.sleep_betas
    # bz_list = torch.arange(0, bz, step = bz / torch.sum(torch.tensor(arg.epochs)))
    C, H, W, trainset, trainloader, testset, testloader = return_data(arg.task, arg.data_dir, arg.batch_size)
    zs = arg.latent_size

    if arg.encoder_type == 1:
        encoder = StyleEncoder((C, H, W), arg.channels, arg.zchannels, zs=zs, k=arg.kernel_size, unmapping=arg.mapping_layers, batch_norm=arg.batch_norm)
    elif arg.encoder_type == 2:
        encoder = StyleEncoder2((C, H, W), arg.channels, arg.zchannels, zs=zs, k=arg.kernel_size, unmapping=arg.mapping_layers, batch_norm=arg.batch_norm)

    if arg.decoder_type == 1:
        decoder = StyleDecoder2((C, H, W), arg.channels, arg.zchannels, zs=zs, k=arg.kernel_size, mapping=arg.mapping_layers, batch_norm=arg.batch_norm, dropouts=arg.dropouts)
    elif arg.decoder_type == 2:
        decoder = StyleDecoder2((C, H, W), arg.channels, arg.zchannels, zs=zs, k=arg.kernel_size, mapping=arg.mapping_layers, batch_norm=arg.batch_norm, dropouts=arg.dropouts)

    discriminator = Discriminator((C,H,W), arg.dchannels)

    if arg.output_distribution == 'siglaplace':
        rec_criterion = util.siglaplace
    elif arg.output_distribution == 'signorm':
        rec_criterion = util.signorm

    discriminator_criterion = nn.BCELoss()

    opte = Adam(list(encoder.parameters()), lr=arg.lr[0])
    optd = Adam(list(decoder.parameters()), lr=arg.lr[1])
    optdi = Adam(list(discriminator.parameters()), lr=arg.lr[2])

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()

    for depth in range(6):

        ## CLASSIV VAE

        print(f'starting depth {depth}, for {arg.epochs[depth]} epochs')
        print('\tLoss\t\tREC\tKL\tZ\tN0\tN1\tN2\tN3\tN4\tN5\tSleep\tREC_RN')
        
        for epoch in range(arg.epochs[depth]):

            # bz = bz_list[epoch]
            epoch_loss = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]


            # Train
            encoder.train(True)
            decoder.train(True)
            discriminator.train(True)

            for i, (input, _) in enumerate(trainloader):
                assert torch.isnan(input).sum() == 0
                assert torch.isinf(input).sum() == 0

                # Prepare the input
                b, c, w, h = input.size()
                if torch.cuda.is_available():
                    input = input.cuda()               


                ## NORMAL VAE
                # -- encoding
                z = encoder(input, depth)

                assert torch.isnan(z).sum() == 0
                assert torch.isinf(z).sum() == 0

                # -- take sample
                zsample  = util.sample(z[:, :zs], z[:, zs:])
        
                assert torch.isnan(zsample).sum() == 0; assert torch.isinf(zsample).sum() == 0

                # -- reconstruct input
                xout = decoder(zsample, depth)
                with torch.no_grad():
                    xout_no_grad = decoder(zsample, depth)

                assert torch.isnan(xout).sum() == 0
                assert torch.isinf(xout).sum() == 0


                ## DISCRIMINATOR
                real_label = torch.full((b,), 1, dtype=torch.float, device=dev)
                fake_label = torch.full((b,), 0, dtype=torch.float, device=dev)
                discriminator_out_real = discriminator(input).view(-1)
                discriminator_out_fake = discriminator(xout_no_grad[:, :C, :, :]).view(-1)
                

                # -- compute losses discriminator
                discriminator_loss_real = discriminator_criterion(discriminator_out_real, real_label)
                discriminator_loss_fake = discriminator_criterion(discriminator_out_fake, fake_label)
                discriminator_loss_real.backward()
                discriminator_loss_fake.backward()
                discriminator_loss = discriminator_loss_real.mean() + discriminator_loss_fake.mean()
                optdi.step()
                optdi.zero_grad()

                # -- compute losses decoder
                rec_loss = rec_criterion(xout, input).view(b, c*h*w)
                rec_loss = rec_loss.mean(dim=1)

                    #disc updated, so second pass trough disc.
                with torch.no_grad():
                    discriminator_out_fake = discriminator(xout[:, :C, :, :]).view(-1)
                generator_loss = discriminator_criterion(discriminator_out_fake, real_label)

                # --compute losses encoder
                zkl  = util.kl_loss(z[:, :zs], z[:, zs:])
                kl_loss = bz * zkl

                loss = br*rec_loss + kl_loss + bg * generator_loss
                loss = loss.mean(dim=0)

                # -- backward pass and update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

                optd.step(); optd.zero_grad()
                opte.step(); opte.zero_grad()

                for ep in encoder.parameters():
                    if torch.isnan(ep).sum() != 0 or torch.isinf(ep).sum() != 0:
                        print(loss, rec_loss, kl_loss)
                    assert torch.isnan(ep).sum() == 0
                    assert torch.isinf(ep).sum() == 0

                for dp in decoder.parameters():
                    assert torch.isnan(dp).sum() == 0
                    assert torch.isinf(dp).sum() == 0


                if arg.use_sleep_update:
                 # SLEEP UPDATE

                        # -- sample random latent
                    zrand, (_, _, _, _, _) = util.latent_sample(b, zs, (C, H, W), depth, arg.zchannels, dev)
                    assert torch.isnan(zrand).sum() == 0
                    assert torch.isinf(zrand).sum() == 0

                    # -- generate x from latent
                    with torch.no_grad():
                        x = decoder(zsample, depth)
                        assert torch.isnan(x).sum() == 0
                        assert torch.isinf(x).sum() == 0
                        xsample = util.sample_images(x, arg.output_distribution)
                        assert torch.isnan(xsample).sum() == 0
                        assert torch.isinf(xsample).sum() == 0

                    # -- reconstruct latent
                    z_prime = encoder(xsample, depth)
                    assert torch.isnan(z_prime).sum() == 0
                    assert torch.isinf(z_prime).sum() == 0

                    # -- compute loss
                    sleep_z = util.sleep_loss(z_prime, zrand)

                    sleep_loss = bsz * sleep_z  
                    assert torch.isnan(sleep_loss).sum() == 0
                    assert torch.isinf(sleep_loss).sum() == 0
                    sleep_loss = sleep_loss.mean()

                    # -- Backward pas
                    sleep_loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
                    opte.step()
                    opte.zero_grad()


                    for ep in encoder.parameters():
                        assert torch.isnan(ep).sum() == 0
                        assert torch.isinf(ep).sum() == 0

                    # -- administration
                with torch.no_grad():
                    epoch_loss[0] += loss.mean(dim=0).item()
                    epoch_loss[1] += rec_loss.mean(dim=0).item() if not arg.train_recon_with_rn else + rec_loss_orignal.mean().item()
                    epoch_loss[2] += kl_loss.mean(dim=0).item()
                    epoch_loss[3] += zkl.mean(dim=0).item()
                    # epoch_loss[4] += n0kl.mean(dim=0).item()
                    # if depth >= 1: epoch_loss[5] += n1kl.mean(dim=0).item()
                    # if depth >= 2: epoch_loss[6] += n2kl.mean(dim=0).item()
                    # if depth >= 3: epoch_loss[7] += n3kl.mean(dim=0).item()
                    # if depth >= 4: epoch_loss[8] += n4kl.mean(dim=0).item()
                    # if depth >= 5: epoch_loss[9] += n5kl.mean(dim=0).item()
                    if arg.use_sleep_update: epoch_loss[10] += sleep_loss.mean(dim=0).item()
                    if arg.train_recon_with_rn: epoch_loss[11] += rec_loss_rn.mean().item()
                    epoch_loss[12] += discriminator_loss.mean(dim=0).item()
                    epoch_loss[13] += generator_loss.mean(dim=0).item()


   
            print(f'Epoch {epoch}\t','\t'.join([str(int(e)) for e in epoch_loss]))
            string = f'Epoch {epoch}:\t\n' +'\t'.join([str(int(e)) for e in epoch_loss])

            ## MAKE PLOTS

            if arg.epochs[depth] <= arg.np or epoch % (arg.epochs[depth]//arg.np) == 0 or epoch == arg.epochs[depth] - 1:
                # if depth ==5:
                with torch.no_grad():
                    err_te = []
                    encoder.train(False)
                    decoder.train(False)

                    ## sample 6x12 images
                    b = 6*12

                    # -- sample latents
                    zrand, (_, _, _, _, _, _) = util.latent_sample(b, zs, (C, H, W), depth, arg.zchannels, dev)

                    # -- construct output
                    sample = decoder(zrand, depth).clamp(0, 1)[:, :C, :, :]

                    ## reconstruct 6x12 images from the testset
                    input = util.readn(testloader, n=6*12)
                    if torch.cuda.is_available():
                        input = input.cuda()

                    # -- encoding
                    z = encoder(input, depth)

                    # -- take samples
                    zsample = util.sample(z[:, :zs], z[:, zs:])

                    # -- decoding
                    xout = decoder(zsample, depth).clamp(0, 1)[:, :C, :, :]

                    images = torch.cat([input.cpu()[:24,:,:], xout.cpu()[:24,:,:], sample.cpu()[:24,:,:],
                                        input.cpu()[24:48,:,:], xout.cpu()[24:48,:,:], sample.cpu()[24:48,:,:],
                                        input.cpu()[48:,:,:], xout.cpu()[48:,:,:], sample.cpu()[48:,:,:]], dim=0)

                    # -- save and slack images
                    try:
                        utils.save_image(images, f'images.{depth}.{epoch}.png', nrow=24, padding=2)
                    except Exception as e:
                        print("Saving images.{depth}.{epoch}.png failed.")
                    slack_util.send_message(f' Depth {depth}, Epoch {epoch}. \nOptions: {arg}')
                    slack_util.send_message(string)
                    slack_util.send_image(f'images.{depth}.{epoch}.png', f'Depth {depth}, Epoch: {epoch}')



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
    parser.add_argument("--dchannels",
                        dest="dchannels",
                        help="Number of channels discriminator(list of 4 integers).",
                        nargs=4,
                        default=[32, 64, 128, 256],
                        type=int)

    parser.add_argument("--zchannels",
                        dest="zchannels",
                        help="Number of channels per noise input.",
                        nargs=6,
                        default=[1, 1, 1, 1, 1, 1],
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
                        nargs=9,
                        type=float,
                        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    parser.add_argument('--sleep-betas',
                        dest='sleep_betas',
                        help="Scaling parameters of the sleep losses.",
                        nargs=7,
                        type=float,
                        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

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
                        nargs=3,
                        default=[0.0001, 0.0001, 0.001], type=float)

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

    parser.add_argument("-DU", "--decoder-update-per-iteration",
                        dest="decoder_update_per_iteration",
                        help="Amount of times the decoder is updated each iteration. (wake phase).",
                        default=1, type=int)

    parser.add_argument("-EN", "--encoder",
                        dest="encoder_type",
                        help="Endoder 1 or 2",
                        default=1, type=int)

    parser.add_argument("-DE", "--decoder",
                        dest="decoder_type",
                        help="Decoder 1 or 2",
                        default=1, type=int)

    parser.add_argument("-OD", "--output-distribution",
                    dest="output_distribution",
                    help="Output distribution ",
                    default='siglaplace', type=str)

    parser.add_argument("--train-recon-with-rn",
                    dest="train_recon_with_rn",
                    help="Train recon with RN",
                    default=None, type=int)

    parser.add_argument("--use-sleep-update",
                    dest="use_sleep_update",
                    help="Uses sleep update",
                    action='store_true')

    options = parser.parse_args()


    print('OPTIONS', options)

    slack_util.send_message(f"Run Started.\nOPTIONS:\n{options}")

    try:
        go(options)
    except Exception as e:
        slack_util.send_message(f"Run Failed.\n{e}")
        raise

    print('Finished succesfully')