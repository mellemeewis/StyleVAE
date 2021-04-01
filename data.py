import torch
import torchvision
import numpy as np
import os
from torchvision.transforms import ToTensor, Compose, Pad, Grayscale, RandomHorizontalFlip, Normalize, ToPILImage
from six.moves import urllib


def return_data(task, data_dir, batch_size):


	opener = urllib.request.build_opener()
	opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
	urllib.request.install_opener(opener)
	## Load the data


	if task == 'mnist':

		transform = Compose([Normalize((20000), (3202340), inplace=True), Pad(2, fill=0, padding_mode='constant'), ToTensor()])

		trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
		print('Min Pixel Value: {} \nMax Pixel Value: {}'.format(trainset.data.min(), trainset.data.max()))
		print('Mean Pixel Value {} \nPixel Values Std: {}'.format(trainset.data.float().mean(), trainset.data.float().std()))
		print('Scaled Mean Pixel Value {} \nScaled Pixel Values Std: {}'.format(trainset.data.float().mean() / 255, trainset.data.float().std() / 255))

		# transform = Compose([Pad(2, fill=0, padding_mode='constant'), ToTensor()])
		# trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
		#                                         download=True, transform=transform)
		# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
		#                                           shuffle=True, num_workers=2)

		# testset = torchvision.datasets.MNIST(root=data_dir, train=False,
		#                                        download=True, transform=transform)
		# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
		#                                          shuffle=False, num_workers=2)

		C, H, W = 1, 32, 32
	elif task == 'cifar10':
		trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
		                                        download=True, transform=ToTensor())
		trainset.targets = torch.tensor(trainset.targets)
		idx = trainset.targets==7
		trainset.targets= trainset.targets[idx]
		trainset.data = trainset.data[idx.numpy().astype(np.bool)]

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
		                                       download=True, transform=ToTensor())

		testset.targets = torch.tensor(testset.targets)
		idx = testset.targets==7
		testset.targets= testset.targets[idx]
		testset.data = testset.data[idx.numpy().astype(np.bool)]

		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
		                                         shuffle=False, num_workers=2)



		
		C, H, W = 3, 32, 32

	elif task == 'cifar-gs':
		transform = Compose([Grayscale(), ToTensor()])

		trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
		                                        download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
		                                       download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
		                                         shuffle=False, num_workers=2)
		C, H, W = 1, 32, 32

	elif task == 'celebA':
		transform = Compose([ToTensor()])

		trainset = torchvision.datasets.ImageFolder(root=data_dir, transform=ToTensor())
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.ImageFolder(root=data_dir+os.sep+'test', transform=ToTensor())
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
		                                         shuffle=False, num_workers=2)
		C, H, W = 3, 32, 32

	elif task == 'imagenet64':
		transform = Compose([ToTensor()])

		trainset = torchvision.datasets.ImageFolder(root=data_dir+os.sep+'train',
		                                            transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.ImageFolder(root=data_dir+os.sep+'valid',
		                                           transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
		                                         shuffle=False, num_workers=2)
		C, H, W = 3, 64, 64

	elif task == 'ffhq':
		tftrain = Compose([RandomHorizontalFlip(0.5), ToTensor()])
		trainset = torchvision.datasets.ImageFolder(root=data_dir,
		                                            transform=tftrain)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
		                                          shuffle=True, num_workers=2)

		tftest = Compose([ToTensor()])
		testset = torchvision.datasets.ImageFolder(root=data_dir+os.sep,
		                                           transform=tftest)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
		                                         shuffle=False, num_workers=2)
		C, H, W = 3, 128, 128

	else:
		raise Exception('Task {} not recognized.'.format(task))

	return C, H, W, trainset, trainloader, testset, testloader
