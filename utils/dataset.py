import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar_ddp(args):
    mean, std = np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0

    train_transform = transforms.Compose([
        torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=test_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, sampler=train_sampler, pin_memory=True)

    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, sampler=test_sampler, pin_memory=True)

    return trainloader, testloader



