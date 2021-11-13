import os
import pathlib
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from htorch import utils


def results_dir():
    """
    Specify where to save the results.
    """
    return os.path.join(pathlib.Path.home(), 'Documents', 'code', 'results')


def archive_dir():
    """
    Directory where results of previous experiments are stored.
    """
    return os.path.join(pathlib.Path.home(), 'Documents', 'code',
                        'results_archive')


def dataset_dir():
    """
    Specify where to save the datasets.
    """
    return os.path.join(pathlib.Path.home(), 'Documents', 'code', 'datasets')


def data_loader(model_to_run, dataset, batch_size):
    """
    Data loader function.
    Has different conditions for different datasets
    and both real and quaternion networks.
    """
    if dataset == 'cifar10':
        data_directory = os.path.join(dataset_dir(), 'cifar10')

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=data_directory,
            train=True,
            download=False,
            transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_directory,
            train=False,
            download=False,
            transform=test_transform
        )

        if model_to_run == 'quat':
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=utils.convert_data_for_quaternion
            )
            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=utils.convert_data_for_quaternion
            )

        else:
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )

    elif dataset == 'cifar100':
        data_directory = os.path.join(dataset_dir(), 'cifar100')

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        trainset = torchvision.datasets.CIFAR100(
            root=data_directory,
            train=True,
            download=False,
            transform=train_transform
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_directory,
            train=False,
            download=False,
            transform=test_transform
        )

        if model_to_run == 'quat':
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=utils.convert_data_for_quaternion
            )
            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=utils.convert_data_for_quaternion
            )

        else:
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )

    elif dataset == 'mnist':
        data_directory = os.path.join(dataset_dir(), 'mnist')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])

        trainset = torchvision.datasets.MNIST(
            root=data_directory,
            train=True,
            download=False,
            transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=data_directory,
            train=False,
            download=False,
            transform=transform
        )

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

    elif dataset == 'pascal':
        data_directory = os.path.join(dataset_dir(), 'pascal')

        image_transform = transforms.Compose([
            transforms.CenterCrop(size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        mask_transform = transforms.Compose([
            transforms.CenterCrop(size=(256, 256)),
            transforms.ToTensor()
        ])

        trainset = torchvision.datasets.VOCSegmentation(
            root=data_directory,
            year='2012',
            image_set='train',
            download=False,
            transform=image_transform,
            target_transform=mask_transform
        )
        testset = torchvision.datasets.VOCSegmentation(
            root=data_directory,
            year='2012',
            image_set='val',
            download=False,
            transform=image_transform,
            target_transform=mask_transform
        )

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

    else:
        raise ValueError("Dataset not known.")

    return trainloader, testloader


def format_text(input, length=40, heading=True):
    """
    Input has to be a string with even length.
    """
    input_len = len(input)
    if input_len % 2 != 0:
        raise ValueError("The input is not of even length.")
    else:
        num_dashes = int((length - len(input) - 4) / 2)
        if heading:
            return ('-'*length + '\n' + '-'*num_dashes + '  ' + input
                    + '  ' + '-'*num_dashes + '\n' + '-'*length + '\n')
        else:
            return ('-'*num_dashes + '  ' + input + '  '
                    + '-'*num_dashes + '\n')


def display_model(model: nn.Module, output_directory=None, show=True):
    """
    Function to print the structure of the neural
    network model (its constituent layers and the
    number of parameters in each) as a table.
    """
    table = PrettyTable(["Layers", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    table.add_row(["Total trainable parameters", total_params])

    if show:
        print(format_text("Model statistics"))
        print(table)
        print('\n')

    if output_directory:
        file_path = os.path.join(output_directory, 'model_structure.txt')
        with open(file_path, 'w') as file:
            file.write(str(table))


def get_trainable_params(model: nn.Module):
    """
    Returns the number of trainable parameters in the network.
    """
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_param
