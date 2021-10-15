import csv
import pickle
import os
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import torch.nn.utils.prune as prune

from htorch import utils

class ModifiedDataset(Dataset):
    """
    Custom Dataset class for modified dataset
    for the quaternion network.
    """
    def __init__(self, file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def data_loader(model_to_run, dataset, batch_size):
    """
    Data loader function for both qcnn and cnn.
    """
    if dataset == 'cifar10':
        data_directory = os.path.join('open_lth_datasets', 'cifar10')

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

        trainset = torchvision.datasets.CIFAR10(root=data_directory, train=True,
                download=False, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=data_directory, train=False,
                download=False, transform=test_transform)

        if model_to_run == 'quaternion':
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
        data_directory = os.path.join('open_lth_datasets', 'mnist')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])

        trainset = torchvision.datasets.MNIST(root=data_directory, train=True,
                download=False, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_directory, train=False,
                download=False, transform=transform)

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
    input has to be a string with even length.
    """
    input_len = len(input)
    if input_len % 2 != 0:
        raise ValueError("The input is not of even length.")
    else:
        num_dashes = int((length - len(input) - 4) / 2)
        if heading:
            return ('\n' + '-'*length + '\n' + '-'*num_dashes + '  ' + input + '  '
                    + '-'*num_dashes + '\n' + '-'*length + '\n')
        else:
            return ('\n' + '-'*num_dashes + '  ' + input + '  '
                    + '-'*num_dashes + '\n')


def display_model(model: nn.Module, output_directory=None):
    """
    Function to print the structure of the neural
    network model (its constituent layers and the
    number of parameters in each) as a table.
    """
    table = PrettyTable(["Layers", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    table.add_row(["Total trainable parameters", total_params])
    print(format_text("Model statistics"))
    print(table)

    if output_directory:
        file_path = os.path.join(output_directory, 'model_structure.txt')
        with open (file_path, 'w') as file:
            file.write(str(table))


def get_trainable_params(model: nn.Module):
    """
    Returns the number of trainable parameters in the network.
    """
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_param


def train_model(model, trainloader, testloader, optimizer, criterion, num_epochs, device, mini_batch, output_directory=None):
    """
    Function to train a model.
    """
    log_output = []
    for epoch in range(num_epochs):
        if epoch == 0:
            accuracy = test_model(model, testloader, device)
            print("ep  {:03d}  loss    {:.3f}  acc  {:.3f}%".format(epoch, 0, accuracy))

        epoch_loss = 0.0
        mini_batch_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            images, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print training statistics
            mini_batch_loss += loss.item()
            epoch_loss += mini_batch_loss

            if (i + 1) % mini_batch == 0:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, mini_batch_loss/mini_batch))
                mini_batch_loss = 0.0

        # Test accuracy at the end of each epoch
        accuracy = test_model(model, testloader, device)
        log_output.append([epoch + 1, epoch_loss / len(trainloader), accuracy])

        print("ep  {:03d}  loss  {:.3f}  acc  {:.3f}%".format(epoch + 1, epoch_loss / len(trainloader), accuracy))

    if output_directory:
        file_path = os.path.join(output_directory, 'logger.csv')
        with open(file_path, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerows(log_output)
    
    print("\nTraining complete.")


def test_model(model, testloader, device):
    """
    Function to train a model.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    
    return accuracy


def layer_sparsity(layer: nn.Module):
    """
    Returns the sparsity of a given layer.
    """
    zeros = 0
    total = 0

    for name, parameter in layer.named_parameters():
        total += parameter.numel()

    for name, buffer in layer.named_buffers():
        zeros += torch.sum(buffer == 0).item()
    
    sparsity = 100 * (total - zeros) / total
    
    return sparsity, zeros, total


def sparsity_check(model: nn.Module, output_directory=None):
    """
    Returns the sparsity of the entire model.
    """
    table = PrettyTable(["Layers", "Percentage of weights left"])
    zeros = 0
    total = 0

    for layer_name, layer in model.named_modules():
        if layer_name not in ['', 'pool', 'abs']:
            module_sparsity, layer_zeros, layer_total = layer_sparsity(layer)
            zeros += layer_zeros
            total += layer_total
            table.add_row([layer_name, '{:.3f}'.format(module_sparsity)])

    sparsity = 100 * (total - zeros) / total

    table.add_row(["Total", '{:.3f}'.format(sparsity)])

    print(format_text("Model Sparsity", heading=False))
    print(table)

    if output_directory:
        file_path = os.path.join(output_directory, 'sparsity_report.txt')
        with open (file_path, 'w') as file:
            file.write(str(table))

def prune_model(parameters_to_prune, percentage, iterations, model, trainloader, testloader,
                optimizer, criterion, num_epochs, device, mini_batch, output_directory):
    """
    Function to iteratively prune the given model.
    """
    iter_percentage = percentage # percentage ** (1 / iterations)
    iter_percentage = 1 - (1 - percentage) ** (1 / iterations)

    for i in range(iterations):
        print(format_text("Pruning iteration {:02d}".format(i + 1)))
        iter_directory = os.path.join(output_directory, f"Level {i + 1}")
        os.mkdir(iter_directory)

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=iter_percentage,
        )
        train_model(model, trainloader, testloader, optimizer, criterion, num_epochs, device, mini_batch, iter_directory)
        sparsity_check(model, iter_directory)
