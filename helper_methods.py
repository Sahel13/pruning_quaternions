import os
import csv
import pathlib
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms

from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.handlers.param_scheduler import LRScheduler

from htorch import utils


def results_dir():
    """
    Specify where to save the results.
    """
    return os.path.join(pathlib.Path.home(), 'Documents', 'code', 'results')


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


def train_model(
        model, trainloader, testloader, optimizer,
        criterion, num_epochs, device,
        output_directory=None, scheduler=None, retrain=False):
    """
    Function to train a model.
    """
    log_output = []
    final_accuracy = 0.0
    for epoch in range(num_epochs):
        if epoch == 0:
            accuracy = test_model(model, testloader, device)
            print("ep  {:03d}  loss    {:.3f}  acc  {:.3f}%".format(epoch,
                  0, accuracy))
            final_accuracy = accuracy

        epoch_loss = 0.0
        for _, data in enumerate(trainloader, 0):

            images, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Test accuracy at the end of each epoch
        accuracy = test_model(model, testloader, device)
        log_output.append([epoch + 1, epoch_loss / len(trainloader), accuracy])

        print("ep  {:03d}  loss  {:.3f}  acc  {:.3f}%".format(
            epoch + 1, epoch_loss / len(trainloader), accuracy))

        if scheduler:
            scheduler.step()

        final_accuracy = accuracy

    if output_directory:
        if not retrain:
            file_path = os.path.join(output_directory, 'logger.csv')
            weight_path = os.path.join(output_directory, 'weights.pth')
        else:
            file_path = os.path.join(output_directory, 'logger_retrain.csv')
            weight_path = os.path.join(output_directory, 'weights_retrain.pth')

        # Save the training log
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(log_output)

        # Save the weights
        torch.save(model.state_dict(), weight_path)

    print("\nTraining complete.\n")

    return final_accuracy


def train_model_ignite(
        model, trainloader, testloader, optimizer,
        criterion, num_epochs, device,
        output_directory=None, scheduler=None, retrain=False):
    """
    Function to train a model.
    """
    metrics = {
        "Accuracy": Accuracy(),
        "Loss": Loss(criterion)
    }

    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )
    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    if scheduler:
        scheduler_engine = LRScheduler(scheduler, save_history=True)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler_engine)

    log_output = []

    @trainer.on(Events.STARTED | Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(testloader)
        metrics = evaluator.state.metrics

        epoch_num = engine.state.epoch
        accuracy = metrics['Accuracy'] * 100
        loss = metrics['Loss']

        print(f"ep  {epoch_num:03d}  loss  {loss:.3f}  acc  {accuracy:.2f}%")
        log_output.append([epoch_num, loss, accuracy])

    @trainer.on(Events.COMPLETED)
    def save_state_dict_and_log(engine):
        if not output_directory:
            pass
        else:
            if not retrain:
                file_path = os.path.join(output_directory, 'logger.csv')
                weight_path = os.path.join(output_directory, 'weights.pth')
            else:
                file_path = os.path.join(output_directory,
                                         'logger_retrain.csv')
                weight_path = os.path.join(output_directory,
                                           'weights_retrain.pth')

            # Save the training log
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(log_output)

            # Save the weights
            torch.save(model.state_dict(), weight_path)

    trainer.run(trainloader, num_epochs)

    return log_output[-1][-1]


def test_model(model, testloader, device):
    """
    Function to test a model.
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

    for _, parameter in layer.named_parameters():
        total += parameter.numel()

    for _, buffer in layer.named_buffers():
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
    print('\n')

    if output_directory:
        file_path = os.path.join(output_directory, 'sparsity_report.txt')
        with open(file_path, 'w') as file:
            file.write(str(table))

    return sparsity


def prune_model(parameters_to_prune, percentage, iterations, model,
                trainloader, testloader, optimizer, criterion, num_epochs,
                device, output_directory, output_file, scheduler=None):
    """
    Function to iteratively prune the given model.
    """
    for i in range(iterations):
        print(format_text("Pruning iteration {:02d}".format(i + 1)))
        iter_directory = os.path.join(output_directory, f"Level {i + 1}")
        os.mkdir(iter_directory)

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=percentage,
        )
        accuracy = train_model(model, trainloader, testloader, optimizer,
                               criterion, num_epochs, device, iter_directory,
                               scheduler)
        sparsity = sparsity_check(model, iter_directory)

        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([sparsity, accuracy])


def load_pruned_model(
        model, parameters_to_prune, pruned_weight_path,
        testloader, device, initial_weight_path=None):
    """
    Function that loads weights to train
    a pruned model from scratch.
    """
    # Initialize the new model in a pruned state
    # (so that the state_dicts match) and load the state_dict.
    for module, parameter in parameters_to_prune:
        prune.identity(module, parameter)

    model.load_state_dict(torch.load(pruned_weight_path))
    model.to(device)

    # Set the initial weights.
    if initial_weight_path:
        initial_weights = torch.load(initial_weight_path, map_location=device)
        for name, parameter in model.named_parameters():

            # If the parameter was not pruned
            if name in initial_weights.keys():
                with torch.no_grad():
                    parameter.data = initial_weights[name]

            # If the parameter was pruned
            elif name[-5:] == '_orig':
                with torch.no_grad():
                    parameter.data = initial_weights[name[:-5]]

            # Not sure if this case will ever arise (kept just in case)
            else:
                raise ValueError("Parameter was not found.")

    # Do a sample forward pass so that the weights are
    # computed from weight_orig and weight_mask.
    dataiter = iter(testloader)
    images, _ = dataiter.next()

    with torch.no_grad():
        _ = model(images.to(device))


def retrain_pruned_model(
        parameters_to_prune, iterations, model,
        trainloader, testloader, optimizer, criterion, num_epochs,
        device, output_directory, output_file, scheduler=None):
    """
    Function to retrain pruned model from scratch.
    """
    for i in range(iterations):
        print(format_text("Retrain iteration {:02d}".format(i + 1)))
        iter_directory = os.path.join(output_directory, f"Level {i + 1}")

        pruned_weight_path = os.path.join(iter_directory, 'weights.pth')
        initial_weight_path = os.path.join(output_directory, 'Level 0',
                                           'init_weights.pth')
        load_pruned_model(
            model, parameters_to_prune, pruned_weight_path,
            testloader, device, initial_weight_path
        )

        accuracy = train_model(
            model, trainloader, testloader, optimizer, criterion,
            num_epochs, device, iter_directory, scheduler, retrain=True)
        sparsity = sparsity_check(model)

        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([sparsity, accuracy])
