import os
import csv
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from utils.misc import format_text
from utils.train import train_model_ignite


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


def sparsity_check(model: nn.Module, output_directory=None, show=False):
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

    if show:
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
                device, output_directory, output_file, lr_scheduler=False):
    """
    Function to iteratively prune the given model.
    """
    prev_acc = 100.0
    curr_acc = 100.0
    acc_threshold = 30.0

    for i in range(iterations):
        print(format_text("Pruning iteration {:02d}".format(i + 1)))
        iter_directory = os.path.join(output_directory, f"Level {i + 1}")
        os.mkdir(iter_directory)

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=percentage,
        )

        accuracy = train_model_ignite(
            model, trainloader, testloader, optimizer,
            criterion, num_epochs, device, iter_directory, lr_scheduler
        )

        sparsity = sparsity_check(model, iter_directory)

        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([sparsity, accuracy])

        # Stop pruning if accuracy is bad (to prevent wasting time).
        curr_acc = accuracy
        prev_acc = curr_acc

        if curr_acc < acc_threshold and prev_acc < acc_threshold:
            break


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
        device, output_directory, output_file, lr_scheduler=False):
    """
    Function to retrain pruned model from scratch.
    """
    prev_acc = 100.0
    curr_acc = 100.0
    acc_threshold = 30.0

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

        accuracy = train_model_ignite(
            model, trainloader, testloader, optimizer, criterion,
            num_epochs, device, iter_directory, lr_scheduler, retrain=True
        )
        sparsity = sparsity_check(model)

        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([sparsity, accuracy])

        # Stop retraining if accuracy is bad (to prevent wasting time).
        curr_acc = accuracy
        prev_acc = curr_acc

        if curr_acc < acc_threshold and prev_acc < acc_threshold:
            break
