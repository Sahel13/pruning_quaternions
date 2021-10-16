import os
import json
import argparse

import torch
import torch.nn as nn

from htorch import layers
import helper_methods as H
import models as M


"""
Command-line arguments.
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model',
    choices=["real", "quaternion"],
    required=True,
    help="Which model to run."
)
parser.add_argument(
    '-d', '--dataset',
    choices=['mnist', 'cifar10'],
    required=True,
    help="Which dataset to use."
)
parser.add_argument(
    '-o', '--output_dir',
    required=True,
    help="Name of output directory."
)
args = parser.parse_args()


"""
Parameters.
"""
# General
dataset = args.dataset
output_dir_name = args.output_dir
model_to_run = args.model
model = M.QLeNet_300_100()

# Pruning
pruning_iterations = 8
pruning_percentage = 0.96

use_gpu = True
mini_batch = 100000

# For MNIST
if dataset == 'mnist':
    batch_size = 128
    num_epochs = 40
    learning_rate = 0.1

# For CIFAR10
elif dataset == 'cifar10':
    batch_size = 4
    num_epochs = 5
    learning_rate = 0.001

else:
    raise ValueError("Dataset not known.")

# ######### No need to change anything below this ######### #

device = torch.device("cuda:0" if use_gpu else "cpu")

output_directory = os.path.join(H.results_dir(), output_dir_name, model_to_run)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

pre_train_directory = os.path.join(output_directory, 'Level 0')
if not os.path.exists(pre_train_directory):
    os.mkdir(pre_train_directory)

# Save the hyper-parameters
par_file_path = os.path.join(pre_train_directory, 'hyper_parameters.json')

hyper_parameters = {
    "general": {
        "dataset": dataset,
        "model_type": model_to_run
    },
    "training": {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate
    },
    "pruning": {
        "pruning_percentage": pruning_percentage,
        "pruning_iterations": pruning_iterations
    }
}
with open(par_file_path, 'w') as file:
    json.dump(hyper_parameters, file)


"""
Loading data and model.
"""
# Get the data
trainloader, testloader = H.data_loader(model_to_run, dataset, batch_size)

# Get the model
# model = M.LeNet_300_100() if model_to_run == 'real' else M.QLeNet_300_100()
model.to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Display model statistics
H.display_model(model, pre_train_directory)


"""
Pretraining.
"""
print(H.format_text('Pre-training'))

weight_path = os.path.join(pre_train_directory, 'weights.pth')

if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path))
    print("Weights have been loaded.\n")
else:
    H.train_model(
        model,
        trainloader,
        testloader,
        optimizer,
        criterion,
        num_epochs,
        device,
        mini_batch,
        pre_train_directory
    )

    torch.save(model.state_dict(), weight_path)

print(f"Accuracy: {H.test_model(model, testloader, device)}%")


"""
Global pruning
"""
parameters_to_prune = []

for child in model.children():
    # Only pruning the weights (no biases) of conv and linear layers.
    if model_to_run == 'quaternion':
        weights = ['r_weight', 'i_weight', 'j_weight', 'k_weight']

        if (isinstance(child, layers.QConv2d) or
                isinstance(child, layers.QLinear)):
            for weight in weights:
                parameters_to_prune.append((child, weight))
    else:
        if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear):
            parameters_to_prune.append((child, 'weight'))

H.prune_model(
    parameters_to_prune,
    pruning_percentage,
    pruning_iterations,
    model,
    trainloader,
    testloader,
    optimizer,
    criterion,
    num_epochs,
    device,
    mini_batch,
    output_directory
)
