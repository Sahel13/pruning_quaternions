import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from htorch import layers
import helper_methods as H
import models.lenet_300_100 as M

# Ensure reproducibility
torch.manual_seed(0)


"""
Parameters.
"""
model_to_run = 'real'

# ############ No need to change anything below this ############### #

model = M.Real() if model_to_run == 'real' else M.Quat()

hparams = M.hyper_params()
dataset = hparams['dataset']
output_dir_name = hparams['output_directory']
tparams = hparams['training']
pparams = hparams['pruning']

batch_size = tparams['batch_size']
num_epochs = tparams['num_epochs']
learning_rate = tparams['learning_rate']
milestones = tparams['milestones']
gamma = tparams['gamma']
weight_decay = tparams['weight_decay']
mini_batch = tparams['mini_batch']

pruning_iterations = pparams['iterations']
pruning_percentage = pparams['percentage']

use_gpu = True
device = torch.device("cuda:0" if use_gpu else "cpu")
model.to(device)

output_directory = os.path.join(H.results_dir(), output_dir_name, model_to_run)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

pre_train_directory = os.path.join(output_directory, 'Level 0')
if not os.path.exists(pre_train_directory):
    os.mkdir(pre_train_directory)


"""
Loading data and model.
"""
# Get the data
trainloader, testloader = H.data_loader(model_to_run, dataset, batch_size)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                            weight_decay=weight_decay)
scheduler = MultiStepLR(optimizer, milestones, gamma)

# Display model statistics
H.display_model(model, pre_train_directory)


"""
Pretraining.
"""
print(H.format_text('Pre-training'))

initial_weight_path = os.path.join(output_directory, 'initial_weights.pth')
weight_path = os.path.join(pre_train_directory, f'weights_{model.name()}.pth')

if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path))
    print("Weights have been loaded.\n")
else:
    torch.save(model.state_dict(), initial_weight_path)
    H.train_model(
        model,
        trainloader,
        testloader,
        optimizer,
        criterion,
        num_epochs,
        device,
        mini_batch,
        pre_train_directory,
        scheduler
    )


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
    output_directory,
    scheduler
)
