import os
import argparse
import numpy

import torch

from utils.misc import results_dir, format_text
from utils.misc import display_model, data_loader
from utils.train import train_model_ignite

# Ensuring reproducibility
torch.manual_seed(0)
numpy.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',
                    required=True,
                    choices=['lenet_300_100', 'conv_2', 'conv_4', 'conv_6'],
                    help="The model architecture to run.")
parser.add_argument('-o', '--output_dir',
                    required=True,
                    help="The directory to save the output files.")
parser.add_argument('-g', '--gpu',
                    default='0',
                    help="Which gpu to use for training.")
parser.add_argument('-lrs', '--lr_scheduler',
                    action='store_true',
                    help="Whether to use the learning rate scheduler.")
args = parser.parse_args()

out_dir_name = args.output_dir
architecture = args.model
gpu = args.gpu
lr_scheduler = args.lr_scheduler

if architecture == 'lenet_300_100':
    import models.lenet_300_100 as M
elif architecture == 'conv_2':
    import models.conv_2 as M
elif architecture == 'conv_4':
    import models.conv_4 as M
elif architecture == 'conv_6':
    import models.conv_6 as M
else:
    raise ValueError("That is not a valid model.")

for model_to_run in ['real', 'quat']:

    # Load the hyper-parameters
    hparams = M.std_hparams()
    dataset = hparams['dataset']
    tparams = hparams['training']

    batch_size = tparams['batch_size']
    num_epochs = tparams['num_epochs']
    learning_rate = tparams['learning_rate']
    optimizer = tparams['optimizer']

    if optimizer == 'sgd':
        momentum = tparams['momentum']
        weight_decay = tparams['weight_decay']

    output_directory = os.path.join(results_dir(), out_dir_name,
                                    model_to_run)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    """
    Loading data and model.
    """
    # Get the data.
    trainloader, testloader = data_loader(model_to_run, dataset,
                                          batch_size)

    # Get the model.
    model = M.Real() if model_to_run == 'real' else M.Quat()

    use_gpu = True
    device = torch.device(f'cuda:{gpu}' if use_gpu else 'cpu')
    model.to(device)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate,
            momentum=momentum, weight_decay=weight_decay
        )

    # Save model statistics
    display_model(model, output_directory, show=False)

    """
    Training.
    """
    print(format_text(model_to_run))

    weight_path = os.path.join(output_directory, 'weights.pth')

    train_model_ignite(
        model,
        trainloader,
        testloader,
        optimizer,
        criterion,
        num_epochs,
        device,
        output_directory,
        lr_scheduler
    )
