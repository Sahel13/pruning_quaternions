import os
import argparse
import numpy

import torch
from torch.optim.lr_scheduler import LambdaLR

import helper_methods as H

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
args = parser.parse_args()

out_dir_name = args.output_dir
architecture = args.model

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
    weight_decay = tparams['weight_decay']

    output_directory = os.path.join(H.results_dir(), out_dir_name,
                                    model_to_run)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    """
    Loading data and model.
    """
    # Get the data.
    trainloader, testloader = H.data_loader(model_to_run, dataset,
                                            batch_size)

    # Get the model.
    model = M.Real() if model_to_run == 'real' else M.Quat()

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
    #                             momentum=0.9, weight_decay=weight_decay)
    if model_to_run == 'real':
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), learning_rate/2)

    scheduler = LambdaLR(optimizer, M.std_lr_scheduler)

    # Save model statistics
    H.display_model(model, output_directory, show=False)

    """
    Training.
    """
    print(H.format_text(model_to_run))

    weight_path = os.path.join(output_directory, 'weights.pth')

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))

    H.train_model_ignite(
        model,
        trainloader,
        testloader,
        optimizer,
        criterion,
        num_epochs,
        device,
        output_directory,
        scheduler
    )
