import os
import csv
import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from htorch import layers
import helper_methods as H

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

num_trials = 5
for trial in range(num_trials):

    for model_to_run in ['real', 'quat']:

        # Load the hyper-parameters
        hparams = M.std_hparams()
        dataset = hparams['dataset']
        tparams = hparams['training']
        pparams = hparams['pruning']

        batch_size = tparams['batch_size']
        num_epochs = tparams['num_epochs']
        learning_rate = tparams['learning_rate']
        weight_decay = tparams['weight_decay']

        pruning_iterations = pparams['iterations']
        pruning_percentage = pparams['percentage']

        output_directory = os.path.join(H.results_dir(), out_dir_name,
                                        f'Trial {trial + 1}', model_to_run)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # File to save sparsity vs accuracy data for inference.
        data_file = os.path.join(output_directory, 'acc_data.csv')
        retrain_data = os.path.join(output_directory, 'acc_data_retrain.csv')

        pre_train_dir = os.path.join(output_directory, 'Level 0')
        if not os.path.exists(pre_train_dir):
            os.mkdir(pre_train_dir)

        """
        Loading data and model.
        """
        # Get the data.
        trainloader, testloader = H.data_loader(model_to_run, dataset,
                                                batch_size)

        # Get the model.
        model = M.Real() if model_to_run == 'real' else M.Quat()

        use_gpu = True
        device = torch.device("cuda:1" if use_gpu else "cpu")
        model.to(device)

        # Loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
        #                             momentum=0.9, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        scheduler = LambdaLR(optimizer, M.std_lr_scheduler)

        # Save model statistics
        H.display_model(model, pre_train_dir, show=False)

        """
        Pre-training.
        """
        print(H.format_text(model_to_run))
        print(H.format_text('Pre-training'))

        initial_weight_path = os.path.join(pre_train_dir, 'init_weights.pth')
        weight_path = os.path.join(pre_train_dir, 'weights.pth')

        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
            print("Weights have been loaded.\n")
            accuracy = H.test_model(model, testloader, device)
        else:
            torch.save(model.state_dict(), initial_weight_path)
            accuracy = H.train_model(
                model,
                trainloader,
                testloader,
                optimizer,
                criterion,
                num_epochs,
                device,
                pre_train_dir,
                scheduler
            )

        for item in [data_file, retrain_data]:
            with open(item, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Sparsity', 'Accuracy'])
                writer.writerow([100.0, accuracy])

        """
        Pruning.
        """
        parameters_to_prune = []

        for child in model.children():
            # Only pruning the weights (no biases) of conv and linear layers.
            if model_to_run == 'quat':
                weights = ['r_weight', 'i_weight', 'j_weight', 'k_weight']

                if (isinstance(child, layers.QConv2d) or
                        isinstance(child, layers.QLinear)):
                    for weight in weights:
                        parameters_to_prune.append((child, weight))
                elif isinstance(child, nn.Linear):
                    parameters_to_prune.append((child, 'weight'))
            else:
                if (isinstance(child, nn.Conv2d) or
                        isinstance(child, nn.Linear)):
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
            output_directory,
            data_file,
            scheduler
        )

        H.retrain_pruned_model(
            parameters_to_prune,
            pruning_iterations,
            model,
            trainloader,
            testloader,
            optimizer,
            criterion,
            num_epochs,
            device,
            output_directory,
            retrain_data,
            scheduler
        )
