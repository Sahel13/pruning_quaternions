import os
import torch
import helper_methods as H
import models.resnet as M
from torch.optim.lr_scheduler import LambdaLR

# Ensure reproducibility
torch.manual_seed(0)


"""
Parameters.
"""
to_run = [(M.Real(), 'real'), (M.Quat(), 'quat')]

hparams = M.hyper_params()
tparams = hparams['training']

output_dir_name = hparams['output_directory']
dataset = hparams['dataset']
batch_size = tparams['batch_size']
num_epochs = tparams['num_epochs']
learning_rate = tparams['learning_rate']
weight_decay = tparams['weight_decay']

for model, model_to_run in to_run:

    """
    Train the model.
    """
    use_gpu = True
    device = torch.device("cuda:1" if use_gpu else "cpu")
    model.to(device)

    # Get the data
    trainloader, testloader = H.data_loader(model_to_run, dataset, batch_size)

    # Display model statistics
    # H.display_model(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=weight_decay,
                                nesterov=True)
    scheduler = LambdaLR(optimizer, M.std_lr_scheduler)

    output_directory = os.path.join(H.results_dir(), output_dir_name)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    H.train_model(
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
