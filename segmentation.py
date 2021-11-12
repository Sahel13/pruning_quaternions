import os
import argparse
import numpy
import models.unet as M
import csv

import torch

from utils.misc import results_dir, format_text
from utils.misc import display_model
from utils.dataloader import pascal_loader, remove_one_hot_enc

# Ensuring reproducibility
torch.manual_seed(0)
numpy.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir',
                    required=True,
                    help="The directory to save the output files.")
parser.add_argument('-g', '--gpu',
                    default='0',
                    help="Which gpu to use for training.")
args = parser.parse_args()

out_dir_name = args.output_dir
gpu = args.gpu

model_to_run = 'real'
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
trainloader, testloader = pascal_loader(batch_size)

# Get the model.
# model = M.Real() if model_to_run == 'real' else M.Quat()
model = M.UNET()

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


def get_iau(preds, labels):
    intersection = 0
    union = 0

    for i in range(21):
        predicted = (preds == i)
        actual = (labels == i)
        intersection += torch.logical_and(predicted, actual).float().sum()
        union += torch.logical_or(predicted, actual).float().sum()

    return intersection, union


def test_model(model, testloader, device):
    """
    Function to test a model.
    """
    EPS = 1e-6
    intersection = 0
    union = 0

    with torch.no_grad():
        for data in testloader:
            images, masks = data[0].to(device), data[1].to(device)
            preds = model(images)
            _, preds = torch.max(preds.data, 1)
            masks = remove_one_hot_enc(masks)
            batch_i, batch_u = get_iau(preds, masks)
            intersection += batch_i
            union += batch_u

    iou = (intersection + EPS) / (union + EPS) * 100
    return iou


def train_model(
        model, trainloader, testloader, optimizer,
        criterion, num_epochs, device,
        output_directory=None):
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

        final_accuracy = accuracy

    if output_directory:
        file_path = os.path.join(output_directory, 'logger.csv')
        weight_path = os.path.join(output_directory, 'weights.pth')

        # Save the training log
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(log_output)

        # Save the weights
        torch.save(model.state_dict(), weight_path)

    print("\nTraining complete.\n")

    return final_accuracy


train_model(
    model,
    trainloader,
    testloader,
    optimizer,
    criterion,
    num_epochs,
    device,
    output_directory
)
