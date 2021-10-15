import os
import torch
import helper_methods as H
import models as M


"""
Parameters.
"""
dataset = 'mnist'

# For MNIST
if dataset == 'mnist':
    batch_size = 128
    num_epochs = 40
    learning_rate = 0.1

# For CIFAR10
elif dataset == 'cifar10':
    batch_size = 128
    num_epochs = 40
    learning_rate = 0.1

else:
    raise ValueError("Dataset not known.")

# General
model_to_run = 'real'
use_gpu = True
mini_batch = 1000000

device = torch.device("cuda:0" if use_gpu else "cpu")


"""
Run both models.
"""
# Get the data
trainloader, testloader = H.data_loader(model_to_run, dataset, batch_size)

# Get the model
model = M.QLeNet_300_100() if model_to_run == 'quaternion' else M.LeNet_300_100()
model.to(device)

# Display model statistics
H.display_model(model)

# Train and test the model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

H.train_model(
    model,
    trainloader,
    testloader,
    optimizer,
    criterion,
    num_epochs,
    device,
    mini_batch
)

weight_path = os.path.join('open_lth_data',
    f"{dataset}_weights_{'qcnn' if model_to_run == 'quaternion' else 'cnn'}.pth")

torch.save(model.state_dict(), weight_path)
