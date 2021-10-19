import os
import torch
import helper_methods as H
import models.resnet as M
from torch.optim.lr_scheduler import MultiStepLR

# Ensure reproducibility
torch.manual_seed(0)


"""
Parameters.
"""
model_to_run = 'quaternion'

# ############ No need to change anything below this ############### #

model = M.Real() if model_to_run == 'real' else M.Quat()

hparams = M.hyper_params()
tparams = hparams['training']

output_directory = hparams['output_directory']
dataset = hparams['dataset']
batch_size = tparams['batch_size']
num_epochs = tparams['num_epochs']
learning_rate = tparams['learning_rate']
milestones = tparams['milestones']
gamma = tparams['gamma']
weight_decay = tparams['weight_decay']
mini_batch = tparams['mini_batch']


"""
Train the model.
"""
use_gpu = True
device = torch.device("cuda:0" if use_gpu else "cpu")
model.to(device)

# Get the data
trainloader, testloader = H.data_loader(model_to_run, dataset, batch_size)

# Display model statistics
H.display_model(model)

# Train and test the model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                            weight_decay=weight_decay)
scheduler = MultiStepLR(optimizer, milestones, gamma)

H.train_model(
    model,
    trainloader,
    testloader,
    optimizer,
    criterion,
    num_epochs,
    device,
    mini_batch,
    scheduler=scheduler
)

weight_path = os.path.join(
    H.results_dir(),
    "{}_weights_{}.pth".format(dataset, model.name())
)

torch.save(model.state_dict(), weight_path)
