import os
import torch
import helper_methods as H
# import models.lenet_300_100 as M
import models.experimentation as M
# import models.lenet as M

torch.manual_seed(0)


"""
Parameters.
"""
dataset = 'cifar10'
model_to_run = 'real'
model = M.Exp2()
use_gpu = True
mini_batch = 1000000

# model = (M.QLeNet_300_100() if model_to_run == 'quaternion'
#          else M.LeNet_300_100())

# For MNIST
if dataset == 'mnist':
    batch_size = 128
    num_epochs = 40
    learning_rate = 0.1

# For CIFAR10
elif dataset == 'cifar10':
    batch_size = 8
    num_epochs = 5
    learning_rate = 0.001

else:
    raise ValueError("Dataset not known.")


# ############ No need to change anything below this ############### #

"""
Train the model.
"""
device = torch.device("cuda:0" if use_gpu else "cpu")
model.to(device)

# Get the data
trainloader, testloader = H.data_loader(model_to_run, dataset, batch_size)

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

weight_path = os.path.join(
    H.results_dir(),
    "{}_weights_{}.pth".format(dataset, model.name())
)

torch.save(model.state_dict(), weight_path)
