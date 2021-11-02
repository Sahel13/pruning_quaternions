# Quaternion LTH

Code to run pruning experiments on quaternion neural networks. The methods to implement quaternion neural networks are borrowed from [hTorch](https://github.com/ispamm/hTorch), and the various pruning experiments are inspired from [open\_lth](https://github.com/facebookresearch/open\_lth).

## Installation
Install PyTorch version specific to the system. Then install packages from `requirements.txt`, followed by the local package in the directory.

## To-do
- [ ] Need to plot train logger for various model sparsities.
- [x] Change code so that the number of re-training iterations <= number of pruning layers.
- [ ] ResNet quaternion model has some issue (training is incredibly slow).
- [ ] Write function to get layer-by-layer sparsity statistics (not urgent, can be done later).