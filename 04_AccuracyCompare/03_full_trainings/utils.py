'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch
import matplotlib.pyplot as plt

LOG_FILENAME = 'log.txt'


##############
# print utils
##############
def set_log_path(log_path):
    global LOG_FILENAME
    LOG_FILENAME = log_path


def print_log(*argv):
    if isinstance(argv, str):
        string = argv
    else:
        string = ' '.join([str(arg) for arg in argv])

    # write to file
    with open(LOG_FILENAME, 'at') as wf:
        wf.write(string + '\n')

    # print stdio
    print(string)


def save_fig(filename='img', folder='images', dpi=400, bbox_inches='tight'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig('{}/{}.png'.format(folder, filename), dpi=dpi, bbox_inches=bbox_inches)


#############
# loss utils
#############
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


################
# serialization
################
def save_checkpoint(save_path, model, epoch):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save({'epoch': epoch + 1, 'state_dict': state_dict}, save_path)


def load_checkpoint(model, model_path):
    if not os.path.isfile(model_path):
        raise ValueError('Invalid checkpoint file: {}'.format(model_path))

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print_log('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    # create state_dict
    state_dict = {}

    # convert data_parallal to model
    tmp_state_dict = checkpoint['state_dict']
    for k in tmp_state_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = tmp_state_dict[k]
        else:
            state_dict[k] = tmp_state_dict[k]

    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print_log('Load parameter partially {}, required shape {}, loaded shape {}'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                tmp = torch.zeros(model_state_dict[k].shape)  # create tensor with zero filled
                tmp[:state_dict[k].shape[0]] = state_dict[k]  # fill valid
                state_dict[k] = tmp
        else:
            print_log('Drop parameter {}'.format(k))

    for k in model_state_dict:
        if not (k in state_dict):
            print_log('No param {}'.format(k))
            state_dict[k] = model_state_dict[k]

    # load state_dict
    model.load_state_dict(state_dict, strict=False)

    return model