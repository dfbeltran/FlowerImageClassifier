# Imports
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
import matplotlib.pyplot as plt
import seaborn as sns
import train_functions as tf
import argparse

#Parser
parser = argparse.ArgumentParser(description='Train neural network for flower category classification.')

parser.add_argument('data_directory', action = 'store', 
                    default='./flowers',
                    help='Enter path where data is located. Train, test and validation data should be in the same path')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_dir', default = 'checkpoint.pth',
                    help = 'Enter location to save checkpoint in.')

parser.add_argument('--arch', action='store', dest='arch', default='vgg11',
                    help='Enter pretrained model to use, You can choose between vgg11, densenet121 \
                    or alexnet architectures. The default is VGG-11.')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'hid_lay', type = int, default = 512,
                    help = 'Enter number of hidden units in the first hidden layer, default is 512.')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=float, default = 0.001,
                    help = 'Enter learning rate for training the model, default is 0.001.')

parser.add_argument('--dropout', action = 'store',
                    dest='dropout', type=int, default = 0.05,
                    help = 'Enter dropout for training the model, default is 0.05.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'epochs', type = int, default = 2,
                    help = 'Enter number of epochs to use during training, default is 1.')

parser.add_argument('--gpu', action="store_true", 
                    default=False,
                    help='Turn GPU mode on or off, default is off.')
                    
results = parser.parse_args()
data_dir = results.data_directory
arch = results.arch
lr = results.lr
hid_lay = results.hid_lay
dropout = results.dropout
epochs = results.epochs
save_model_dir = results.save_dir
device = results.gpu

def main():
    data_transforms, image_datasets, dataloaders = tf.load_data(data_dir)
    model, optimizer, criterion = tf.trainNetwork(structure = arch, dropout = dropout, hidden_layer1 = hid_lay, lr=lr)
    tf.do_deep_learning(model, dataloaders['trainloader'], dataloaders['validloader'], 3, 10, criterion = criterion, optimizer = optimizer, device = device)
    tf.check_accuracy_on_test(dataloaders['testloader'], model, device)
    tf.save_model(model, arch, image_datasets['train_data'], hid_lay, dropout, lr, epochs, save_model_dir)
main()

