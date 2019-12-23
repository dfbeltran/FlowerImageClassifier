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

def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train_transforms': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])]),
        'test_transforms': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]),
        'valid_transforms': transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])}
    image_datasets = {'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
                      'test_data' : datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms']),
                      'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_transforms'])}
    
    dataloaders = {'trainloader' : torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
                   'testloader' : torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=32),
                   'validloader' : torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32)}
        
    return data_transforms, image_datasets, dataloaders

arch = {'densenet121' : 1024,
        'vgg11' : 25088,
        'alexnet' : 9216}

def trainNetwork(structure, dropout, hidden_layer1, lr):
    
    if structure in list(arch.keys()):
        if structure == 'vgg11':
            model = models.vgg11(pretrained=True)
        elif structure == 'densenet121':
            model = models.densenet121(pretrained=True)
        else: 
            model = models.alexnet(pretrained = True)
    else:
        print('Please input vgg11, densenet121 or alexnet')
        
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(arch[structure],hidden_layer1)),
                          ('relu1', nn.ReLU()),
                          ('d_out1',nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_layer1, 256)),
                          ('relu2', nn.ReLU()),
                          ('d_out2',nn.Dropout(dropout)),
                          ('fc3', nn.Linear(256, 128)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)    
    
    
    return model, optimizer, criterion

def validation(model, testloader, criterion, optimizer, device):
    model.eval()
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        optimizer.zero_grad()
        model.to(device)
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            output = model.forward(inputs)
            test_loss = criterion(output, labels)
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(dim = 1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()
            
    test_loss = test_loss / len(testloader)
    accuracy = accuracy /len(testloader)
    
    return test_loss, accuracy

def do_deep_learning(model, trainloader, testloader, epochs, print_every, criterion, optimizer, device):
    
    optimizer = optimizer
    epochs = epochs
    print_every = print_every
    steps = 0
    
    device = torch.device("cuda:0" if device == True else "cpu")

    # change to device
    model.to(device)
    print('Beginning training.....')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            
            if steps % print_every == 0:
                test_loss, accuracy = validation(model, testloader, criterion, optimizer, device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(test_loss),
                      "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0
    print('End of training...')


def check_accuracy_on_test(testloader, model, device):
    correct = 0
    total = 0
    device = torch.device("cuda:0" if device == True else "cpu")
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def save_model(model, arch, train_loader, hid_lay, dropout, lr, epochs, save_model_dir):
    model.class_to_idx =  train_loader.class_to_idx
    model.cpu
    torch.save({'structure' : arch,
                'hidden_layer1': hid_lay,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                save_model_dir)