import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
from train_functions import trainNetwork


def openCat(path, name):
    with open(path + name, 'r') as f:
        cat_file = json.load(f)
    return cat_file

def load_model(path, device):
    
    if device == True:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage) #load without gpu param
    
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']
    model,_,_ = trainNetwork(structure, dropout, hidden_layer1, lr)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
  
    return model

def predict(image_path, model, topk, cat_file, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda:0" if device == True else "cpu")
    model.to(device)
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.to(device))
        
    probability = F.softmax(output.data,dim=1)
    topk_package = probability.topk(topk)
    probs = np.array(topk_package[0][0])
    model_cat = {x: y for y, x  in model.class_to_idx.items()}
    flowers = [cat_file[model_cat[i]] for i in np.array(topk_package[1][0])]
    
    return dict(zip(flowers, probs))

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    img_pipeline = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                      ])
    img_tensor = img_pipeline(img_pil)
    return img_tensor
