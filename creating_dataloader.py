# Importing all necessary packages
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose 
# Setting randomness seed to ensure reproducability 
torch.manual_seed(5) #5

# The following sets the transformations for all images in the test set. This converts it to a tensor, and normalizes it. 

transform = transforms.Compose(
    [transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# The following transforms the training images in the same way as the testing images, with the only
# difference being a 0.35 chance of flipping the image horizontally. 

train_transform = Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Sets batch size of 8
batch_size = 8

# Creates trainset and trainloader, applying transformations and batchsize. 

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Creates trainset and trainloader, applying transformations and batchsize. 

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Determines classes for the classification. 
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           
