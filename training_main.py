import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
from data_loading import DeepDriveDataset, load_datasets
from fcn import FCN, train



"""
Main function to create the dataset object, initialize the model
and train it
"""
if __name__ == '__main__':
    # #TODO parse command line arguments
    DEFAULT_EPOCHS = 1000
    epochs = DEFAULT_EPOCHS

    #load datasets
    train_loader, test_loader = load_datasets()

    #intialize model
    segmentation_model = FCN()

    #intialize optimizer
    optimizer = optim.Adam(segmentation_model.parameters, lr = .001)

    #train the model for a set number of epochs
    for epoch in range(epochs):
        train(segmentation_model, torch.device("cpu"), train_loader, optimizer, epoch)
        segmentation_model.save()
        test(segmentation_model, torch.device("cpu"), test_loader)
