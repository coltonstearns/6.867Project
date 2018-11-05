import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
from data_loading import DeepDriveDataset, load_datasets
from fcn import FCN, train, test



"""
Main function to create the dataset object, initialize the model
and train it
"""
if __name__ == '__main__':

    # #TODO parse command line arguments
    DEFAULT_EPOCHS = 1000
    epochs = DEFAULT_EPOCHS
    DEFAULT_DEVICE = "cpu"
    DEFAULT_BATCH = 1
    USE_CUDA = (DEFAULT_DEVICE == "cuda")

    img_path = "/home/arjun/MIT/6.867/project/bdd100k_images/bdd100k/images/100k"
    test_path = "/home/arjun/MIT/6.867/project/bdd100k_drivable_maps/bdd100k/drivable_maps/labels"
    #img_path = "C:/Users/Arjun/6.867Project/images/bdd100k/images/100k"
    #test_path = "C:/Users/Arjun/6.867Project/images/bdd100k/drivable_maps/labels"

    print("Initializing Dataset ... ")
    #load datasets
    train_dataset, test_dataset = load_datasets(img_path, test_path)
    train_loader = DataLoader(train_dataset, batch_size = DEFAULT_BATCH, shuffle = False,
                             num_workers = 1 if USE_CUDA else 0, pin_memory = USE_CUDA)
    test_loader = DataLoader(test_dataset, batch_size = DEFAULT_BATCH, shuffle = False,
                             num_workers = 1 if USE_CUDA else 0, pin_memory = USE_CUDA)
    

    print("Initializing FCN for Segmentation...")

    #intialize model
    params = sys.argv[1:]
    if(len(params) >= 1):
        segmentation_model = FCN(params[0])
    else:
        assert(False), "Please specify the save directory"

    if(len(params) >= 3):
        if(params[1] == "load"):
            segmentation_model.load_state_dict(torch.load(params[2]))
    

    if(params[-1] == "train"):
        print("Initializing Optimizer...")
        #intialize optimizer
        optimizer = optim.Adam(segmentation_model.parameters(), lr = .001)
        print("Successful initialization!")

        # push model to either cpu or gpu
        segmentation_model.to(torch.device(DEFAULT_DEVICE))

        #train the model for a set number of epochs
        for epoch in range(epochs):
            train(segmentation_model, torch.device(DEFAULT_DEVICE), train_loader, optimizer, epoch)
            segmentation_model.save_state_dict(segmentation_model.save_dir)
            test(segmentation_model, torch.device(DEFAULT_DEVICE), test_loader)

    elif(params[-1] == "test"):
        print("Successful initialization!")
        print("testing...")
        test(segmentation_model, torch.device(DEFAULT_DEVICE), test_loader, iters_per_log=1)

    else:
        assert(False), "Final Argument must be either \"test\" or \"train\""

