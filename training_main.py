import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from tqdm import tqdm
from data_loading import DeepDriveDataset, load_datasets
from fcn import FCN, train, test



"""
Main function to create the dataset object, initialize the model
and train it
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A training module for an FCN for the Berkley Deep Drive Dataset')
    parser.add_argument('--load', '-l', type=str, nargs=1, help= 'A file location to load the model from',
                         dest = 'load_dir', default = '')
    parser.add_argument('--test', '-t', action = "store_true", help = "A flag to say that we are testing the model only")
    parser.add_argument('--save-to', '-s', type = str, help = "A file location to store the model", dest = "save_dir", required=True)
    parser.add_argument('--log_iters', '-log', type = int, help = "The spacing between log printouts for training and testing", default = 7200)
    parser.add_argument('-lr', type = float, help = "the learning rate to use", default = .001)
    parser.add_argument('--cuda', '-c', action = "store_true", help = "Flag to use cuda for training and testing")
    parser.add_argument('--per_class', action="store_true", help="Flag to output per class data during training")
    parser.add_argument('--batch_size', action= "store", help = "set the batch size for training and testing", default=1)

    args = parser.parse_args()

    # #TODO parse command line arguments
    DEFAULT_EPOCHS = 1000
    epochs = DEFAULT_EPOCHS
    USE_CUDA = args.cuda
    DEFAULT_DEVICE = "cuda" if args.cuda else "cpu"
    DEFAULT_BATCH = args.batch_size

    img_path = "/home/arjun/MIT/6.867/project/bdd100k_images/bdd100k/images/100k"
    test_path = "/home/arjun/MIT/6.867/project/bdd100k_drivable_maps/bdd100k/drivable_maps/labels"
    #img_path = "C:/Users/Arjun/6.867Project/images/bdd100k/images/100k"
    #test_path = "C:/Users/Arjun/6.867Project/images/bdd100k/drivable_maps/labels"

    print("Initializing Dataset ... ")
    #load datasets
    train_dataset, test_dataset = load_datasets()
    train_loader = DataLoader(train_dataset, batch_size = DEFAULT_BATCH, shuffle = False,
                             num_workers = 1 if USE_CUDA else 0, pin_memory = USE_CUDA)
    test_loader = DataLoader(test_dataset, batch_size = DEFAULT_BATCH, shuffle = False,
                             num_workers = 1 if USE_CUDA else 0, pin_memory = USE_CUDA)
    

    print("Initializing FCN for Segmentation...")

    #intialize model
    segmentation_model = FCN(args.save_dir)

    if not args.load_dir == '':
            segmentation_model.load_state_dict(torch.load(args.load_dir))
    

    if not args.test:
        print("Initializing Optimizer...")
        #intialize optimizer
        optimizer = optim.Adam(segmentation_model.parameters(), lr = args.lr)
        print("Successful initialization!")

        # push model to either cpu or gpu
        segmentation_model.to(torch.device(DEFAULT_DEVICE))

        #train the model for a set number of epochs
        for epoch in range(epochs):
            train(segmentation_model, torch.device(DEFAULT_DEVICE), train_loader, optimizer, epoch, per_class=args.per_class)
            segmentation_model.save()
            test(segmentation_model, torch.device(DEFAULT_DEVICE), test_loader, iters_per_log = args.log_iters)

    else:
        print("Successful initialization!")
        print("testing...")
        test(segmentation_model, torch.device(DEFAULT_DEVICE), test_loader, iters_per_log=args.log_iters)


