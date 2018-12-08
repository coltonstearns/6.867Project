import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import subprocess
from tqdm import tqdm

from utils.data_loading import DeepDriveDataset, load_datasets
from training.segmentation_trainer import SegmentationTrainer

from architectures.network1 import Network_1
from architectures.network2 import Network_2
from architectures.network3 import Network_3
from architectures.network4 import Network_4
from architectures.network5 import Network_5

from utils.data_stats import DataStats



"""
Main function to create the dataset object, initialize the model
and train it
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A training module for an FCN for the Berkley Deep Drive Dataset')
    parser.add_argument('--load', '-l', action = "store", type=str, help= 'A file location to load the model from',
                         dest = 'load_dir', default = '')
    parser.add_argument('--test', '-t', action = "store_true", help = "A flag to say that we are testing the model only")
    parser.add_argument('--save-to', '-s', type = str, help = "A file location to store the model", dest = "save_dir", required=True)
    parser.add_argument('--log_iters', '-log', type = int, help = "The spacing between log printouts for training and testing", default = 500)
    parser.add_argument('-lr', type = float, help = "the learning rate to use", default = .001)
    parser.add_argument('--cuda', '-c', action = "store_true", help = "Flag to use cuda for training and testing")
    parser.add_argument('--per_class', action="store_true", help="Flag to output per class data during training")
    parser.add_argument('--batch_size', type = int, action= "store", help = "set the batch size for training and testing", default=1)
    parser.add_argument('--visualize_output', "-vis", action = "store_true", help = "visualize the output every <log_iters> for testing")
    parser.add_argument('--use_crf', "-crf", action = "store_true", help = "postprocess data with the CRF for testing")
    parser.add_argument('--two_class', '-2', action = "store_true", help = "train on just 2 classes")
    parser.add_argument('--prior', action = "store_true", help = "post process using prior data")
    parser.add_argument('--L2', action = "store", dest = "l2", type = float, help = "sets how much l2 regularization to add", default = 0)
    parser.add_argument('--start-idx', action = "store", dest = "start_idx", type = int, help = "tells where to resume in data", default = 0)
    args = parser.parse_args()

    # ====================================== Parameters From Command Line ================================
    USE_CUDA = args.cuda
    DEFAULT_DEVICE = "cuda" if args.cuda else "cpu"
    DEFAULT_BATCH = args.batch_size
    NUM_CLASSES = 2 if args.two_class else 3
    # ====================================================================================================

    print("using " + DEFAULT_DEVICE + " ---- batch_size = " + str(DEFAULT_BATCH) + " ----- number_of_classes = " + str(NUM_CLASSES))

    # =================================== More Parameters =============================================
    prior_distribution_file = "priors/python3_prior.out" if sys.version_info[0] > 2 else "priors/python2_prior.out"
    EPOCHS = 1
    # IMG_PATH = "/home/arjun/MIT/6.867/project/bdd100k_images/bdd100k/images/100k"
    # TEST_PATH = "/home/arjun/MIT/6.867/project/bdd100k_drivable_maps/bdd100k/drivable_maps/labels"
    #IMG_PATH = "C:/Users/Arjun/6.867Project/images/bdd100k/images/100k"
    #TEST_PATH = "C:/Users/Arjun/6.867Project/images/bdd100k/drivable_maps/labels"
    # IMG_PATH = "C:/Users/cstea/Documents/6.867 Final Project/bdd100k_images/bdd100k/images/100k"
    # TEST_PATH = "C:/Users/cstea/Documents/6.867 Final Project/bdd100k_drivable_maps/bdd100k/drivable_maps/labels"
    #IMG_PATH = "C:/Users/sarah/Documents/6.867 Project/bdd100k_images/bdd100k/images/100k"
    #TEST_PATH = "C:/Users/sarah/Documents/6.867 Project/bdd100k_drivable_maps/bdd100k/drivable_maps/labels"
    IMG_PATH = "/home/arjun/6.867Project/images/bdd100k/images/100k"
    TEST_PATH = "/home/arjun/6.867Project/images/bdd100k/drivable_maps/labels"
    # ================================================================================================


    print("Initializing Dataset ... ")
    #load datasets
    train_dataset, test_dataset = load_datasets(IMG_PATH, TEST_PATH, num_classes = NUM_CLASSES)
    train_loader = DataLoader(train_dataset, batch_size = DEFAULT_BATCH, shuffle = False,
                             num_workers = 4 if USE_CUDA else 0)
    test_loader = DataLoader(test_dataset, batch_size = DEFAULT_BATCH, shuffle = False,
                             num_workers = 4 if USE_CUDA else 0)

    # load dataset statistics
    data_statistics = DataStats(train_dataset, NUM_CLASSES)
    data_statistics.load_stats(prior_distribution_file)

    print("Initializing FCN for Segmentation...")

    #intialize model
    networks = {"network1": Network_1, "network2": Network_2, "network3": Network_3, "network4": Network_4, "network5": Network_5}
    network = None
    if args.load_dir:
        if len(args.load_dir.split("/")) > 1:
            network_key = args.load_dir.split("/")[1]
            if network_key in networks:
                network = networks[network_key]  # second subfolder contains network
    if args.save_dir:
        if len(args.save_dir.split("/")) > 1:
            network_key = args.save_dir.split("/")[1]
            if network_key in networks:
                network = networks[network_key]
         
                

    if not network:
        raise RuntimeError("Please specify a model folder in which to save the current model.")

    segmentation_model = network(args.save_dir, NUM_CLASSES)

    if not args.load_dir == '':
        try:
            segmentation_model.load(args.load_dir, DEFAULT_DEVICE)
        except:
            print("Loading Legacy Model")
            segmentation_model.legacy_load(args.load_dir, DEFAULT_DEVICE)
    
    # push model to either cpu or gpu
    segmentation_model.to(torch.device(DEFAULT_DEVICE))
    optimizer = optim.Adam(segmentation_model.parameters(), lr = args.lr, weight_decay = args.l2)
    trainer = SegmentationTrainer(segmentation_model, DEFAULT_DEVICE, train_loader, test_loader, optimizer, data_statistics,
                 num_classes = NUM_CLASSES, log_spacing = args.log_iters, per_class = args.per_class)
    print("Successful initialization!")

    if not args.test:        
        #train the model for a set number of epochs
        for epoch in range(EPOCHS):
            trainer.train(EPOCHS, args.start_idx)
            segmentation_model.save()
            #trainer.test(use_crf = args.use_crf, iters_per_log = args.log_iters, visualize = args.visualize_output, use_prior = args.prior)

    else:
        print("testing...")
        trainer.test(use_crf = args.use_crf, iters_per_log = args.log_iters, visualize = args.visualize_output, use_prior = args.prior)
        segmentation_model.save()

