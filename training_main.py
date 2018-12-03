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

from utils.data_loading import DeepDriveDataset, load_datasets
from architectures.fcn import FCN
from training.segmentation_trainer import SegmentationTrainer
from architectures.vgg16 import VGG16
from architectures.vgg16_deconv import VGG16Deconv
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
    parser.add_argument('--log_iters', '-log', type = int, help = "The spacing between log printouts for training and testing", default = 100)
    parser.add_argument('-lr', type = float, help = "the learning rate to use", default = .001)
    parser.add_argument('--cuda', '-c', action = "store_true", help = "Flag to use cuda for training and testing")
    parser.add_argument('--per_class', action="store_true", help="Flag to output per class data during training")
    parser.add_argument('--batch_size', type = int, action= "store", help = "set the batch size for training and testing", default=1)
    parser.add_argument('--visualize_output', "-vis", action = "store_true", help = "visualize the output every <log_iters> for testing")
    parser.add_argument('--use_crf', "-crf", action = "store_true", help = "postprocess data with the CRF for testing")
    parser.add_argument('--two_class', '-2', action = "store_true", help = "train on just 2 classes")
    parser.add_argument('--prior', action = "store_true", help = "post process using prior data")

    args = parser.parse_args()

    # #TODO parse command line arguments
    DEFAULT_EPOCHS = 10
    epochs = DEFAULT_EPOCHS
    USE_CUDA = args.cuda
    DEFAULT_DEVICE = "cuda" if args.cuda else "cpu"
    DEFAULT_BATCH = args.batch_size
    NUM_CLASSES = 2 if args.two_class else 3

    print("using " + DEFAULT_DEVICE + " ---- batch_size = " + str(DEFAULT_BATCH) + " ----- number_of_classes = " + str(NUM_CLASSES))

    # img_path = "/home/arjun/MIT/6.867/project/bdd100k_images/bdd100k/images/100k"
    # test_path = "/home/arjun/MIT/6.867/project/bdd100k_drivable_maps/bdd100k/drivable_maps/labels"
    #img_path = "C:/Users/Arjun/6.867Project/images/bdd100k/images/100k"
    #test_path = "C:/Users/Arjun/6.867Project/images/bdd100k/drivable_maps/labels"
    # img_path = "C:/Users/cstea/Documents/6.867 Final Project/bdd100k_images/bdd100k/images/100k"
    # test_path = "C:/Users/cstea/Documents/6.867 Final Project/bdd100k_drivable_maps/bdd100k/drivable_maps/labels"
    img_path = "C:/Users/sarah/Documents/6.867 Project/bdd100k_images/bdd100k/images/100k"
    test_path = "C:/Users/sarah/Documents/6.867 Project/bdd100k_drivable_maps/bdd100k/drivable_maps/labels"

    print("Initializing Dataset ... ")
    #load datasets
    train_dataset, test_dataset = load_datasets(img_path, test_path, num_classes = NUM_CLASSES)
    train_loader = DataLoader(train_dataset, batch_size = DEFAULT_BATCH, shuffle = False,
                             num_workers = 1 if USE_CUDA else 0, pin_memory = USE_CUDA)
    test_loader = DataLoader(test_dataset, batch_size = DEFAULT_BATCH, shuffle = False,
                             num_workers = 1 if USE_CUDA else 0, pin_memory = USE_CUDA)


    # generate a prior
    # data_statistics = DataStats(train_dataset, NUM_CLASSES)
    # data_statistics.collect_all_stats("python3_prior_distribution.out")

    # load dataset statistics
    data_statistics = DataStats(train_dataset, NUM_CLASSES)
    data_statistics.load_stats("dataset_statistics.out")

    print("Initializing FCN for Segmentation...")

    #intialize model
    segmentation_model = VGG16Deconv(args.save_dir, NUM_CLASSES)

    if not args.load_dir == '':
        with open(args.load_dir, 'rb') as f:
            segmentation_model.load_state_dict(torch.load(f, map_location = DEFAULT_DEVICE))
    
    # push model to either cpu or gpu
    segmentation_model.to(torch.device(DEFAULT_DEVICE))
    optimizer = optim.Adam(segmentation_model.parameters(), lr = args.lr)
    trainer = SegmentationTrainer(segmentation_model, DEFAULT_DEVICE, train_loader, test_loader, optimizer, data_statistics,
                 num_classes = NUM_CLASSES, log_spacing = args.log_iters, per_class = args.per_class)
    print("Successful initialization!")

    if not args.test:        
        #train the model for a set number of epochs
        for epoch in range(epochs):
            trainer.train(epoch)
            segmentation_model.save()
            trainer.test(use_crf = args.use_crf, iters_per_log = args.log_iters, visualize = args.visualize_output, use_prior = args.prior)

    else:
        print("testing...")
        trainer.test(use_crf = args.use_crf, iters_per_log = args.log_iters, visualize = args.visualize_output, use_prior = args.prior)


