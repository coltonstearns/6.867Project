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
from architectures.network6 import Network_6
from architectures.network7 import Network_7
from architectures.network8 import Network_8
from utils.data_stats import DataStats
from IPython import embed


"""
Main function to create the dataset object, initialize the model
and train it
"""
if __name__ == '__main__':

    # =================================== Command Line Parsing =========================================
    parser = argparse.ArgumentParser(description = 'A training module for an FCN for the Berkley Deep Drive Dataset')
    parser.add_argument('--load', '-l', action = "store", type = str, help = 'A file location to load the model from',
                        dest = 'load_dir', default = '')
    parser.add_argument('--two_class', '-2', action = "store_true", help = "train on just 2 classes")
    args = parser.parse_args()
    NUM_CLASSES = 2 if args.two_class else 3

    # ===================================================================================================
    # ===================================== Load Pre-trained Model=======================================
    # ===================================================================================================
    print("Initializing FCN for Segmentation...")
    #intialize model
    networks = {"network1": Network_1, "network2": Network_2, "network3": Network_3, "network4": Network_4,
                "network5": Network_5, "network6": Network_6, "network7": Network_7, "network8": Network_8}
    network = None
    if args.load_dir:
        if len(args.load_dir.split("/")) > 1:
            network_key = args.load_dir.split("/")[1]
            if network_key in networks:
                network = networks[network_key]  # second subfolder contains network

    if not network:
        raise RuntimeError("Please specify a model folder in which to load the current model.")

    model = network("", NUM_CLASSES)

    if not args.load_dir == '':
        try:
            segmentation_model.load(args.load_dir, "cpu")
        except:
            print("Loading Legacy Model")
            segmentation_model.legacy_load(args.load_dir, "cpu")

    # ===================================================================================================
    # ===================================================================================================

    embed()
