import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from architectures.model_stats import ModelStats

'''
Base class that all networks inherit from
'''
class NetworkBase(nn.Module):  # inherit from base class torch.nn.Module
    def __init__(self, save_dir, num_classes):
        super(NetworkBase, self).__init__()  # initialize Module characteristics
        self.train_stats = ModelStats(num_classes)
        self.test_stats = ModelStats(num_classes)
        self.stats = {"train": self.train_stats, 
                      "test": self.test_stats}
        self.save_dir = save_dir
        self.num_classes = num_classes

    """
    defines how we save our model, save all info about model to file
    """
    def save(self):
        torch.save([self.state_dict(), self.train_stats, 
                    self.test_stats, self.num_classes], self.save_dir)

    """
    defines how we load the model, load in all the data
    """
    def load(self, load_dir, device):
        with open(load_dir, 'rb') as f:
            [state_dict, self.train_stats, 
            self.test_stats, num_classes] = torch.load(f, map_location = device)

            assert(self.num_classes == num_classes), "wrong number of classes"
            self.load_state_dict(state_dict)


    """
    loads models we trained before the representation update
    """
    def legacy_load(self, load_dir, device):
        with open(load_dir, 'rb') as f:
            self.load_state_dict(torch.load(f, map_location = device))
