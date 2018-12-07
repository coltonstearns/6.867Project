import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Base class that all networks inherit from
'''
class NetworkBase(nn.Module):  # inherit from base class torch.nn.Module
    def __init__(self, save_dir, num_classes):
        super(NetworkBase, self).__init__()  # initialize Module characteristics
        self.training_loss = []
        self.training_accuracy = []
        self.training_confusion = np.zeros((num_classes, num_classes))
        
        self.test_loss = []
        self.test_accuracy = []
        self.test_confusion = np.zeros((num_classes, num_classes))

        self.per_class_training_loss = [] 

        self.save_dir = save_dir
        self.num_classes = num_classes

    """
    defines how we save our model
    """
    def save(self):
        torch.save([self.state_dict(), self.training_loss, 
                    self.test_loss, self.training_confusion, 
                    self.test_confusion, self.per_class_training_loss,
                    self.training_accuracy, self.test_accuracy, 
                    self.num_classes], self.save_dir)

    """
    defines how we load the model
    """
    def load(self, load_dir, device):
        with open(load_dir, 'rb') as f:
            [state_dict, self.training_loss, self.test_loss,
            self.training_confusion, self.test_confusion, 
            self.per_class_training_loss, 
            self.training_accuracy, self.test_accuracy,
            num_classes] = torch.load(f, map_location = device)

            assert(self.num_classes == num_classes), "wrong number of classes"
            self.load_state_dict(state_dict)


    """
    loads models we trained before the representation update
    """
    def legacy_load(self, load_dir, device):
        with open(load_dir, 'rb') as f:
            self.load_state_dict(torch.load(f, map_location = device))
