import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# import our pytorch formatted datasets from data_loading.py script
from data_loading_2class import load_datasets

'''
Encoder based on VGG16 architecture (without final fully connected layers)
'''
class VGG16(nn.Module):  # inherit from base class torch.nn.Module
    def __init__(self, save_dir):
        super(VGG16, self).__init__()  # initialize Module characteristics
        
        self.save_dir = save_dir
        self.section_outputs = [None, None, None, None, None]
        self.sections = []
        self.section_pools = []
        #TODO compute the mean RGB value over pixels in image and subtract from image before forward pass
        # goes from (1280 X 720 X 3) -> (1280 X 720 X 64) VERY LARGE
        self.conv3_1_64 = nn.Conv2d(3, 32, kernel_size=3, stride = 1, padding = 1)
        self.conv3_2_64 = nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding = 1)
        self.max_pool_1 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_64, self.conv3_2_64])
        self.section_pools.append(self.max_pool_1)
        self.reduction_layers = []
        # max pooling between layers 

        # goes from (640 x 360 x 64) to (640 x 360 x 128)
        self.conv3_1_128 = nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1)
        self.conv3_2_128 = nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding = 1)
        self.max_pool_2 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_128, self.conv3_2_128])
        self.section_pools.append(self.max_pool_2)


        # goes from (320 x 1 x 90) to (72 x 72 x 90)
        self.conv3_1_256 = nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding = 1)
        self.conv3_2_256 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1)
        self.conv3_3_256 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1)
        self.max_pool_3 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_256, self.conv3_2_256, self.conv3_3_256])
        self.section_pools.append(self.max_pool_3)

        self.conv3_1_512 = nn.Conv2d(128, 256, kernel_size=3, stride = 1, padding = 1)
        self.conv3_2_512 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.conv3_3_512 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.max_pool_4 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_512, self.conv3_2_512, self.conv3_3_512])
        self.section_pools.append(self.max_pool_4)

        self.conv3_4_512 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.conv3_5_512 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.conv3_6_512 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.sections.append([self.conv3_4_512, self.conv3_5_512, self.conv3_6_512])

        # reduce tensor to simple 3D value; this will be our result
        self.reduction_layer_1 = nn.Conv2d(256, 64, kernel_size = 1, stride = 1)
        self.reduction_layer_2 = nn.Conv2d(256, 64, kernel_size = 1, stride = 1)
        self.reduction_layer_3 = nn.Conv2d(128, 64, kernel_size = 1, stride = 1)
        self.reduction_layer_4 = nn.Conv2d(64, 64, kernel_size = 1, stride = 1)
        self.intermediate_reduction = nn.Conv2d(128, 64, kernel_size = 1, stride = 1)
        
        self.reduction_layers = [self.reduction_layer_2, self.reduction_layer_3, self.reduction_layer_4]

        self.classify_layer = nn.Conv2d(96, 2, kernel_size = 1, stride = 1)  # change for smaller network!
        

    """
    Defines how we perform a forward pass of the VGG16 neural network
    """
    def forward(self, x):
        # downsample
        for index, section in enumerate(self.sections):
            for layer in section:
                x = F.relu(layer(x))
            if(index < 4):
                x = self.section_pools[index](x)
            self.section_outputs[index] = x

        #upsample
        #reduce dimensionality for performance
        x = F.relu(self.reduction_layer_1(x))

        for index, reduction_layer in enumerate(self.reduction_layers):
            x_prev = F.relu(reduction_layer(self.section_outputs[-index-2]))  # iterate backwards through section oututs --> propogate up
            x = torch.cat((x, x_prev), dim = 1)  # did Arjun get this from a paper?
            x = F.relu(nn.functional.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners=True))
            x = F.relu(self.intermediate_reduction(x))
        
        x_prev = self.section_outputs[0]
        x = torch.cat((x, x_prev), dim = 1)
        x = F.relu(nn.functional.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners=True))                

        x = self.classify_layer(x)
        return nn.LogSoftmax(dim = 1)(x)

    def save(self):
        torch.save(self.state_dict(), self.save_dir)