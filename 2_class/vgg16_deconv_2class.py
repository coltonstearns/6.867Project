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
        self.section_outputs = [None, None, None]
        self.sections = []
        self.section_pools = []
        
        # 1: goes from (1280 X 720 X 3) -> (640 X 320 X 32) VERY LARGE
        self.conv3_1_64 = nn.Conv2d(3, 32, kernel_size=3, stride = 1, padding = 1)
        self.conv3_2_64 = nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding = 1)
        self.max_pool_1 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_64, self.conv3_2_64])
        self.section_pools.append(self.max_pool_1)
        self.reduction_layers = []
        # max pooling between layers 

        # 2: goes from (640 x 360 x 32) to (320 x 180 x 64)
        self.conv3_1_128 = nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1, dilation = 1)
        self.conv3_2_128 = nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding = 1, dilation = 1)
        self.max_pool_2 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_128, self.conv3_2_128])
        self.section_pools.append(self.max_pool_2)

        # 3: goes from (320 x 180 x 64) to (160, 90, 128)
        self.conv3_1_256 = nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding = 1, dilation = 1)
        self.conv3_2_256 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1, dilation = 1)
        self.conv3_3_256 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1, dilation = 1)
        self.max_pool_3 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_256, self.conv3_2_256, self.conv3_3_256])
        self.section_pools.append(self.max_pool_3)

        # 4: goes from (160 x 90 x 128) to (80, 45, 256)
        self.conv3_1_512 = nn.Conv2d(128, 256, kernel_size=3, stride = 1, padding = 1)
        self.conv3_2_512 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.conv3_3_512 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.max_pool_4 = nn.MaxPool2d(2, stride = 2)
        self.sections.append([self.conv3_1_512, self.conv3_2_512, self.conv3_3_512])
        self.section_pools.append(self.max_pool_4)

        # 5: goes from (80, 45, 256) to (80, 45, 256)
        self.conv3_4_512 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.conv3_5_512 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.conv3_6_512 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.sections.append([self.conv3_4_512, self.conv3_5_512, self.conv3_6_512])

        # 6: bottom transition layer
        self.bottom_transition = nn.Conv2d(256, 256, kernel_size = 1, stride = 1)

        # 7: (80, 45, 256) to (160 x 90 x 128) 
        self.deconv_1 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.skip_connect_1 = nn.Conv2d(256, 128, kernel_size = 1, stride = 1)
        
        # 8: (160 x 90 x 128) to (320 x 180 x 64) 
        self.deconv_2 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.skip_connect_2 = nn.Conv2d(128, 64, kernel_size = 1, stride = 1)

        # 9: (320 x 180 x 64) to (640 x 320 x 32)
        self.deconv_3 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.skip_connect_3 = nn.Conv2d(64, 32, kernel_size = 1, stride = 1)

        self.deconvolutions = [self.deconv_1, self.deconv_2, self.deconv_3]
        self.skip_connections = [self.skip_connect_1, self.skip_connect_2, self.skip_connect_3]

        # 10: (640 x 320 x 32) to (1280 x 720 x 16)
        self.final_deconv = nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)      
        self.classify_layer = nn.Conv2d(16, 3, kernel_size = 1, stride = 1)
        

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
            if index < 3:
                self.section_outputs[index] = x

        # perform final convolution at bottom layer
        x = F.relu(self.bottom_transition(x))

        # upsample
        for index, deconv_layer in enumerate(self.deconvolutions):
            x = F.relu(deconv_layer(x))  # upsample through deconvolution
            x = torch.cat((x, self.section_outputs[-index-1]), dim = 1)  # concatenate skiplayer to channels
            x = F.relu(self.skip_connections[index](x))  # compute upsample given skip-connection info
        
        x = F.relu(self.final_deconv(x))
        x = self.classify_layer(x)

        return nn.LogSoftmax(dim = 1)(x)

    def save(self):
        torch.save(self.state_dict(), self.save_dir)